"""An AI agent that answers questions about the shop's sales by writing its own SQL queries."""
import json
import os
import time
from dataclasses import dataclass, field

import httpx
from google import genai
from google.genai import errors, types

from .db import QueryError, ShopDatabase

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# tried in order when the main model is still overloaded after the retries (both free tier, with function calling)
FALLBACK_MODELS = [m.strip() for m in os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite").split(",")
                   if m.strip()]
MAX_STEPS = 8  # how many rounds of tool calls the agent gets before we stop it

SYSTEM_PROMPT = """You are a data analyst for an online shop that sells coffee, tea, accessories and sweets.
You answer questions about sales using the shop's SQLite database, which you can only reach through your tools.

- Look up the tables you need with list_tables and describe_table before writing a query.
- Take every number in your answer from a query result. Don't guess or estimate.
- Dates are stored as text in the format 'YYYY-MM-DD HH:MM:SS'. Group by month with strftime('%Y-%m', column).
- Revenue is the sum of quantity * unit_price in order_items (or the sum of total in orders).
- Amounts are in PLN. Write them with two decimals, e.g. 1,234.56 PLN.
- Answer in English, briefly: the result first, then one sentence on how you got it.
- If a question asks for data you can't access or asks you to change the database, say so plainly."""

FUNCTIONS = [
    types.FunctionDeclaration(
        name="list_tables",
        description="Returns the names of all tables in the shop database.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    types.FunctionDeclaration(
        name="describe_table",
        description="Returns the column names and types of one table. Blocked columns (emails, passwords) are not listed.",
        parameters_json_schema={
            "type": "object",
            "properties": {"table": {"type": "string", "description": "table name, e.g. orders"}},
            "required": ["table"],
        },
    ),
    types.FunctionDeclaration(
        name="run_query",
        description=("Runs one SELECT query on the shop's SQLite database and returns at most 50 rows as JSON. "
                     "The database is read-only and the email and password_hash columns are blocked."),
        parameters_json_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "a single SQLite SELECT query"}},
            "required": ["query"],
        },
    ),
]

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=FUNCTIONS)],
    # we run the tools ourselves, so the SDK must not call Python functions on its own
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    max_output_tokens=8192,
)

BLOCKED_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

# errors that make the agent try the next model: overloaded, rate-limited, or not available to this key
FALLBACK_CODES = (404, 429, 500, 502, 503, 504)

RETRY_CODES = (429, 500, 502, 503)  # worth another try on the same model (504 already waited, so it moves on)
RETRY_DELAYS = (2, 4)  # seconds to wait before the 2nd and 3rd attempt
REQUEST_TIMEOUT_MS = 60_000

# network problems (connection reset, timeout, dropped connection)
try:
    import httpx2
    TRANSPORT_ERRORS = (httpx.TransportError, httpx2.TransportError)
    TIMEOUT_ERRORS = (httpx.TimeoutException, httpx2.TimeoutException)
except ImportError:
    TRANSPORT_ERRORS = (httpx.TransportError,)
    TIMEOUT_ERRORS = (httpx.TimeoutException,)


def make_client():
    """Gemini client that reads the API key from GEMINI_API_KEY. Retries happen in SalesAgent, where they can be logged."""
    return genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def friendly_error(exc: errors.APIError) -> str:
    """One readable line for an API error that is still there after the retries."""
    if exc.code in (502, 503, 504):
        return f"The model is busy right now ({exc.code}). Please try again in a minute."
    if exc.code == 429:
        return "The free-tier request limit is used up for now (429). Please try again later."
    if exc.code in (401, 403):
        return f"The API key was rejected ({exc.code}). Check GEMINI_API_KEY."
    return f"The Gemini API returned an error ({exc.code}): {exc.message}"


def friendly_network_error(_exc) -> str:
    return "Connection to the Gemini API was interrupted. Check your internet/VPN and try again."


@dataclass
class AgentAnswer:
    text: str
    queries: list = field(default_factory=list)  # SQL the agent ran, to show the user
    steps: int = 0
    model: str = ""  # the model that gave the answer


def _finish_reason(response):
    reason = response.candidates[0].finish_reason if response.candidates else None
    return getattr(reason, "name", reason)


class SalesAgent:
    def __init__(self, db: ShopDatabase, client=None, model=MODEL, fallback_models=None, on_status=None):
        self.db = db
        self.client = client or make_client()
        self.on_status = on_status or (lambda _message: None)  # told about retries and fallbacks while waiting
        self.models = [model] + [m for m in (FALLBACK_MODELS if fallback_models is None else fallback_models)
                                 if m != model]

    def run_tool(self, name, args, answer):
        """Runs one tool and returns its result as a dict (errors go under "error")."""
        try:
            if name == "list_tables":
                return {"tables": self.db.list_tables()}
            if name == "describe_table":
                return {"columns": self.db.describe_table(args["table"])}
            if name == "run_query":
                answer.queries.append(args["query"])
                return json.loads(json.dumps(self.db.run(args["query"]), default=str))
        except QueryError as exc:
            return {"error": f"Query error: {exc}"}
        return {"error": f"Unknown tool: {name}"}

    def ask(self, question):
        """Answers with the main model; if it's still unavailable after the retries, starts over on a fallback."""
        for i, model in enumerate(self.models):
            try:
                return self._ask_with(model, question)
            except (errors.APIError, *TRANSPORT_ERRORS) as exc:
                # a fresh start keeps the history consistent: thought signatures belong to one model
                can_fall_back = not isinstance(exc, errors.APIError) or exc.code in FALLBACK_CODES
                if not can_fall_back or i == len(self.models) - 1:
                    raise
                self.on_status(f"[fallback] switching to {self.models[i + 1]}")

    def _generate(self, model, contents):
        """One model call, with a few quick retries when the model is busy or the connection drops."""
        for delay in (*RETRY_DELAYS, None):
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=CONFIG)
            except TIMEOUT_ERRORS:
                raise  # it already waited a full minute, so move on to the next model
            except errors.APIError as exc:
                if exc.code not in RETRY_CODES or delay is None:
                    raise
                problem = f"busy ({exc.code})"
            except TRANSPORT_ERRORS:
                if delay is None:
                    raise
                problem = "connection dropped"
            self.on_status(f"[retry] {model}: {problem}, retrying in {delay} s...")
            time.sleep(delay)

    def _ask_with(self, model, question):
        contents = [types.Content(role="user", parts=[types.Part(text=question)])]
        answer = AgentAnswer(text="", model=model)
        while answer.steps < MAX_STEPS:
            response = self._generate(model, contents)
            reason = _finish_reason(response)
            if not response.candidates or reason in BLOCKED_REASONS:
                answer.text = "The model declined to answer this question."
                return answer
            if reason == "MAX_TOKENS":
                answer.text = "The answer got too long and was cut off. Try a narrower question."
                return answer
            # the model turn goes back into the history unchanged (it carries the thought signatures)
            contents.append(response.candidates[0].content)
            calls = response.function_calls or []
            if not calls:
                answer.text = (response.text or "").strip()
                return answer

            answer.steps += 1
            parts = []
            for call in calls:  # the model can ask for several tools at once
                result = self.run_tool(call.name, call.args or {}, answer)
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=call.id, name=call.name, response=result)))
            contents.append(types.Content(role="user", parts=parts))  # all results in one turn

        answer.text = f"Stopped after {MAX_STEPS} steps without an answer."
        return answer
