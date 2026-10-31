"""Tests for the agent loop without the real API: FakeClient plays the model with prepared responses."""
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from agent.agent import MAX_STEPS, SalesAgent, friendly_error, make_client


def call(name, args, call_id):
    return SimpleNamespace(name=name, args=args, id=call_id)


def response(*calls, text=None, finish_reason="STOP"):
    content = SimpleNamespace(role="model", parts=[])
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason), content=content)
    return SimpleNamespace(candidates=[candidate], function_calls=list(calls) or None, text=text)


class FakeClient:
    """Stands in for client.models.generate_content and records what the agent sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        self.requests.append({**kwargs, "contents": list(kwargs["contents"])})
        return self.responses.pop(0)


def last_tool_results(client):
    """The function responses the agent sent in its last request."""
    return [part.function_response for part in client.requests[-1]["contents"][-1].parts]


def test_agent_uses_tools_and_answers(db):
    client = FakeClient([
        response(call("list_tables", {}, "t1")),
        response(call("describe_table", {"table": "orders"}, "t2")),
        response(call("run_query", {"query": "SELECT ROUND(SUM(total), 2) FROM orders"}, "t3")),
        response(text="Total revenue is 145.50 PLN (the sum of total in the orders table)."),
    ])
    answer = SalesAgent(db, client=client).ask("What is the total revenue?")

    assert answer.text.startswith("Total revenue is 145.50 PLN")
    assert answer.queries == ["SELECT ROUND(SUM(total), 2) FROM orders"]
    assert answer.steps == 3
    # the SQL result went back to the model as a function response with the matching id
    result = last_tool_results(client)[0]
    assert (result.id, result.name) == ("t3", "run_query")
    assert result.response["rows"] == [[145.5]]


def test_list_and_describe_tools(db):
    client = FakeClient([
        response(call("list_tables", {}, "a"), call("describe_table", {"table": "customers"}, "b")),
        response(text="Done."),
    ])
    SalesAgent(db, client=client).ask("What's in the database?")
    tables, columns = last_tool_results(client)
    assert "orders" in tables.response["tables"]
    assert [c["name"] for c in columns.response["columns"]] == ["customer_id", "name"]


def test_unknown_table_is_an_error(db):
    client = FakeClient([response(call("describe_table", {"table": "clients"}, "t1")), response(text="No such table.")])
    SalesAgent(db, client=client).ask("?")
    assert "error" in last_tool_results(client)[0].response


def test_sql_error_goes_back_to_model(db):
    client = FakeClient([
        response(call("run_query", {"query": "DELETE FROM orders"}, "t1")),
        response(text="I can't change data, I only have read-only access."),
    ])
    answer = SalesAgent(db, client=client).ask("Delete the orders")
    assert "SELECT" in last_tool_results(client)[0].response["error"]
    assert "read-only" in answer.text


def test_parallel_tool_calls_answered_in_one_message(db):
    client = FakeClient([
        response(call("run_query", {"query": "SELECT COUNT(*) FROM orders"}, "a"),
                 call("run_query", {"query": "SELECT COUNT(*) FROM products"}, "b")),
        response(text="2 orders and 2 products."),
    ])
    SalesAgent(db, client=client).ask("How many orders and products?")
    assert [r.id for r in last_tool_results(client)] == ["a", "b"]


def test_model_turn_kept_in_history(db):
    first = response(call("list_tables", {}, "t1"))
    client = FakeClient([first, response(text="OK")])
    SalesAgent(db, client=client).ask("?")
    assert client.requests[-1]["contents"][1] is first.candidates[0].content


def test_agent_stops_after_max_steps(db):
    endless = [response(call("list_tables", {}, f"t{i}")) for i in range(MAX_STEPS + 1)]
    answer = SalesAgent(db, client=FakeClient(endless)).ask("?")
    assert answer.steps == MAX_STEPS
    assert "Stopped" in answer.text


def test_blocked_response_is_handled(db):
    answer = SalesAgent(db, client=FakeClient([response(finish_reason="SAFETY")])).ask("?")
    assert "declined" in answer.text


def test_request_settings(db):
    client = FakeClient([response(text="OK")])
    SalesAgent(db, client=client, model="gemini-3.8-flash").ask("Hi")
    request = client.requests[0]
    assert request["model"] == "gemini-3.8-flash"
    config = request["config"]
    assert {f.name for f in config.tools[0].function_declarations} == {"list_tables", "describe_table", "run_query"}
    assert config.automatic_function_calling.disable is True


def api_error(code):
    return errors.ServerError(code, {"error": {"code": code, "message": "busy", "status": "UNAVAILABLE"}})


class FlakyClient(FakeClient):
    """Fails with the given error on the first `times` calls, then answers normally."""

    def __init__(self, responses, error, times=1):
        super().__init__(responses)
        self.error = error
        self.times = times

    def generate_content(self, **kwargs):
        if self.times:
            self.times -= 1
            raise self.error
        return super().generate_content(**kwargs)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr("agent.agent.time.sleep", lambda _: None)


def test_busy_model_is_retried_and_reported(db):
    status = []
    client = FlakyClient([response(text="OK")], api_error(503))
    answer = SalesAgent(db, client=client, model="main-model", fallback_models=[], on_status=status.append).ask("?")
    assert (answer.text, answer.model) == ("OK", "main-model")
    assert status == ["[retry] main-model: busy (503), retrying in 2 s..."]


def test_fallback_is_reported(db):
    status = []
    client = FlakyClient([response(text="OK")], api_error(503), times=3)
    answer = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a"],
                        on_status=status.append).ask("?")
    assert answer.model == "backup-a"
    assert status[-1] == "[fallback] switching to backup-a"
    assert len(status) == 3  # two retries, then the switch


def test_timeout_moves_straight_to_fallback(db):
    client = FlakyClient([response(text="OK")], httpx.ReadTimeout("too slow"))
    answer = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a"]).ask("?")
    assert answer.model == "backup-a"
    assert client.times == 0


def test_session_survives_an_api_error(db):
    agent = SalesAgent(db, client=FlakyClient([response(text="2 orders.")], api_error(503), times=3), fallback_models=[])
    with pytest.raises(errors.APIError) as exc:
        agent.ask("How many orders?")
    assert "busy" in friendly_error(exc.value)
    assert agent.ask("How many orders?").text == "2 orders."  # the next question works


def test_evals_count_api_errors_separately(db, db_path):
    from evals.run_evals import run

    questions = [
        {"id": "a", "question": "How many orders?", "reference_sql": "SELECT COUNT(*) FROM orders"},
        {"id": "b", "question": "How many orders again?", "reference_sql": "SELECT COUNT(*) FROM orders"},
    ]
    agent = SalesAgent(db, client=FlakyClient([response(text="There are 2 orders.")], api_error(503), times=3),
                       fallback_models=[])
    assert run(agent, sqlite3.connect(db_path), questions, out=lambda _: None) == (1, 0, 1)


def test_real_client_has_a_timeout_and_no_hidden_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")  # building the client makes no request
    options = make_client()._api_client._http_options
    assert options.timeout == 60_000
    assert options.retry_options is None


class OverloadedModelClient(FakeClient):
    """The listed models always answer 503; the others use the prepared responses."""

    def __init__(self, responses, overloaded):
        super().__init__(responses)
        self.overloaded = set(overloaded)

    def generate_content(self, **kwargs):
        if kwargs["model"] in self.overloaded:
            self.requests.append({**kwargs, "contents": list(kwargs["contents"])})
            raise api_error(503)
        return super().generate_content(**kwargs)


def test_falls_back_when_main_model_is_overloaded(db):
    client = OverloadedModelClient([
        response(call("run_query", {"query": "SELECT COUNT(*) FROM orders"}, "t1")),
        response(text="There are 2 orders."),
    ], overloaded={"main-model"})
    agent = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a", "backup-b"])
    answer = agent.ask("How many orders?")
    assert (answer.text, answer.model) == ("There are 2 orders.", "backup-a")
    assert [r["model"] for r in client.requests] == ["main-model"] * 3 + ["backup-a", "backup-a"]
    assert answer.queries == ["SELECT COUNT(*) FROM orders"]


def test_error_raised_when_every_model_is_overloaded(db):
    client = OverloadedModelClient([], overloaded={"main-model", "backup-a"})
    agent = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a"])
    with pytest.raises(errors.APIError):
        agent.ask("?")


def test_other_errors_are_not_retried_on_fallback(db):
    client = FlakyClient([response(text="never used")], errors.ClientError(
        400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}))
    agent = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a"])
    with pytest.raises(errors.ClientError):
        agent.ask("?")
    assert client.requests == []  # the fallback was not tried



def test_model_not_available_moves_to_fallback(db):
    client = FlakyClient([response(text="OK")], errors.ClientError(
        404, {"error": {"code": 404, "message": "model not found", "status": "NOT_FOUND"}}))
    answer = SalesAgent(db, client=client, model="old-model", fallback_models=["backup-a"]).ask("?")
    assert (answer.text, answer.model) == ("OK", "backup-a")


class DroppingClient(FakeClient):
    """The connection drops on the first `drops` calls, then the prepared responses come back."""

    def __init__(self, responses, drops):
        super().__init__(responses)
        self.drops = drops

    def generate_content(self, **kwargs):
        if self.drops:
            self.drops -= 1
            raise httpx.ReadError("[WinError 10054] connection reset")
        return super().generate_content(**kwargs)


def test_dropped_connection_is_retried(db):
    answer = SalesAgent(db, client=DroppingClient([response(text="OK")], drops=2), fallback_models=[]).ask("?")
    assert answer.text == "OK"


def test_connection_that_keeps_dropping_moves_to_fallback(db):
    agent = SalesAgent(db, client=DroppingClient([response(text="OK")], drops=3), model="main-model",
                       fallback_models=["backup-a"])
    assert agent.ask("?").model == "backup-a"


def test_network_error_raised_when_nothing_works(db):
    agent = SalesAgent(db, client=DroppingClient([], drops=99), fallback_models=[])
    with pytest.raises(httpx.TransportError):
        agent.ask("?")


def test_evals_count_network_errors(db, db_path):
    from evals.run_evals import run

    questions = [{"id": "a", "question": "How many orders?", "reference_sql": "SELECT COUNT(*) FROM orders"},
                 {"id": "b", "question": "And again?", "reference_sql": "SELECT COUNT(*) FROM orders"}]
    agent = SalesAgent(db, client=DroppingClient([response(text="2 orders")], drops=3), fallback_models=[])
    assert run(agent, sqlite3.connect(db_path), questions, out=lambda _: None) == (1, 0, 1)


def test_server_deadline_moves_straight_to_fallback(db):
    status = []
    client = FlakyClient([response(text="OK")], api_error(504))
    answer = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a"],
                        on_status=status.append).ask("?")
    assert answer.model == "backup-a"
    assert status == ["[fallback] switching to backup-a"]


def test_busy_message_names_the_status_code():
    assert friendly_error(api_error(504)) == "The model is busy right now (504). Please try again in a minute."


class ScriptedErrorsClient(FakeClient):
    """Raises the listed errors in order on the main model; other models use the prepared responses."""

    def __init__(self, responses, main_model, script):
        super().__init__(responses)
        self.main_model = main_model
        self.script = list(script)

    def generate_content(self, **kwargs):
        if kwargs["model"] == self.main_model:
            raise self.script.pop(0)
        return super().generate_content(**kwargs)


def test_503_503_504_on_main_model_ends_on_fallback(db):
    status = []
    client = ScriptedErrorsClient([response(text="OK")], "main-model", [api_error(503), api_error(503), api_error(504)])
    answer = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a"],
                        on_status=status.append).ask("?")
    assert (answer.text, answer.model) == ("OK", "backup-a")
    assert status == ["[retry] main-model: busy (503), retrying in 2 s...",
                      "[retry] main-model: busy (503), retrying in 4 s...",
                      "[fallback] switching to backup-a"]


@pytest.mark.parametrize("code", [429, 500, 502, 503])
def test_every_transient_status_falls_back_after_retries(db, code):
    client = ScriptedErrorsClient([response(text="OK")], "main-model", [api_error(code)] * 3)
    answer = SalesAgent(db, client=client, model="main-model", fallback_models=["backup-a"]).ask("?")
    assert answer.model == "backup-a"
    assert client.script == []  # all three attempts were used first
