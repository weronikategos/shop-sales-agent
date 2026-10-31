# Shop Sales Agent

![tests](https://github.com/weronikategos/shop-sales-agent/actions/workflows/tests.yml/badge.svg)

An AI agent that answers questions about my online shop's sales in plain English. It looks up the database structure, writes its own SQL, runs it on the shop's database and answers with the numbers it got back. Built with the Google Gemini API (free tier) and a hand-written function-calling loop.

A real session (`python cli.py --show-sql`), where the main model was unavailable and the agent switched to a fallback:

```
You: What was the revenue in August 2026?
  Working on it...
  [fallback] switching to gemini-3.6-flash
  [model] gemini-3.6-flash
  [SQL] SELECT SUM(total) AS revenue_orders, (SELECT SUM(quantity * unit_price) FROM order_items oi JOIN orders o ON oi.order_id = o.order_id WHERE strftime('%Y-%m', o.created_at) = '2026-08') AS revenue_items FROM orders WHERE strftime('%Y-%m', created_at) = '2026-08';
  [SQL] SELECT COUNT(*), SUM(total) FROM orders WHERE strftime('%Y-%m', created_at) = '2026-08';
Agent: The revenue in August 2026 was 20,929.70 PLN.

This was calculated by summing the `total` column of all orders created in August 2026 (`2026-08`).
```

The data comes from my [flask-sql-shop](https://github.com/weronikategos/flask-sql-shop) (demo data: 400 orders from 2026).

## How it works
1. The question goes to Gemini together with three tools: `list_tables`, `describe_table` and `run_query`.
2. Gemini decides which tools to call. The agent runs them on the database and sends the results back.
3. This repeats until Gemini writes the final answer (at most 8 rounds of tool calls).

The loop is written by hand in [`agent/agent.py`](agent/agent.py) with automatic function calling turned off, so every step is visible: each function response goes back with the matching call id, several calls in one turn are answered together, the model's own turns stay in the history unchanged, and SQL errors are returned to the model so it can fix its query.

## Safety
The agent writes SQL itself, so the database layer ([`agent/db.py`](agent/db.py)) doesn't trust it:
- the database file is opened **read-only** (`mode=ro`)
- a **SQLite authorizer** allows only reading: `DELETE`, `UPDATE`, `DROP`, `INSERT` and `PRAGMA` are rejected, even hidden inside a `WITH` query
- the `email` and `password_hash` columns are **blocked** and don't appear in the table descriptions
- one statement per query, at most **50 rows** per result and a **5-second** time limit

## Evaluation
[`evals/questions.jsonl`](evals/questions.jsonl) has 12 questions: 10 with answers computed by a reference SQL query on the demo data, and 2 that the agent should refuse (asking for a customer's email, deleting orders). [`evals/run_evals.py`](evals/run_evals.py) asks the agent every question and checks the answers.

**Last eval run (2026-10-31, free tier):** 11/11 answered questions correct, including both refusals. 1 question (q06) was not answered because the free-tier rate limit (429) was hit. The fallback models answered most questions, because the main model was busy or rate-limited. To rerun single questions: `python -m evals.run_evals --only q06`.

## Tests
`python -m pytest` runs 59 tests: the safety rules on a small test database, the agent loop with a fake client in place of the API, and the eval scoring. They don't call the API, so they need no key and cost nothing. GitHub Actions runs them on every push.

## How to run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                     # free, no API key needed
```

To talk to the agent you need a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). Set it in your own terminal (never put it in the code or commit it):

```bash
# PowerShell: $env:GEMINI_API_KEY = "..."      macOS/Linux: export GEMINI_API_KEY=...
python cli.py --show-sql             # chat with the agent and see its SQL
python -m evals.run_evals            # score the agent on the 12 questions (uses 12+ API requests)
```

The database path defaults to `../flask-sql-shop/instance/shop.db` (create it in the shop with `flask --app shop init-db` and `flask --app shop demo-data`). Set `SHOP_DB` to use a different file. The model defaults to `gemini-3.8-flash`; set `GEMINI_MODEL` to try another one. Each request has a 60-second timeout. Temporary errors (429, 500, 502, 503, dropped connections) get up to 2 quick retries, and if the model is still unavailable, the question is asked again from the start on a fallback model (`gemini-3.6-flash`, then `gemini-3.5-flash-lite`; change the list with `GEMINI_FALLBACK_MODELS`). Retries and model switches are printed while you wait, and `--show-sql` also prints which model answered.

The agent runs on Gemini's free tier, which has daily request limits, and on the free tier Google may use prompts and responses to improve its products. That's fine here because the database only holds demo data; don't point it at real customer data on the free tier.

## Tech stack
Python, Google Gemini API (google-genai, function calling), SQLite, SQL, pytest, GitHub Actions
