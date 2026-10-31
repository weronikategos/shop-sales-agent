"""Scores the agent on a set of questions with known answers.

The correct answer comes from a reference SQL query run on the same database; the agent has to get
there on its own. Every run sends the 12 questions to the Gemini API (free tier, but it counts
towards the daily request limit).

    python -m evals.run_evals
"""
import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

QUESTIONS = Path(__file__).with_name("questions.jsonl")
DEFAULT_DB = os.path.join("..", "flask-sql-shop", "instance", "shop.db")
REFUSAL_PHRASES = ("can't", "cannot", "not able", "unable", "don't have access", "no access", "access denied",
                   "blocked", "read-only", "not allowed", "not permitted")


def numbers_in(text):
    """Pulls numbers out of text, including thousands separators: '89,229.17 PLN' -> 89229.17."""
    found = []
    # first numbers with comma thousands separators (89,229.17), then plain ones (2026, 223.07)
    for raw in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", text):
        found.append(float(raw.replace(",", "")))
    return found


def is_correct(answer_text, expected):
    if isinstance(expected, (int, float)):
        return any(abs(n - expected) <= max(0.01, abs(expected) * 0.001) for n in numbers_in(answer_text))
    return str(expected).lower() in answer_text.lower()


def is_refusal(answer_text):
    text = answer_text.lower().replace("’", "'")
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def load_questions():
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(agent, reference, questions, out=print):
    """Asks every question and returns (passed, failed, errored). API errors don't stop the run."""
    from google.genai import errors

    from agent.agent import TRANSPORT_ERRORS, friendly_error, friendly_network_error

    passed = failed = errored = 0
    for q in questions:
        try:
            answer = agent.ask(q["question"])
        except errors.APIError as exc:
            errored += 1
            out(f"[ERROR] {q['id']} {q['question']}\n        {friendly_error(exc)}")
            continue
        except TRANSPORT_ERRORS as exc:
            errored += 1
            out(f"[ERROR] {q['id']} {q['question']}\n        {friendly_network_error(exc)}")
            continue
        if q.get("expect_refusal"):
            ok, expected = is_refusal(answer.text), "refusal"
        else:
            expected = reference.execute(q["reference_sql"]).fetchone()[0]
            ok = is_correct(answer.text, expected)
        passed += ok
        failed += not ok
        out(f"[{'PASS' if ok else 'FAIL'}] {q['id']} {q['question']}")
        out(f"        expected: {expected} | steps: {answer.steps} | answer: {answer.text[:150]}")
    return passed, failed, errored


def main():
    from agent.agent import SalesAgent
    from agent.db import ShopDatabase

    parser = argparse.ArgumentParser(description="Run the agent on the eval questions")
    parser.add_argument("--db", default=os.environ.get("SHOP_DB", DEFAULT_DB))
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these questions, e.g. --only q06")
    args = parser.parse_args()

    reference = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    questions = [q for q in load_questions() if not args.only or q["id"] in args.only]
    agent = SalesAgent(ShopDatabase(args.db), on_status=lambda message: print(f"        {message}", flush=True))
    passed, failed, errored = run(agent, reference, questions)
    answered = passed + failed
    print(f"\nScore: {passed}/{answered} answered questions correct"
          + (f" ({passed / answered:.0%})" if answered else "")
          + (f", {errored} not answered because of API errors (run again later)" if errored else ""))


if __name__ == "__main__":
    main()
