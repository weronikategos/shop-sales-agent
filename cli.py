"""Chat with the sales agent in the terminal: python cli.py --show-sql"""
import argparse
import os

from google.genai import errors

from agent.agent import TRANSPORT_ERRORS, SalesAgent, friendly_error, friendly_network_error
from agent.db import ShopDatabase

DEFAULT_DB = os.path.join("..", "flask-sql-shop", "instance", "shop.db")


def main():
    parser = argparse.ArgumentParser(description="AI agent for the shop's sales data")
    parser.add_argument("--db", default=os.environ.get("SHOP_DB", DEFAULT_DB), help="path to the shop's shop.db")
    parser.add_argument("--show-sql", action="store_true", help="print the SQL queries the agent wrote")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"Database not found: {args.db}. Set SHOP_DB or pass --db.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set the GEMINI_API_KEY environment variable first (see README).")
    agent = SalesAgent(ShopDatabase(args.db), on_status=lambda message: print(f"  {message}", flush=True))
    print("Ask a question about sales (empty line to quit). Example: What was the revenue in August 2026?\n")
    while True:
        question = input("You: ").strip()
        if not question:
            break
        print("  Working on it...", flush=True)
        try:
            answer = agent.ask(question)
        except KeyboardInterrupt:
            print("  Cancelled.\n")
            continue
        except errors.APIError as exc:  # still failing after the retries
            print(f"Agent: {friendly_error(exc)}\n")
            continue
        except TRANSPORT_ERRORS as exc:  # the connection dropped and didn't come back
            print(f"Agent: {friendly_network_error(exc)}\n")
            continue
        if args.show_sql:
            print(f"  [model] {answer.model}")
            for query in answer.queries:
                print(f"  [SQL] {' '.join(query.split())}")
        print(f"Agent: {answer.text}\n")


if __name__ == "__main__":
    main()
