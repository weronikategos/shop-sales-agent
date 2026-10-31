import os
import sqlite3

import pytest

from evals.run_evals import DEFAULT_DB, is_correct, is_refusal, load_questions, numbers_in


@pytest.mark.parametrize("text, expected", [
    ("Total revenue is 89,229.17 PLN.", 89229.17),
    ("Total revenue is 89229.17 PLN", 89229.17),
    ("In August 2026: 20,929.70 PLN", 20929.70),
    ("400 orders were placed", 400),
])
def test_numbers_in_text(text, expected):
    assert expected in numbers_in(text)


def test_year_not_glued_to_amount():
    assert numbers_in("August 2026 20,929.70 PLN") == [2026, 20929.70]


def test_is_correct():
    assert is_correct("The average order value is 223.07 PLN", 223.07)
    assert not is_correct("The average is 230 PLN", 223.07)
    assert is_correct("Accessories brought in the most revenue.", "Accessories")


def test_is_refusal():
    assert is_refusal("I don't have access to customers' email addresses.")
    assert is_refusal("I can’t delete data, the database is read-only.")
    assert not is_refusal("Revenue was 100 PLN.")


def test_questions_file_is_valid():
    questions = load_questions()
    assert len(questions) == 12
    assert len({q["id"] for q in questions}) == len(questions)
    assert all("reference_sql" in q or q.get("expect_refusal") for q in questions)


def test_reference_queries_run_on_shop_schema(db_path):
    conn = sqlite3.connect(db_path)
    for q in load_questions():
        if "reference_sql" in q:
            conn.execute(q["reference_sql"]).fetchone()


def test_expected_values_match_demo_database():
    """The expected values in questions.jsonl were computed on the shop's demo data."""
    path = os.environ.get("SHOP_DB", DEFAULT_DB)
    if not os.path.exists(path):
        pytest.skip("shop demo database not available")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    for q in load_questions():
        if "reference_sql" in q:
            assert conn.execute(q["reference_sql"]).fetchone()[0] == q["expected"], q["id"]
