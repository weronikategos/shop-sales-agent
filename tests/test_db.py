import pytest

from agent.db import MAX_ROWS, QueryError


def test_select_works(db):
    result = db.run("SELECT ROUND(SUM(total), 2) AS revenue FROM orders")
    assert result == {"columns": ["revenue"], "rows": [[145.5]], "truncated": False}


def test_with_query_allowed(db):
    result = db.run("WITH m AS (SELECT strftime('%Y-%m', created_at) AS month, total FROM orders) "
                    "SELECT month, SUM(total) FROM m GROUP BY month ORDER BY month")
    assert result["rows"] == [["2026-08", 105.5], ["2026-09", 40.0]]


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE products SET stock = 0",
    "DROP TABLE orders",
    "INSERT INTO categories VALUES (3, 'X')",
    "PRAGMA table_info(customers)",
])
def test_only_select_allowed(db, sql):
    with pytest.raises(QueryError):
        db.run(sql)


def test_cannot_hide_write_in_second_statement(db):
    with pytest.raises(QueryError, match="one query"):
        db.run("SELECT 1; DELETE FROM orders")


def test_cannot_write_through_cte(db):
    with pytest.raises(QueryError):
        db.run("WITH x AS (SELECT 1) DELETE FROM orders")
    assert db.run("SELECT COUNT(*) FROM orders")["rows"] == [[2]]


@pytest.mark.parametrize("sql", [
    "SELECT email FROM customers",
    "SELECT * FROM customers",
    "SELECT password_hash FROM customers WHERE customer_id = 1",
    "SELECT o.order_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.email LIKE '%a%'",
])
def test_personal_data_blocked(db, sql):
    with pytest.raises(QueryError, match="Access denied"):
        db.run(sql)


def test_customer_name_still_available(db):
    assert db.run("SELECT name FROM customers")["rows"] == [["Ala"]]


def test_list_tables(db):
    assert db.list_tables() == ["categories", "customers", "order_items", "orders", "products"]


def test_describe_table_hides_blocked_columns(db):
    assert db.describe_table("customers") == [{"name": "customer_id", "type": "INTEGER"},
                                              {"name": "name", "type": "TEXT"}]


def test_describe_unknown_table(db):
    with pytest.raises(QueryError, match="no table"):
        db.describe_table("clients")


def test_rows_are_limited(db):
    result = db.run("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 200) SELECT x FROM n")
    assert len(result["rows"]) == MAX_ROWS
    assert result["truncated"] is True


def test_sql_error_message_returned(db):
    with pytest.raises(QueryError, match="no such table"):
        db.run("SELECT * FROM clients")
