import sqlite3

import pytest

from agent.db import ShopDatabase

SCHEMA = """
CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, category_id INTEGER, name TEXT, price REAL, stock INTEGER);
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, email TEXT, password_hash TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL, created_at TEXT);
CREATE TABLE order_items (order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price REAL);
INSERT INTO categories VALUES (1, 'Coffee'), (2, 'Tea');
INSERT INTO products VALUES (1, 1, 'Coffee A', 40.0, 10), (2, 2, 'Tea B', 25.5, 3);
INSERT INTO customers VALUES (1, 'Ala', 'ala@example.com', 'scrypt:secret');
INSERT INTO orders VALUES (1, 1, 105.5, '2026-08-10 12:00:00'), (2, 1, 40.0, '2026-09-02 09:30:00');
INSERT INTO order_items VALUES (1, 1, 2, 40.0), (1, 2, 1, 25.5), (2, 1, 1, 40.0);
"""


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
    return path


@pytest.fixture
def db(db_path):
    return ShopDatabase(db_path)
