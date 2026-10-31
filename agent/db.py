"""Safe database access for the agent: read-only, SELECT only, no personal data."""
import sqlite3
import time

MAX_ROWS = 50
TIMEOUT_SECONDS = 5
# columns the agent must never read (personal data and password hashes)
BLOCKED_COLUMNS = {("customers", "email"), ("customers", "password_hash")}
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


class QueryError(Exception):
    """An error the agent gets back as a tool result, so it can fix its query."""


def _authorizer(action, arg1, arg2, _db_name, _trigger):
    # SQLite asks for permission on every operation. For SQLITE_READ, arg1 is the table and arg2 the column.
    if action not in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and (arg1, arg2) in BLOCKED_COLUMNS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class ShopDatabase:
    def __init__(self, path):
        # mode=ro: even if something got past the authorizer, the file is opened read-only
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.set_authorizer(_authorizer)

    def _metadata(self, sql, params=()):
        """Runs a schema lookup with the authorizer switched off (PRAGMA is blocked for the agent)."""
        self.conn.set_authorizer(None)
        try:
            return self.conn.execute(sql, params).fetchall()
        finally:
            self.conn.set_authorizer(_authorizer)

    def list_tables(self):
        rows = self._metadata(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [r["name"] for r in rows]

    def describe_table(self, table):
        """Column names and types of one table, without the blocked columns."""
        if table not in self.list_tables():
            raise QueryError(f"There is no table called '{table}'. Use list_tables to see the tables.")
        columns = self._metadata("SELECT name, type FROM pragma_table_info(?)", (table,))
        return [{"name": c["name"], "type": c["type"]} for c in columns
                if (table, c["name"]) not in BLOCKED_COLUMNS]

    def run(self, sql):
        """Runs one SELECT query and returns the column names and at most MAX_ROWS rows."""
        sql = sql.strip().rstrip(";")
        if not sql.lower().startswith(("select", "with")):
            raise QueryError("Only SELECT queries are allowed.")
        if ";" in sql:
            raise QueryError("Send one query at a time.")
        deadline = time.monotonic() + TIMEOUT_SECONDS
        self.conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)  # stop slow queries
        try:
            cursor = self.conn.execute(sql)
            rows = cursor.fetchmany(MAX_ROWS + 1)
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if "not authorized" in message or "prohibited" in message:
                message = "Access denied: the query touches blocked data (emails, passwords) or tries to change the database."
            elif "interrupted" in message:
                message = f"The query took longer than {TIMEOUT_SECONDS} s."
            raise QueryError(message) from exc
        finally:
            self.conn.set_progress_handler(None, 0)
        columns = [d[0] for d in cursor.description]
        return {"columns": columns, "rows": [list(r) for r in rows[:MAX_ROWS]], "truncated": len(rows) > MAX_ROWS}
