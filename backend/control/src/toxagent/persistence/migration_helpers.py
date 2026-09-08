"""Making the migration chain deterministic from an empty database.

`0001_baseline` creates the schema by calling `metadata.create_all()` on the
*current* ORM metadata rather than by writing out frozen DDL. That is a
deliberate choice — one definition of the schema — but it has a consequence
its author documented in `0002` and nobody applied afterwards: on a fresh
database, revision 0001 already produces the head schema, so every later
revision must tolerate finding its own work done.

0003 through 0006 did not, and `alembic upgrade head` on an empty PostgreSQL
database failed at `ALTER TABLE sessions ADD COLUMN title_source` — meaning a
brand-new deployment could not start, since `deploy/entrypoint.sh` migrates
before binding a port. Every local suite passed because SQLite's
`batch_alter_table` recreates the table instead of altering it.

These helpers make "already present" a no-op and nothing else. They do not
compare types or make a column match a definition: a column that exists with
the wrong shape is drift, and drift must fail loudly rather than be papered
over here.
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect


def _inspector():
    return inspect(op.get_bind())


def table_exists(table: str) -> bool:
    return table in _inspector().get_table_names()


def column_exists(table: str, column: str) -> bool:
    if not table_exists(table):
        return False
    return any(c["name"] == column for c in _inspector().get_columns(table))


def index_exists(table: str, index: str) -> bool:
    if not table_exists(table):
        return False
    return any(i["name"] == index for i in _inspector().get_indexes(table))


def missing_columns(table: str, columns) -> list:
    """The subset of `columns` this database does not have yet.

    Takes SQLAlchemy `Column` objects and returns them, so a caller can pass
    the result straight to `batch.add_column` without restating names.
    """
    if not table_exists(table):
        return list(columns)
    present = {c["name"] for c in _inspector().get_columns(table)}
    return [column for column in columns if column.name not in present]
