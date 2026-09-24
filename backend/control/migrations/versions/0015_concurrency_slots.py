"""Concurrency slots held in the database, not in a process (WS08 / PR-15).

P1-7: the only cap on concurrent work was `max_concurrent_runs_per_session`,
checked at admission. Nothing bounded runs across sessions, per tenant, or per
model provider, and a process-local semaphore cannot: with N workers it is N
caps, and a crashed worker's permits vanish with it.

A slot is a row keyed by (scope, scope_key, slot_index) with a lease. Taking
one is an insert that does nothing on conflict, or a conditional update of a
row whose lease expired — so two workers racing for the last slot produce one
holder, and a dead worker's slot becomes available when its lease does.

Revision ID: 0015_concurrency_slots
Revises: 0014_run_job_queues
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0015_concurrency_slots"
down_revision = "0014_run_job_queues"
branch_labels = None
depends_on = None

_TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("concurrency_slots"):
        op.create_table(
            "concurrency_slots",
            sa.Column("scope", sa.String(16), nullable=False),
            sa.Column("scope_key", sa.String(128), nullable=False),
            sa.Column("slot_index", sa.Integer, nullable=False),
            sa.Column("run_id", sa.String(40), nullable=False),
            sa.Column("worker_id", sa.String(64), nullable=False),
            sa.Column("expires_at", _TS, nullable=False),
            sa.Column("acquired_at", _TS, nullable=False),
            sa.PrimaryKeyConstraint("scope", "scope_key", "slot_index"),
        )
    if not index_exists("concurrency_slots", "ix_concurrency_slots_run"):
        op.create_index("ix_concurrency_slots_run", "concurrency_slots", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_concurrency_slots_run", table_name="concurrency_slots")
    op.drop_table("concurrency_slots")
