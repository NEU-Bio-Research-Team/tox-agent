"""Add title lifecycle fields without rewriting existing sessions.

Revision ID: 0003_session_titles
Revises: 0002_runtime_usage_events
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import missing_columns

revision = "0003_session_titles"
down_revision = "0002_runtime_usage_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 creates the baseline from live metadata, so on a fresh database
    # these columns already exist; on a database created at 0001 before this
    # revision was written they do not. See migration_helpers' module
    # docstring — this used to fail outright on an empty PostgreSQL database.
    pending = missing_columns("sessions", [
        sa.Column("title_source", sa.String(24), nullable=True),
        sa.Column("title_status", sa.String(24), nullable=False, server_default="pending"),
        sa.Column("title_updated_at", sa.DateTime(timezone=True), nullable=True),
    ])
    if not pending:
        return
    with op.batch_alter_table("sessions") as batch:
        for column in pending:
            batch.add_column(column)


def downgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.drop_column("title_updated_at")
        batch.drop_column("title_status")
        batch.drop_column("title_source")
