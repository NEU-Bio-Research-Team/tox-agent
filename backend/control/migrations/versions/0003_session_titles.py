"""Add title lifecycle fields without rewriting existing sessions.

Revision ID: 0003_session_titles
Revises: 0002_runtime_usage_events
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_session_titles"
down_revision = "0002_runtime_usage_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.add_column(sa.Column("title_source", sa.String(24), nullable=True))
        batch.add_column(sa.Column("title_status", sa.String(24), nullable=False, server_default="pending"))
        batch.add_column(sa.Column("title_updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.drop_column("title_updated_at")
        batch.drop_column("title_status")
        batch.drop_column("title_source")
