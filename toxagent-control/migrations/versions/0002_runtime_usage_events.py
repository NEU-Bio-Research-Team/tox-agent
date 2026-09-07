"""Persist normalized runtime usage reports.

Revision ID: 0002_runtime_usage_events
Revises: 0001_baseline
Create Date: 2026-09-06
"""
from __future__ import annotations

from alembic import op

from toxagent.persistence.schema import runtime_usage_events

revision = "0002_runtime_usage_events"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 intentionally creates the metadata baseline. On a fresh database
    # it may already include this table; on an existing 0001 database it will
    # not. ``checkfirst`` makes both upgrade paths correct.
    runtime_usage_events.create(op.get_bind(), checkfirst=True)


def downgrade() -> None:
    runtime_usage_events.drop(op.get_bind(), checkfirst=True)
