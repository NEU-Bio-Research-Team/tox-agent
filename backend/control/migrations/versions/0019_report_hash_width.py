"""Widen report content_sha256 columns to fit prefixed digests.

The v3 report compiler (report/compiler_v3.py) and the synthesis stage
publish ``sha256:<64 hex>`` — 71 characters — into ``varchar(64)`` columns.
SQLite does not enforce varchar length, so every test passed; PostgreSQL
refuses the insert, and every orchestrated report build failed at the
rendering stage with StringDataRightTruncationError. Found by the first live
TAB-Suite report batch (2026-09-17).

Widening a varchar is a metadata-only change in PostgreSQL and loses nothing.

Revision ID: 0019_report_hash_width
Revises: 0018_decision_support_state
"""
from alembic import op
import sqlalchemy as sa

revision = "0019_report_hash_width"
down_revision = "0018_decision_support_state"
branch_labels = None
depends_on = None

TABLES = ("report_artifacts", "report_figures", "report_renderings")


def upgrade() -> None:
    for table in TABLES:
        with op.batch_alter_table(table) as batch:
            batch.alter_column(
                "content_sha256", type_=sa.String(80),
                existing_type=sa.String(64), existing_nullable=False,
            )


def downgrade() -> None:
    for table in TABLES:
        with op.batch_alter_table(table) as batch:
            batch.alter_column(
                "content_sha256", type_=sa.String(64),
                existing_type=sa.String(80), existing_nullable=False,
            )
