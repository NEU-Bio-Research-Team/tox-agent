"""Skill drafts awaiting expert review (RETHINK §4.8, W9-11).

* ``skill_drafts`` — one row per proposed skill package, with its author,
  source run, status and review. The draft document is the row's ``draft``
  column; ``skill_id``, ``status`` and ``author_subject`` are there to list by.

Additive only: one new table. Nothing writes it unless the ``skill_drafts_v1``
flag is on, and nothing in the catalog a run is offered ever reads it.

Revision ID: 0021_skill_drafts
Revises: 0020_scientific_cases
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0021_skill_drafts"
down_revision = "0020_scientific_cases"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("skill_drafts"):
        op.create_table(
            "skill_drafts",
            sa.Column("id", ID, primary_key=True),
            sa.Column("skill_id", sa.String(64), nullable=False),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("author_subject", sa.String(200), nullable=False),
            sa.Column("draft", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
            sa.Column("updated_at", TS, nullable=False),
        )
    if not index_exists("skill_drafts", "ix_skill_drafts_status"):
        op.create_index("ix_skill_drafts_status", "skill_drafts", ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_skill_drafts_status", table_name="skill_drafts")
    op.drop_table("skill_drafts")
