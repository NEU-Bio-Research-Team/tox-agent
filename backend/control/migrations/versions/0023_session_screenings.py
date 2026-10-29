"""Frozen activity screenings attached to sessions (#44).

* ``session_screenings`` — one row per (session, screening), holding the
  session's own copy of the frozen record from the deployment's replay file.

Additive only: one new table. Nothing writes it unless the
``activity_replay_v1`` flag is on.

Revision ID: 0023_session_screenings
Revises: 0022_evidence_full_text
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import table_exists

revision = "0023_session_screenings"
down_revision = "0022_evidence_full_text"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("session_screenings"):
        op.create_table(
            "session_screenings",
            sa.Column("session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("screening_id", ID, primary_key=True),
            sa.Column("record", JSON, nullable=False),
            sa.Column("record_sha256", sa.String(64), nullable=False),
            sa.Column("source", sa.String(16), nullable=False),
            sa.Column("attached_at", TS, nullable=False),
        )


def downgrade() -> None:
    op.drop_table("session_screenings")
