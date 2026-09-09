"""Per-target explanation checkpoints.

A bundle computed predict plus every requested explanation before committing
anything, so a crash between targets threw away all the completed ones and the
retry paid for them again (I19).

Revision ID: 0008_explanation_ckpt
Revises: 0007_run_job_leases
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0008_explanation_ckpt"
down_revision = "0007_run_job_leases"
branch_labels = None
depends_on = None

_ID = sa.String(40)
_TS = sa.DateTime(timezone=True)
_JSON = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    # Guarded like 0003-0007: on an empty database `0001_baseline` already
    # created the current metadata, this table included.
    if not table_exists("explanation_checkpoints"):
        op.create_table(
            "explanation_checkpoints",
            sa.Column("key", sa.String(72), primary_key=True),
            sa.Column(
                "session_id", _ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
            ),
            sa.Column("endpoint", sa.String(32), nullable=False),
            sa.Column("task", sa.String(64)),
            sa.Column("model_id", sa.String(128)),
            sa.Column("canonical_smiles", sa.Text, nullable=False),
            sa.Column("payload", _JSON, nullable=False),
            sa.Column("created_at", _TS, nullable=False),
        )
    if not index_exists("explanation_checkpoints", "ix_explanation_checkpoints_session"):
        op.create_index(
            "ix_explanation_checkpoints_session",
            "explanation_checkpoints",
            ["session_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index(
        "ix_explanation_checkpoints_session", table_name="explanation_checkpoints"
    )
    op.drop_table("explanation_checkpoints")
