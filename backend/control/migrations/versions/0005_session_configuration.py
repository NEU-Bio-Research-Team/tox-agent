"""Persist user-selected AI and predictor configuration per session/run.

Revision ID: 0005_session_configuration
Revises: 0004_investigation_kernel
"""
from alembic import op
import sqlalchemy as sa

revision = "0005_session_configuration"
down_revision = "0004_investigation_kernel"
branch_labels = None
depends_on = None


def upgrade() -> None:
    ident = sa.String(40)
    ts = sa.DateTime(timezone=True)
    op.create_table(
        "session_settings",
        sa.Column("session_id", ident, sa.ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("ai_profile_id", ident),
        sa.Column("predictor_bindings", sa.JSON(), nullable=False),
        sa.Column("updated_at", ts, nullable=False),
    )
    op.create_table(
        "run_configuration_snapshots",
        sa.Column("run_id", ident, sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("ai_profile_id", ident),
        sa.Column("predictor_bindings", sa.JSON(), nullable=False),
        sa.Column("created_at", ts, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("run_configuration_snapshots")
    op.drop_table("session_settings")
