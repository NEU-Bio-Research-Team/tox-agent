"""The model a session runs on, chosen in the chat (not in Settings).

A connection is a credential: an API key or a ChatGPT sign-in for one
provider. Which of that provider's models a run uses is the user's choice per
session, made in the composer's model picker, so it lives next to the
connection id in ``session_settings`` and is copied into each run's immutable
``run_configuration_snapshots`` row.

Additive only. NULL means "the connection's own model", which is what every
session and run before this revision used.

Revision ID: 0030_session_ai_model
Revises: 0029_tool_call_phase
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import column_exists

revision = "0030_session_ai_model"
down_revision = "0029_tool_call_phase"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("session_settings", "run_configuration_snapshots"):
        if not column_exists(table, "ai_model_id"):
            with op.batch_alter_table(table) as batch:
                batch.add_column(sa.Column("ai_model_id", sa.String(200)))


def downgrade() -> None:
    for table in ("run_configuration_snapshots", "session_settings"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("ai_model_id")
