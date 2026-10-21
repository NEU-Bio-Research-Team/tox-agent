"""Which phase of a run made each tool call (#46/#53).

* ``tool_calls.phase`` -- investigator, workflow_acquisition,
  workflow_synthesis, hard_correction, review_initial, semantic_repair or
  review_final, so a run's calls and cost can be read per phase.

Additive only. NULL on calls made before this revision.

Revision ID: 0029_tool_call_phase
Revises: 0028_run_results
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import column_exists

revision = "0029_tool_call_phase"
down_revision = "0028_run_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not column_exists("tool_calls", "phase"):
        with op.batch_alter_table("tool_calls") as batch:
            batch.add_column(sa.Column("phase", sa.String(24)))


def downgrade() -> None:
    with op.batch_alter_table("tool_calls") as batch:
        batch.drop_column("phase")
