"""What a decision-support run ended with (#47).

* ``run_results`` -- one row per run under a declared release policy
  (strict-reviewed-v1, hard-only-v1): the publication disposition (publish,
  partial_facts, withhold), the review status, the execution and knowledge
  outcomes, the policy and the hash of its table, the public answer if any
  and the last candidate (for audit), and the detail a reader needs. It can
  say "nothing was published", which ``answer_releases`` (one row per
  released answer) cannot.
* ``answer_drafts.phase`` -- which submission phase wrote a draft: initial,
  hard_correction or semantic_repair.

Additive only. Nothing writes it unless a strict or hard-only policy is set.

Revision ID: 0028_run_results
Revises: 0027_full_text_provenance
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import column_exists, table_exists

revision = "0028_run_results"
down_revision = "0027_full_text_provenance"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("run_results"):
        op.create_table(
            "run_results",
            sa.Column("run_id", ID, sa.ForeignKey("runs.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("disposition", sa.String(24), nullable=False),
            sa.Column("review_status", sa.String(32), nullable=False),
            sa.Column("execution_outcome", sa.String(32), nullable=False),
            sa.Column("knowledge_outcome", sa.String(32), nullable=False),
            sa.Column("release_policy", sa.String(32), nullable=False),
            sa.Column("release_policy_sha256", sa.String(64), nullable=False),
            sa.Column("public_answer_id", ID),
            sa.Column("candidate_answer_id", ID),
            sa.Column("detail", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
        )
        op.create_index("ix_run_results_public_answer", "run_results", ["public_answer_id"])
    if not column_exists("answer_drafts", "phase"):
        with op.batch_alter_table("answer_drafts") as batch:
            batch.add_column(sa.Column("phase", sa.String(24)))


def downgrade() -> None:
    with op.batch_alter_table("answer_drafts") as batch:
        batch.drop_column("phase")
    op.drop_index("ix_run_results_public_answer", table_name="run_results")
    op.drop_table("run_results")
