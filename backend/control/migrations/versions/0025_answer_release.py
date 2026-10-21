"""How an answer was released, and which answer a relation belongs to (#47).

* ``answer_releases`` — one row per released answer: the outcome (reviewed,
  repaired_and_reviewed, released_with_unresolved_review, ungated, fallback)
  and its detail (the review rounds, what the repair changed). One per run.
* ``evidence_relation_assessments.answer_id`` — the answer a relation was
  submitted with, so a repaired run's superseded candidate keeps its
  relations without them reading as the released answer's.

Additive only. Nothing writes ``answer_releases`` unless ``release_gate_v1``
is on; ``answer_id`` is NULL on rows written before this revision.

Revision ID: 0025_answer_release
Revises: 0024_answer_lineage
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import column_exists, table_exists

revision = "0025_answer_release"
down_revision = "0024_answer_lineage"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("answer_releases"):
        op.create_table(
            "answer_releases",
            sa.Column("answer_id", ID, sa.ForeignKey("answers.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("run_id", ID, sa.ForeignKey("runs.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("outcome", sa.String(40), nullable=False),
            sa.Column("detail", JSON, nullable=False),
            sa.Column("released_at", TS, nullable=False),
            sa.UniqueConstraint("run_id", name="uq_answer_release_run"),
        )
    if not column_exists("evidence_relation_assessments", "answer_id"):
        with op.batch_alter_table("evidence_relation_assessments") as batch:
            batch.add_column(sa.Column("answer_id", ID))


def downgrade() -> None:
    with op.batch_alter_table("evidence_relation_assessments") as batch:
        batch.drop_column("answer_id")
    op.drop_table("answer_releases")
