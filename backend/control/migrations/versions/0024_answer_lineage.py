"""Answer lineage: the drafts a run submitted, and each claim's label (#47).

* ``answer_drafts`` — every submission of a run, as the model sent it, with
  its outcome (accepted / rejected / fallback) and violations; primary key
  ``(run_id, generation)``.
* ``claims.local_ref`` — the grounded-answer v2 label, which ties a claim in a
  repaired answer to the same claim before the repair (the server issues a
  new claim id every time).

Additive only: one new table and one nullable column. Rows written before
this have no local_ref, and a repair comparison falls back to claim ids.

Revision ID: 0024_answer_lineage
Revises: 0023_session_screenings
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import column_exists, table_exists

revision = "0024_answer_lineage"
down_revision = "0023_session_screenings"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("answer_drafts"):
        op.create_table(
            "answer_drafts",
            sa.Column("run_id", ID, sa.ForeignKey("runs.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("generation", sa.Integer(), primary_key=True),
            sa.Column("session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("schema_version", sa.String(32), nullable=False),
            sa.Column("payload", JSON, nullable=False),
            sa.Column("outcome", sa.String(16), nullable=False),
            sa.Column("answer_id", ID),
            sa.Column("violations", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
        )
    if not column_exists("claims", "local_ref"):
        with op.batch_alter_table("claims") as batch:
            batch.add_column(sa.Column("local_ref", sa.String(32)))


def downgrade() -> None:
    with op.batch_alter_table("claims") as batch:
        batch.drop_column("local_ref")
    op.drop_table("answer_drafts")
