"""Scientific cases: a durable investigation across turns (ADR 0012).

* ``scientific_cases`` — the current ScientificCaseV1 snapshot, one row per
  case, revision-checked.
* ``scientific_case_events`` — the append-only log the snapshot is the fold
  of; primary key ``(case_id, revision)``.
* ``scientific_case_dossiers`` — the DecisionDossierV1 a run compiled over its
  case, one per run.

Additive only: three new tables and no change to an existing one, so a
rollback of the application code leaves a schema the previous release reads.
Nothing writes these tables unless the ``scientific_case_v1`` flag is on.

Revision ID: 0020_scientific_cases
Revises: 0019_report_hash_width
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0020_scientific_cases"
down_revision = "0019_report_hash_width"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("scientific_cases"):
        op.create_table(
            "scientific_cases",
            sa.Column("id", ID, primary_key=True),
            sa.Column(
                "session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("subject_key", sa.String(80), nullable=False),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.Column("state", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
            sa.Column("updated_at", TS, nullable=False),
        )
    if not index_exists("scientific_cases", "ix_scientific_cases_session_subject"):
        op.create_index(
            "ix_scientific_cases_session_subject", "scientific_cases",
            ["session_id", "subject_key", "status"],
        )
    if not table_exists("scientific_case_events"):
        op.create_table(
            "scientific_case_events",
            sa.Column(
                "case_id", ID, sa.ForeignKey("scientific_cases.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.Column("op", sa.String(32), nullable=False),
            sa.Column("actor", sa.String(16), nullable=False),
            sa.Column("run_id", ID),
            sa.Column("payload", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
            sa.PrimaryKeyConstraint("case_id", "revision", name="pk_scientific_case_events"),
        )
    if not index_exists("scientific_case_events", "ix_scientific_case_events_run"):
        op.create_index("ix_scientific_case_events_run", "scientific_case_events", ["run_id"])
    if not table_exists("scientific_case_dossiers"):
        op.create_table(
            "scientific_case_dossiers",
            sa.Column(
                "run_id", ID, sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True,
            ),
            sa.Column(
                "case_id", ID, sa.ForeignKey("scientific_cases.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("case_revision", sa.Integer(), nullable=False),
            sa.Column("dossier", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
        )
    if not index_exists("scientific_case_dossiers", "ix_scientific_case_dossiers_case"):
        op.create_index(
            "ix_scientific_case_dossiers_case", "scientific_case_dossiers",
            ["case_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("ix_scientific_case_dossiers_case", table_name="scientific_case_dossiers")
    op.drop_table("scientific_case_dossiers")
    op.drop_index("ix_scientific_case_events_run", table_name="scientific_case_events")
    op.drop_table("scientific_case_events")
    op.drop_index("ix_scientific_cases_session_subject", table_name="scientific_cases")
    op.drop_table("scientific_cases")
