"""Structured evidence-relation assessments for decision support (ADS plan
section 9.3, ADR 0010).

One row per source-vs-proposition assessment on a decision_support run:
``relation`` (supports/contradicts/contextual/insufficient/not_applicable),
``directness``, ``applicability``, ``strength`` and reason codes — never a
single pseudo-precise score. Distinct from ``report_evidence_links``, which
is the report-build capability's own, narrower relation link and is
untouched by this migration.

New table; nothing existing is altered.

Revision ID: 0016_evidence_relation_assessments
Revises: 0015_concurrency_slots
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0016_evidence_relation_assessments"
down_revision = "0015_concurrency_slots"
branch_labels = None
depends_on = None

JSON = sa.JSON()
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    # 0001 builds the baseline from live metadata, so a fresh database already
    # has this table; a database created before this revision does not. See
    # toxagent/persistence/migration_helpers.py.
    if not table_exists("evidence_relation_assessments"):
        op.create_table(
            "evidence_relation_assessments",
            sa.Column("id", ID, primary_key=True),
            sa.Column(
                "session_id", ID,
                sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("run_id", ID, sa.ForeignKey("runs.id"), nullable=False),
            sa.Column("proposition_id", ID, nullable=False),
            sa.Column("source_class", sa.String(32), nullable=False),
            sa.Column("source_id", ID, nullable=False),
            sa.Column("relation", sa.String(24), nullable=False),
            sa.Column("directness", sa.String(16), nullable=False),
            sa.Column("applicability", sa.String(16), nullable=False),
            sa.Column("strength", sa.String(16), nullable=False),
            sa.Column("reason_codes", JSON, nullable=False),
            sa.Column("scope", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
        )
    if not index_exists("evidence_relation_assessments", "ix_evidence_relation_run"):
        op.create_index(
            "ix_evidence_relation_run", "evidence_relation_assessments", ["run_id"],
        )
    if not index_exists(
        "evidence_relation_assessments", "ix_evidence_relation_proposition"
    ):
        op.create_index(
            "ix_evidence_relation_proposition", "evidence_relation_assessments",
            ["session_id", "proposition_id"],
        )


def downgrade() -> None:
    op.drop_index(
        "ix_evidence_relation_proposition",
        table_name="evidence_relation_assessments",
    )
    op.drop_index(
        "ix_evidence_relation_run", table_name="evidence_relation_assessments"
    )
    op.drop_table("evidence_relation_assessments")
