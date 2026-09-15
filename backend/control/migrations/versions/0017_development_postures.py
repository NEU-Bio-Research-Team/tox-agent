"""Development posture rows for decision support (ADS plan section 10.1/10.2,
ADR 0010, W6).

One row per answer that carried a development posture (proceed/hold/
deprioritize/insufficient/not_applicable), keyed by ``answer_id`` — the
relationship is 1:1, since ``domain/development_posture.py``'s
``DevelopmentPosture`` is a value object with no identity of its own.

New table; nothing existing is altered.

Revision ID: 0017_development_postures
Revises: 0016_evidence_relations

See 0016_evidence_relations.py's own note on why revision ids here are kept
short: alembic_version.version_num is character varying(32).
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0017_development_postures"
down_revision = "0016_evidence_relations"
branch_labels = None
depends_on = None

JSON = sa.JSON()
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    # 0001 builds the baseline from live metadata, so a fresh database already
    # has this table; a database created before this revision does not. See
    # toxagent/persistence/migration_helpers.py.
    if not table_exists("development_postures"):
        op.create_table(
            "development_postures",
            sa.Column(
                "answer_id", ID,
                sa.ForeignKey("answers.id", ondelete="CASCADE"), primary_key=True,
            ),
            sa.Column(
                "session_id", ID,
                sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("run_id", ID, sa.ForeignKey("runs.id"), nullable=False),
            sa.Column("value", sa.String(24), nullable=False),
            sa.Column("scope", sa.String(24), nullable=False),
            sa.Column("confidence_band", sa.String(16), nullable=False),
            sa.Column("basis_claim_ids", JSON, nullable=False),
            sa.Column("contrary_claim_ids", JSON, nullable=False),
            sa.Column("rationale", sa.Text(), nullable=False),
            sa.Column("conditions", JSON, nullable=False),
            sa.Column("recommended_next_steps", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
        )
    if not index_exists("development_postures", "ix_development_postures_run"):
        op.create_index(
            "ix_development_postures_run", "development_postures", ["run_id"],
        )


def downgrade() -> None:
    op.drop_index("ix_development_postures_run", table_name="development_postures")
    op.drop_table("development_postures")
