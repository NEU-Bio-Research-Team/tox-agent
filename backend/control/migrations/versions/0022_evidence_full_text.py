"""Full text an evidence record was read in, and the passages claims quote (#43).

* ``evidence_full_texts`` — one row per record whose text was asked for and
  definitively answered: read (``full_text_retrieved``) or refused
  (``access_failed``, with the reason).
* ``evidence_passages`` — the located passages of a read text, in document
  order; primary key ``(evidence_id, locator)``.
* ``claim_evidence_spans`` — which passage of a cited record a claim quotes.

Additive only: three new tables and no change to an existing one, so a
rollback of the application code leaves a schema the previous release reads.
Nothing writes these tables unless the ``evidence_full_text_v1`` flag is on.

Revision ID: 0022_evidence_full_text
Revises: 0021_skill_drafts
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0022_evidence_full_text"
down_revision = "0021_skill_drafts"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("evidence_full_texts"):
        op.create_table(
            "evidence_full_texts",
            sa.Column("evidence_id", ID,
                      sa.ForeignKey("evidence_records.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("run_id", ID, sa.ForeignKey("runs.id"), nullable=False),
            sa.Column("state", sa.String(32), nullable=False),
            sa.Column("full_text_id", sa.String(64)),
            sa.Column("reason", sa.Text()),
            sa.Column("section_titles", JSON, nullable=False),
            sa.Column("omitted", JSON, nullable=False),
            sa.Column("truncated", sa.Boolean(), nullable=False),
            sa.Column("document_sha256", sa.String(64), nullable=False),
            sa.Column("fetched_at", TS, nullable=False),
        )
    if not table_exists("evidence_passages"):
        op.create_table(
            "evidence_passages",
            sa.Column("evidence_id", ID,
                      sa.ForeignKey("evidence_full_texts.evidence_id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("locator", sa.String(80), primary_key=True),
            sa.Column("position", sa.Integer(), nullable=False),
            sa.Column("section", sa.Text(), nullable=False),
            sa.Column("kind", sa.String(24), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("truncated", sa.Boolean(), nullable=False),
            sa.Column("cells", JSON, nullable=False),
        )
    if not index_exists("evidence_passages", "ix_evidence_passages_order"):
        op.create_index(
            "ix_evidence_passages_order", "evidence_passages", ["evidence_id", "position"]
        )
    if not table_exists("claim_evidence_spans"):
        op.create_table(
            "claim_evidence_spans",
            sa.Column("claim_id", ID, sa.ForeignKey("claims.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("evidence_id", ID, sa.ForeignKey("evidence_records.id"),
                      primary_key=True),
            sa.Column("locator", sa.String(80), primary_key=True),
            sa.Column("position", sa.Integer(), nullable=False),
        )


def downgrade() -> None:
    op.drop_table("claim_evidence_spans")
    op.drop_index("ix_evidence_passages_order", table_name="evidence_passages")
    op.drop_table("evidence_passages")
    op.drop_table("evidence_full_texts")
