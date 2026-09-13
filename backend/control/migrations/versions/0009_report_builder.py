"""Report builder tables.

The report is a first-class artifact, not a rendering of a chat answer
(docs/spec/TOXAGENT_REPORT_BUILDER_PLAN.md sections 5.2 and 12.1): it is
immutable, versioned, cited and downloaded, so it needs rows of its own rather
than living inside a run transcript.

Bytes stay in the object store throughout. These tables hold ownership,
provenance, content hashes and object refs.

Revision ID: 0009_report_builder
Revises: 0008_explanation_ckpt
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0009_report_builder"
down_revision = "0008_explanation_ckpt"
branch_labels = None
depends_on = None

_ID = sa.String(40)
_TS = sa.DateTime(timezone=True)
_JSON = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    # Guarded like 0003-0008: on an empty database `0001_baseline` already
    # created the current metadata, these tables included.
    if not table_exists("report_builds"):
        op.create_table(
            "report_builds",
            sa.Column("id", _ID, primary_key=True),
            sa.Column(
                "session_id", _ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("run_id", _ID, sa.ForeignKey("runs.id"), nullable=False),
            sa.Column(
                "analysis_id", _ID, sa.ForeignKey("analysis_snapshots.id"), nullable=False
            ),
            sa.Column("request", _JSON, nullable=False),
            sa.Column("stage", sa.String(32), nullable=False),
            sa.Column("report_id", _ID),
            sa.Column("correction_attempts", sa.Integer, nullable=False, server_default="0"),
            sa.Column("failure_code", sa.String(64)),
            sa.Column("failure_detail", sa.Text),
            sa.Column("stage_state", _JSON, nullable=False),
            sa.Column("deadline_at", _TS),
            sa.Column("created_at", _TS, nullable=False),
            sa.Column("updated_at", _TS, nullable=False),
        )
    if not index_exists("report_builds", "ix_report_builds_session"):
        op.create_index("ix_report_builds_session", "report_builds", ["session_id", "created_at"])

    if not table_exists("report_artifacts"):
        op.create_table(
            "report_artifacts",
            sa.Column("id", _ID, primary_key=True),
            sa.Column(
                "report_build_id", _ID, sa.ForeignKey("report_builds.id"), nullable=False
            ),
            sa.Column(
                "session_id", _ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "analysis_id", _ID, sa.ForeignKey("analysis_snapshots.id"), nullable=False
            ),
            sa.Column("schema_version", sa.String(32), nullable=False),
            sa.Column("title", sa.Text, nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("report_language", sa.String(8), nullable=False),
            sa.Column("document", _JSON, nullable=False),
            sa.Column("content_sha256", sa.String(64), nullable=False),
            sa.Column(
                "supersedes_report_id", _ID, sa.ForeignKey("report_artifacts.id")
            ),
            sa.Column("version", sa.Integer, nullable=False, server_default="1"),
            sa.Column("created_at", _TS, nullable=False),
        )
    if not index_exists("report_artifacts", "ix_report_artifacts_session"):
        op.create_index(
            "ix_report_artifacts_session", "report_artifacts", ["session_id", "created_at"]
        )
    if not index_exists("report_artifacts", "ix_report_artifacts_build"):
        op.create_index("ix_report_artifacts_build", "report_artifacts", ["report_build_id"])

    if not table_exists("report_figures"):
        op.create_table(
            "report_figures",
            sa.Column("figure_id", _ID, primary_key=True),
            sa.Column(
                "session_id", _ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "attachment_id", _ID, sa.ForeignKey("attachments.id"), nullable=False
            ),
            sa.Column("observation_id", _ID, sa.ForeignKey("observations.id")),
            sa.Column("endpoint", sa.String(32)),
            sa.Column("task", sa.String(64)),
            sa.Column("media_type", sa.String(64), nullable=False),
            sa.Column("caption", sa.Text, nullable=False),
            sa.Column("alt_text", sa.Text, nullable=False),
            sa.Column("content_sha256", sa.String(64), nullable=False),
            sa.Column("renderer_version", sa.String(64), nullable=False),
            sa.Column("created_at", _TS, nullable=False),
        )
    if not index_exists("report_figures", "ix_report_figures_session"):
        op.create_index("ix_report_figures_session", "report_figures", ["session_id", "created_at"])

    if not table_exists("report_renderings"):
        op.create_table(
            "report_renderings",
            sa.Column("id", _ID, primary_key=True),
            sa.Column(
                "report_id", _ID,
                sa.ForeignKey("report_artifacts.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("format", sa.String(16), nullable=False),
            sa.Column("media_type", sa.String(64), nullable=False),
            sa.Column("object_uri", sa.Text, nullable=False),
            sa.Column("content_sha256", sa.String(64), nullable=False),
            sa.Column("size_bytes", sa.Integer, nullable=False),
            sa.Column("renderer_version", sa.String(64), nullable=False),
            sa.Column("created_at", _TS, nullable=False),
            sa.UniqueConstraint("report_id", "format", name="uq_report_rendering_format"),
        )

    if not table_exists("report_claim_links"):
        op.create_table(
            "report_claim_links",
            sa.Column(
                "report_id", _ID,
                sa.ForeignKey("report_artifacts.id", ondelete="CASCADE"), primary_key=True,
            ),
            sa.Column("claim_id", _ID, primary_key=True),
            sa.Column("section_id", sa.String(64), nullable=False),
            sa.Column("kind", sa.String(24), nullable=False),
            sa.Column("source_class", sa.String(24), nullable=False),
            sa.Column("observation_id", _ID),
            sa.Column("field_path", sa.Text),
        )
    if not index_exists("report_claim_links", "ix_report_claim_links_observation"):
        op.create_index(
            "ix_report_claim_links_observation", "report_claim_links", ["observation_id"]
        )

    if not table_exists("report_evidence_links"):
        op.create_table(
            "report_evidence_links",
            sa.Column(
                "report_id", _ID,
                sa.ForeignKey("report_artifacts.id", ondelete="CASCADE"), primary_key=True,
            ),
            sa.Column("evidence_id", _ID, primary_key=True),
            sa.Column("relation", sa.String(24), nullable=False),
            sa.Column("section_id", sa.String(64), nullable=False),
        )
    if not index_exists("report_evidence_links", "ix_report_evidence_links_evidence"):
        op.create_index(
            "ix_report_evidence_links_evidence", "report_evidence_links", ["evidence_id"]
        )


def downgrade() -> None:
    op.drop_index("ix_report_evidence_links_evidence", table_name="report_evidence_links")
    op.drop_table("report_evidence_links")
    op.drop_index("ix_report_claim_links_observation", table_name="report_claim_links")
    op.drop_table("report_claim_links")
    op.drop_table("report_renderings")
    op.drop_index("ix_report_figures_session", table_name="report_figures")
    op.drop_table("report_figures")
    op.drop_index("ix_report_artifacts_build", table_name="report_artifacts")
    op.drop_index("ix_report_artifacts_session", table_name="report_artifacts")
    op.drop_table("report_artifacts")
    op.drop_index("ix_report_builds_session", table_name="report_builds")
    op.drop_table("report_builds")
