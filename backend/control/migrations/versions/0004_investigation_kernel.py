"""Product-owned investigation state and model connections.

Revision ID: 0004_investigation_kernel
Revises: 0003_session_titles
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import (
    index_exists,
    missing_columns,
    table_exists,
)

revision = "0004_investigation_kernel"
down_revision = "0003_session_titles"
branch_labels = None
depends_on = None

JSON = sa.JSON()
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    # Every statement below is guarded: 0001 builds the baseline from live
    # metadata, so a fresh database arrives here with this revision's work
    # already done, while a database created at 0001 before this revision
    # existed does not. See toxagent/persistence/migration_helpers.py.
    pending = missing_columns("runtime_bindings", [
        sa.Column("auth_mode", sa.String(32), nullable=False, server_default="none"),
        sa.Column("connection_id", ID, nullable=True),
    ])
    if pending:
        with op.batch_alter_table("runtime_bindings") as batch:
            for column in pending:
                batch.add_column(column)
    if not table_exists("cases"):
        op.create_table(
            "cases", sa.Column("id", ID, primary_key=True),
            sa.Column("session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
            sa.Column("goal", sa.String(64), nullable=False), sa.Column("subject", JSON, nullable=False),
            sa.Column("active_analysis_id", ID, sa.ForeignKey("analysis_snapshots.id")),
            sa.Column("active_plan_id", ID), sa.Column("state", JSON, nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False), sa.Column("revision_reason", sa.Text(), nullable=False),
            sa.Column("created_at", TS, nullable=False), sa.Column("updated_at", TS, nullable=False),
            sa.UniqueConstraint("session_id", "id", name="uq_cases_session_id"),
        )
    if not index_exists("cases", "ix_cases_session_updated"):
        op.create_index("ix_cases_session_updated", "cases", ["session_id", "updated_at"])
    if not table_exists("case_revisions"):
        op.create_table(
            "case_revisions", sa.Column("case_id", ID, sa.ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("revision", sa.Integer(), primary_key=True), sa.Column("reason", sa.Text(), nullable=False),
            sa.Column("state", JSON, nullable=False), sa.Column("created_at", TS, nullable=False),
        )
    if not table_exists("investigation_plans"):
        op.create_table(
            "investigation_plans", sa.Column("id", ID, primary_key=True),
            sa.Column("case_id", ID, sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False), sa.Column("reason", sa.Text(), nullable=False),
            sa.Column("created_at", TS, nullable=False),
            sa.UniqueConstraint("case_id", "revision", name="uq_plan_case_revision"),
        )
    if not table_exists("investigation_steps"):
        op.create_table(
            "investigation_steps", sa.Column("id", ID, primary_key=True),
            sa.Column("plan_id", ID, sa.ForeignKey("investigation_plans.id", ondelete="CASCADE"), nullable=False),
            sa.Column("position", sa.Integer(), nullable=False), sa.Column("question", sa.Text(), nullable=False),
            sa.Column("capability", sa.String(64), nullable=False), sa.Column("input_refs", JSON, nullable=False),
            sa.Column("expected_output", sa.Text(), nullable=False), sa.Column("success_condition", sa.Text(), nullable=False),
            sa.Column("case_revision", sa.Integer(), nullable=False), sa.Column("status", sa.String(24), nullable=False),
            sa.Column("output_refs", JSON, nullable=False), sa.Column("failure_reason", sa.Text()),
            sa.UniqueConstraint("plan_id", "position", name="uq_plan_step_position"),
        )
    if not table_exists("kernel_transitions"):
        op.create_table(
            "kernel_transitions", sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("case_id", ID, sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
            sa.Column("state", sa.String(32), nullable=False), sa.Column("detail", sa.Text(), nullable=False, server_default=""),
            sa.Column("occurred_at", TS, nullable=False),
        )
    if not index_exists("kernel_transitions", "ix_kernel_transition_case"):
        op.create_index("ix_kernel_transition_case", "kernel_transitions", ["case_id", "id"])
    if not table_exists("model_connections"):
        op.create_table(
            "model_connections", sa.Column("id", ID, primary_key=True), sa.Column("owner_id", sa.String(255), nullable=False),
            sa.Column("provider_id", sa.String(128), nullable=False), sa.Column("model_id", sa.String(128), nullable=False),
            sa.Column("base_url", sa.Text()), sa.Column("auth_mode", sa.String(32), nullable=False),
            sa.Column("credential_ref", sa.String(255)), sa.Column("capabilities", JSON, nullable=False),
            sa.Column("status", sa.String(24), nullable=False), sa.Column("created_at", TS, nullable=False),
            sa.Column("updated_at", TS, nullable=False),
        )
    if not index_exists("model_connections", "ix_model_connections_owner"):
        op.create_index("ix_model_connections_owner", "model_connections", ["owner_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_model_connections_owner", table_name="model_connections")
    op.drop_table("model_connections")
    op.drop_index("ix_kernel_transition_case", table_name="kernel_transitions")
    op.drop_table("kernel_transitions")
    op.drop_table("investigation_steps")
    op.drop_table("investigation_plans")
    op.drop_table("case_revisions")
    op.drop_index("ix_cases_session_updated", table_name="cases")
    op.drop_table("cases")
    with op.batch_alter_table("runtime_bindings") as batch:
        batch.drop_column("connection_id")
        batch.drop_column("auth_mode")
