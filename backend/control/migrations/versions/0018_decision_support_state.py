"""Decision-support state and relation lineage (TAB-Suite Wave 2).

* ``decision_support_states`` — DecisionSupportStateV1, one row per
  decision_support run (domain/decision_state.py). New table.
* ``run_configuration_snapshots.effective_budget`` — the EffectiveRunBudgetV1
  a run was admitted under (P1-06).
* ``evidence_relation_assessments`` gains three nullable columns:
  ``input_refs`` (an agent_synthesis relation's resolved inputs, P1-03),
  ``assessor`` and ``method_version`` (the canonical ontology's provenance,
  P1-02). Existing rows are not rewritten; they dual-read as a model
  assessment under grounded-answer-v2, which is what they were.

Additive only: nothing is dropped or narrowed, and a rollback of the
application code leaves a schema the previous release still reads.

Revision ID: 0018_decision_support_state
Revises: 0017_development_postures
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import column_exists, index_exists, table_exists

revision = "0018_decision_support_state"
down_revision = "0017_development_postures"
branch_labels = None
depends_on = None

JSON = sa.JSON()
ID = sa.String(40)
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    if not table_exists("decision_support_states"):
        op.create_table(
            "decision_support_states",
            sa.Column("run_id", ID, sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
            sa.Column(
                "session_id", ID, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.Column("stop_reason", sa.String(32)),
            sa.Column("state", JSON, nullable=False),
            sa.Column("created_at", TS, nullable=False),
            sa.Column("updated_at", TS, nullable=False),
        )
    if not index_exists("decision_support_states", "ix_decision_support_states_session"):
        op.create_index(
            "ix_decision_support_states_session", "decision_support_states",
            ["session_id", "updated_at"],
        )
    for name, column_type in (
        ("input_refs", JSON), ("assessor", sa.String(16)), ("method_version", sa.String(40)),
    ):
        if not column_exists("evidence_relation_assessments", name):
            op.add_column("evidence_relation_assessments", sa.Column(name, column_type))
    if not column_exists("run_configuration_snapshots", "effective_budget"):
        op.add_column("run_configuration_snapshots", sa.Column("effective_budget", JSON))


def downgrade() -> None:
    op.drop_column("run_configuration_snapshots", "effective_budget")
    for name in ("method_version", "assessor", "input_refs"):
        op.drop_column("evidence_relation_assessments", name)
    op.drop_index("ix_decision_support_states_session", table_name="decision_support_states")
    op.drop_table("decision_support_states")
