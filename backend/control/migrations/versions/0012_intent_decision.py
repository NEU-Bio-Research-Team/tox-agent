"""Record which router decided a run's intent, and why.

P1-9 of the 2026-09-13 audit: the router matched its term lists with plain
substring containment, so "executive summary" contained "execute" and a
question about a report section routed to OUT_OF_SCOPE. Nothing in the database
said which rule fired, which meant a routing complaint could not be settled
without re-reading the code as it stood that day.

Additive and nullable. Runs routed before this migration have no decision,
which is the truth about them.

Revision ID: 0012_intent_decision
Revises: 0011_usage_source_identity
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from toxagent.persistence.migration_helpers import column_exists

revision = "0012_intent_decision"
down_revision = "0011_usage_source_identity"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    if not column_exists("run_configuration_snapshots", "intent_decision"):
        op.add_column(
            "run_configuration_snapshots", sa.Column("intent_decision", _JSON)
        )


def downgrade() -> None:
    raise NotImplementedError("0012_intent_decision is forward-only")
