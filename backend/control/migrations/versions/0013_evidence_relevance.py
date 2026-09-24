"""Record why a piece of evidence is, or is not, about this question.

P1-2 of the 2026-09-13 audit: a search for ethanol and hERG persisted five
durable `accepted` records about asthma, cannabinoids, breast cancer,
remdesivir and neurocardiology. `accepted` meant the provider payload parsed
and its host was on the allowlist — a statement about bytes, sharing one
durable state with a statement about subject matter.

Additive and nullable. Historical records are not rewritten and not re-hashed:
their assessment is empty, which is true of them. A new run that wants to cite
one needs a fresh assessment rather than the benefit of the doubt.

Revision ID: 0013_evidence_relevance
Revises: 0012_intent_decision
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from toxagent.persistence.migration_helpers import column_exists

revision = "0013_evidence_relevance"
down_revision = "0012_intent_decision"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    if not column_exists("evidence_records", "relevance_assessment"):
        op.add_column("evidence_records", sa.Column("relevance_assessment", _JSON))


def downgrade() -> None:
    raise NotImplementedError("0013_evidence_relevance is forward-only")
