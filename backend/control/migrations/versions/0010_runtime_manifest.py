"""Record what a run's runtime was actually asked to be.

P0-1 of the 2026-09-13 audit: the run manifest claimed a 64-step report budget
while the adapter dispatched the 32-step Q&A agent. Nothing in the database
could have caught that, because the binding recorded neither the agent name nor
the cap the runtime really enforces.

Additive and nullable. Bindings written before this migration keep a NULL
manifest, which reads as "unknown" — the honest value, and not something a
backfill could invent.

Revision ID: 0010_runtime_manifest
Revises: 0009_report_builder
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from toxagent.persistence.migration_helpers import column_exists

revision = "0010_runtime_manifest"
down_revision = "0009_report_builder"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    # Guarded like 0003-0009: on an empty database `0001_baseline` created the
    # current metadata, this column included.
    if not column_exists("runtime_bindings", "runtime_manifest"):
        op.add_column("runtime_bindings", sa.Column("runtime_manifest", _JSON))


def downgrade() -> None:
    # Schema migration is forward-only (plan section 9). A rollback drops the
    # binary, not the column: an old reader ignores a field it does not know.
    raise NotImplementedError("0010_runtime_manifest is forward-only")
