"""Give a usage report an identity, so it can only be stored once.

P1-1 of the 2026-09-13 audit: the same cumulative snapshot was persisted three
times per assistant message, because a row recorded the numbers and nothing
about where they came from. Summing the column overstated a run's cost by
roughly a factor of three.

Additive and nullable throughout. Historical rows keep ``semantics='unknown'``
and ``is_normalized=false``: their semantics were never recorded, and inventing
one would make a fabricated number look like a measured one. They stay readable
and stay out of every total.

The unique index is partial for the same reason — without the predicate, every
pre-existing row would collide on a single NULL key.

Revision ID: 0011_usage_source_identity
Revises: 0010_runtime_manifest
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import column_exists, index_exists

revision = "0011_usage_source_identity"
down_revision = "0010_runtime_manifest"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("source_event_id", sa.String(64)),
    ("source_event_type", sa.String(64)),
    ("provider_message_id", sa.String(128)),
    ("provider_step_id", sa.String(128)),
    ("revision", sa.Integer),
    ("raw_payload_hash", sa.String(64)),
)


def upgrade() -> None:
    # Guarded like 0003-0010: on an empty database `0001_baseline` created the
    # current metadata, these columns included.
    for name, type_ in _COLUMNS:
        if not column_exists("runtime_usage_events", name):
            op.add_column("runtime_usage_events", sa.Column(name, type_))
    if not column_exists("runtime_usage_events", "semantics"):
        op.add_column(
            "runtime_usage_events",
            sa.Column("semantics", sa.String(16), nullable=False, server_default="unknown"),
        )
    if not column_exists("runtime_usage_events", "is_normalized"):
        op.add_column(
            "runtime_usage_events",
            sa.Column("is_normalized", sa.Boolean, nullable=False, server_default="0"),
        )
    if not index_exists("runtime_usage_events", "uq_runtime_usage_source"):
        op.create_index(
            "uq_runtime_usage_source",
            "runtime_usage_events",
            ["runtime_binding_id", "source_event_id"],
            unique=True,
            sqlite_where=sa.text("source_event_id IS NOT NULL"),
            postgresql_where=sa.text("source_event_id IS NOT NULL"),
        )


def downgrade() -> None:
    raise NotImplementedError("0011_usage_source_identity is forward-only")
