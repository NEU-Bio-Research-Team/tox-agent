"""Durable run execution records, owned by a lease.

Before this, execution input lived only in the accepting process's memory and
ownership was inferred from "no process knows about it", which made a second
replica fail the first replica's live runs (I17) and made a `kill -9`
unrecoverable (I18).

Revision ID: 0007_run_job_leases
Revises: 0006_model_conn_name
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from toxagent.persistence.migration_helpers import index_exists, table_exists

revision = "0007_run_job_leases"
down_revision = "0006_model_conn_name"
branch_labels = None
depends_on = None

_ID = sa.String(40)
_TS = sa.DateTime(timezone=True)
_JSON = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    # Guarded for the same reason as 0003-0006: on an empty database
    # `0001_baseline` has already created the current metadata, this table
    # included, so this revision must find its own work done.
    if not table_exists("run_jobs"):
        op.create_table(
            "run_jobs",
            sa.Column("run_id", _ID, sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("envelope", _JSON, nullable=False),
            sa.Column("worker_id", sa.String(64)),
            sa.Column("lease_expires_at", _TS),
            sa.Column("lease_epoch", sa.Integer, nullable=False, server_default="0"),
            sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
            sa.Column("created_at", _TS, nullable=False),
            sa.Column("updated_at", _TS, nullable=False),
        )
    if not index_exists("run_jobs", "ix_run_jobs_claimable"):
        op.create_index("ix_run_jobs_claimable", "run_jobs", ["lease_expires_at"])
    # Nothing backfills runs that were in flight across this upgrade: they have
    # no envelope, so they are exactly the case startup reconciliation still
    # closes out. Inventing one would mean guessing at a request.


def downgrade() -> None:
    op.drop_index("ix_run_jobs_claimable", table_name="run_jobs")
    op.drop_table("run_jobs")
