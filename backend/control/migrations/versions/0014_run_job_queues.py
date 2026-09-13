"""Queue classes for run jobs, so a report cannot starve a question (WS08 / PR-14).

Until now every job was claimed by the process that accepted it, and the only
other claimant was a replica adopting an expired lease. With the API and the
workers separated, a job waits in `run_jobs` unowned until a worker takes it,
and which worker may take it depends on what kind of work it is: a 150-second
report and a 15-second answer must not compete for the same slots.

Additive and nullable. A job written before this revision has no `queue_name`;
readers derive one from the intent in its envelope rather than this migration
guessing at JSON inside a portable UPDATE.

Revision ID: 0014_run_job_queues
Revises: 0013_evidence_relevance
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import column_exists, index_exists

revision = "0014_run_job_queues"
down_revision = "0013_evidence_relevance"
branch_labels = None
depends_on = None

_TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    columns = (
        sa.Column("queue_name", sa.String(32)),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column("available_at", _TS),
        sa.Column("last_error_code", sa.String(64)),
    )
    for column in columns:
        if not column_exists("run_jobs", column.name):
            op.add_column("run_jobs", column)
    if not index_exists("run_jobs", "ix_run_jobs_queue"):
        op.create_index(
            "ix_run_jobs_queue", "run_jobs", ["queue_name", "priority", "created_at"]
        )


def downgrade() -> None:
    op.drop_index("ix_run_jobs_queue", table_name="run_jobs")
    for name in ("last_error_code", "available_at", "priority", "queue_name"):
        op.drop_column("run_jobs", name)
