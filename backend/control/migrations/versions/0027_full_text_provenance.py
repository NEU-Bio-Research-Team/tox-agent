"""Where a read full text came from (#43).

* ``evidence_full_texts.provenance`` -- the provider that served the text,
  whether it was replayed from a pinned frozen store, and for a replay the
  live capture it reproduces (provider, time, sha256 of the bytes received,
  parser version).

Additive only. NULL on every text read before this revision.

Revision ID: 0027_full_text_provenance
Revises: 0026_run_subject_binding
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import column_exists

revision = "0027_full_text_provenance"
down_revision = "0026_run_subject_binding"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    if not column_exists("evidence_full_texts", "provenance"):
        with op.batch_alter_table("evidence_full_texts") as batch:
            batch.add_column(sa.Column("provenance", JSON))


def downgrade() -> None:
    with op.batch_alter_table("evidence_full_texts") as batch:
        batch.drop_column("provenance")
