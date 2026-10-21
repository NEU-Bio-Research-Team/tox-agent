"""The subject a run is bound to at admission (#44).

* ``run_configuration_snapshots.subject_binding`` -- the RunSubjectBinding
  (run-subject-binding-v1) the run was admitted with: the subject's kind,
  reference and identity, and the artifacts it was given with their hashes.
  Tools read it to default to the run's subject instead of the session's
  current selection.

Additive only. NULL on every run admitted before this revision and on every
run admitted with ``subject_binding_v1`` off.

Revision ID: 0026_run_subject_binding
Revises: 0025_answer_release
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from toxagent.persistence.migration_helpers import column_exists

revision = "0026_run_subject_binding"
down_revision = "0025_answer_release"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    if not column_exists("run_configuration_snapshots", "subject_binding"):
        with op.batch_alter_table("run_configuration_snapshots") as batch:
            batch.add_column(sa.Column("subject_binding", JSON))


def downgrade() -> None:
    with op.batch_alter_table("run_configuration_snapshots") as batch:
        batch.drop_column("subject_binding")
