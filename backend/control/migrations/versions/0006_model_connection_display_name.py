"""Give provider profiles a product-facing display name.

Revision ID: 0006_model_conn_name
Revises: 0005_session_configuration
"""
from alembic import op
import sqlalchemy as sa

from toxagent.persistence.migration_helpers import column_exists

revision = "0006_model_conn_name"
down_revision = "0005_session_configuration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guarded for the same reason as 0003/0004/0005.
    if column_exists("model_connections", "display_name"):
        return
    with op.batch_alter_table("model_connections") as batch:
        batch.add_column(sa.Column("display_name", sa.String(160), nullable=False, server_default=""))


def downgrade() -> None:
    with op.batch_alter_table("model_connections") as batch:
        batch.drop_column("display_name")
