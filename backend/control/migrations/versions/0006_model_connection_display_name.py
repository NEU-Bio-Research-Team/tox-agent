"""Give provider profiles a product-facing display name.

Revision ID: 0006_model_conn_name
Revises: 0005_session_configuration
"""
from alembic import op
import sqlalchemy as sa

revision = "0006_model_conn_name"
down_revision = "0005_session_configuration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("model_connections") as batch:
        batch.add_column(sa.Column("display_name", sa.String(160), nullable=False, server_default=""))


def downgrade() -> None:
    with op.batch_alter_table("model_connections") as batch:
        batch.drop_column("display_name")
