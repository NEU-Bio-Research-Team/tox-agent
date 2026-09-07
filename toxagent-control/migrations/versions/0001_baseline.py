"""Baseline schema.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-09-04

The baseline creates the tables from ``toxagent.persistence.schema``. Writing it
this way rather than as three hundred lines of generated ``op.create_table``
keeps a single definition of the schema; from here on, every revision is
handwritten and reviewed against an autogenerate diff.
"""
from __future__ import annotations

from alembic import op

from toxagent.persistence.schema import metadata

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata.create_all(op.get_bind())


def downgrade() -> None:
    metadata.drop_all(op.get_bind())
