"""The migration chain is well-formed, without needing a real database.

Caught the hard way (2026-09-15): a migration named
``0016_evidence_relation_assessments`` (34 chars) was fine in every local
SQLite-backed test, then failed against a real PostgreSQL database with
``value too long for type character varying(32)`` — alembic's own
``alembic_version.version_num`` column is ``VARCHAR(32)``, and nothing in the
SQLite-only suite exercises that column's width.
"""
from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

SERVICE_ROOT = Path(__file__).resolve().parents[2]

#: alembic_version.version_num, every backend this project supports.
_VERSION_NUM_MAX_LENGTH = 32


def _script_directory() -> ScriptDirectory:
    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(SERVICE_ROOT / "migrations"))
    return ScriptDirectory.from_config(config)


def test_the_chain_has_exactly_one_head():
    heads = _script_directory().get_heads()
    assert len(heads) == 1, f"expected one head, found {heads}"


def test_every_revision_id_fits_the_alembic_version_column():
    sd = _script_directory()
    too_long = [
        rev.revision
        for rev in sd.walk_revisions()
        if len(rev.revision) > _VERSION_NUM_MAX_LENGTH
    ]
    assert not too_long, (
        f"revision id(s) exceed alembic_version.version_num's "
        f"VARCHAR({_VERSION_NUM_MAX_LENGTH}): {too_long}"
    )
