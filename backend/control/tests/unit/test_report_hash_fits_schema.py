"""A value the report path writes must fit the column PostgreSQL enforces.

SQLite ignores varchar lengths, so the suite could not see that a v3 report's
``sha256:``-prefixed digest (71 chars) overflowed ``varchar(64)`` — every
orchestrated build failed on PostgreSQL at rendering. This checks the width
against the schema directly, independent of the test database.
"""
from __future__ import annotations

import hashlib

from toxagent.persistence.schema import report_artifacts, report_figures, report_renderings


def test_prefixed_report_digests_fit_their_columns():
    prefixed = "sha256:" + hashlib.sha256(b"x").hexdigest()
    for table in (report_artifacts, report_figures, report_renderings):
        assert table.c.content_sha256.type.length >= len(prefixed), table.name
