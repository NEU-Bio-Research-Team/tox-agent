"""Every fixed issue names the check that would fail without the fix (W1-12).

The audit closed 34 issues. "Closed" was recorded as prose in
`docs/audit/SYSTEM_ISSUES_VI.md`, which cannot notice when the test that made
an issue closed is renamed, moved or deleted — at which point the issue is
open again and the document still says otherwise.

`docs/audit/regression_guards.json` names, for each issue, the file and the
anchor inside it that constitutes its guard. This checks the registry against
the tree: every issue is covered, every named file exists, and every anchor is
still in it. It does not run the guards — the service suites and CI do that —
it checks that they are still there to be run.

Deliberately not asserted: that a guard *fails* without its fix. Only writing
the guard alongside the fix can establish that, and the commit that adds each
one says whether it was checked that way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "audit" / "regression_guards.json"

#: I01..I34, the catalogue in SYSTEM_ISSUES_VI.md plus the two the K01 work
#: found (I33, I34). A new issue with no entry fails here rather than being
#: closed on a description alone.
EXPECTED_ISSUES = tuple(f"I{n:02d}" for n in range(1, 35))

KINDS = {"pytest", "vitest", "ci", "config"}


def registry() -> dict:
    return json.loads(REGISTRY.read_text())


def guards() -> list[tuple[str, dict]]:
    return [(issue, g) for issue, gs in registry()["issues"].items() for g in gs]


def test_the_registry_covers_every_issue_in_the_catalogue():
    assert tuple(sorted(registry()["issues"])) == EXPECTED_ISSUES


def test_every_issue_names_at_least_one_guard():
    empty = [issue for issue, gs in registry()["issues"].items() if not gs]
    assert empty == [], f"closed with nothing guarding it: {empty}"


@pytest.mark.parametrize("issue,guard", guards(), ids=lambda v: v if isinstance(v, str) else "")
def test_the_guard_still_exists(issue: str, guard: dict):
    assert guard["kind"] in KINDS, (issue, guard["kind"])
    path = ROOT / guard["path"]
    assert path.is_file(), f"{issue}: guard file is gone: {guard['path']}"
    assert guard["anchor"] in path.read_text(), (
        f"{issue}: {guard['path']} no longer contains its anchor "
        f"{guard['anchor']!r} — the issue is unguarded, whatever the audit says"
    )


def test_the_issues_that_stay_open_are_still_written_down_as_open():
    """The five with work remaining are named in the report, not silently
    absorbed into the closed set by having a guard for their closed half."""
    report = (ROOT / "docs" / "audit" / "SYSTEM_ISSUES_VI.md").read_text()
    for issue in ("I07", "I16", "I17/I18", "I30", "I31"):
        assert f"| {issue} |" in report, issue
