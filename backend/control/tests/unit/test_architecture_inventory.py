"""The checked-in architecture inventory matches the code (review P2-03).

When this fails, the product's shape changed — an intent, flag, profile, tool,
queue, provider, eval pack, grader or schema version. Regenerate with
``python -m evals.architecture_inventory --write`` and review the diff: it is
the architecture change, stated once.
"""
from __future__ import annotations

import subprocess

from evals.architecture_inventory import INVENTORY_PATH, REPO_ROOT, build_inventory, render


def test_the_checked_in_inventory_is_current():
    assert INVENTORY_PATH.read_text() == render(build_inventory()), (
        "docs/reference/architecture-inventory.json is stale; run "
        "`python -m evals.architecture_inventory --write`"
    )


def test_every_live_agentic_intent_names_a_real_profile():
    inventory = build_inventory()
    for intent, entry in inventory["intents"].items():
        if entry["default_lane"] != "deterministic":
            assert entry["capability_profile"] in inventory["capability_profiles"], intent


def test_no_two_tracked_paths_differ_only_by_case():
    try:
        files = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a checkout
        return
    seen: dict[str, str] = {}
    collisions = []
    for path in files:
        key = path.lower()
        if key in seen and seen[key] != path:
            collisions.append((seen[key], path))
        seen[key] = path
    assert not collisions, collisions
