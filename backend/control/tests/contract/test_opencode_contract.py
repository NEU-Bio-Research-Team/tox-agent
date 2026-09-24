"""The pinned OpenCode V1 contract (plan §11.1, §11.4; progress log §4.4).

The hand-written mocks in ``test_opencode_v1_adapter.py`` prove the adapter's
*logic*. This file proves its *assumptions about OpenCode* against a document
captured from the real pinned binary by
``scripts/snapshot_opencode_contract.py``. Without that snapshot the structural
checks skip — but once it exists (it should, on any host that has run the Phase
3 stack), a moved or renamed endpoint fails here instead of at the next live
run, the way ``/app/agents`` vs ``/agent`` did (progress log §3.1).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from toxagent.harness.adapters.opencode_v1 import OPENCODE_V1_PIN

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1].parent
    / "toxagent"
    / "harness"
    / "adapters"
    / "opencode_v1_contract.json"
)

# Every OpenCode path the V1 adapter calls, with its methods. Path parameters
# are normalized to ``{}`` so this does not depend on the spec's parameter
# names. Anything not listed here the adapter does not depend on.
REQUIRED_PATHS = {
    "/agent": {"get"},
    "/session": {"post"},
    "/session/status": {"get"},
    "/session/{}": {"delete"},
    "/session/{}/prompt_async": {"post"},
    "/session/{}/abort": {"post"},
    "/mcp": {"post"},
    "/mcp/{}/connect": {"post"},
    "/mcp/{}/disconnect": {"post"},
    "/global/event": {"get"},
}

_PARAM = re.compile(r"\{[^}]+\}")


def _normalize(path: str) -> str:
    return _PARAM.sub("{}", path)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            "no OpenCode contract snapshot; run "
            "scripts/snapshot_opencode_contract.py against the pinned server"
        )
    return json.loads(SNAPSHOT_PATH.read_text())


@pytest.fixture(scope="module")
def normalized_paths(snapshot) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for raw_path, item in snapshot["openapi"].get("paths", {}).items():
        methods = {m.lower() for m in item if m.lower() in {"get", "post", "put", "patch", "delete"}}
        merged.setdefault(_normalize(raw_path), set()).update(methods)
    return merged


def test_snapshot_is_for_the_pinned_binary(snapshot):
    assert snapshot["pin"] == OPENCODE_V1_PIN
    version = snapshot.get("binary", {}).get("version", "unknown")
    assert version in ("unknown", OPENCODE_V1_PIN), (
        f"snapshot captured from OpenCode {version!r}, adapter pin is {OPENCODE_V1_PIN!r}"
    )


@pytest.mark.parametrize("path,methods", sorted(REQUIRED_PATHS.items()))
def test_required_paths_exist(normalized_paths, path, methods):
    assert path in normalized_paths, f"OpenCode {OPENCODE_V1_PIN} no longer serves {path}"
    assert methods <= normalized_paths[path], f"{path} lost {methods - normalized_paths[path]}"


def test_the_removed_app_agents_route_did_not_come_back_under_a_confusing_name(normalized_paths):
    """``app.agents`` is only an operationId; the path is ``GET /agent``
    (progress log §3.1). A ``/app/agents`` path reappearing is a red flag."""
    assert "/app/agents" not in normalized_paths
