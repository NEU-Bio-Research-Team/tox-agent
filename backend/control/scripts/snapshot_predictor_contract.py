#!/usr/bin/env python3
"""Pin the ToxPred OpenAPI document this control plane was built against.

Run from the monorepo root with the predictor importable. The snapshot is the
compatibility gate between the two boundaries (ADR 0001): a predictor change
that removes a field or renames an endpoint fails
``tests/contract/test_predictor_contract.py`` here before it reaches
production, and the diff in this file is what a reviewer reads.

    python toxagent-control/scripts/snapshot_predictor_contract.py

The snapshot is generated from the predictor's own app factory rather than from
a running server on purpose: it must not depend on model artifacts being
present, so it can be regenerated in CI.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
TARGET = HERE.parents[1] / "toxagent" / "predictor" / "contract_snapshot.json"


def predictor_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from toxpred.api.app import create_app  # noqa: PLC0415 — import after path setup

    document = create_app().openapi()
    snapshot = {
        "captured_at_commit": predictor_commit(),
        "openapi": document,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"wrote {TARGET.relative_to(REPO_ROOT)} ({len(document['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
