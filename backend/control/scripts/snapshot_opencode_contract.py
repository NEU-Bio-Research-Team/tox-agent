#!/usr/bin/env python3
"""Pin the OpenCode V1 OpenAPI document this adapter was built against.

Unlike the predictor snapshot, OpenCode's contract cannot be regenerated from
importable source — it is a third-party binary. So this captures ``GET /doc``
from a *running, pinned* OpenCode server and records the binary's version and
digest alongside it. ``tests/contract/test_opencode_contract.py`` then checks
the paths the adapter calls against the snapshot; three of the four blocking
bugs in the first live Phase 3 run were contract drift the hand-written mocks
could not see (progress log §3.1, §4.4).

Usage — with the pinned server already running (scripts/run_local_phase3.sh
starts one on :4096)::

    OPENCODE_BIN=~/.opencode/bin/opencode \\
      python toxagent-control/scripts/snapshot_opencode_contract.py \\
      --url http://127.0.0.1:4096

Re-run and review the diff whenever the OpenCode pin moves (ADR 0004).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
TARGET = HERE.parents[1] / "toxagent" / "harness" / "adapters" / "opencode_v1_contract.json"
PIN = "1.17.11"


def _binary_report(opencode_bin: str | None) -> dict[str, str]:
    if not opencode_bin:
        return {"version": "unknown", "sha256": "unknown"}
    path = Path(opencode_bin)
    report: dict[str, str] = {}
    try:
        out = subprocess.run(
            [opencode_bin, "--version"], capture_output=True, text=True, check=True
        )
        report["version"] = out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        report["version"] = "unknown"
    try:
        report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        report["sha256"] = "unknown"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.getenv("TOXAGENT_OPENCODE_URL", "http://127.0.0.1:4096"),
        help="Base URL of a running, pinned OpenCode server.",
    )
    parser.add_argument("--doc-path", default="/doc", help="OpenAPI document path (default /doc).")
    args = parser.parse_args()

    url = args.url.rstrip("/") + args.doc_path
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - localhost dev tool
            document = json.loads(response.read())
    except (OSError, ValueError) as exc:
        print(f"could not fetch {url}: {exc}", file=sys.stderr)
        print("start the pinned server first (scripts/run_local_phase3.sh).", file=sys.stderr)
        return 2

    binary = _binary_report(os.getenv("OPENCODE_BIN"))
    if binary["version"] not in ("unknown", PIN):
        print(
            f"refusing to snapshot: OPENCODE_BIN is {binary['version']!r}, adapter pin is {PIN!r}",
            file=sys.stderr,
        )
        return 2

    snapshot = {
        "pin": PIN,
        "binary": binary,
        "source_url": url,
        "openapi": document,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    paths = document.get("paths", {})
    print(f"wrote {TARGET.name} ({len(paths)} paths, binary {binary['version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
