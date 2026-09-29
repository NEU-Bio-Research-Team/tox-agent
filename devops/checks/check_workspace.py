#!/usr/bin/env python3
"""Warn about untracked, source-like directories beside the canonical tree.

A local checkout accumulates roots the repository no longer tracks — earlier
`agents/`, `services/`, `model_server/`, `src/`, `tools/`, `scripts/` layouts,
mostly `.gitignore`d. They are not the product (docs/explanation/workspace-layout.md), but a
person or an agent reading the workspace cannot tell that from the directory
name, and an audit or an edit made against one of them describes code that
never ships. This lists them.

It warns and exits 0 by default: an ignored directory on a workstation is not
a defect of the repository. `--strict` exits 1, for a CI checkout or a clean
benchmark host, where no such root should exist.

    python devops/checks/check_workspace.py [--strict]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".mjs"}
#: Tooling and caches that are neither source nor product.
NOT_SOURCE = {"node_modules", ".venv", ".git", ".cache", ".pytest_cache", ".codegraph",
              ".claude", ".vscode", ".data", "dist", "build"}


def tracked_roots() -> set[str]:
    output = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return {line.split("/", 1)[0] for line in output.splitlines() if "/" in line}


def looks_like_source(directory: Path, limit: int = 2000) -> bool:
    for seen, path in enumerate(directory.rglob("*")):
        if seen > limit:
            return False
        if any(part in NOT_SOURCE for part in path.parts):
            continue
        if path.suffix in SOURCE_SUFFIXES:
            return True
    return False


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    tracked = tracked_roots()
    stray = sorted(
        entry.name
        for entry in REPO_ROOT.iterdir()
        if entry.is_dir() and entry.name not in tracked and entry.name not in NOT_SOURCE
        and looks_like_source(entry)
    )
    if not stray:
        print("workspace OK: no untracked source-like roots")
        return 0
    print("untracked source-like roots (NOT the product; see docs/explanation/workspace-layout.md):")
    for name in stray:
        print(f"  {name}/")
    return 1 if strict else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
