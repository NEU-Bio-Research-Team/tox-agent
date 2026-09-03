#!/usr/bin/env python3
"""Phase 7B exit gate: no agent, LLM or web dependency in the runtime.

Run in CI. Scans the runtime source and dependency files for tokens belonging to
the layer this service replaced. Two kinds of file are excluded on purpose:

* benchmarks/manifests/openapi-legacy-*.json — an archived snapshot of the old
  API, kept as the before picture;
* tests/unit/test_import_boundaries.py — the list of banned imports has to name
  them to forbid them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCAN = ["toxpred", "backend", "deploy", "artifacts", "config", "scripts", "benchmarks", "tests"]
SCAN_FILES = ["pyproject.toml", "requirements.txt", "environment.yml"]

EXCLUDE = {
    "benchmarks/manifests/openapi-legacy-e6882b2.json",
    "tests/unit/test_import_boundaries.py",
    "scripts/check_no_agent_deps.py",
}

BANNED = [
    r"google[-_]adk", r"google[-_]genai", r"\bgemini\b", r"LLM_RUNTIME",
    r"firebase[-_]admin", r"\bfirestore\b", r"sentence[-_]transformers",
    r"\bmolscribe\b", r"\bagents\.", r"/agent/",
]


def files() -> list[Path]:
    out: list[Path] = []
    for d in SCAN:
        out += [p for p in (ROOT / d).rglob("*") if p.is_file()]
    out += [ROOT / f for f in SCAN_FILES if (ROOT / f).exists()]
    return [
        p for p in out
        if "__pycache__" not in p.parts
        and str(p.relative_to(ROOT)) not in EXCLUDE
        and p.suffix not in {".pt", ".pkl", ".png", ".gz"}
    ]


def main() -> int:
    patterns = [re.compile(b, re.IGNORECASE) for b in BANNED]
    hits: list[str] = []
    for path in files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # a comment recording what was removed is not a dependency
            for pat in patterns:
                if pat.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:100]}")
    if hits:
        print(f"FAIL — {len(hits)} agent/LLM/web reference(s) in the runtime:")
        for h in hits:
            print(f"  {h}")
        return 1
    print(f"PASS — scanned {len(files())} files, no agent/LLM/web reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
