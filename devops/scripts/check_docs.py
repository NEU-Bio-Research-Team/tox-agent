#!/usr/bin/env python3
"""Documentation that still resolves against the tree it describes.

Two checks, both of which the audit found failing (I32):

1. Every relative link in a Markdown file points at something that exists.
2. Operational documents name no path that the relocation removed.

The second check is deliberately narrow. Historical records — the progress
log, the spec plans, the refactor audit, anything under docs/archive — are
evidence of what was true when written and must not be edited to match today's
tree; they are exempt, and `docs/README.md` marks them as historical instead.
A runbook is not evidence: someone follows it, so a dead path there is a bug.

    python devops/scripts/check_docs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Historical records. Exempt from the stale-path check, never from links.
HISTORICAL = (
    "docs/archive/",
    "docs/spec/",
    "docs/refactor/",
    "docs/audit/",
    "docs/unified-v2/BASELINE.md",
    "docs/unified-v2/IMPLEMENTATION_STATUS.md",
    # An analysis of the pre-refactor monolith. Its inventory of what
    # model_server/ contained is the finding, not a stale reference.
    "docs/BM1_explainer_benchmark_analysis.md",
    # A source plan, on the same footing as docs/spec.
    "new_plan.md",
    # An earlier audit, superseded by docs/audit/. Its inventory of the
    # pre-relocation tree is the record.
    "audit_5_9.md",
)

#: Paths the workspace consolidation removed, and what replaced them.
RETIRED = {
    "toxagent-control/": "backend/control/",
    "model_server/": "backend/predictor/ (the endpoint it served no longer exists)",
    "backend/control/agent_profiles/": "backend/control/src/toxagent/agent_profiles/",
    "artifacts/predictor-manifest.yaml": "backend/predictor/registry/predictor-manifest.yaml",
}

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^```", re.MULTILINE)


def _is_historical(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in HISTORICAL)


def _strip_code_fences(text: str) -> str:
    """Blank out fenced blocks so shell samples are not scanned for links."""
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return "\n".join(out)


def check_links(path: Path, rel: str) -> list[str]:
    problems = []
    for target in LINK.findall(_strip_code_fences(path.read_text(encoding="utf-8"))):
        target = target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        # An absolute path is a machine-local file reference someone pasted
        # (`/home/.../routes.py:42`), not a link into this repository. It
        # cannot resolve on any other machine and there is nothing to fix.
        if target.startswith("/"):
            continue
        if not (path.parent / target).exists():
            problems.append(f"{rel}: broken link -> {target}")
    return problems


def check_retired_paths(path: Path, rel: str) -> list[str]:
    problems = []
    text = path.read_text(encoding="utf-8")
    for dead, replacement in RETIRED.items():
        if dead in text:
            line = next(
                (i for i, l in enumerate(text.splitlines(), 1) if dead in l), 0
            )
            problems.append(f"{rel}:{line}: names retired path {dead!r}; use {replacement}")
    return problems


def main() -> int:
    problems: list[str] = []
    for path in sorted(REPO_ROOT.glob("docs/**/*.md")) + sorted(REPO_ROOT.glob("*.md")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        problems += check_links(path, rel)
        if not _is_historical(rel):
            problems += check_retired_paths(path, rel)

    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} documentation problem(s).", file=sys.stderr)
        return 1
    print("documentation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
