"""eval-manifest-v2: enough to rebuild the product and environment a run graded.

The v1 manifest named a commit, a runtime kind and a trial count. Two runs with
equal v1 manifests could still have graded different products (a rollout flag,
a worker topology, a tool profile) on different environments (a starved host,
a different timeout). v2 adds, without removing any v1 key:

* source: commit, dirty-worktree flag, image digest when one is supplied;
* effective_product: the deployment's own ``effective-product-v1`` document —
  flags, intent -> lane -> profile -> tools, runtime binding, budgets,
  topology, prompt/tool hashes — read in process for a scripted run and over
  ``GET /v1/system/effective-product`` for a live one;
* discovery: selected packs and the discovered/executed/skipped/invalid/
  infra_error conservation;
* graders: the version of every grader that could have run;
* environment: host resources and the driver's timeout policy;
* release_evidence: whether this manifest can be attached to a release
  decision, and every reason it cannot.

A manifest is always written. What changes is whether it is evidence.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "eval-manifest-v2"

#: Fields of effective-product-v1 a release manifest cannot do without.
REQUIRED_PRODUCT_FIELDS = ("flags", "intents", "runtime", "topology", "hashes")


def git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_snapshot() -> dict[str, Any]:
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD") or "unknown",
        "dirty_worktree": None if status is None else bool(status),
        "image_digest": os.environ.get("TOXAGENT_IMAGE_DIGEST") or None,
    }


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _cgroup_cpu_limit() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    return round(int(quota) / int(period), 3)


def environment_snapshot(timeout_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cgroup_cpu_limit": _cgroup_cpu_limit(),
        "memory_bytes": _memory_bytes(),
        "ci": bool(os.environ.get("CI")),
        "timeout_policy": timeout_policy,
    }


def release_blockers(manifest: dict[str, Any]) -> list[str]:
    """Every reason this manifest is not release evidence. Empty means it is."""
    blockers: list[str] = []
    source = manifest.get("source") or {}
    if source.get("commit") in (None, "unknown"):
        blockers.append("commit unknown")
    if source.get("dirty_worktree") is not False:
        blockers.append("worktree dirty or unknown")
    product = manifest.get("effective_product") or {}
    if product.get("unavailable"):
        blockers.append(f"effective product unavailable: {product['unavailable']}")
    else:
        missing = [f for f in REQUIRED_PRODUCT_FIELDS if f not in product]
        if missing:
            blockers.append(f"effective product missing {missing}")
        if product.get("expired_flags"):
            blockers.append(f"expired rollout flags still present: {product['expired_flags']}")
    summary = manifest.get("summary") or {}
    if summary.get("invalid"):
        blockers.append(f"{summary['invalid']} invalid task(s)")
    if summary.get("conservation_violations"):
        blockers.extend(summary["conservation_violations"])
    if summary.get("infra_error"):
        blockers.append(f"{summary['infra_error']} task(s) ended in infra_error")
    not_evaluated = summary.get("not_evaluated_packs") or []
    if not_evaluated:
        blockers.append(f"selected pack(s) not evaluated: {not_evaluated}")
    if manifest.get("runtime_kind") == "scripted":
        blockers.append("scripted runs are CI evidence, not live release evidence")
    return blockers


def build_manifest(
    *,
    runtime: str,
    trials: int,
    fixture_mode: str,
    summary: dict[str, Any],
    suite_hash: str,
    discovery: dict[str, Any],
    effective_product: dict[str, Any],
    grader_versions: dict[str, str],
    timeout_policy: dict[str, Any],
    predictor_commit: str,
    suite: str | None = None,
) -> dict[str, Any]:
    source = source_snapshot()
    product_runtime = (effective_product or {}).get("runtime") or {}
    manifest: dict[str, Any] = {
        "manifest_schema": SCHEMA_VERSION,
        # v1 keys, unchanged, so evals.gates and older readers keep working.
        "eval_suite_hash": suite_hash,
        "toxagent_commit": source["commit"],
        "toxpred_commit": predictor_commit,
        "runtime_kind": runtime,
        "fixture_mode": fixture_mode,
        "runtime_version": (
            "in-process-scripted" if runtime == "scripted"
            else product_runtime.get("runtime_version") or "live-stack"
        ),
        "trial_count": trials,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        # v2.
        "suite": suite,
        "source": source,
        "discovery": discovery,
        "effective_product": effective_product,
        "graders": grader_versions,
        "environment": environment_snapshot(timeout_policy),
    }
    blockers = release_blockers(manifest)
    manifest["release_evidence"] = {"eligible": not blockers, "blockers": blockers}
    return manifest
