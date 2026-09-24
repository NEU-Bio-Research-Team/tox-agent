"""What exactly is in a release (K12).

A tag names a commit. It does not name the versions of `torch`, `rdkit` and
`transformers` that produced the numbers, the base image the services were
built on, or the model artifacts they load — and this project reports
scientific measurements, so "which build produced this probability" has to be
answerable after the fact rather than reconstructed from a date.

    release_manifest.py                      # write to stdout
    release_manifest.py --out release.json
    release_manifest.py --check release.json # compare this tree to a manifest

The manifest is deliberately made of things that can be read here rather than
things a human types: git state, the resolved dependency versions of whichever
environment ran this, the pinned base image digests out of the Dockerfiles,
and the predictor's artifact manifest hash. A field this script cannot
determine is recorded as null, never as a guess — a manifest that quietly
substitutes a plausible value for an unknown one is worse than no manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

#: Read from the installed environment, not from pyproject: the ranges in
#: pyproject say what is allowed, and a release has to record what was
#: actually resolved. `torch` and `rdkit` are unpinned there, and the golden
#: numbers depend on both.
TRACKED = (
    "fastapi", "uvicorn", "pydantic", "sqlalchemy", "alembic", "httpx",
    "sse-starlette", "mcp", "pyjwt", "anyio", "asyncpg", "psycopg",
    "numpy", "torch", "transformers", "rdkit", "pyyaml", "captum",
    "torch-geometric", "scikit-learn",
)

#: `FROM python:3.10-slim@sha256:...` — the digest, because a tag moves.
_FROM = re.compile(r"^FROM\s+(\S+?)@(sha256:[0-9a-f]{64})", re.MULTILINE)

DOCKERFILES = {
    "control": "backend/control/deploy/Dockerfile",
    "predictor": "backend/predictor/deploy/Dockerfile",
    "ocr": "backend/ocr/deploy/Dockerfile",
}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict[str, Any]:
    dirty = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        # A release built from a dirty tree is not reproducible from the
        # commit it names, so the manifest says so rather than implying it is.
        "clean": None if dirty is None else dirty == "",
    }


def resolved_dependencies() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # Absent from *this* environment. The control image has no torch
            # and the predictor image has no alembic; null says "not here",
            # which is different from "unknown version".
            versions[name] = None
    return versions


def base_images() -> dict[str, str | None]:
    images: dict[str, str | None] = {}
    for service, relative in DOCKERFILES.items():
        path = ROOT / relative
        if not path.is_file():
            images[service] = None
            continue
        match = _FROM.search(path.read_text())
        images[service] = f"{match.group(1)}@{match.group(2)}" if match else None
    return images


def unpinned_base_images() -> list[str]:
    """Dockerfiles whose FROM names a tag rather than a digest.

    A tag is not an artifact: `python:3.10-slim` is a different image next
    month, so a rebuild of the same commit is a different release.
    """
    loose = []
    for service, relative in DOCKERFILES.items():
        path = ROOT / relative
        if path.is_file() and not _FROM.search(path.read_text()):
            loose.append(service)
    return loose


def predictor_registry() -> dict[str, Any]:
    """The model registry's own hash, so the weights a release serves are named."""
    registry = ROOT / "backend" / "predictor" / "registry" / "models"
    if not registry.is_dir():
        return {"sha256": None, "models": []}
    digest = hashlib.sha256()
    models = []
    for path in sorted(registry.glob("*.yaml")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        models.append(path.stem)
    return {"sha256": f"sha256:{digest.hexdigest()}", "models": models}


def contract_snapshot() -> str | None:
    path = (
        ROOT / "backend" / "control" / "src" / "toxagent" / "predictor" / "contract_snapshot.json"
    )
    try:
        return json.loads(path.read_text()).get("captured_at_commit")
    except (OSError, ValueError):
        return None


def build(version: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "release-manifest-v1",
        "version": version or (ROOT / "VERSION").read_text().strip(),
        "git": git_state(),
        "base_images": base_images(),
        "unpinned_base_images": unpinned_base_images(),
        "dependencies": resolved_dependencies(),
        "predictor_registry": predictor_registry(),
        "predictor_contract_snapshot_commit": contract_snapshot(),
    }


def differences(manifest: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """What changed, ignoring the fields that legitimately differ per build.

    Not compared: which environment ran the script, so a dependency recorded
    in one and absent from the other is not a difference — the control image
    has no torch. A *changed* version is.
    """
    problems: list[str] = []
    for field in ("version", "predictor_contract_snapshot_commit"):
        if manifest.get(field) != current.get(field):
            problems.append(f"{field}: {manifest.get(field)!r} -> {current.get(field)!r}")
    if manifest.get("git", {}).get("commit") != current.get("git", {}).get("commit"):
        problems.append(
            f"git.commit: {manifest.get('git', {}).get('commit')!r} -> "
            f"{current.get('git', {}).get('commit')!r}"
        )
    for service, image in (manifest.get("base_images") or {}).items():
        if (current.get("base_images") or {}).get(service) != image:
            problems.append(f"base_images.{service} changed")
    if manifest.get("predictor_registry", {}).get("sha256") != current.get(
        "predictor_registry", {}
    ).get("sha256"):
        problems.append("predictor_registry.sha256 changed: the served weights are not the same")
    for name, version in (manifest.get("dependencies") or {}).items():
        here = (current.get("dependencies") or {}).get(name)
        if version is not None and here is not None and version != here:
            problems.append(f"dependencies.{name}: {version} -> {here}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the manifest here instead of stdout")
    parser.add_argument("--check", type=Path, help="compare this tree against a saved manifest")
    parser.add_argument("--version", help="override the release version")
    args = parser.parse_args(argv)

    current = build(args.version)
    if args.check:
        problems = differences(json.loads(args.check.read_text()), current)
        if problems:
            print("release manifest does not describe this tree:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("release manifest matches this tree")
        return 0

    rendered = json.dumps(current, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
