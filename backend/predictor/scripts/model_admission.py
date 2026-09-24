#!/usr/bin/env python3
"""Safe, read-only checkpoint discovery and manifest validation.

This tool intentionally never turns a discovered ``.pt`` into a served model.
Admission still requires a reviewed manifest with hashes and a compatible
provider factory.  It is useful for triaging the existing model directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PREDICTOR_ROOT = Path(__file__).resolve().parents[1]
ROOT = PREDICTOR_ROOT.parents[1]
PREDICTOR_SOURCE = PREDICTOR_ROOT / "src"
sys.path.insert(0, str(PREDICTOR_SOURCE))

from toxpred.scientific.artifacts import ArtifactError, load_manifest, sha256_file
from toxpred.scientific.bootstrap import build_registry


def checkpoints(root: Path):
    return sorted(path for path in root.rglob("*.pt") if path.is_file())


def inspect(path: Path) -> int:
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2
    print(f"path: {path}")
    print(f"bytes: {path.stat().st_size}")
    print(f"sha256: {sha256_file(path)}")
    try:
        import torch
        # ``weights_only`` prevents arbitrary pickle code from running. Some
        # old releases cannot be inspected this way and are reported, not
        # silently loaded through the unsafe legacy mode.
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            print("format: state mapping")
            print("keys:", ", ".join(map(str, list(value)[:12])))
        else:
            print(f"format: {type(value).__name__}")
    except Exception as exc:  # a checkpoint can be invalid without being unsafe
        print(f"safe_load: unavailable ({type(exc).__name__}: {exc})")
    return 0


def scan(root: Path, manifest: Path) -> int:
    rows = checkpoints(root)
    print(f"Found {len(rows)} checkpoints under {root}")
    try:
        specs = load_manifest(manifest, models_root=root)
    except ArtifactError as exc:
        print(f"Manifest unavailable: {exc}", file=sys.stderr)
        return 1
    declared = {
        (spec.root / entry.relative_path).resolve(): spec
        for spec in specs.values()
        for entry in spec.files
    }
    for path in rows:
        spec = declared.get(path.resolve())
        if spec is None:
            state = "discovered"
        elif spec.blocked_reason:
            state = "blocked"
        elif spec.required:
            state = "admitted"
        else:
            state = "declared"
        suffix = f" — {spec.model_id}" if spec else ""
        print(f"[{state}] {path.relative_to(root)} ({path.stat().st_size} bytes){suffix}")
    return 0


def validate(manifest: Path, model_id: str | None) -> int:
    try:
        specs = load_manifest(manifest)
        if model_id and model_id not in specs:
            raise ArtifactError(f"no manifest entry for {model_id!r}")
        targets = [model_id] if model_id else sorted(specs)
        for target in targets:
            specs[target].verify()
            print(f"OK {target}: checksums verified")
        return 0
    except ArtifactError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1


def admit(manifest: Path, model_id: str) -> int:
    # Admission means artifact + provider compatibility, never just file
    # existence. Eager load is intentionally disabled for a quick CI check.
    code = validate(manifest, model_id)
    if code:
        return code
    try:
        registry = build_registry(manifest, eager_load=False)
        registry.get(model_id)
        print(f"ADMITTED {model_id}: artifact and provider contract are valid")
        return 0
    except Exception as exc:
        print(f"NOT ADMITTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="toxagent models")
    parser.add_argument("--models-root", type=Path, default=ROOT / ".data" / "models")
    parser.add_argument("--manifest", type=Path, default=PREDICTOR_ROOT / "registry" / "predictor-manifest.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("path", type=Path)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("model_id", nargs="?")
    admit_parser = sub.add_parser("admit")
    admit_parser.add_argument("model_id")
    args = parser.parse_args()
    if args.command == "scan": return scan(args.models_root, args.manifest)
    if args.command == "inspect": return inspect(args.path)
    if args.command == "validate": return validate(args.manifest, args.model_id)
    return admit(args.manifest, args.model_id)


if __name__ == "__main__":
    raise SystemExit(main())
