"""Promote an approved skill draft into the shipped catalog (RETHINK §4.8, W9-11).

    python scripts/promote_skill_draft.py package.json [--profiles-dir DIR] [--replace]

``package.json`` is what ``GET /v1/skill-drafts/{id}/package`` returned for an
approved draft. The script checks the package digest the reviewer saw, writes
the files under ``agent_profiles/scientific_skills/<skill_id>/`` (manifest
status ``active``), and loads the whole catalog again, so a package that would
break start-up never lands. The result is a change to review and commit like
any other: a skill reaches runs only in a release.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxagent.application.investigation.skill_catalog import (
    CATALOG_DIR,
    SkillCatalogError,
    load_catalog,
)  # noqa: E402
from toxagent.application.investigation.skill_drafts import package_digest  # noqa: E402

DEFAULT_PROFILES = ROOT / "src" / "toxagent" / "agent_profiles"


def promote(package: dict, profiles_dir: Path, *, replace: bool = False) -> Path:
    files = package["files"]
    if package_digest(files) != package["package_sha256"]:
        raise SystemExit("the package does not match the digest the reviewer approved")
    if not package.get("review") or package["review"].get("decision") != "approve":
        raise SystemExit("only an approved draft is promoted")
    manifest = json.loads(files["skill.manifest.json"])
    if manifest.get("status") != "active":
        raise SystemExit("an exported package carries status 'active'")
    target = profiles_dir / CATALOG_DIR / package["skill_id"]
    if target.exists() and not replace:
        raise SystemExit(f"{target} exists; pass --replace to ship a revision")
    backup = None
    if target.exists():
        backup = target.with_name(target.name + ".previous")
        shutil.move(str(target), str(backup))
    try:
        for relative, text in files.items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        load_catalog(profiles_dir)
    except (SkillCatalogError, OSError) as exc:
        shutil.rmtree(target, ignore_errors=True)
        if backup is not None:
            shutil.move(str(backup), str(target))
        raise SystemExit(f"the catalog does not load with this package: {exc}") from None
    if backup is not None:
        shutil.rmtree(backup)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", type=Path)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    target = promote(json.loads(args.package.read_text()), args.profiles_dir, replace=args.replace)
    print(f"promoted to {target}; review and commit it like any other change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
