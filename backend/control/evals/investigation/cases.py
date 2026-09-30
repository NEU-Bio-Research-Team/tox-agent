"""Investigation cases: build from the hand-authored specs, load, validate.

A case is fixed before any system runs: the question turns, the structure
(resolved from PubChem, never typed from memory), the tags that decide which
skills it measures, and a reference outcome for the graders. The reference is
never shown to a system.

    python -m evals.investigation.cases --build   # resolve structures, write cases/
    python -m evals.investigation.cases --check   # validate cases/ against the schema
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
SPECS_PATH = HERE / "case_specs.json"
CASES_DIR = HERE / "cases"
SCHEMA_PATH = HERE / "schema" / "investigation-case.schema.json"
SCHEMA_VERSION = "investigation-case-v1"

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/SMILES,MolecularFormula/JSON"

Resolver = Callable[[str], dict[str, Any]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(value: Any) -> str:
    text = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pubchem_resolver(name: str) -> dict[str, Any]:
    import httpx

    url = PUBCHEM.format(name=quote(name))
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    properties = response.json()["PropertyTable"]["Properties"][0]
    return {
        "pubchem_cid": int(properties["CID"]),
        "smiles": properties["SMILES"],
        "molecular_formula": properties.get("MolecularFormula", ""),
        "source_url": url,
    }


def build_case(spec: dict[str, Any], *, set_id: str, resolver: Resolver, now: str) -> dict[str, Any]:
    resolved = resolver(spec["compound"])
    disclosed = spec.get("name_disclosed", True)
    turns = []
    for turn in spec["turns"]:
        text = turn["text"].format(smiles=resolved["smiles"], name=spec["compound"])
        if not disclosed and spec["compound"].lower() in text.lower():
            raise ValueError(f"{spec['case_id']}: the name is withheld but a turn states it")
        rendered = {"text": text}
        if turn.get("context"):
            rendered["context"] = dict(turn["context"])
        turns.append(rendered)
    return {
        "schema_version": SCHEMA_VERSION,
        "set_id": set_id,
        "case_id": spec["case_id"],
        "tags": list(spec["tags"]),
        "compound": {
            "query_name": spec["compound"],
            "name_disclosed": disclosed,
            "pubchem_cid": resolved["pubchem_cid"],
            "smiles": resolved["smiles"],
            "molecular_formula": resolved["molecular_formula"],
            "resolved_from": "PubChem PUG REST",
            "resolved_at": now,
            "source_url": resolved["source_url"],
        },
        "turns": turns,
        "reference": {**spec["reference"], "status": "pending_lab_verification"},
        "spec_sha256": sha256(spec),
    }


def build(resolver: Resolver = pubchem_resolver, *, out_dir: Path = CASES_DIR,
          specs_path: Path = SPECS_PATH) -> list[dict[str, Any]]:
    specs = json.loads(specs_path.read_text())
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    cases = [build_case(spec, set_id=specs["set_id"], resolver=resolver, now=now)
             for spec in specs["cases"]]
    validate(cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        (out_dir / f"{case['case_id']}.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n"
        )
    return cases


def validate(cases: list[dict[str, Any]]) -> None:
    import jsonschema

    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    problems = [
        f"{case.get('case_id', '?')}: {list(error.path)} {error.message}"
        for case in cases for error in validator.iter_errors(case)
    ]
    ids = [case.get("case_id") for case in cases]
    if len(ids) != len(set(ids)):
        problems.append("duplicate case ids")
    if problems:
        raise ValueError("invalid investigation cases:\n" + "\n".join(problems))


def load_cases(cases_dir: Path = CASES_DIR, *, only: list[str] | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(p.read_text()) for p in sorted(cases_dir.glob("*.json"))]
    validate(cases)
    if only:
        unknown = sorted(set(only) - {c["case_id"] for c in cases})
        if unknown:
            raise ValueError(f"unknown case ids: {unknown}")
        cases = [c for c in cases if c["case_id"] in only]
    return cases


def case_set_sha256(cases: list[dict[str, Any]]) -> str:
    return sha256([sha256(case) for case in sorted(cases, key=lambda c: c["case_id"])])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--specs", type=Path, default=SPECS_PATH,
                        help="case spec file to build from (default: the pilot set)")
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR,
                        help="where the built cases live; a set of its own keeps studies separate")
    args = parser.parse_args(argv)
    if args.build:
        cases = build(out_dir=args.cases_dir, specs_path=args.specs)
        print(f"wrote {len(cases)} cases to {args.cases_dir}")
    else:
        cases = load_cases(args.cases_dir)
        print(f"{len(cases)} cases valid; set sha256 {case_set_sha256(cases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
