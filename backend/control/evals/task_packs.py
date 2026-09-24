"""Declared task packs, discovery and the v1 -> v3 loader (TAB-Suite Wave 0).

The runner used to glob ``evals/tasks`` and nothing else, so a regression task
written under ``evals/regression/tasks`` existed in the repository, changed no
suite hash, ran in no suite and blocked no release. The fix is not "add a
second glob": it is making the set of task locations a declared, reviewed
thing, and making every JSON file under a declared location either load or
show up as ``invalid`` — never silently absent.

* A pack is declared by ``evals/packs/<name>.json`` (``eval-pack-v1``).
* ``discover(packs)`` returns a :class:`Discovery`: loaded tasks plus invalid
  files, per pack, with an unavailable external pack recorded as such.
* Every loaded task is normalised to the v3 shape. A v1 file is adapted, not
  rewritten, so the frozen 50-task history keeps its bytes and its hash.
* ``suite_hash`` covers the selected pack manifests, their task files, every
  fixture, both schemas and the grader code — a grader edit is a suite change.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
PACKS_DIR = HERE / "packs"
SCHEMA_DIR = HERE / "schema"
FIXTURES_DIR = HERE / "fixtures"
GRADERS_DIR = HERE / "graders"

SCHEMAS = {
    "eval-task-v1": SCHEMA_DIR / "task.schema.json",
    "eval-task-v3": SCHEMA_DIR / "task.schema.v3.json",
}

#: What a pull request runs when nobody chose. Regression is in it on purpose:
#: a regression that only runs when someone remembers is not a guard.
DEFAULT_PACKS: tuple[str, ...] = ("core", "regression")

SUITES = ("pr", "nightly", "release")

_DETERMINISTIC_INTENTS = {"out_of_scope", "clarification_required"}


@dataclass(frozen=True)
class PackManifest:
    name: str
    owner: str
    description: str
    task_dirs: tuple[Path, ...]
    suites: tuple[str, ...]
    external: bool = False
    external_dir_env: str | None = None
    sealed_ids: frozenset[str] = frozenset()
    path: Path | None = None

    def resolved_dirs(self) -> tuple[Path, ...] | None:
        """Directories to scan, or ``None`` when an external pack is absent."""
        if not self.external:
            return self.task_dirs
        location = os.environ.get(self.external_dir_env or "", "")
        if not location or not Path(location).is_dir():
            return None
        return (Path(location),)


@dataclass
class InvalidTask:
    pack: str
    path: str
    problems: list[str]


@dataclass
class PackDiscovery:
    pack: str
    available: bool
    files: list[str] = field(default_factory=list)
    loaded: int = 0
    invalid: int = 0
    unavailable_reason: str | None = None


@dataclass
class Discovery:
    selected_packs: tuple[str, ...]
    tasks: list[dict[str, Any]]
    invalid: list[InvalidTask]
    packs: dict[str, PackDiscovery]

    @property
    def discovered(self) -> int:
        return sum(len(p.files) for p in self.packs.values())

    def summary(self) -> dict[str, Any]:
        return {
            "selected_packs": list(self.selected_packs),
            "discovered": self.discovered,
            "loaded": len(self.tasks),
            "invalid": [
                {"pack": i.pack, "path": i.path, "problems": i.problems} for i in self.invalid
            ],
            "packs": {
                name: {
                    "available": p.available,
                    "files": len(p.files),
                    "loaded": p.loaded,
                    "invalid": p.invalid,
                    **({"unavailable_reason": p.unavailable_reason} if p.unavailable_reason else {}),
                }
                for name, p in self.packs.items()
            },
        }


# ------------------------------------------------------------------ manifests

def load_pack_manifests(packs_dir: Path = PACKS_DIR) -> dict[str, PackManifest]:
    manifests: dict[str, PackManifest] = {}
    for path in sorted(packs_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        if raw.get("schema_version") != "eval-pack-v1":
            raise ValueError(f"{path.name}: schema_version must be eval-pack-v1")
        name = raw["pack"]
        if name != path.stem:
            raise ValueError(f"{path.name}: pack name {name!r} must match the file name")
        if not raw.get("owner"):
            raise ValueError(f"{path.name}: a pack must name an owner")
        unknown_suites = set(raw.get("suites", ())) - set(SUITES)
        if unknown_suites:
            raise ValueError(f"{path.name}: unknown suites {sorted(unknown_suites)}")
        external = bool(raw.get("external", False))
        if external and raw.get("task_dirs"):
            raise ValueError(f"{path.name}: an external pack keeps no task_dirs in the repository")
        manifests[name] = PackManifest(
            name=name,
            owner=raw["owner"],
            description=raw.get("description", ""),
            task_dirs=tuple(HERE / d for d in raw.get("task_dirs", ())),
            suites=tuple(raw.get("suites", ())),
            external=external,
            external_dir_env=raw.get("external_dir_env"),
            sealed_ids=frozenset(raw.get("sealed_ids", ())),
            path=path,
        )
    return manifests


def packs_for_suite(suite: str, manifests: dict[str, PackManifest] | None = None) -> tuple[str, ...]:
    manifests = manifests if manifests is not None else load_pack_manifests()
    return tuple(name for name, m in manifests.items() if suite in m.suites)


def parse_packs(value: str | None, manifests: dict[str, PackManifest] | None = None) -> tuple[str, ...]:
    manifests = manifests if manifests is not None else load_pack_manifests()
    if not value:
        return DEFAULT_PACKS
    names: list[str] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("suite:"):
            names.extend(packs_for_suite(token.removeprefix("suite:"), manifests))
            continue
        if token == "all":
            names.extend(manifests)
            continue
        if token not in manifests:
            raise SystemExit(f"unknown task pack {token!r}; declared: {sorted(manifests)}")
        names.append(token)
    return tuple(dict.fromkeys(names))


# -------------------------------------------------------------------- loading

def _validator(schema_version: str):
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a dev dependency
        return None
    return jsonschema.Draft202012Validator(json.loads(SCHEMAS[schema_version].read_text()))


def validate_task(raw: dict[str, Any]) -> list[str]:
    version = raw.get("schema_version")
    if version not in SCHEMAS:
        return [f"unknown schema_version {version!r}"]
    validator = _validator(version)
    if validator is None:
        return []
    return [f"{list(e.path)} {e.message}" for e in validator.iter_errors(raw)]


def _derived_lane(raw: dict[str, Any]) -> str:
    run = raw.get("expect", {}).get("run", {})
    if run.get("lane"):
        return run["lane"]
    return "deterministic" if _is_deterministic_v1(raw) else "agentic"


def _is_deterministic_v1(task: dict[str, Any]) -> bool:
    run_expect = task.get("expect", {}).get("run", {})
    if run_expect.get("lane") == "deterministic":
        return True
    if run_expect.get("intent") in _DETERMINISTIC_INTENTS:
        return True
    if run_expect.get("intent") == "analysis" and run_expect.get("status") == "failed":
        return True
    return task.get("expect", {}).get("error_code") in {
        "invalid_smiles", "predictor_not_ready", "predictor_protocol_error"
    }


def adapt_v1(raw: dict[str, Any], pack: str) -> dict[str, Any]:
    """A v1 task in the v3 shape. Every v1 field is kept verbatim."""
    task = copy.deepcopy(raw)
    task["source_schema_version"] = raw["schema_version"]
    task.setdefault("capability_pack", pack)
    intent = raw.get("expect", {}).get("run", {}).get("intent")
    if intent:
        task.setdefault("intent", intent)
    task.setdefault("lane", _derived_lane(raw))
    task.setdefault("risk_tier", "critical" if raw.get("critical") else "medium")
    expect = raw.get("expect", {})
    if _is_deterministic_v1(raw):
        requirement = "scripted"
    elif expect.get("state", {}).get("reconstructable_after_restart") or (
        expect.get("error_code") == "runtime_unavailable"
    ):
        requirement = "process_control"
    else:
        requirement = "agentic_runtime"
    task.setdefault("runtime_requirement", requirement)
    task.setdefault("required_graders", list(raw.get("graders", ["schema", "state"])))
    task.setdefault(
        "trial_policy",
        {"aggregation": "worst_of_n" if raw.get("critical") else "all"},
    )
    return task


def normalise(raw: dict[str, Any], pack: str) -> dict[str, Any]:
    if raw.get("schema_version") == "eval-task-v1":
        return adapt_v1(raw, pack)
    task = copy.deepcopy(raw)
    task["source_schema_version"] = raw["schema_version"]
    task.setdefault("required_graders", list(raw.get("graders", ["schema", "state"])))
    task.setdefault(
        "trial_policy",
        {"aggregation": "worst_of_n" if raw.get("risk_tier") == "critical" else "all"},
    )
    # `critical` is the v1 name the summary and graders still read.
    task.setdefault("critical", raw.get("risk_tier") == "critical")
    return task


def _check_task(raw: Any, pack: PackManifest, fixtures: set[str]) -> list[str]:
    if not isinstance(raw, dict):
        return ["a task file must hold one JSON object"]
    problems = validate_task(raw)
    if problems:
        return problems
    declared = raw.get("capability_pack")
    if declared is not None and declared != pack.name:
        problems.append(f"capability_pack {declared!r} does not match pack {pack.name!r}")
    if raw.get("fixture") not in fixtures:
        problems.append(f"fixture {raw.get('fixture')!r} does not exist")
    if pack.external:
        sealed_id = raw.get("sealed_id")
        if not sealed_id:
            problems.append("a task in an external pack must carry a sealed_id")
        elif sealed_id not in pack.sealed_ids:
            problems.append(f"sealed_id {sealed_id!r} is not declared in the pack manifest")
    elif raw.get("sealed_id"):
        problems.append("sealed_id is only valid for a task served from the external sealed pack")
    snapshot = raw.get("fixture_snapshot")
    if snapshot and raw.get("fixture") in fixtures:
        fixture = json.loads((FIXTURES_DIR / f"{raw['fixture']}.json").read_text())
        if fixture.get("content_sha256") != snapshot:
            problems.append("fixture_snapshot no longer matches the fixture's content_sha256")
    return problems


def discover(
    packs: Iterable[str] = DEFAULT_PACKS,
    manifests: dict[str, PackManifest] | None = None,
) -> Discovery:
    manifests = manifests if manifests is not None else load_pack_manifests()
    selected = tuple(packs)
    fixtures = {p.stem for p in FIXTURES_DIR.glob("*.json")}
    tasks: list[dict[str, Any]] = []
    invalid: list[InvalidTask] = []
    per_pack: dict[str, PackDiscovery] = {}
    seen_ids: dict[str, str] = {}

    for name in selected:
        manifest = manifests[name]
        dirs = manifest.resolved_dirs()
        if dirs is None:
            per_pack[name] = PackDiscovery(
                name, available=False,
                unavailable_reason=f"${manifest.external_dir_env} is unset or not a directory",
            )
            continue
        record = PackDiscovery(name, available=True)
        per_pack[name] = record
        for directory in dirs:
            for path in sorted(directory.glob("*.json")):
                rel = _rel(path)
                record.files.append(rel)
                try:
                    raw = json.loads(path.read_text())
                except ValueError as exc:
                    problems = [f"not valid JSON: {exc}"]
                else:
                    problems = _check_task(raw, manifest, fixtures)
                    if not problems:
                        task_id = raw["task_id"]
                        if task_id in seen_ids:
                            problems = [f"task_id {task_id!r} is also declared by {seen_ids[task_id]}"]
                        else:
                            seen_ids[task_id] = rel
                if problems:
                    invalid.append(InvalidTask(name, rel, problems))
                    record.invalid += 1
                    continue
                task = normalise(raw, name)
                task["capability_pack"] = name
                task["_source_path"] = rel
                tasks.append(task)
                record.loaded += 1
    return Discovery(selected, tasks, invalid, per_pack)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(HERE).as_posix()
    except ValueError:
        # An external (sealed) task: never record where on disk it came from.
        return f"<external>/{path.name}"


# ----------------------------------------------------------------- suite hash

def suite_hash(discovery: Discovery, manifests: dict[str, PackManifest] | None = None) -> str:
    """Content hash of everything that decides what a suite run grades."""
    from toxagent.domain.provenance import content_sha256

    manifests = manifests if manifests is not None else load_pack_manifests()
    payload: dict[str, Any] = {"selected_packs": list(discovery.selected_packs)}
    for name in discovery.selected_packs:
        manifest = manifests[name]
        if manifest.path is not None:
            payload[f"pack:{name}"] = manifest.path.read_text()
        record = discovery.packs.get(name)
        payload[f"pack_available:{name}"] = bool(record and record.available)
    for task in discovery.tasks:
        source = task["_source_path"]
        if source.startswith("<external>/"):
            # Sealed content must not leak into a hash that can be brute-forced
            # against a guess; the id and the normalised task hash are enough.
            payload[f"sealed:{task.get('sealed_id')}"] = content_sha256(
                {k: v for k, v in task.items() if not k.startswith("_")}
            )
        else:
            payload[f"task:{source}"] = (HERE / source).read_text()
    for invalid in discovery.invalid:
        if not invalid.path.startswith("<external>/"):
            payload[f"invalid:{invalid.path}"] = (HERE / invalid.path).read_text()
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        payload[f"fixture:{path.name}"] = path.read_text()
    for version, path in sorted(SCHEMAS.items()):
        payload[f"schema:{version}"] = path.read_text()
    for path in sorted(GRADERS_DIR.glob("*.py")):
        payload[f"grader:{path.name}"] = path.read_text()
    return content_sha256(payload)


def check_conservation(discovery: Discovery, *, executed: int, skipped: int, invalid: int,
                       infra_error: int = 0) -> list[str]:
    """``discovered = executed + skipped + invalid + infra_error``, or why not.

    A task that was discovered and appears in none of those buckets is the
    silent omission this module exists to make impossible.
    """
    total = executed + skipped + invalid + infra_error
    if total != discovery.discovered:
        return [
            f"task conservation violated: discovered={discovery.discovered} but "
            f"executed={executed} + skipped={skipped} + invalid={invalid} + "
            f"infra_error={infra_error} = {total}"
        ]
    return []
