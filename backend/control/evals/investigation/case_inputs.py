"""Protocol v2 cases: what a system is given, kept apart from what it is graded on (#51).

A v3 case is three files, each with one job:

* the **case** (``investigation-case-v3``): what the case measures -- its task
  group, workflow context, answerability, pairing, fault -- and the hashes of
  the other two;
* the **inputs** (``frozen-case-inputs-v1``): everything a system is given --
  the subject by reference, the frozen artifacts with their hashes, the
  evidence corpus and full-text store, the contract versions, and the turns
  with their actions (add_context, attach_evidence_ref, switch_subject). The
  same file goes to every arm;
* the **gold** (``case-gold-v1``): what a grader checks the answer against.
  It is never read by an adapter, and the audit checks that nothing of it
  leaked into the inputs.

``cohort`` keeps meaning where a measurement disagreement was sampled from;
whether a case is about toxicity alone or an activity-guided screening is
``workflow_context``, a separate field.

The audit (:func:`audit_set`) returns findings rather than raising, so a set is
checked and fixed in one pass: shape against the three schemas, hashes,
artifact refs against the replay file, spans against the frozen full-text
store (or a declared access failure), pairing symmetry, answerability against
the gold, T3 polarity, T6 faults, gold isolation, and dependency groups --
which :func:`dependency_split` turns into a dev/test split no compound,
document or measurement crosses.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .case_validator import Finding, validate_case

HERE = Path(__file__).resolve().parent
SCHEMA_DIR = HERE / "schema"
CASE_SCHEMA = "investigation-case-v3"
INPUTS_SCHEMA = "frozen-case-inputs-v1"
GOLD_SCHEMA = "case-gold-v1"

#: Dependency kinds whose sharing makes two cases one statistical unit.
DEPENDENCY_KINDS = ("compounds", "documents", "measurements", "sources")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validator(name: str):
    import jsonschema

    return jsonschema.Draft202012Validator(
        json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())
    )


@dataclass
class LoadedCase:
    case: dict[str, Any]
    inputs: dict[str, Any]
    path: Path
    #: Loaded only for grading; ``None`` in anything a system sees.
    gold: dict[str, Any] | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def case_id(self) -> str:
        return str(self.case.get("case_id"))


def _resolve(base: Path, ref: Mapping[str, Any]) -> Path:
    path = Path(str(ref.get("path") or ""))
    return path if path.is_absolute() else (base / path)


def load_case(path: Path, *, with_gold: bool = False) -> LoadedCase:
    """One v3 case and its inputs (and, for a grader, its gold). Hash
    mismatches are findings, not exceptions."""
    path = Path(path)
    case = json.loads(path.read_text())
    loaded = LoadedCase(case=case, inputs={}, path=path)
    fail = lambda code, detail: loaded.findings.append(  # noqa: E731
        Finding(case_id=str(case.get("case_id") or path.stem), code=code, detail=detail))
    inputs_path = _resolve(path.parent, case.get("inputs_ref") or {})
    if not inputs_path.is_file():
        fail("inputs_missing", f"{inputs_path} does not exist")
    else:
        if file_sha256(inputs_path) != (case.get("inputs_ref") or {}).get("sha256"):
            fail("inputs_hash_mismatch", f"{inputs_path.name} is not the file the case pins")
        loaded.inputs = json.loads(inputs_path.read_text())
    if with_gold:
        gold_path = _resolve(path.parent, case.get("gold_ref") or {})
        if not gold_path.is_file():
            fail("gold_missing", f"{gold_path} does not exist")
        else:
            if file_sha256(gold_path) != (case.get("gold_ref") or {}).get("sha256"):
                fail("gold_hash_mismatch", f"{gold_path.name} is not the file the case pins")
            loaded.gold = json.loads(gold_path.read_text())
    return loaded


def load_set(directory: Path, *, with_gold: bool = False) -> list[LoadedCase]:
    return [load_case(p, with_gold=with_gold) for p in sorted(Path(directory).glob("*.json"))]


def as_v2_view(loaded: LoadedCase) -> dict[str, Any]:
    """The v2 shape the v2 validator checks, assembled from case + gold, so a
    v3 case is held to every rule a v2 case is."""
    case, gold = loaded.case, loaded.gold or {}
    view = {
        "case_id": case.get("case_id"), "task_group": case.get("task_group"),
        "answerability": case.get("answerability"), "evidence_mode": case.get("evidence_mode"),
        "acceptable_conclusions": gold.get("acceptable_conclusions") or [],
        "required_limitations": gold.get("required_limitations") or [],
        "optional_limitations": gold.get("optional_limitations") or [],
        "limitation_scoring": gold.get("limitation_scoring"),
        "forbidden_inferences": [f for f in gold.get("forbidden_inferences") or ()],
        "evidence_spans": gold.get("evidence_spans") or [],
        "turns": loaded.inputs.get("turns") or [],
        "stress_set": case.get("stress_set", False),
    }
    for key in ("fault", "paired_with", "cohort"):
        if key in case:
            view[key] = case[key]
    if "predictor_error" in gold:
        view["predictor_error"] = gold["predictor_error"]
    return view


def _model_facing_text(inputs: Mapping[str, Any]) -> str:
    return json.dumps(inputs, sort_keys=True, ensure_ascii=False).casefold()


def audit_case(
    loaded: LoadedCase, *, replay_records: Mapping[str, Mapping[str, Any]] | None = None,
    frozen_analyses: Mapping[str, Mapping[str, Any]] | None = None,
    full_text_documents: Mapping[str, Mapping[str, Any]] | None = None,
    snapshot_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Finding]:
    """Every rule one case (with its inputs and gold) can break."""
    case, inputs, gold = loaded.case, loaded.inputs, loaded.gold or {}
    findings = list(loaded.findings)
    case_id = loaded.case_id

    def fail(code: str, detail: str) -> None:
        findings.append(Finding(case_id=case_id, code=code, detail=detail))

    for name, document in ((CASE_SCHEMA, case), (INPUTS_SCHEMA, inputs), (GOLD_SCHEMA, gold)):
        if not document:
            continue
        for error in _validator(name).iter_errors(document):
            fail(f"{name}_shape", f"{list(error.path)} {error.message}")
    if inputs and inputs.get("case_id") != case_id:
        fail("inputs_of_another_case", f"inputs name {inputs.get('case_id')!r}")
    if gold and gold.get("case_id") != case_id:
        fail("gold_of_another_case", f"gold names {gold.get('case_id')!r}")
    if inputs and inputs.get("workflow_context") != case.get("workflow_context"):
        fail("workflow_context_disagrees", "the case and its inputs name different contexts")
    if gold:
        findings.extend(validate_case(as_v2_view(loaded)))

    # Artifacts: every frozen artifact the inputs name exists, with its hash.
    for artifact in inputs.get("artifacts") or ():
        ref, digest = str(artifact.get("ref")), artifact.get("content_sha256")
        if artifact.get("kind") == "screening" and replay_records is not None:
            record = replay_records.get(ref)
            if record is None:
                fail("artifact_not_in_replay_file", f"screening {ref}")
            elif record.get("content_sha256") != digest:
                fail("artifact_hash_mismatch", f"screening {ref}")
        if artifact.get("kind") == "frozen_analysis" and frozen_analyses is not None:
            frozen = frozen_analyses.get(ref)
            if frozen is None:
                fail("artifact_not_in_replay_file", f"analysis {ref}")
            elif frozen.get("content_sha256") != digest:
                fail("artifact_hash_mismatch", f"analysis {ref}")

    # The subject is one of the artifacts, and every turn's subject is too.
    known = {str(a.get("ref")) for a in inputs.get("artifacts") or ()}
    subjects = [((inputs.get("subject") or {}).get("subject_ref") or {})]
    for turn in inputs.get("turns") or ():
        if turn.get("subject_ref"):
            subjects.append(turn["subject_ref"])
        for action in turn.get("actions") or ():
            kind = action.get("type")
            if kind == "switch_subject":
                if not action.get("subject_ref"):
                    fail("switch_without_subject", "switch_subject names no subject_ref")
                else:
                    subjects.append(action["subject_ref"])
            elif kind == "add_context" and not (action.get("key") and action.get("value")):
                fail("context_without_value", "add_context needs key and value")
            elif kind == "attach_evidence_ref" and not action.get("ref"):
                fail("evidence_ref_missing", "attach_evidence_ref names no ref")
    for ref in subjects:
        identifier = ref.get("screening_id") or ref.get("analysis_ref")
        if identifier and identifier not in known:
            fail("subject_not_an_input_artifact", f"{identifier} is not among the inputs")

    # Spans: the frozen full text holds them, or the gold says it could not.
    for span in gold.get("evidence_spans") or ():
        record_id = str(span.get("evidence_ref"))
        document = (full_text_documents or {}).get(record_id)
        if span.get("access") == "access_failed":
            if document is not None and document.get("outcome") == "retrieved":
                fail("declared_access_failure_but_text_held", record_id)
            continue
        if full_text_documents is None:
            continue
        if document is None or document.get("outcome") != "retrieved":
            fail("span_text_not_frozen", f"{record_id} has no frozen text")
            continue
        locators = {p.get("locator") for p in document.get("passages") or ()}
        locator = str(span.get("locator"))
        if not any(loc == locator or str(loc).startswith(locator + "/") for loc in locators):
            fail("span_locator_not_in_text", f"{record_id} has no passage {locator}")

    # Sources the gold relies on are in the corpus every arm reads.
    for source in gold.get("sources") or ():
        record_id = source.get("record_id")
        if record_id and snapshot_records is not None and record_id not in snapshot_records:
            fail("gold_source_not_in_corpus", str(record_id))

    # Gold isolation: no gold statement appears in what a system is given.
    facing = _model_facing_text(inputs)
    for conclusion in gold.get("acceptable_conclusions") or ():
        statement = str(conclusion.get("statement") or "").strip().casefold()
        if len(statement) >= 24 and statement in facing:
            fail("gold_leaked_into_inputs", f"conclusion {statement[:60]!r}")
    reference = str(gold.get("reference_outcome") or "").strip().casefold()
    if len(reference) >= 24 and reference in facing:
        fail("gold_leaked_into_inputs", "reference_outcome")
    if "gold" in facing and "gold_ref" in facing:
        fail("gold_referenced_from_inputs", "the inputs point at a gold file")

    if case.get("provenance", {}).get("kind") == "synthetic_engineering_fixture" and \
            gold.get("status") not in (None, "synthetic_engineering_gold"):
        fail("synthetic_case_with_verified_gold", "a synthetic fixture cannot carry SME gold")
    sair = (inputs.get("provenance") or {}).get("sair_status")
    if case.get("cohort") == "sair_span" and sair != "available":
        fail("sair_cohort_without_sair", "SAIR is unavailable; name the cohort used instead")
    return findings


def audit_set(
    loaded: Sequence[LoadedCase], **stores: Any,
) -> list[Finding]:
    """Every rule the whole set can break, and every case's own."""
    from .case_validator import validate_set

    findings: list[Finding] = []
    for item in loaded:
        findings.extend(audit_case(item, **stores))
    views = [as_v2_view(item) for item in loaded if item.gold]
    # Pairing, T3 polarity: the v2 set rules, over the v3 cases.
    set_findings = [f for f in validate_set(views)
                    if f.case_id == "<set>" or f.code.startswith(("paired", "pairing"))]
    findings.extend(set_findings)
    ids = [item.case_id for item in loaded]
    if len(ids) != len(set(ids)):
        findings.append(Finding("<set>", "duplicate_case_ids", "case ids repeat"))
    return findings


def _keys(case: Mapping[str, Any]) -> set[str]:
    groups = case.get("dependency_groups") or {}
    return {f"{kind}:{value}" for kind in DEPENDENCY_KINDS for value in groups.get(kind) or ()}


def dependency_components(cases: Iterable[Mapping[str, Any]]) -> list[list[str]]:
    """Cases joined by any shared compound, document, measurement or source
    (and by pairing): each component is one statistical unit, and one split."""
    cases = list(cases)
    parent = {c["case_id"]: c["case_id"] for c in cases}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    owner: dict[str, str] = {}
    for case in cases:
        for key in _keys(case):
            if key in owner:
                union(owner[key], case["case_id"])
            else:
                owner[key] = case["case_id"]
        partner = (case.get("paired_with") or {}).get("case_id")
        if partner in parent:
            union(partner, case["case_id"])
    components: dict[str, list[str]] = {}
    for case in cases:
        components.setdefault(find(case["case_id"]), []).append(case["case_id"])
    return sorted((sorted(c) for c in components.values()), key=lambda c: c[0])


def dependency_split(cases: Iterable[Mapping[str, Any]], *, test_fraction: float,
                     seed: int) -> dict[str, str]:
    """dev/test by whole components, assigned by a seeded shuffle: nothing a
    test case depends on is seen in dev. The fraction is approximate -- a
    component is never cut to hit it."""
    import random

    components = dependency_components(cases)
    rng = random.Random(seed)
    order = list(range(len(components)))
    rng.shuffle(order)
    total = sum(len(c) for c in components)
    split: dict[str, str] = {}
    in_test = 0
    for index in order:
        component = components[index]
        target = "test" if in_test + len(component) <= round(total * test_fraction) else "dev"
        if target == "test":
            in_test += len(component)
        for case_id in component:
            split[case_id] = target
    return split


def stores_from_files(*, replay_path: Path | None = None, full_text_path: Path | None = None,
                      snapshot_path: Path | None = None) -> dict[str, Any]:
    """The frozen stores an audit checks a set against, read from disk."""
    stores: dict[str, Any] = {}
    if replay_path is not None:
        document = json.loads(Path(replay_path).read_text())
        stores["replay_records"] = {
            str((r.get("screening_state") or {}).get("screening_id")): r
            for r in document.get("screenings") or ()
        }
        stores["frozen_analyses"] = dict(document.get("analyses") or {})
    if full_text_path is not None:
        stores["full_text_documents"] = dict(
            json.loads(Path(full_text_path).read_text()).get("documents") or {}
        )
    if snapshot_path is not None:
        evidence = json.loads(Path(snapshot_path).read_text()).get("evidence") or {}
        stores["snapshot_records"] = dict(evidence.get("records") or {})
        if "full_text_documents" not in stores and evidence.get("full_text"):
            stores["full_text_documents"] = dict(evidence["full_text"].get("documents") or {})
    return stores


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit a protocol v2 (v3 case) set.")
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--full-text", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    loaded = load_set(args.cases_dir, with_gold=True)
    findings = audit_set(loaded, **stores_from_files(
        replay_path=args.replay, full_text_path=args.full_text, snapshot_path=args.snapshot,
    ))
    for finding in findings:
        print(finding)
    print(f"{len(loaded)} case(s), {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
