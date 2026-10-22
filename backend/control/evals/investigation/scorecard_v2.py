"""Protocol v2 scoring: what the arms did, measured apart (#53).

Every number here is ``n``/``N`` with the denominator stated, and no number
is combined with another into a total. The pieces, and why each is its own
measurement:

* **Task Success** over every *assigned* case and trial. A response that
  never came -- an error, a timeout, a withheld result with nothing useful --
  is in the denominator and is not a success. A critical error makes a
  response unsuccessful whatever else a grader marked. Reported per workflow
  context (toxicity-only / activity-guided), per task group, and with the
  stress group (T6) kept out of the scientific partition.
* **The primary difference**, D - B, over the cases both arms were assigned:
  the paired discordance ``(n_D_only - n_B_only) / N_paired``, with a 95%
  interval from a bootstrap that resamples *dependency components* -- cases
  that share a compound, document, measurement or source are one unit, so an
  interval is not narrowed by counting them twice.
* **Mandatory information** the case requires: recall over the facts the gold
  names. Dropping every claim cannot buy this.
* **Limitations**: recall of the required codes, precision over what was
  stated, and whether every required one was stated. A case that requires
  none is ``NA``, never a free 100%.
* **Forbidden inferences and critical errors**: n/N each, by code.
* **Retrieval**, from the product trace alone (``retrieval.py``): record
  found / read / cited and span read / quoted, so a failure is placed before
  or after the source was read, apart from extraction, which a grader judges.
* **Cost and usage**, over every attempt including failed and diagnostic
  ones, with unknown usage kept unknown rather than read as zero.
* **Completeness**: ungraded cells, missing responses and grader
  disagreements awaiting adjudication. While any remain, the card says the
  scientific table is not publishable; it does not quietly drop them.

Grades are read from the sheets the packet shipped (``read_grades_v2``). A
grader's marks are facts about one response; the key says which arm it was.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evals.investigation.scorecard import GradeError, clopper_pearson

SCHEMA_VERSION = "investigation-scorecard-v2"

#: Why a response is not a success. One cause per response, so the table says
#: why as well as how often.
CAUSES = ("no_response", "critical_error", "not_graded", "graders_disagree", "task_not_met",
          "fallback", "withheld")

_LIST_COLUMNS = ("mandatory_facts_met", "limitations_stated", "forbidden_inferences_committed")


def _flag(value: str, where: str, column: str) -> bool | None:
    text = (value or "").strip()
    if text == "":
        return None
    if text not in {"0", "1"}:
        raise GradeError(f"{where}: {column} must be 0 or 1, got {text!r}")
    return text == "1"


def read_grades_v2(paths: Sequence[Path], *, criticals: Sequence[str]) -> list[dict[str, Any]]:
    """The grading sheets as rows. An empty cell is ungraded -- never a 0."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for number, row in enumerate(csv.DictReader(handle), 2):
                where = f"{Path(path).name}:{number}"
                for column in ("grader_id", "case_id", "response_id", "task_success"):
                    if column not in row:
                        raise GradeError(f"{where}: the sheet has no {column} column")
                grade: dict[str, Any] = {
                    "grader_id": (row["grader_id"] or "").strip() or Path(path).stem,
                    "case_id": row["case_id"].strip(),
                    "response_id": row["response_id"].strip(),
                    "task_success": _flag(row["task_success"], where, "task_success"),
                    "critical": {c: _flag(row.get(c, ""), where, c) for c in criticals},
                    "comments": row.get("comments", ""),
                }
                for column in _LIST_COLUMNS:
                    grade[column] = [v.strip() for v in (row.get(column) or "").split(";")
                                     if v.strip()]
                # Predictor adjudication marks (adjudication.py), on cases that have them.
                for column in ("agent_stance", "agent_label", "evidence_grounded"):
                    grade[column] = (row.get(column) or "").strip()
                rows.append(grade)
    return rows


def is_graded(grade: Mapping[str, Any]) -> bool:
    """Whether a row carries any judgement. A sheet returned with a row per
    response but nothing filled in is not grading, and must not read as a
    response judged unsuccessful."""
    return (grade.get("task_success") is not None
            or any(v is not None for v in (grade.get("critical") or {}).values())
            or any(grade.get(column) for column in _LIST_COLUMNS))


def _by_response(grades: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for grade in grades:
        if is_graded(grade):
            out[(grade["case_id"], grade["response_id"])].append(dict(grade))
    return out


def response_outcome(judged: Sequence[Mapping[str, Any]], *, missing_reason: str | None = None,
                     is_fallback: bool = False,
                     disposition: str | None = None) -> tuple[bool, str | None]:
    """Whether one response succeeded, and the one cause when it did not."""
    if missing_reason is not None:
        return False, f"no_response:{missing_reason}"
    if any(v for g in judged for v in g["critical"].values() if v is not None):
        return False, "critical_error"
    verdicts = [g["task_success"] for g in judged if g["task_success"] is not None]
    if not verdicts:
        return False, "not_graded"
    if len(set(verdicts)) > 1:
        return False, "graders_disagree"
    if verdicts[0]:
        return True, None
    if is_fallback:
        return False, "fallback"
    if disposition in ("withhold", "partial_facts"):
        return False, "withheld"
    return False, "task_not_met"


def _rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r["success"])
    return {"successes": k, "assigned": n, "rate": round(k / n, 4) if n else None,
            "ci95_clopper_pearson": clopper_pearson(k, n) if n else None}


def outcome_rows(*, key: Mapping[str, Any], grades: Sequence[Mapping[str, Any]],
                 cases: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """One row per assigned (case, trial) per arm: did it succeed, and why not."""
    judged = _by_response(grades)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in key["responses"]:
        case = cases.get(response["case_id"]) or {}
        success, cause = response_outcome(
            judged.get((response["case_id"], response.get("response_id")), []),
            missing_reason=None if response.get("response_id") else str(
                response.get("missing_reason") or "not_run"),
            is_fallback=bool(response.get("is_fallback")),
            disposition=response.get("disposition"),
        )
        rows[response["system_id"]].append({
            "case_id": response["case_id"], "trial": response.get("trial"),
            "response_id": response.get("response_id"), "success": success, "cause": cause,
            "task_group": case.get("task_group"),
            "workflow_context": case.get("workflow_context"),
            "stress": bool(case.get("stress_set")),
            "fault": ((case.get("fault") or {}).get("mechanism")),
        })
    return dict(rows)


def task_success(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Task Success per arm, partitioned the way the report prints it."""
    table: dict[str, Any] = {}
    for system_id, items in sorted(rows.items()):
        scientific = [r for r in items if not r["stress"]]
        stress = [r for r in items if r["stress"]]
        groups = sorted({r["task_group"] for r in items if r["task_group"]})
        contexts = sorted({r["workflow_context"] for r in items if r["workflow_context"]})
        faults = sorted({r["fault"] for r in stress if r["fault"]})
        table[system_id] = {
            "all_assigned": _rate(items),
            "scientific_tasks": _rate(scientific),
            "stress_set": _rate(stress),
            "by_workflow_context": {c: _rate([r for r in scientific
                                              if r["workflow_context"] == c]) for c in contexts},
            "by_task_group": {g: _rate([r for r in items if r["task_group"] == g])
                              for g in groups},
            "by_fault_mechanism": {f: _rate([r for r in stress if r["fault"] == f])
                                   for f in faults},
            "failure_causes": dict(sorted(Counter(
                r["cause"].split(":")[0] if r["cause"] else "none"
                for r in items if r["cause"]).items())),
            "note": "the denominator is every assigned case and trial; a response that never "
                    "came is in it and is not a success",
        }
    return table


# ------------------------------------------------------------------ pairing

def _component_of(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    from evals.investigation.case_inputs import dependency_components

    mapping: dict[str, str] = {}
    for component in dependency_components(
            [{**dict(c), "case_id": cid} for cid, c in cases.items()]):
        for case_id in component:
            mapping[case_id] = component[0]
    return mapping


def paired_delta(rows: Mapping[str, Sequence[Mapping[str, Any]]], *, system: str, baseline: str,
                 cases: Mapping[str, Mapping[str, Any]], reps: int = 2000,
                 seed: int = 20261022) -> dict[str, Any]:
    """D - B over the cases both arms were assigned.

    A multi-turn case is one unit, and cases joined by a shared compound,
    document, measurement or source are one cluster: the bootstrap resamples
    clusters, not cases, so dependence does not shrink the interval.
    """
    left = {(r["case_id"], r["trial"]): r["success"] for r in rows.get(system, ())}
    right = {(r["case_id"], r["trial"]): r["success"] for r in rows.get(baseline, ())}
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"system": system, "baseline": baseline, "paired_assigned": 0, "delta": None,
                "reason": "the arms share no assigned case"}
    component = _component_of(cases)
    by_cluster: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for case_id, trial in shared:
        by_cluster[component.get(case_id, case_id)].append((left[(case_id, trial)],
                                                            right[(case_id, trial)]))
    pairs = [p for items in by_cluster.values() for p in items]

    def delta_of(items: Sequence[tuple[bool, bool]]) -> float | None:
        if not items:
            return None
        only_left = sum(1 for a, b in items if a and not b)
        only_right = sum(1 for a, b in items if b and not a)
        return (only_left - only_right) / len(items)

    clusters = sorted(by_cluster)
    rng = random.Random(seed)
    deltas: list[float] = []
    if len(clusters) >= 2:
        for _ in range(reps):
            sample: list[tuple[bool, bool]] = []
            for _ in clusters:
                sample.extend(by_cluster[rng.choice(clusters)])
            value = delta_of(sample)
            if value is not None:
                deltas.append(value)
        deltas.sort()
    ci = ((round(deltas[int(0.025 * len(deltas))], 4),
           round(deltas[int(0.975 * len(deltas)) - 1], 4)) if deltas else None)
    return {
        "system": system, "baseline": baseline, "paired_assigned": len(pairs),
        "n_system_only": sum(1 for a, b in pairs if a and not b),
        "n_baseline_only": sum(1 for a, b in pairs if b and not a),
        "n_both": sum(1 for a, b in pairs if a and b),
        "n_neither": sum(1 for a, b in pairs if not a and not b),
        "delta": round(delta_of(pairs) or 0.0, 4),
        "clusters": len(clusters),
        "ci95_cluster_bootstrap": ci,
        "interval_method": (f"95% percentile bootstrap over {len(clusters)} dependency "
                            f"clusters, {reps} resamples, seed {seed}"),
        "note": "one unit per assigned case and trial; cases sharing a compound, document, "
                "measurement or source are one cluster",
    }


# -------------------------------------------------- required information

def mandatory_information(*, key: Mapping[str, Any], grades: Sequence[Mapping[str, Any]],
                          gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Recall of the facts a case requires, over the cases that require any.

    A response that was never produced owes every fact of its case, so its
    case counts with zero met: deleting an answer cannot raise this.
    """
    judged = _by_response(grades)
    out: dict[str, Any] = {}
    per_system: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for response in key["responses"]:
        required = [f["id"] for f in (gold.get(response["case_id"]) or {}).get(
            "mandatory_facts") or ()]
        if not required:
            continue
        marks = judged.get((response["case_id"], response.get("response_id")), [])
        if response.get("response_id") and not marks:
            continue  # ungraded: counted in completeness, not imputed here
        met = set(required)
        for grade in marks:
            met &= set(grade["mandatory_facts_met"])
        if not response.get("response_id"):
            met = set()
        per_system[response["system_id"]].append((len(met), len(required)))
    for system_id, items in sorted(per_system.items()):
        met = sum(m for m, _ in items)
        owed = sum(n for _, n in items)
        out[system_id] = {
            "facts_met": met, "facts_required": owed,
            "recall": round(met / owed, 4) if owed else None,
            "cases_with_required_facts": len(items),
            "all_facts_cases": sum(1 for m, n in items if m == n),
        }
    return out


def limitations(*, key: Mapping[str, Any], grades: Sequence[Mapping[str, Any]],
                gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Required-limitation recall, stated precision and all-required pass.

    Precision counts a stated code as correct when the case's gold names it
    required or optional. A case whose ``limitation_scoring`` is
    ``not_applicable``, or that requires none, is NA and is left out of every
    denominator here.
    """
    judged = _by_response(grades)
    per_system: dict[str, dict[str, int]] = defaultdict(
        lambda: {"required": 0, "recalled": 0, "stated": 0, "stated_ok": 0,
                 "cases": 0, "all_required": 0, "na": 0})
    for response in key["responses"]:
        case_gold = gold.get(response["case_id"]) or {}
        required = set(case_gold.get("required_limitations") or ())
        allowed = required | set(case_gold.get("optional_limitations") or ())
        row = per_system[response["system_id"]]
        if case_gold.get("limitation_scoring") == "not_applicable" or not required:
            row["na"] += 1
            continue
        marks = judged.get((response["case_id"], response.get("response_id")), [])
        if response.get("response_id") and not marks:
            continue
        stated: set[str] = set()
        if marks:
            stated = set.intersection(*(set(g["limitations_stated"]) for g in marks))
        row["cases"] += 1
        row["required"] += len(required)
        row["recalled"] += len(required & stated)
        row["stated"] += len(stated)
        row["stated_ok"] += len(stated & allowed)
        row["all_required"] += int(required <= stated)
    out: dict[str, Any] = {}
    for system_id, row in sorted(per_system.items()):
        out[system_id] = {
            **row,
            "required_recall": round(row["recalled"] / row["required"], 4) if row["required"] else None,
            "stated_precision": round(row["stated_ok"] / row["stated"], 4) if row["stated"] else None,
            "all_required_pass_rate": (round(row["all_required"] / row["cases"], 4)
                                       if row["cases"] else None),
            "note": "NA cases (no required limitation) are excluded from every denominator",
        }
    return out


def error_rates(rows: Mapping[str, Sequence[Mapping[str, Any]]], *,
                key: Mapping[str, Any], grades: Sequence[Mapping[str, Any]],
                criticals: Sequence[str]) -> dict[str, Any]:
    """Critical errors and forbidden inferences, n/N by code."""
    judged = _by_response(grades)
    graded_by_system: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for response in key["responses"]:
        if response.get("response_id"):
            graded_by_system[response["system_id"]].append(
                (response["case_id"], response["response_id"]))
    out: dict[str, Any] = {}
    for system_id, responses in sorted(graded_by_system.items()):
        graded = [r for r in responses if judged.get(r)]
        by_code: dict[str, int] = {}
        for code in criticals:
            by_code[code] = sum(
                1 for r in graded if any(g["critical"].get(code) for g in judged[r]))
        forbidden = Counter(
            code for r in graded for g in judged[r]
            for code in g["forbidden_inferences_committed"])
        any_critical = sum(
            1 for r in graded
            if any(v for g in judged[r] for v in g["critical"].values() if v is not None))
        out[system_id] = {
            "responses_graded": len(graded),
            "critical_errors": {code: {"n": n, "of": len(graded)} for code, n in by_code.items()},
            "any_critical_error": {"n": any_critical, "of": len(graded),
                                   "rate": round(any_critical / len(graded), 4) if graded else None},
            "forbidden_inferences": {code: {"n": n, "of": len(graded)}
                                     for code, n in sorted(forbidden.items())},
        }
    return out


# -------------------------------------------------------------- agreement

def agreement(grades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Inter-grader agreement on Task Success, and what is awaiting
    adjudication. Reported, never resolved by picking a grader."""
    by_cell: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for grade in grades:
        if grade["task_success"] is not None:
            by_cell[(grade["case_id"], grade["response_id"])][grade["grader_id"]] = \
                bool(grade["task_success"])
    graders = sorted({g["grader_id"] for g in grades})
    overlapping = {k: v for k, v in by_cell.items() if len(v) >= 2}
    agreed = sum(1 for v in overlapping.values() if len(set(v.values())) == 1)
    kappa = None
    if len(graders) >= 2 and overlapping:
        first, second = graders[0], graders[1]
        pairs = [(v[first], v[second]) for v in overlapping.values()
                 if first in v and second in v]
        if pairs:
            n = len(pairs)
            observed = sum(a == b for a, b in pairs) / n
            p_left = sum(a for a, _ in pairs) / n
            p_right = sum(b for _, b in pairs) / n
            expected = p_left * p_right + (1 - p_left) * (1 - p_right)
            kappa = round((observed - expected) / (1 - expected), 4) if expected < 1 else None
    return {
        "graders": graders, "cells_graded": len(by_cell), "cells_overlapping": len(overlapping),
        "exact_agreement": round(agreed / len(overlapping), 4) if overlapping else None,
        "cohens_kappa_task_success": kappa,
        "pending_adjudication": sorted(
            f"{case_id}/{response_id}" for (case_id, response_id), v in overlapping.items()
            if len(set(v.values())) > 1),
    }


def completeness(*, key: Mapping[str, Any], grades: Sequence[Mapping[str, Any]],
                 rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """What is not finished. While anything here is non-zero the scientific
    table is not publishable, and the card says so rather than printing one."""
    judged = _by_response(grades)
    expected = [r for r in key["responses"] if r.get("response_id")]
    ungraded = [f"{r['case_id']}/{r['response_id']}" for r in expected
                if not judged.get((r["case_id"], r["response_id"]))]
    missing = [{"case_id": r["case_id"], "system_id": r["system_id"], "trial": r.get("trial"),
                "reason": r.get("missing_reason")} for r in key["responses"]
               if not r.get("response_id")]
    disagreements = agreement(grades)["pending_adjudication"]
    gold_status = sorted({str(s) for s in (key.get("gold_status") or {}).values()})
    publishable = not ungraded and not disagreements and gold_status in ([], ["sme_verified"])
    return {
        "responses_in_packet": len(expected), "ungraded_responses": ungraded,
        "missing_responses": missing, "pending_adjudication": disagreements,
        "gold_status": gold_status,
        "scientific_table_publishable": publishable,
        "why_not": [reason for reason, holds in (
            ("responses are ungraded", bool(ungraded)),
            ("grader disagreements are unadjudicated", bool(disagreements)),
            ("the gold is not SME-verified", gold_status not in ([], ["sme_verified"])),
        ) if holds],
    }


def build_scorecard_v2(
    *, key: Mapping[str, Any], grades: Sequence[Mapping[str, Any]],
    cases: Mapping[str, Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]],
    criticals: Sequence[str], compare: Sequence[tuple[str, str]] = (),
    reps: int = 2000, seed: int = 20261022, cost: Mapping[str, Any] | None = None,
    retrieval: Mapping[str, Any] | None = None,
    claims: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    from evals.investigation import adjudication as adjudication_module
    from evals.investigation import claims as claims_module

    rows = outcome_rows(key=key, grades=grades, cases=cases)
    return {
        "schema_version": SCHEMA_VERSION, "study_id": key.get("study_id"),
        "packet_id": key.get("packet_id"), "rubric_version": key.get("rubric_version"),
        "unit_of_analysis": "one assigned case and trial; a multi-turn case is one unit",
        "task_success": task_success(rows),
        "paired": [paired_delta(rows, system=left, baseline=right, cases=cases, reps=reps,
                                seed=seed) for left, right in compare],
        "mandatory_information": mandatory_information(key=key, grades=grades, gold=gold),
        "limitations": limitations(key=key, grades=grades, gold=gold),
        "errors": error_rates(rows, key=key, grades=grades, criticals=criticals),
        "agreement": agreement(grades),
        "completeness": completeness(key=key, grades=grades, rows=rows),
        "retrieval": dict(retrieval or {}),
        "hallucination": (claims_module.hallucination(key=key, claims=claims)
                          if claims else {}),
        "adjudication": adjudication_module.adjudication(key=key, grades=grades, gold=gold),
        "cost": dict(cost or {}),
        "note": "Dimensions are separate measurements; there is no total score. Engineering "
                "conformance on synthetic fixtures is not scientific performance.",
    }


def render_markdown(card: Mapping[str, Any]) -> str:
    lines = [f"# Scorecard v2 — {card['study_id']} / {card['packet_id']}", "",
             f"Unit: {card['unit_of_analysis']}. {card['note']}", ""]
    completion = card["completeness"]
    if not completion["scientific_table_publishable"]:
        lines += [f"> **Not publishable as a scientific result:** "
                  f"{'; '.join(completion['why_not'])}.", ""]
    lines += ["## Task success (every assigned case and trial)", "",
              "| Arm | All | Scientific | Stress (T6) | Failure causes |", "|---|---|---|---|---|"]

    def cell(rate: Mapping[str, Any]) -> str:
        if rate.get("rate") is None:
            return "—"
        ci = rate.get("ci95_clopper_pearson")
        text = f"{rate['successes']}/{rate['assigned']} ({rate['rate']:.0%})"
        return text + (f" [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else "")

    for system_id, table in card["task_success"].items():
        causes = ", ".join(f"{k}: {v}" for k, v in table["failure_causes"].items()) or "—"
        lines.append(f"| `{system_id}` | {cell(table['all_assigned'])} | "
                     f"{cell(table['scientific_tasks'])} | {cell(table['stress_set'])} | "
                     f"{causes} |")
    for system_id, table in card["task_success"].items():
        if table["by_workflow_context"]:
            lines += ["", f"### `{system_id}` by workflow context and task group", "",
                      "| Partition | Task success |", "|---|---|"]
            for name, rate in table["by_workflow_context"].items():
                lines.append(f"| context {name} | {cell(rate)} |")
            for name, rate in table["by_task_group"].items():
                lines.append(f"| {name} | {cell(rate)} |")
            for name, rate in table["by_fault_mechanism"].items():
                lines.append(f"| fault {name} | {cell(rate)} |")
    for paired in card["paired"]:
        if paired.get("delta") is None:
            continue
        ci = paired.get("ci95_cluster_bootstrap")
        lines += ["", f"## `{paired['system']}` − `{paired['baseline']}` (paired)", "",
                  f"- paired assigned: {paired['paired_assigned']} "
                  f"({paired['clusters']} dependency cluster(s))",
                  f"- {paired['system']} only: {paired['n_system_only']}; "
                  f"{paired['baseline']} only: {paired['n_baseline_only']}; "
                  f"both: {paired['n_both']}; neither: {paired['n_neither']}",
                  f"- delta: {paired['delta']:+.4f}"
                  + (f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else " (no interval: one cluster)"),
                  f"- {paired['interval_method']}"]
    if card["mandatory_information"]:
        lines += ["", "## Mandatory information", "",
                  "| Arm | Facts met / required | Recall | Cases with every fact |",
                  "|---|---|---|---|"]
        for system_id, row in card["mandatory_information"].items():
            recall = "—" if row["recall"] is None else f"{row['recall']:.0%}"
            lines.append(f"| `{system_id}` | {row['facts_met']}/{row['facts_required']} | "
                         f"{recall} | {row['all_facts_cases']}/{row['cases_with_required_facts']} |")
    if card["limitations"]:
        lines += ["", "## Limitations", "",
                  "| Arm | Required recall | Stated precision | All-required | NA cases |",
                  "|---|---|---|---|---|"]
        for system_id, row in card["limitations"].items():
            fmt = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
            lines.append(f"| `{system_id}` | {fmt(row['required_recall'])} | "
                         f"{fmt(row['stated_precision'])} | {fmt(row['all_required_pass_rate'])} "
                         f"| {row['na']} |")
    if card["errors"]:
        lines += ["", "## Critical errors", "", "| Arm | Any | By code |", "|---|---|---|"]
        for system_id, row in card["errors"].items():
            any_error = row["any_critical_error"]
            codes = ", ".join(f"{k} {v['n']}/{v['of']}" for k, v in row["critical_errors"].items()
                              if v["n"]) or "—"
            lines.append(f"| `{system_id}` | {any_error['n']}/{any_error['of']} | {codes} |")
    cost = card.get("cost") or {}
    if cost:
        lines += ["", "## Cost and usage (every attempt)", "",
                  "| Arm | Attempts | Turns | Median s/turn | Tokens | Cost | Unknown usage |",
                  "|---|---|---|---|---|---|---|"]
        for system_id, row in cost.items():
            tokens = ", ".join(f"{k.replace('tokens_', '')}={v:g}"
                               for k, v in sorted(row.get("tokens", {}).items())) or "—"
            money = ", ".join(f"{k}={v:g}" for k, v in sorted(row.get("cost", {}).items())) or "—"
            lines.append(f"| `{system_id}` | {row.get('attempts')} | {row.get('turns')} | "
                         f"{row.get('median_turn_seconds') or '—'} | {tokens} | {money} | "
                         f"{row.get('records_with_unknown_usage')} |")
    from evals.investigation import adjudication as adjudication_module
    from evals.investigation import claims as claims_module

    if card.get("hallucination"):
        lines += claims_module.render_markdown(card["hallucination"])
    lines += adjudication_module.render_markdown(card.get("adjudication") or {})
    agree = card["agreement"]
    lines += ["", "## Grading", "",
              f"- graders: {', '.join(agree['graders']) or 'none'}",
              f"- cells graded: {agree['cells_graded']}, overlapping: "
              f"{agree['cells_overlapping']}, exact agreement: {agree['exact_agreement']}, "
              f"kappa: {agree['cohens_kappa_task_success']}",
              f"- pending adjudication: {', '.join(agree['pending_adjudication']) or 'none'}",
              f"- ungraded responses: {', '.join(completion['ungraded_responses']) or 'none'}",
              f"- missing responses: {len(completion['missing_responses'])}"]
    return "\n".join(lines) + "\n"


def usage_summary(records: Iterable[Any]) -> dict[str, Any]:
    """Cost and latency per arm over every attempt, failed and diagnostic
    ones included. A record whose usage is unknown is counted as unknown, not
    as zero, and the count is reported beside the totals."""
    per_system: dict[str, dict[str, Any]] = {}
    for record in records:
        row = per_system.setdefault(record.system_id, {
            "attempts": 0, "primary_attempts": 0, "diagnostic_attempts": 0, "turns": 0,
            "tokens": {}, "cost": {}, "records_with_unknown_usage": 0, "durations": [],
            "phases": {},
        })
        row["attempts"] += 1
        row["primary_attempts"] += int(record.attempt_kind != "diagnostic_rerun")
        row["diagnostic_attempts"] += int(record.attempt_kind == "diagnostic_rerun")
        row["turns"] += len(record.turns)
        usage = record.usage or {}
        if usage.get("status") != "reported":
            row["records_with_unknown_usage"] += 1
        for name, value in usage.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            bucket = "tokens" if name.startswith("tokens_") else "cost" if name.startswith("cost_") \
                else None
            if bucket:
                row[bucket][name] = round(row[bucket].get(name, 0) + value, 8)
        for turn in record.turns:
            if isinstance(turn.duration_s, (int, float)):
                row["durations"].append(float(turn.duration_s))
            for phase, entry in ((turn.meta.get("phase_usage") or {}).get("phases") or {}).items():
                into = row["phases"].setdefault(phase, {"dispatches": 0, "calls": 0})
                into["dispatches"] += int(entry.get("dispatches") or 0)
                into["calls"] += int(sum((entry.get("calls") or {}).values())
                                     if isinstance(entry.get("calls"), dict) else 0)
    for row in per_system.values():
        durations = row.pop("durations")
        row["median_turn_seconds"] = (round(statistics.median(durations), 1)
                                      if durations else None)
    return dict(sorted(per_system.items()))


def retrieval_summary(*, key: Mapping[str, Any], gold: Mapping[str, Mapping[str, Any]],
                      traces: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> dict[str, Any]:
    """Record- and span-level retrieval per arm, from the traces alone."""
    from evals.investigation.retrieval import retrieval_recall

    per_system: dict[str, dict[str, Any]] = {}
    for response in key["responses"]:
        trace = traces.get((response["case_id"], response["system_id"],
                            int(response.get("trial") or 1)))
        spans = (gold.get(response["case_id"]) or {}).get("evidence_spans") or []
        if trace is None or not spans:
            continue
        measured = retrieval_recall({"evidence_spans": spans}, trace)
        if not measured.get("scored"):
            continue
        row = per_system.setdefault(response["system_id"], {
            "cases_scored": 0, "records": {}, "spans": {}, "unusable_refs": []})
        row["cases_scored"] += 1
        for level in ("records", "spans"):
            for name, counts in measured[level].items():
                into = row[level].setdefault(name, {"n": 0, "of": 0})
                into["n"] += counts["n"]
                into["of"] += counts["of"]
        row["unusable_refs"] = sorted(set(row["unusable_refs"]) | set(measured["unusable_refs"]))
    for row in per_system.values():
        for level in ("records", "spans"):
            for counts in row[level].values():
                counts["rate"] = round(counts["n"] / counts["of"], 4) if counts["of"] else None
    return dict(sorted(per_system.items()))


def write_card(card: Mapping[str, Any], directory: Path) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scorecard.json").write_text(
        json.dumps(card, indent=2, default=str) + "\n", encoding="utf-8")
    (directory / "scorecard.md").write_text(render_markdown(card), encoding="utf-8")
    return {"json": str(directory / "scorecard.json"), "markdown": str(directory / "scorecard.md")}


def score_study(*, study_dir: Path, case_set_dir: Path, packet_id: str, grades: Sequence[Path],
                compare: Sequence[tuple[str, str]] = (), reps: int = 2000,
                seed: int = 20261022, claims: Sequence[Path] = ()) -> dict[str, Any]:
    """The card of one packet: its grades, the study's traces, its case set."""
    from evals.investigation.case_inputs import load_case, load_case_set
    from evals.investigation.packet import RUBRIC_PATH
    from evals.investigation.record import StudyStore

    store = StudyStore(study_dir)
    key = json.loads((study_dir / "keys" / f"{packet_id}.unblinding.json").read_text())
    rubric = json.loads(RUBRIC_PATH.read_text())
    criticals = [c["id"] for c in rubric["critical_errors"]]
    case_set = load_case_set(case_set_dir)
    cases = {c.case_id: c.case for c in case_set.cases}
    gold: dict[str, Any] = {}
    for loaded in case_set.cases:
        with_gold = load_case(loaded.path, with_gold=True)
        if with_gold.gold is not None:
            gold[loaded.case_id] = with_gold.gold
    records = store.records()
    traces: dict[tuple[str, str, int], Any] = {}
    for record in store.primary_records().values():
        path = store.root / record.artifacts.get("trace", "")
        if record.artifacts.get("trace") and path.exists():
            traces[(record.case_id, record.system_id, record.trial)] = json.loads(
                path.read_text())
    return build_scorecard_v2(
        key=key, grades=read_grades_v2(grades, criticals=criticals), cases=cases, gold=gold,
        criticals=criticals, compare=compare, reps=reps, seed=seed,
        cost=usage_summary(records),
        retrieval=retrieval_summary(key=key, gold=gold, traces=traces),
        claims=_claim_rows(claims),
    )


def _claim_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    from evals.investigation.claims import read_claim_sheets

    return read_claim_sheets(paths) if paths else []


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from evals.investigation.run import DEFAULT_ROOT

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-set", type=Path, required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--grades", type=Path, nargs="+", required=True)
    parser.add_argument("--compare", action="append", default=[], metavar="SYSTEM:BASELINE")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--claims", type=Path, nargs="*", default=[],
                        help="atomic-claim sheets (claims.py) for the hallucination section")
    args = parser.parse_args(argv)
    study_dir = args.root / args.study
    card = score_study(
        study_dir=study_dir, case_set_dir=args.case_set, packet_id=args.packet_id,
        grades=args.grades, reps=args.reps, claims=args.claims,
        compare=[tuple(item.split(":", 1)) for item in args.compare],
    )
    written = write_card(card, study_dir / "scorecards" / args.packet_id)
    print(render_markdown(card))
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
