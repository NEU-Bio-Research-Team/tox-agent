"""Predictor adjudication: does the agent know when the predictor is right?

The predictor layer gives a call -- a hERG blocker label, an activity estimate
that clears or misses a target threshold. An investigating agent earns its
place if it can tell, from the evidence it can reach, when that call holds and
when it does not: **corroborate** a right call, **challenge** a wrong one with
the evidence that shows it, and say the evidence is **insufficient** when it
cannot decide -- instead of agreeing by default or contradicting at random.

Each adjudication case's gold says (``adjudication``):

* ``predictor_call`` and ``verified`` -- ``positive`` / ``negative`` (blocker or
  not; active above the threshold or not), the latter from a measurement the
  case declares, so ``predictor_correct`` is known;
* ``decidable_from_corpus`` -- whether the corpus the system can search holds
  the deciding measurement;
* ``expected_stance`` -- ``corroborate`` / ``challenge`` / ``insufficient``.

A grader marks, per response: ``agent_stance`` (corroborate, challenge,
insufficient, or none when the reply takes no position), ``agent_label``
(positive, negative, undetermined -- the reply's own final call), and
``evidence_grounded`` (1 when a corroboration or challenge cites the deciding
source). A missing reply is stance ``none``, kept in every denominator.

The measures, per arm, kept apart:

* stance accuracy against the expected stance;
* on decidable cases: correction rate (challenges of wrong calls), uncaught
  error rate (corroborations of wrong calls -- the critical one), corroboration
  rate of right calls, false-alarm rate (challenges of right calls);
* unsupported corroboration on undecidable cases;
* final-label accuracy against the verified label, beside the predictor's own
  accuracy on the same cases -- the agent's lift, with its coverage;
* grounded share of corroborations and challenges.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "predictor-adjudication-v1"
STANCES = ("corroborate", "challenge", "insufficient", "none")
LABELS = ("positive", "negative", "undetermined")
COLUMNS = ("agent_stance", "agent_label", "evidence_grounded")

INSTRUCTIONS = """## Predictor adjudication columns (cases whose reference has a "Predictor adjudication" section)

The reference states what the predictor called, the verified answer, and whether
the corpus can decide it. For each response, judge the **final result**:

- `agent_stance`: `corroborate` (the reply says the evidence supports the
  predictor's call), `challenge` (it says the evidence contradicts the call),
  `insufficient` (it says the evidence cannot decide), or `none` (it takes no
  position on the predictor's call at all, including no reply).
- `agent_label`: the reply's own final call on the endpoint -- `positive`
  (blocker / active above the threshold), `negative`, or `undetermined`.
- `evidence_grounded`: 1 if a corroboration or challenge cites the deciding
  source the reference names, 0 if it does not; empty for `insufficient`/`none`.

Leave these three columns empty on cases without an adjudication section.
"""


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _stance_of(grade: Mapping[str, Any] | None) -> str:
    stance = (grade or {}).get("agent_stance") or "none"
    return stance if stance in STANCES else "none"


def adjudication(*, key: Mapping[str, Any], grades: Sequence[Mapping[str, Any]],
                 gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per grader and arm. Cases without an ``adjudication`` gold are skipped."""
    cases = {cid: g["adjudication"] for cid, g in gold.items() if g.get("adjudication")}
    if not cases:
        return {}
    by_grader: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for grade in grades:
        if grade.get("agent_stance") or grade.get("agent_label"):
            by_grader[grade["grader_id"]][(grade["case_id"], grade["response_id"])] = grade
    assigned = [r for r in key.get("responses") or () if r["case_id"] in cases]
    out: dict[str, dict[str, Any]] = {}
    stance_by_cell: dict[tuple[str, str, int], dict[str, str]] = defaultdict(dict)
    for grader, marks in sorted(by_grader.items()):
        table: dict[str, dict[str, Any]] = {}
        per_system: dict[str, list[tuple[Mapping, str, str, bool]]] = defaultdict(list)
        for response in assigned:
            spec = cases[response["case_id"]]
            mark = marks.get((response["case_id"], response.get("response_id")))
            if response.get("response_id") and mark is None:
                continue  # not graded by this grader; reported as ungraded below
            stance = _stance_of(mark)
            label = (mark or {}).get("agent_label") or "undetermined"
            grounded = str((mark or {}).get("evidence_grounded") or "") == "1"
            per_system[response["system_id"]].append((spec, stance, label, grounded))
            stance_by_cell[(response["case_id"], response["system_id"],
                            int(response.get("trial") or 1))][grader] = stance
        for system, items in sorted(per_system.items()):
            decidable = [i for i in items if i[0].get("decidable_from_corpus")]
            undecidable = [i for i in items if not i[0].get("decidable_from_corpus")]
            wrong = [i for i in decidable if i[0]["predictor_call"] != i[0]["verified"]]
            right = [i for i in decidable if i[0]["predictor_call"] == i[0]["verified"]]
            committed = [i for i in decidable if i[2] in ("positive", "negative")]
            agent_correct = sum(i[2] == i[0]["verified"] for i in decidable)
            predictor_correct = sum(i[0]["predictor_call"] == i[0]["verified"] for i in decidable)
            positions = [i for i in items if i[1] in ("corroborate", "challenge")]
            table[system] = {
                "cases": len(items),
                "stance_correct": sum(i[1] == i[0]["expected_stance"] for i in items),
                "stance_accuracy": _rate(sum(i[1] == i[0]["expected_stance"] for i in items),
                                         len(items)),
                "stance_counts": {s: sum(i[1] == s for i in items) for s in STANCES},
                "decidable": len(decidable), "predictor_wrong": len(wrong),
                "predictor_right": len(right),
                "correction_rate": _rate(sum(i[1] == "challenge" for i in wrong), len(wrong)),
                "uncaught_error_rate": _rate(sum(i[1] == "corroborate" for i in wrong), len(wrong)),
                "corroboration_rate": _rate(sum(i[1] == "corroborate" for i in right), len(right)),
                "false_alarm_rate": _rate(sum(i[1] == "challenge" for i in right), len(right)),
                "abstained_on_decidable": sum(i[1] in ("insufficient", "none") for i in decidable),
                "undecidable": len(undecidable),
                "unsupported_corroboration_rate": _rate(
                    sum(i[1] == "corroborate" for i in undecidable), len(undecidable)),
                "agent_label_accuracy": _rate(agent_correct, len(decidable)),
                "predictor_accuracy": _rate(predictor_correct, len(decidable)),
                "lift": (round((agent_correct - predictor_correct) / len(decidable), 4)
                         if decidable else None),
                "agent_label_coverage": _rate(len(committed), len(decidable)),
                "grounded_positions": sum(i[3] for i in positions),
                "positions": len(positions),
                "grounded_rate": _rate(sum(i[3] for i in positions), len(positions)),
            }
        out[grader] = table
    both = [v for v in stance_by_cell.values() if len(v) >= 2]
    agree = sum(len(set(v.values())) == 1 for v in both)
    return {
        "schema_version": SCHEMA_VERSION,
        "by_grader": out,
        "agreement": {"cells_by_two": len(both), "stance_agree": agree,
                      "rate": _rate(agree, len(both)),
                      "disagreements": sorted(f"{c}/{s}" for (c, s, _), v in stance_by_cell.items()
                                              if len(v) >= 2 and len(set(v.values())) > 1)},
        "note": "The predictor's accuracy here is the set's designed prevalence, not an estimate "
                "of the real predictor. Lift compares the agent's own final call with the "
                "predictor's on the same decidable cases; an undetermined call counts as wrong "
                "and shows in coverage.",
    }


def render_markdown(section: Mapping[str, Any]) -> list[str]:
    if not section:
        return []

    def pct(value):
        return "—" if value is None else f"{value:.0%}"

    lines = ["", "## Predictor adjudication (does the agent know when the predictor is right?)",
             "", "| Grader | Arm | Stance accuracy | Correction (wrong call challenged) | "
             "Uncaught error (wrong call corroborated) | Corroboration (right call) | "
             "False alarm (right call challenged) | Unsupported corroboration | "
             "Agent label acc. vs predictor | Coverage | Grounded positions |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for grader, table in section.get("by_grader", {}).items():
        for system, r in table.items():
            lines.append(
                f"| {grader} | `{system}` | {r['stance_correct']}/{r['cases']} "
                f"({pct(r['stance_accuracy'])}) | {pct(r['correction_rate'])} of "
                f"{r['predictor_wrong']} | {pct(r['uncaught_error_rate'])} | "
                f"{pct(r['corroboration_rate'])} of {r['predictor_right']} | "
                f"{pct(r['false_alarm_rate'])} | {pct(r['unsupported_corroboration_rate'])} of "
                f"{r['undecidable']} | {pct(r['agent_label_accuracy'])} vs "
                f"{pct(r['predictor_accuracy'])} (lift {r['lift']:+.2f}) | "
                f"{pct(r['agent_label_coverage'])} | {r['grounded_positions']}/{r['positions']} |"
                if r["lift"] is not None else
                f"| {grader} | `{system}` | {r['stance_correct']}/{r['cases']} | — | — | — | — | — "
                f"| — | — | — |")
    agree = section.get("agreement") or {}
    lines += ["", f"- stance agreement between graders: {agree.get('stance_agree')}/"
                  f"{agree.get('cells_by_two')}; disagreements: "
                  f"{', '.join(agree.get('disagreements') or ()) or 'none'}",
              f"- {section.get('note')}"]
    return lines
