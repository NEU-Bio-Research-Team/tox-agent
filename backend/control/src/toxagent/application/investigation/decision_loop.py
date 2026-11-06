"""The gap-driven decision loop (#70, §6.2–6.3) and discriminating actions (#71, §6.4).

§6.2's seven steps: read the case view → pick the most important unresolved
requirement → pick one allowed action (or answer / ask / stop) → run it,
bounded → validate what came back → commit a small patch → loop only while a
required question is open. Steps one, four, five and six already exist (the
case view, the tools, the validators, the case operations). This module is
steps two, three and seven, as pure functions over a recorded *ledger*:

* **gaps** — one per requirement of the goal contract, typed by
  ``domain.investigation_gap.Gap``, with the claim each one limits. A stop
  that says "insufficient" reads its reasons from here, never from prose;
* **actions** — each one the run took, with the gap it served, what it was
  meant to discriminate, and one public sentence of why (§10.4: no long
  private reasoning is kept);
* **explanations** — the competing explanations of a disagreement §6.4 asks
  the investigator to hold, each typed.

Ranking follows §6.4's order exactly: (1) does it resolve an identity/scope
blocker; (2) does it target an unsupported decisive required claim; (3) is the
source reachable and likely to carry the data; (4) does it reuse an
observation already held before searching anew; (5) does its cost fit what is
left. This is a *directed heuristic*, not Bayesian information gain — the doc
says so and so does this code. Criterion (3)'s "likely to carry the data" is
an estimate to be checked on the dev set; it is not a learned probability and
nothing here presents it as one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ...domain.investigation_gap import GAP_PRIORITY, Gap, insufficiency, open_gaps  # noqa: F401

LEDGER_VERSION = "decision-loop-v1"

#: Actions that are not tool calls.
TERMINAL_ACTIONS = frozenset({"answer", "ask_user", "stop"})
#: Actions that look for new material, and so are subject to repeat refusal.
SEARCH_ACTIONS = frozenset({
    "search_toxicology_evidence", "get_chembl_activities", "get_evidence_record",
    "read_evidence_text", "delegate_evidence_workers",
})


class Explanation(str, Enum):
    """§6.4's competing explanations of a disagreement. One of these, never prose."""

    #: The prediction disagrees with a measurement taken in the same scope.
    PREDICTION_MISMATCH_IN_SCOPE = "prediction_mismatch_in_scope"
    #: The assays or readouts differ, so the two are not directly comparable.
    ASSAY_OR_READOUT_DIFFERS = "assay_or_readout_differs"
    #: An identifier or variant was mapped to the wrong entity.
    IDENTITY_MISMAPPED = "identity_mismapped"
    #: The second source repeats the same experiment; it is not independent.
    SAME_EXPERIMENT_REPEATED = "same_experiment_repeated"


#: A dev-set estimate, by source, of how often a reachable source of that
#: kind carries a measured value for a toxicology question. NOT a learned
#: probability: a prior to be checked on the dev set (#71), recorded with each
#: ranking so a reader sees what the order rested on.
CARRIES_DATA_ESTIMATE: Mapping[str, float] = {
    "observation": 1.0,        # already held: it is the data
    "chembl": 0.7,             # assay records with values
    "full_text": 0.6,          # tables and methods of a record found
    "europepmc": 0.4,          # abstracts first; values often in the body
    "pubchem": 0.3,
    "unknown": 0.3,
}


#: The procedure that closes each gap, where one does (§7.1). Live (29 arm-E
#: turns, 2026-10-29) the model never called run_scientific_procedure: it was
#: listed among twenty tools with nothing tying it to the moment it helps.
#: Each open gap now names it.
GAP_PROCEDURE: Mapping[str, str] = {
    Gap.IDENTITY_OR_SCOPE.value: "resolve_subject_context",
    Gap.MEASUREMENT_UNRESOLVED.value: "normalise_measurements",
    Gap.DISAGREEMENT_UNEXPLAINED.value: "compare_measurements (then deduplicate_by_origin)",
    Gap.NO_RELEVANT_EVIDENCE.value: "collect_assay_records over what the searches returned",
}


def with_procedures(gaps: Sequence[Mapping[str, Any]], *, available: bool) -> list[dict[str, Any]]:
    """Open gaps, each with the run_scientific_procedure that closes it."""
    out = []
    for gap in gaps:
        item = dict(gap)
        if available and gap.get("gap") in GAP_PROCEDURE:
            item["procedure"] = "run_scientific_procedure " + GAP_PROCEDURE[gap["gap"]]
        out.append(item)
    return out


def normalise_target(target: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.:/-]", " ", (target or "").lower())).strip()


@dataclass(frozen=True)
class Candidate:
    action: str
    target: str = ""
    gap: str = ""
    requirement_id: str = ""
    discriminates: tuple[str, ...] = ()
    source: str = "unknown"
    reuses: str = ""
    cost: int = 1
    reason: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Candidate":
        return cls(
            action=str(data.get("action", "")), target=str(data.get("target") or ""),
            gap=str(data.get("gap") or ""), requirement_id=str(data.get("requirement_id") or ""),
            discriminates=tuple(str(d) for d in data.get("discriminates") or ()),
            source=str(data.get("source") or "unknown"), reuses=str(data.get("reuses") or ""),
            cost=int(data.get("cost") or 1), reason=str(data.get("reason") or "")[:240],
        )


def seed_ledger(contract: Mapping[str, Any]) -> dict[str, Any]:
    """One gap per required question of the goal contract."""
    gaps = []
    for question in contract.get("must_answer") or ():
        needs = list(question.get("needs") or ())
        gap = question.get("gap") or (
            Gap.NO_RELEVANT_EVIDENCE.value if "evidence" in needs
            else Gap.ONLY_RESEARCHER_HAS_IT.value if needs == ["researcher_input"]
            else ""
        )
        gaps.append({
            "requirement_id": question.get("id"), "limits": question.get("question", ""),
            "gap": gap, "status": "open" if gap else "unassessed", "refs": [],
            "decisive": bool(question.get("decisive", "evidence" in needs)),
        })
    return {"version": LEDGER_VERSION, "gaps": gaps, "actions": [], "explanations": [],
            "counter_evidence": {}}


def _accessible(source: str, access_failures: Iterable[Mapping[str, Any]]) -> bool:
    failed = {str(f.get("provider", "")).lower() for f in access_failures}
    return source.lower() not in failed


def rank(
    candidates: Sequence[Candidate], *, ledger: Mapping[str, Any],
    access_failures: Iterable[Mapping[str, Any]] = (), budget_left: int | None = None,
) -> list[dict[str, Any]]:
    """Candidates in §6.4 order, each with the criteria it was ranked on."""
    failures = list(access_failures)
    by_requirement = {g.get("requirement_id"): g for g in ledger.get("gaps") or ()}
    ranked = []
    for index, candidate in enumerate(candidates):
        gap = by_requirement.get(candidate.requirement_id) or {}
        accessible = _accessible(candidate.source, failures)
        criteria = {
            "resolves_identity_or_scope": candidate.gap == Gap.IDENTITY_OR_SCOPE.value
            or Explanation.IDENTITY_MISMAPPED.value in candidate.discriminates,
            "targets_open_decisive_claim": bool(gap.get("decisive")) and gap.get("status") == "open",
            "accessible": accessible,
            "carries_data_estimate": (
                CARRIES_DATA_ESTIMATE.get(candidate.source, CARRIES_DATA_ESTIMATE["unknown"])
                if accessible else 0.0
            ),
            "reuses_observation": bool(candidate.reuses),
            "fits_budget": budget_left is None or candidate.cost <= budget_left,
        }
        key = (
            criteria["resolves_identity_or_scope"],
            criteria["targets_open_decisive_claim"],
            criteria["accessible"] * criteria["carries_data_estimate"],
            criteria["reuses_observation"],
            criteria["fits_budget"],
            -index,
        )
        ranked.append((key, {"index": index, "action": candidate.action,
                             "target": candidate.target, "gap": candidate.gap,
                             "requirement_id": candidate.requirement_id,
                             "discriminates": list(candidate.discriminates),
                             "criteria": criteria}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in ranked]


def refusals(
    candidates: Sequence[Candidate], *, ledger: Mapping[str, Any], admissible: Iterable[str],
) -> dict[int, str]:
    """Why a candidate may not be chosen, by index. Absent means admissible."""
    allowed = set(admissible) | TERMINAL_ACTIONS
    gap_values = {g.value for g in Gap}
    explanation_values = {e.value for e in Explanation}
    done = {
        (a.get("action"), normalise_target(a.get("target", "")))
        for a in ledger.get("actions") or ()
        if a.get("action") in SEARCH_ACTIONS
    }
    out: dict[int, str] = {}
    for index, candidate in enumerate(candidates):
        if candidate.action not in allowed:
            out[index] = f"{candidate.action!r} is not an action this run may take"
        elif candidate.action not in TERMINAL_ACTIONS and not candidate.discriminates:
            out[index] = "name the gap or explanation its result could tell apart (discriminates)"
        elif any(d not in gap_values | explanation_values for d in candidate.discriminates):
            out[index] = "discriminates must name gaps or explanations from the lists given"
        elif candidate.action in SEARCH_ACTIONS and (
            candidate.action, normalise_target(candidate.target)
        ) in done:
            # §6.2: no repeated query; the bound is the unresolved gap, not a count.
            out[index] = "this run already took this exact action; change the query or the gap"
    return out


def counter_evidence_due(
    ledger: Mapping[str, Any], candidate: Candidate, *, search_allowed: bool, budget_left: int | None,
) -> str | None:
    """§6.3: before answering a decisive claim, check for counter-evidence once —
    only when a search is allowed and affordable. A lookup is never asked to."""
    if candidate.action != "answer" or not search_allowed or (budget_left is not None and budget_left < 1):
        return None
    if ledger.get("counter_evidence", {}).get("checked"):
        return None
    decisive = [g for g in ledger.get("gaps") or () if g.get("decisive")]
    if not decisive:
        return None
    took_evidence = any(a.get("action") in SEARCH_ACTIONS for a in ledger.get("actions") or ())
    if not took_evidence:
        return None
    return ("you are about to answer a decisive question from evidence: run one search aimed "
            "at contradicting it (opposite result, other assay, other species), then answer")


def mark_counter_evidence(ledger: dict[str, Any], candidate: Candidate) -> None:
    if candidate.action in SEARCH_ACTIONS and "counter" in candidate.reason.lower():
        ledger.setdefault("counter_evidence", {})["checked"] = True


def resolve(ledger: dict[str, Any], updates: Iterable[Mapping[str, Any]]) -> None:
    """Move gaps on: resolved with refs, or re-typed to the gap that now blocks."""
    by_requirement = {g.get("requirement_id"): g for g in ledger.get("gaps") or ()}
    for update in updates:
        gap = by_requirement.get(update.get("requirement_id"))
        if gap is None:
            continue
        if update.get("status") == "resolved":
            gap["status"] = "resolved"
            gap["refs"] = list(dict.fromkeys([*gap.get("refs", []), *(update.get("refs") or [])]))
        elif update.get("gap"):
            gap["gap"] = Gap(update["gap"]).value
            gap["status"] = "open"
