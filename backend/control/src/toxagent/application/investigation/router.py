"""The four serving routes (#68, §4.1 and §4.2).

§4.1 splits what one product surface was doing into four routes with
different costs: reading an artifact, investigating a question, updating an
assessment after new input, and comparing candidates. The product already
routes by *intent* -- analysis, decision support, report -- which is a
different axis: it says which pipeline runs, not how much investigation the
question deserves. A value lookup and a contradiction assessment are both
`decision_support` today, and they were both paying for the full loop.

The rules, in §4.2's order, and the reasons they are in that order:

* **Explicit subject and artifact first.** A question about a bound screening
  with a named artifact is answerable from what the run already holds, and a
  router that reaches for a classifier before checking that spends a model
  call to learn something the binding already stated.
* **A classifier only for the ambiguous rest, and it may return
  "undetermined".** §4.2 requires that route to exist: asking the researcher
  which subject they mean is a better answer than guessing between two
  compounds.
* **Keywords never decide a research question.** §4.2 names this directly --
  "Không dùng keyword 'safe' để tự quyết toàn bộ câu hỏi nghiên cứu". So the
  lookup markers here can only route a question whose subject and artifact are
  *already* bound, and any comparison or assessment marker in the text takes
  the question out of the lookup route regardless of what else it contains.
* **Every decision carries reason codes.** Routing errors and the classifier's
  own cost have to be reported separately from task success (§4.2), and a
  decision nobody can attribute cannot be counted as an error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

ROUTE_SCHEMA_VERSION = "serving-route-v1"

#: Read or explain what the run already holds.
READ_EXPLAIN = "read_explain"
#: Investigate a question against sources.
INVESTIGATE = "investigate"
#: Re-assess after new input or a changed premise.
UPDATE = "update"
#: Compare several candidates under declared criteria.
COMPARE_BATCH = "compare_batch"
#: The subject or the task is not determined; ask rather than guess.
UNDETERMINED = "undetermined"

ROUTES = (READ_EXPLAIN, INVESTIGATE, UPDATE, COMPARE_BATCH, UNDETERMINED)

#: Phrases that mark a question as reading a value, label or status. They may
#: only move a question that is already bound to a subject and an artifact,
#: and never one that also carries an assessment marker.
_LOOKUP_MARKERS = (
    "what is", "what's", "what was", "show me", "which stage", "did it run",
    "has it run", "what does the model say", "what did the model",
    "là gì", "bao nhiêu", "đã chạy", "trạng thái", "cho tôi xem",
)

#: The shapes of a question that asks what an artifact says, rather than the
#: literal phrases above. Live (2026-10-29, arm E) the literal list matched
#: none of "What does the predictor say about hERG", "What did the activity
#: model estimate" or "What does the screening say", and every read went
#: down the full loop. Same rule as the markers: these only move a question
#: whose subject and artifact are bound and that asks for no judgement.
_LOOKUP_PATTERNS = tuple(re.compile(p) for p in (
    r"\bwhat (?:does|did|do) (?:[\w-]+ ){0,5}?(?:say|says|estimate|predict|report|show|give)\b",
    r"\bwhat (?:is|was|are|were) (?:[\w-]+ ){0,4}?(?:label|score|status|probability|"
    r"estimate|value|decision|threshold|interval)s?\b",
    r"\b(?:did|has|was) (?:[\w-]+ ){0,4}?(?:stage|step) (?:run|ran|fail|failed|complete|completed)\b",
))

#: Phrases that mean the researcher is asking for a judgement, not a field.
#: Any of these outranks every lookup marker: a question that asks whether two
#: things disagree is not answered by printing one of them.
_ASSESSMENT_MARKERS = (
    "contradict", "disagree", "conflict", "safe", "risk", "should we", "why",
    "explain why", "is it consistent", "margin", "recommend", "interpret",
    "mâu thuẫn", "bất đồng", "có nên", "vì sao", "tại sao", "giải thích",
    "khuyến nghị", "an toàn", "rủi ro",
)

#: Phrases that mean the researcher just supplied something new.
_UPDATE_MARKERS = (
    "i measured", "i have a", "i found", "now that", "update", "recompute",
    "instead of", "correction", "corrected",
    "tôi đo", "tôi có", "tôi tìm thấy", "cập nhật", "tính lại", "sửa lại",
)

_COMPARE_MARKERS = (
    "compare", "rank", "which of", "versus", " vs ", "shortlist", "candidates",
    "so sánh", "xếp hạng", "ứng viên",
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    #: ``high`` when a declared fact decided it, ``low`` when wording did.
    confidence: str = "low"
    reason_codes: tuple[str, ...] = ()
    #: What the researcher has to answer before a route can be chosen.
    clarification: str = ""
    #: True when a model call was spent deciding. Measured separately from
    #: task success, per §4.2.
    classifier_used: bool = False
    schema_version: str = ROUTE_SCHEMA_VERSION
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version, "route": self.route,
            "confidence": self.confidence, "reason_codes": list(self.reason_codes),
            "classifier_used": self.classifier_used,
        }
        if self.clarification:
            data["clarification"] = self.clarification
        if self.detail:
            data["detail"] = dict(self.detail)
        return data


def _has(text: str, markers: Sequence[str]) -> str | None:
    lowered = f" {text.casefold()} "
    for marker in markers:
        if marker in lowered:
            return marker
    return None


def _matches(text: str, patterns: Sequence[re.Pattern]) -> str | None:
    lowered = text.casefold()
    for pattern in patterns:
        found = pattern.search(lowered)
        if found:
            return found.group(0)
    return None


def classify(
    *,
    question: str,
    has_subject: bool,
    artifact_refs: Sequence[str] = (),
    stale_claims: Sequence[str] = (),
    new_researcher_input: bool = False,
    candidate_count: int = 1,
) -> RouteDecision:
    """Which route this turn takes, from what the server already knows.

    Deterministic, and deliberately so: it is the cheap check §4.2 puts before
    any classifier. ``classifier_used`` stays false here, so a deployment that
    adds a model-backed fallback for the undetermined case can report its cost
    without this function's decisions being counted into it.
    """
    text = question or ""
    codes: list[str] = []

    # A premise moved, or the researcher supplied something: the question is
    # about what changes, whatever else its wording suggests. This outranks
    # the lookup route because a value read from a stale claim is wrong.
    if stale_claims:
        codes.append("stale_claims_present")
        return RouteDecision(UPDATE, confidence="high", reason_codes=tuple(codes),
                             detail={"stale_claims": list(stale_claims)})
    if new_researcher_input or _has(text, _UPDATE_MARKERS):
        codes.append("researcher_supplied_input" if new_researcher_input else "update_wording")
        return RouteDecision(
            UPDATE, confidence="high" if new_researcher_input else "low",
            reason_codes=tuple(codes),
        )

    compare = _has(text, _COMPARE_MARKERS)
    if compare:
        # Wording asks for a comparison; the session decides whether one can
        # be served. Several subjects in a session is not, on its own, a
        # request to compare them -- that would send every question asked in
        # a multi-compound session down the batch route.
        if candidate_count > 1:
            codes.extend(["compare_wording", "several_candidates"])
            return RouteDecision(
                COMPARE_BATCH, confidence="high", reason_codes=tuple(codes),
                detail={"candidates": candidate_count, "marker": compare},
            )
        return RouteDecision(
            UNDETERMINED, confidence="high",
            reason_codes=("compare_wording", "one_candidate"),
            clarification="This session holds one candidate. Which others should it be "
                          "compared with?",
            detail={"candidates": candidate_count, "marker": compare},
        )

    if not has_subject and not artifact_refs:
        # §4.2: the route has to exist. Guessing between two compounds is the
        # error asking a question avoids.
        return RouteDecision(
            UNDETERMINED, confidence="high", reason_codes=("no_subject_bound",),
            clarification="Which compound, target and assay context is this about?",
        )

    assessment = _has(text, _ASSESSMENT_MARKERS)
    if assessment:
        codes.append("assessment_wording")
        return RouteDecision(INVESTIGATE, confidence="low", reason_codes=tuple(codes),
                             detail={"marker": assessment})

    lookup = _has(text, _LOOKUP_MARKERS) or _matches(text, _LOOKUP_PATTERNS)
    if lookup and has_subject and artifact_refs:
        # Only here: subject bound, artifact in hand, and nothing in the text
        # asking for a judgement.
        codes.extend(["subject_bound", "artifact_available", "lookup_wording"])
        return RouteDecision(READ_EXPLAIN, confidence="high", reason_codes=tuple(codes),
                             detail={"marker": lookup, "artifacts": list(artifact_refs)})

    codes.append("default_investigate")
    return RouteDecision(INVESTIGATE, confidence="low", reason_codes=tuple(codes))


#: Routes whose turn must not open hypotheses or load skills (§4.2): a value
#: read does not need eight competing explanations.
LIGHT_ROUTES: frozenset[str] = frozenset({READ_EXPLAIN, UNDETERMINED})

#: The capability profile each route runs under (§4.1 "Cách chạy"). A light
#: route gets the reads and the answer and nothing else: no search, no case
#: writes, no skills -- which is what makes it cheaper rather than merely
#: labelled cheaper.
READ_PROFILE = "decision_support_read"
FULL_PROFILE = "decision_support"


def opens_hypotheses(route: str) -> bool:
    return route not in LIGHT_ROUTES


def profile_for(route: str) -> str:
    return READ_PROFILE if route in LIGHT_ROUTES else FULL_PROFILE


_WORD = re.compile(r"[a-z0-9]+")


def query_signature(query: str) -> str:
    """A query's identity for "have I already searched this" (§5.4, §6.2).

    Word set, not the string: a turn that reorders two words has not asked a
    new question, and letting it count as one is how a run spends its search
    budget three times on one query.
    """
    return " ".join(sorted(set(_WORD.findall((query or "").casefold()))))
