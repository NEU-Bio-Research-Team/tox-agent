"""The deterministic router (plan section 4.3).

No LLM decides whether to call an LLM. Routing reads request fields and a small
set of literal parsers, and when it cannot tell, it returns a structured
clarification instead of guessing — a wrong guess here spends a provider request
and, worse, can answer about the wrong molecule.

The keyword lists are deliberately narrow. They exist to recognise an explicit
request for literature, not to infer intent from tone; anything ambiguous falls
through to clarification, which is cheap and honest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from ..domain.run import Intent, Lane

INTENT_HINTS: Final[dict[str, Intent]] = {
    "analyze": Intent.ANALYSIS,
    "ask_report": Intent.REPORT_QA,
    "research_evidence": Intent.EVIDENCE_RESEARCH,
    "request_attribution": Intent.ATTRIBUTION,
}

#: Explicit asks for external literature, in both supported languages (DEC-08).
RESEARCH_TERMS: Final[tuple[str, ...]] = (
    "literature", "publication", "published", "paper", "pubmed", "europe pmc",
    "reference", "citation", "cite", "evidence", "study", "studies", "research",
    "tài liệu", "bài báo", "nghiên cứu", "công bố", "trích dẫn", "nguồn tham khảo",
    "bằng chứng",
)

#: Asks for a per-token explanation of one endpoint.
ATTRIBUTION_TERMS: Final[tuple[str, ...]] = (
    "attribution", "attribute", "which atoms", "which tokens", "contribut",
    "quy gán", "nguyên tử nào", "đóng góp",
)

#: Requests this product does not serve at all. Routed without touching a tool.
OUT_OF_SCOPE_TERMS: Final[tuple[str, ...]] = (
    "run this code", "execute", "shell command", "browse the web", "open a website",
    "prescribe", "dosage for a patient", "treat my", "diagnose",
    "kê đơn", "liều dùng cho bệnh nhân", "chẩn đoán",
)

#: Marks text as a question rather than a bare submission. Punctuation alone is
#: not enough, since "CCO?" is a typo, not a question about a report.
QUESTION_TERMS: Final[tuple[str, ...]] = (
    "what", "why", "how", "which", "is it", "does", "explain", "compare", "should",
    "gì", "sao", "thế nào", "tại sao", "giải thích", "so sánh", "có nên", "bao nhiêu",
)


@dataclass(frozen=True)
class Clarification:
    code: str
    question: str
    options: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "question": self.question, "options": list(self.options)}


@dataclass(frozen=True)
class RouteRequest:
    text: str = ""
    molecule_smiles: str | None = None
    batch_smiles: tuple[str, ...] = ()
    has_image: bool = False
    intent_hint: str = "auto"
    has_active_analysis: bool = False
    analysis_id: str | None = None
    requested_endpoints: tuple[str, ...] = ()
    include_attribution: bool = False

    @property
    def normalised_text(self) -> str:
        return self.text.strip().lower()

    def mentions(self, terms: tuple[str, ...]) -> bool:
        text = self.normalised_text
        return any(term in text for term in terms)

    @property
    def looks_like_a_question(self) -> bool:
        text = self.normalised_text
        if not text:
            return False
        return "?" in text and len(text) > 12 or any(t in text for t in QUESTION_TERMS)


@dataclass(frozen=True)
class Route:
    intent: Intent
    lane: Lane
    reason: str
    needs_snapshot_first: bool = False
    clarification: Clarification | None = None

    @property
    def calls_a_runtime(self) -> bool:
        return self.lane in (Lane.AGENTIC, Lane.MIXED)


def route(request: RouteRequest) -> Route:
    """Decide the lane and intent for one request. Pure and total."""
    hinted = INTENT_HINTS.get(request.intent_hint)

    if request.mentions(OUT_OF_SCOPE_TERMS):
        return Route(
            Intent.OUT_OF_SCOPE, Lane.DETERMINISTIC,
            "the request asks for something outside this product's scope",
        )

    if request.batch_smiles:
        return Route(Intent.ANALYSIS_BATCH, Lane.DETERMINISTIC, "batch of molecules submitted")

    if request.has_image:
        # Lane.DETERMINISTIC — REBUILD_PLAN section 26.6's own transcript-shape
        # table puts `structure_recognition` in the Lane D row (with
        # analysis/analysis_batch), not the Lane A row (report_qa/
        # evidence_research/attribution). It never reasons over the image with
        # a model turn either way: a configured toxocr/ service resolves to a
        # SMILES through MolScribe and the run never binds a runtime
        # (application/recognize_structure.py); an unconfigured one answers
        # `capability_unavailable` the same way (ADR 0006,
        # submit_message.py's `structure_recognition_available` gate) — both
        # are a deterministic lookup, so the run's lane must say so.
        return Route(
            Intent.STRUCTURE_RECOGNITION, Lane.DETERMINISTIC, "an image was submitted for structure recognition"
        )

    if hinted is Intent.ATTRIBUTION or (
        hinted is None and request.mentions(ATTRIBUTION_TERMS)
    ):
        if not (request.analysis_id or request.has_active_analysis or request.molecule_smiles):
            return _clarify(
                "attribution_target_missing",
                "Which analysis and endpoint should the attribution explain?",
            )
        return Route(
            Intent.ATTRIBUTION,
            Lane.MIXED,
            "attribution requested; the tool is deterministic and the synthesis is not",
            # Any explicitly submitted molecule always gets a fresh snapshot,
            # even if a *different* analysis is already active — otherwise a
            # new molecule silently answers against the stale one.
            needs_snapshot_first=bool(request.molecule_smiles),
        )

    wants_research = hinted is Intent.EVIDENCE_RESEARCH or (
        hinted is None and request.mentions(RESEARCH_TERMS)
    )
    if wants_research:
        if not (request.analysis_id or request.has_active_analysis or request.molecule_smiles):
            return _clarify(
                "research_subject_missing",
                "Which molecule or analysis should the evidence search be about?",
            )
        return Route(
            Intent.EVIDENCE_RESEARCH,
            Lane.AGENTIC,
            "the request explicitly asks for external literature",
            needs_snapshot_first=bool(request.molecule_smiles),
        )

    if hinted is Intent.ANALYSIS:
        if not request.molecule_smiles:
            return _clarify("smiles_missing", "Which SMILES should be analysed?")
        return Route(Intent.ANALYSIS, Lane.DETERMINISTIC, "analysis requested for a SMILES")

    if hinted is Intent.REPORT_QA:
        if not (request.analysis_id or request.has_active_analysis or request.molecule_smiles):
            return _clarify(
                "report_subject_missing",
                "Which analysis is the question about? Submit a SMILES or select one.",
            )
        needs_snapshot = bool(request.molecule_smiles)
        return Route(
            Intent.REPORT_QA,
            Lane.MIXED if needs_snapshot else Lane.AGENTIC,
            "the caller asked to question a report",
            needs_snapshot_first=needs_snapshot,
        )

    if request.molecule_smiles and not request.looks_like_a_question:
        return Route(
            Intent.ANALYSIS, Lane.DETERMINISTIC, "a molecule was submitted with no question"
        )

    if request.molecule_smiles and request.looks_like_a_question:
        # A new molecule plus a question: snapshot first, deterministically,
        # then answer against that snapshot. The question never reaches a model
        # before the numbers it is about exist.
        return Route(
            Intent.REPORT_QA, Lane.MIXED,
            "a new molecule and a question; the snapshot is taken before the question is answered",
            needs_snapshot_first=True,
        )

    if request.analysis_id or request.has_active_analysis:
        if not request.text.strip():
            return _clarify("question_missing", "What would you like to know about this analysis?")
        return Route(Intent.REPORT_QA, Lane.AGENTIC, "a question about an existing analysis")

    if request.text.strip():
        # This clarification is only ever reached when there is no active
        # analysis and none was named — "select an existing one" would always
        # be a dead end here (there is nothing to pick from and no UI to pick
        # it with), which is exactly what made this button loop in practice.
        return _clarify(
            "molecule_missing",
            "Provide a SMILES string to analyse.",
            options=("submit_smiles",),
        )

    return _clarify("empty_request", "The request contained neither a molecule nor a question.")


def _clarify(code: str, question: str, options: tuple[str, ...] = ()) -> Route:
    return Route(
        Intent.CLARIFICATION_REQUIRED,
        Lane.DETERMINISTIC,
        "the request is missing something the router will not guess at",
        clarification=Clarification(code, question, options),
    )
