"""The deterministic router (plan section 4.3).

No LLM decides whether to call an LLM. Routing reads request fields and a small
set of literal parsers, and when it cannot tell, it returns a structured
clarification instead of guessing — a wrong guess here spends a provider request
and, worse, can answer about the wrong molecule.

The keyword lists are deliberately narrow. They exist to recognise an explicit
request for literature, not to infer intent from tone; anything ambiguous falls
through to clarification, which is cheap and honest.

Matching is by **whole word and whole phrase** (``intent_matching``), not by
substring. P1-9 of the 2026-09-13 audit: ``"execute" in text`` fires on
"**exec**utive summary" and routed a question about a report section to
OUT_OF_SCOPE, and ``"contribut"`` fires on "**contribut**ing factors". A term
list here therefore spells out the forms it means; three forms of a word are
three decisions, not an accident of a prefix.

Every decision carries reason codes and the terms that produced them. A router
that cannot say why it chose cannot be argued with, and the frontend's own
guess must be a hint this code validates, never a second source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from ..domain.run import Intent, Lane
from .intent_matching import matched_terms, mentions

INTENT_HINTS: Final[dict[str, Intent]] = {
    "analyze": Intent.ANALYSIS,
    # ADS plan section 7.1 / ADR 0010: these three legacy hints are still
    # accepted from old clients, but no longer name distinct destinations —
    # all three resolve to the one adaptive capability. Which flavour was
    # asked for is preserved separately in REQUESTED_HINT_FLAVOR for audit
    # (IntentDecision.requested_hint / reason_codes), never as a different
    # Intent value.
    "ask_report": Intent.DECISION_SUPPORT,
    "research_evidence": Intent.DECISION_SUPPORT,
    "request_attribution": Intent.DECISION_SUPPORT,
    "build_report": Intent.BUILD_REPORT,
}

#: The pre-ADS name of the flavour a legacy hint asked for, kept only as a
#: reason code on the routing decision — never used to pick the Intent.
REQUESTED_HINT_FLAVOR: Final[dict[str, str]] = {
    "ask_report": "requested_report_qa",
    "research_evidence": "requested_evidence_research",
    "request_attribution": "requested_attribution",
}

#: Explicit asks for external literature, in both supported languages (DEC-08).
RESEARCH_TERMS: Final[tuple[str, ...]] = (
    "literature", "publication", "published", "paper", "pubmed", "europe pmc",
    "reference", "citation", "cite", "evidence", "study", "studies", "research",
    "tài liệu", "bài báo", "nghiên cứu", "công bố", "trích dẫn", "nguồn tham khảo",
    "bằng chứng",
)

#: Explicit asks for a whole report document. Deliberately narrow, and checked
#: *before* the research and attribution terms: a report build launches
#: explanation generation and bounded external research, which is far more
#: expensive than either, and inferring it from "tell me about this molecule"
#: would spend a provider budget nobody asked for (report spec section 3.1:
#: "must not launch research or explanation generation for a normal prediction
#: request"). A bare "report" is not enough on its own — this product already
#: calls its analysis view a report, so the phrase has to name the *making* of
#: one.
REPORT_BUILD_TERMS: Final[tuple[str, ...]] = (
    "build a report", "build me a report", "generate a report", "generate the report",
    "write a report", "write me a report", "produce a report", "create a report",
    "full report", "complete report", "detailed report", "comprehensive report",
    "report document", "download a report", "export a report", "pdf report",
    "tạo báo cáo", "lập báo cáo", "viết báo cáo", "xuất báo cáo", "báo cáo đầy đủ",
    "báo cáo chi tiết", "báo cáo hoàn chỉnh",
)

#: Asks for a per-token explanation of one endpoint.
ATTRIBUTION_TERMS: Final[tuple[str, ...]] = (
    "attribution", "attributions", "attribute", "attributes",
    "which atoms", "which tokens", "what atoms",
    # Spelled out rather than the old "contribut" prefix, which matched
    # "contributing factors" and turned a question about uncertainty into an
    # attribution run.
    "contributor", "contributors", "contribution", "contributions",
    "quy gán", "nguyên tử nào", "đóng góp",
)

#: Requests this product does not serve at all. Routed without touching a tool.
OUT_OF_SCOPE_TERMS: Final[tuple[str, ...]] = (
    "run this code", "execute this", "execute the following", "shell command",
    "browse the web", "open a website",
    "prescribe", "dosage for a patient", "treat my", "diagnose",
    "kê đơn", "liều dùng cho bệnh nhân", "chẩn đoán",
)

#: Marks text as a question rather than a bare submission. Punctuation alone is
#: not enough, since "CCO?" is a typo, not a question about a report.
QUESTION_TERMS: Final[tuple[str, ...]] = (
    "what", "why", "how", "which", "is it", "does", "do", "explain", "compare",
    "should", "can you", "tell me",
    "gì", "sao", "thế nào", "tại sao", "giải thích", "so sánh", "có nên", "bao nhiêu",
)

#: The router's own version, recorded on every decision and in the run
#: configuration snapshot. A routing complaint six weeks old is unanswerable
#: without knowing which rules were in force.
ROUTER_VERSION: Final = "router-2"


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
    #: Flag subjectless_research_v1, passed in so routing stays pure: a
    #: literature question without a molecule runs instead of asking for one.
    allow_subjectless_research: bool = False

    @property
    def normalised_text(self) -> str:
        return self.text.strip().lower()

    def matches(self, terms: tuple[str, ...]) -> tuple[str, ...]:
        """Which of ``terms`` appear as whole words or phrases."""
        return matched_terms(self.text, terms)

    def mentions(self, terms: tuple[str, ...]) -> bool:
        return bool(self.matches(terms))

    @property
    def looks_like_a_question(self) -> bool:
        text = self.normalised_text
        if not text:
            return False
        # No negation handling here: "do not explain" is still a question,
        # and suppressing the word would make it look like a bare submission.
        if mentions(text, QUESTION_TERMS, negators=None):
            return True
        # Punctuation alone is not enough: "CCO?" is a typo, not a question.
        return "?" in text and len(text) > 12


@dataclass(frozen=True)
class IntentDecision:
    """Why the router chose what it chose.

    Carried on every route so a disagreement can be settled with evidence
    rather than by re-reading the term lists. ``matched`` holds the phrases
    that actually fired, ``reason_codes`` the rules they triggered, and
    ``required_context`` what a clarification is waiting for.

    ``confidence`` is a band, not a number. ``high`` means an explicit signal —
    a caller's ``intent_hint``, or a phrase that names the workflow. ``medium``
    means a structural inference (a molecule with a question attached).
    ``low`` means the router declined and asked.
    """

    intent: Intent
    confidence: str
    reason_codes: tuple[str, ...] = ()
    matched: tuple[str, ...] = ()
    required_context: tuple[str, ...] = ()
    clarification_options: tuple[str, ...] = ()
    router_version: str = ROUTER_VERSION
    #: What the caller asked for, kept even when it was not honoured, so a UI
    #: that disagrees with the backend can be shown where it diverged.
    requested_hint: str = "auto"
    hint_honoured: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "matched": list(self.matched),
            "required_context": list(self.required_context),
            "clarification_options": list(self.clarification_options),
            "router_version": self.router_version,
            "requested_hint": self.requested_hint,
            "hint_honoured": self.hint_honoured,
        }


@dataclass(frozen=True)
class Route:
    intent: Intent
    lane: Lane
    reason: str
    needs_snapshot_first: bool = False
    clarification: Clarification | None = None
    decision: IntentDecision | None = None

    @property
    def calls_a_runtime(self) -> bool:
        return self.lane in (Lane.AGENTIC, Lane.MIXED)


def _subject_context(request: RouteRequest) -> tuple[str, ...]:
    """What a subject-bound intent still needs before it can run."""
    if request.analysis_id or request.has_active_analysis or request.molecule_smiles:
        return ()
    return ("analysis_id_or_smiles",)


def route(request: RouteRequest) -> Route:
    """Decide the lane and intent for one request. Pure and total.

    Precedence is explicit and ordered, because two rules can both match one
    sentence ("build me a report citing the literature" names a report build
    *and* research) and the expensive workflow has to win deliberately rather
    than by whichever list happens to be checked first.
    """
    hinted = INTENT_HINTS.get(request.intent_hint)
    hint_given = request.intent_hint not in ("", "auto", None)
    # An unknown hint is not a silent fall-through to inference: the caller
    # asked for something this deployment does not have a name for.
    hint_unknown = hint_given and hinted is None

    def decide(
        intent: Intent,
        *,
        confidence: str,
        codes: tuple[str, ...],
        matched: tuple[str, ...] = (),
    ) -> IntentDecision:
        return IntentDecision(
            intent=intent,
            confidence=confidence,
            reason_codes=codes + (("unknown_intent_hint",) if hint_unknown else ()),
            matched=matched,
            router_version=ROUTER_VERSION,
            requested_hint=request.intent_hint or "auto",
            hint_honoured=not hint_given or (hinted is not None and hinted is intent),
        )

    out_of_scope = request.matches(OUT_OF_SCOPE_TERMS)
    if out_of_scope:
        return Route(
            Intent.OUT_OF_SCOPE, Lane.DETERMINISTIC,
            "the request asks for something outside this product's scope",
            decision=decide(
                Intent.OUT_OF_SCOPE,
                confidence="high",
                codes=("out_of_scope_phrase",),
                matched=out_of_scope,
            ),
        )

    if request.batch_smiles:
        return Route(
            Intent.ANALYSIS_BATCH, Lane.DETERMINISTIC, "batch of molecules submitted",
            decision=decide(
                Intent.ANALYSIS_BATCH, confidence="high", codes=("batch_submitted",)
            ),
        )

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
            Intent.STRUCTURE_RECOGNITION, Lane.DETERMINISTIC,
            "an image was submitted for structure recognition",
            decision=decide(
                Intent.STRUCTURE_RECOGNITION,
                confidence="high",
                codes=("image_submitted",),
            ),
        )

    report_terms = request.matches(REPORT_BUILD_TERMS)
    wants_report = hinted is Intent.BUILD_REPORT or (hinted is None and report_terms)
    if wants_report:
        missing = _subject_context(request)
        if missing:
            return _clarify(
                "report_subject_missing",
                "Which molecule or analysis should the report be about?",
                decision=decide(
                    Intent.CLARIFICATION_REQUIRED,
                    confidence="low",
                    codes=("report_requested", "subject_missing"),
                    matched=report_terms,
                ),
                required_context=missing,
            )
        return Route(
            Intent.BUILD_REPORT,
            # MIXED even without a new molecule: a report build is a
            # deterministic assembly stage (substance, predictions,
            # explanations) followed by an agent synthesis stage, which is
            # exactly what MIXED describes. Calling it AGENTIC would say the
            # whole thing is a model turn, and the expensive half is not.
            Lane.MIXED,
            "the request explicitly asks for a report document to be produced",
            needs_snapshot_first=bool(request.molecule_smiles),
            decision=decide(
                Intent.BUILD_REPORT,
                confidence="high",
                codes=("explicit_hint",) if hinted else ("report_phrase",),
                matched=report_terms,
            ),
        )

    # ADS plan section 7.1 / ADR 0010: attribution, evidence-research and
    # report-QA keyword/hint triggers all resolve to the one adaptive
    # capability now. Precedence among them is preserved from the pre-ADS
    # router (attribution phrase, then research phrase, then the explicit
    # ask_report hint) purely to keep `matched`/reason_codes deterministic
    # when a sentence fires more than one term list; it no longer picks
    # between different destinations.
    attribution_terms = request.matches(ATTRIBUTION_TERMS)
    wants_attribution = request.intent_hint == "request_attribution" or (
        hinted is None and attribution_terms
    )
    if wants_attribution:
        missing = _subject_context(request)
        if missing:
            return _clarify(
                "attribution_target_missing",
                "Which analysis and endpoint should the attribution explain?",
                decision=decide(
                    Intent.CLARIFICATION_REQUIRED,
                    confidence="low",
                    codes=("attribution_requested", "subject_missing"),
                    matched=attribution_terms,
                ),
                required_context=missing,
            )
        return Route(
            Intent.DECISION_SUPPORT,
            Lane.MIXED,
            "attribution requested; the tool is deterministic and the synthesis is not",
            # Any explicitly submitted molecule always gets a fresh snapshot,
            # even if a *different* analysis is already active — otherwise a
            # new molecule silently answers against the stale one.
            needs_snapshot_first=bool(request.molecule_smiles),
            decision=decide(
                Intent.DECISION_SUPPORT,
                confidence="high",
                codes=(("explicit_hint", REQUESTED_HINT_FLAVOR["request_attribution"])
                       if hinted else ("attribution_phrase",)),
                matched=attribution_terms,
            ),
        )

    research_terms = request.matches(RESEARCH_TERMS)
    wants_research = request.intent_hint == "research_evidence" or (
        hinted is None and research_terms
    )
    if wants_research:
        missing = _subject_context(request)
        if missing and request.allow_subjectless_research:
            return Route(
                Intent.DECISION_SUPPORT,
                Lane.AGENTIC,
                "a literature question with no molecule in the session",
                needs_snapshot_first=False,
                decision=decide(
                    Intent.DECISION_SUPPORT,
                    confidence="medium",
                    codes=("research_phrase", "subject_absent"),
                    matched=research_terms,
                ),
            )
        if missing:
            return _clarify(
                "research_subject_missing",
                "Which molecule or analysis should the evidence search be about?",
                decision=decide(
                    Intent.CLARIFICATION_REQUIRED,
                    confidence="low",
                    codes=("research_requested", "subject_missing"),
                    matched=research_terms,
                ),
                required_context=missing,
            )
        return Route(
            Intent.DECISION_SUPPORT,
            Lane.AGENTIC,
            "the request explicitly asks for external literature",
            needs_snapshot_first=bool(request.molecule_smiles),
            decision=decide(
                Intent.DECISION_SUPPORT,
                confidence="high",
                codes=(("explicit_hint", REQUESTED_HINT_FLAVOR["research_evidence"])
                       if hinted else ("research_phrase",)),
                matched=research_terms,
            ),
        )

    if hinted is Intent.ANALYSIS:
        if not request.molecule_smiles:
            return _clarify(
                "smiles_missing",
                "Which SMILES should be analysed?",
                decision=decide(
                    Intent.CLARIFICATION_REQUIRED,
                    confidence="low",
                    codes=("analysis_requested", "smiles_missing"),
                ),
                required_context=("molecule_smiles",),
            )
        return Route(
            Intent.ANALYSIS, Lane.DETERMINISTIC, "analysis requested for a SMILES",
            decision=decide(
                Intent.ANALYSIS, confidence="high", codes=("explicit_hint",)
            ),
        )

    if request.intent_hint == "ask_report":
        missing = _subject_context(request)
        if missing:
            return _clarify(
                "report_subject_missing",
                "Which analysis is the question about? Submit a SMILES or select one.",
                decision=decide(
                    Intent.CLARIFICATION_REQUIRED,
                    confidence="low",
                    codes=("question_requested", "subject_missing"),
                ),
                required_context=missing,
            )
        needs_snapshot = bool(request.molecule_smiles)
        return Route(
            Intent.DECISION_SUPPORT,
            Lane.MIXED if needs_snapshot else Lane.AGENTIC,
            "the caller asked to question a report",
            needs_snapshot_first=needs_snapshot,
            decision=decide(
                Intent.DECISION_SUPPORT,
                confidence="high",
                codes=("explicit_hint", REQUESTED_HINT_FLAVOR["ask_report"]),
            ),
        )

    if request.molecule_smiles and not request.looks_like_a_question:
        return Route(
            Intent.ANALYSIS, Lane.DETERMINISTIC, "a molecule was submitted with no question",
            decision=decide(
                Intent.ANALYSIS, confidence="medium", codes=("bare_molecule",)
            ),
        )

    if request.molecule_smiles and request.looks_like_a_question:
        # A new molecule plus a question: snapshot first, deterministically,
        # then answer against that snapshot. The question never reaches a model
        # before the numbers it is about exist.
        return Route(
            Intent.DECISION_SUPPORT, Lane.MIXED,
            "a new molecule and a question; the snapshot is taken before the question is answered",
            needs_snapshot_first=True,
            decision=decide(
                Intent.DECISION_SUPPORT,
                confidence="medium",
                codes=("molecule_with_question",),
            ),
        )

    if request.analysis_id or request.has_active_analysis:
        if not request.text.strip():
            return _clarify(
                "question_missing",
                "What would you like to know about this analysis?",
                decision=decide(
                    Intent.CLARIFICATION_REQUIRED,
                    confidence="low",
                    codes=("active_analysis", "question_missing"),
                ),
                required_context=("question_text",),
            )
        return Route(
            Intent.DECISION_SUPPORT, Lane.AGENTIC, "a question about an existing analysis",
            decision=decide(
                Intent.DECISION_SUPPORT, confidence="medium", codes=("question_about_active",)
            ),
        )

    if request.text.strip():
        # This clarification is only ever reached when there is no active
        # analysis and none was named — "select an existing one" would always
        # be a dead end here (there is nothing to pick from and no UI to pick
        # it with), which is exactly what made this button loop in practice.
        return _clarify(
            "molecule_missing",
            "Provide a SMILES string to analyse.",
            options=("submit_smiles",),
            decision=decide(
                Intent.CLARIFICATION_REQUIRED,
                confidence="low",
                codes=("text_without_subject",),
            ),
            required_context=("molecule_smiles",),
        )

    return _clarify(
        "empty_request",
        "The request contained neither a molecule nor a question.",
        decision=decide(
            Intent.CLARIFICATION_REQUIRED, confidence="low", codes=("empty_request",)
        ),
        required_context=("molecule_smiles", "question_text"),
    )


def _clarify(
    code: str,
    question: str,
    options: tuple[str, ...] = (),
    *,
    decision: IntentDecision | None = None,
    required_context: tuple[str, ...] = (),
) -> Route:
    from dataclasses import replace as _replace

    if decision is not None:
        decision = _replace(
            decision, required_context=required_context, clarification_options=options
        )
    return Route(
        Intent.CLARIFICATION_REQUIRED,
        Lane.DETERMINISTIC,
        "the request is missing something the router will not guess at",
        clarification=Clarification(code, question, options),
        decision=decision,
    )
