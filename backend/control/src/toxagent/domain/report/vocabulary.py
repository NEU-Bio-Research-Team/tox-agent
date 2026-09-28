"""Schema versions, section and language constants, and the enums a report's parts use."""
from __future__ import annotations

from enum import Enum
from typing import Final

#: v2 adds ``references``: the resolved, immutable snapshot of every cited
#: source (REP-01). v1 artifacts carried only ``evidence_id`` strings, which a
#: renderer could print but not turn into anything a reader could open, so the
#: readers below accept both and treat a v1 report as one with no references
#: rather than as one with none *cited*.
SCHEMA_VERSION: Final = "toxagent-report-v2"


#: Every artifact schema version a reader must handle, newest first.
SCHEMA_VERSIONS: Final[tuple[str, ...]] = ("toxagent-report-v2", "toxagent-report-v1")


BUILD_REQUEST_SCHEMA_VERSION: Final = "report-build-request-v1"


#: Spec section 5.3. Stable IDs; the heading a person reads is a rendering
#: decision and may change without changing these.
REQUIRED_SECTION_IDS: Final[tuple[str, ...]] = (
    "executive_summary",
    "substance_profile",
    "predictor_results",
    "explanation_and_visuals",
    "external_evidence",
    "integrated_interpretation",
    "conclusions",
    "recommendations",
    "limitations",
    "references",
    "provenance_appendix",
)


#: The English-first scope (spec section 5.1). A second language is a product
#: decision with its own limitation wording, not a parameter.
SUPPORTED_REPORT_LANGUAGES: Final[frozenset[str]] = frozenset({"en"})


#: Media types a figure may be. An allowlist rather than a denylist because the
#: renderer inlines these, and "whatever the explainer produced" is how a
#: remote-embed or script payload gets into a report (spec section 11).
ALLOWED_FIGURE_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"image/svg+xml", "image/png"}
)


class SourceClass(str, Enum):
    """Spec section 3.4's table, as a type.

    The report must never blur these, so nothing that carries content in this
    module is constructible without saying which one it is.
    """

    STRUCTURE_FACT = "structure_fact"
    PREDICTOR_FACT = "predictor_fact"
    EXPLANATION_FACT = "explanation_fact"
    EXTERNAL_EVIDENCE = "external_evidence"
    AGENT_SYNTHESIS = "agent_synthesis"
    RECOMMENDATION = "recommendation"


class EvidenceRelation(str, Enum):
    """Spec section 5.6. ``INSUFFICIENT`` is a real outcome: "we looked and
    found nothing that bears on this" is information, and suppressing it in
    favour of a weak match is the failure this enum exists to make nameable."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    INSUFFICIENT = "insufficient"


class GapReason(str, Enum):
    """Why something the report was asked for is not in it.

    A closed set because ``completed_with_gaps`` is a product outcome that has
    to be countable per reason (spec section 14's tracked metrics), and free
    text is not countable.
    """

    ENDPOINT_NOT_SERVED = "endpoint_not_served"
    EXPLANATION_FAILED = "explanation_failed"
    EXPLANATION_PARTIAL = "explanation_partial"
    COMPOUND_IDENTITY_UNRESOLVED = "compound_identity_unresolved"
    NO_RELEVANT_EVIDENCE = "no_relevant_evidence"
    #: Nobody looked. Distinct from NO_RELEVANT_EVIDENCE, which means a search
    #: ran and came back empty or unusable: telling a reader the evidence base
    #: is thin, when in fact the build was configured not to search, misreports
    #: the state of the literature (P0-2).
    EXTERNAL_EVIDENCE_NOT_REQUESTED = "external_evidence_not_requested"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ExplanationStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


#: URL schemes a rendered citation may become a link to. An allowlist, and
#: HTTPS-only: a report is a downloadable document that gets opened later, and
#: ``javascript:``/``data:`` in an href is the one thing a citation must never
#: be able to smuggle past a renderer (spec REP-01, section 11).
SAFE_URL_SCHEMES: Final[frozenset[str]] = frozenset({"https"})


def is_safe_citation_url(url: str | None) -> bool:
    """Whether this URL may be rendered as a clickable link.

    Refusing the link is not refusing the citation: a source with an unusable
    URL still appears in the references with its title, authors and identifier,
    because dropping the source entirely would hide that a claim had a basis.
    """
    if not url:
        return False
    scheme, separator, rest = url.partition("://")
    if not separator or scheme.lower() not in SAFE_URL_SCHEMES:
        return False
    # A host is required. "https://" alone, or a URL whose authority is empty,
    # resolves to nothing and would render as a dead link.
    return bool(rest.split("/", 1)[0].strip())
