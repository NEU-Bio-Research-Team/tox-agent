"""The deterministic report validator (spec section 11): draft in, decision out.

The acceptance boundary. A ``preflight-report-draft`` skill can improve a
first-pass draft, and does not decide anything; this module does. Every gate in
spec section 11's table is applied here, in one place, and passing all of them
is what produces the compiled artifact — there is no path that stores a draft
without going through every check.

Almost nothing here is new logic. Numbers, classifications, citations,
limitations and prohibited wording are the *same* validators a grounded answer
goes through (``answer_validator.py``), because a claim in a report is a claim,
and a report that could assert a number an answer could not would be a way to
route around the trust chain. What is genuinely new is the structural half:
section coverage, endpoint coverage, explanation-to-figure linkage, source
separation and render integrity — the things a document has and a chat reply
does not.

Split by what is checked: ``structure`` (scope, sections, source separation,
endpoint coverage, figures, references, citation tokens), ``claims`` (claims,
explanations, evidence), ``conclusions`` (conclusions, recommendations,
limitations, gaps), ``safety`` (content safety, semantic consistency).
``draft.validate_report_draft`` runs them all. Every name is re-exported here.
"""
from __future__ import annotations

from ._common import (  # noqa: F401
    _KNOWN_LIMITATION_CODES,
    _DERIVED_TRANSFORMS,
    _BASIS_REQUIRED_KINDS,
    _KIND_TO_ALLOWED_CLASSES,
    _MUST_NOT_BE_EMPTY,
    _RAW_HTML,
    _HTML_EVENT_ATTR,
    _REMOTE_IMAGE,
    _GUARANTEE,
    ReportValidationResult,
    ReportValidationContext,
    _prefixed,
)
from .draft import (  # noqa: F401
    validate_report_draft,
)
from .structure import (  # noqa: F401
    _validate_scope,
    _validate_sections,
    _validate_source_separation,
    _validate_endpoint_coverage,
    _validate_figures,
    _validate_citation_tokens,
    _validate_references,
)
from .claims import (  # noqa: F401
    _validate_claims,
    _validate_explanations,
    _validate_evidence,
)
from .conclusions import (  # noqa: F401
    _validate_conclusions,
    _validate_recommendations,
    _validate_limitations,
    _validate_gaps,
)
from .safety import (  # noqa: F401
    _validate_content_safety,
    _validate_semantic_consistency,
)
