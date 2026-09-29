"""``validate_report_draft``: every check, in order, over one draft."""
from __future__ import annotations

from datetime import datetime

from ....domain.errors import Violation
from ..draft_wire import ReportDraftCandidate
from ._common import ReportValidationContext, ReportValidationResult
from .claims import _validate_claims, _validate_evidence, _validate_explanations
from .conclusions import (
    _validate_conclusions,
    _validate_gaps,
    _validate_limitations,
    _validate_recommendations,
)
from .safety import _validate_content_safety, _validate_semantic_consistency
from .structure import (
    _validate_citation_tokens,
    _validate_endpoint_coverage,
    _validate_figures,
    _validate_references,
    _validate_scope,
    _validate_sections,
)


def validate_report_draft(
    draft: ReportDraftCandidate,
    *,
    context: ReportValidationContext,
    now: datetime,
) -> ReportValidationResult:
    """Apply every gate. Order matches spec section 11's table."""
    violations: list[Violation] = []

    violations += _validate_scope(draft, context)
    violations += _validate_sections(draft)
    violations += _validate_claims(draft, context)
    violations += _validate_endpoint_coverage(draft, context)
    violations += _validate_explanations(draft, context)
    violations += _validate_figures(draft, context)
    violations += _validate_evidence(draft, context)
    violations += _validate_conclusions(draft, context)
    violations += _validate_recommendations(draft)
    violations += _validate_limitations(draft, context)
    violations += _validate_gaps(draft)
    violations += _validate_content_safety(draft)
    violations += _validate_citation_tokens(draft, context)
    violations += _validate_references(draft, context)
    violations += _validate_semantic_consistency(draft, context)

    if violations:
        return ReportValidationResult(violations=tuple(violations))
    return ReportValidationResult(violations=(), accepted_draft=draft)
