"""``submit_report_draft``'s workflow (spec section 10, stages 4 and 5).

Correction policy, in one sentence and deliberately narrower than the
conversational one: a draft is validated, an invalid first draft returns typed
violations for exactly one correction attempt, and a second invalid draft fails
the build. There is no fallback report.

That asymmetry with ``submit_answer`` is the point. A fallback *answer* is an
honest, server-authored sentence saying what could not be established — a
person reads it in a chat and moves on. A fallback *report* would be a
downloadable, citable, eleven-section document produced by nobody, and there is
no wording that makes that safe (spec section 10: "Never accept invalid agent
output").

What happens on the accepted path: the draft is compiled into the immutable
artifact, the renderings are produced and stored, the claim and evidence link
rows are written, and the build reaches ``completed`` or, when gaps exist,
``completed_with_gaps`` — all in one transaction, so a report can never exist
whose citations were not recorded.

``SubmitReportDraft`` is assembled from mixins: ``checkpoints`` (save, check,
patch and submit a saved draft) and ``assembly`` (evidence, explanations,
figures, provenance and rendering of the accepted report). ``service`` holds
admission, validation and ``execute``.
"""
from __future__ import annotations

from ._common import log, SUBMIT_TOOL_NAME, SUBMIT_SAVED_TOOL_NAME, WORKING_DRAFT_KEY, WORKING_DRAFT_VERSION_KEY, WORKING_DRAFT_SHA_KEY, RENDERERS, _now, ReportValidationFailed, SubmitReportOutcome, DraftCheckpointOutcome, _draft_sha256, _pointer_parts, _apply_draft_patch  # noqa: F401
from .service import SubmitReportDraft  # noqa: F401
