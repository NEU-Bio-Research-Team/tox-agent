"""ReportArtifact and its build aggregate — the canonical report contract.

Spec: docs/spec/TOXAGENT_REPORT_BUILDER_PLAN.md sections 5 and 10. The artifact
is the source of truth; Markdown, HTML and PDF are renderings of it, and a
renderer that disagrees with the artifact is a renderer bug, never a second
opinion about what the report says.

Three things here are deliberate and load-bearing:

**No aggregate verdict field.** Exactly as ``GroundedAnswer`` has no overall
toxicity score (ADR 0002), ``ReportArtifact`` has no field into which "this
compound is safe" could be written. Endpoint-level conclusions each name their
own endpoint; there is nowhere to put a collapsed one (spec section 3.5).

**Missing data is a value, not an absence.** ``REQUIRED_SECTION_IDS`` must all
be present even when a section reports only a gap, so a failed provider or a
failed explainer cannot make a section quietly disappear (spec sections 3.6,
11 "Missing-data visibility").

**Source class is carried on every piece of content.** A predictor number, an
explanation, an external record and an agent synthesis are four different kinds
of claim about the world, and the report is required never to blur them
(spec section 3.4), so the distinction lives in the type rather than in
whichever sentence a model happened to write.

The report domain: its vocabulary, the parts a report is made of, the
artifact that is the report, and the build that produces it.

Every name is re-exported here, so ``from ..domain.report import X`` keeps
working.
"""
from __future__ import annotations

from .vocabulary import (  # noqa: F401
    SCHEMA_VERSION,
    SCHEMA_VERSIONS,
    BUILD_REQUEST_SCHEMA_VERSION,
    REQUIRED_SECTION_IDS,
    SUPPORTED_REPORT_LANGUAGES,
    ALLOWED_FIGURE_MEDIA_TYPES,
    SourceClass,
    EvidenceRelation,
    GapReason,
    ExplanationStatus,
    SAFE_URL_SCHEMES,
    is_safe_citation_url,
)
from .build import (  # noqa: F401
    BuildStage,
    TERMINAL_STAGES,
    _FORWARD,
    ALLOWED_STAGE_TRANSITIONS,
    ReportBuildRequest,
    BuildTransitionError,
    ReportBuild,
)
from .parts import (  # noqa: F401
    ReportFigure,
    ExplanationHighlights,
    ExplanationPackage,
    SubstanceProfile,
    EvidenceSynthesis,
    ReportReference,
    ReportTable,
    ReportSection,
    ReportGap,
    ReportConclusion,
    ReportRecommendation,
    ReportRendering,
)
from .artifact import (  # noqa: F401
    ReportArtifact,
)
