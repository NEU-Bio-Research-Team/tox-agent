"""Module-level helpers the gateway class and its mixins share."""
from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from ....domain.errors import Conflict, ToxAgentError, Violation
from ....domain.report import (
    ReportArtifact,
    ReportRendering,
)
from ....report.renderers import (
    HTML_RENDERER_VERSION,
    MARKDOWN_BUNDLE_RENDERER_VERSION,
    MARKDOWN_RENDERER_VERSION,
    PDF_RENDERER_VERSION,
    render_html,
    render_markdown,
    render_markdown_bundle,
    render_pdf,
)
from ....validation.report.draft_wire import ReportDraftCandidate

log = logging.getLogger("toxagent.report")


SUBMIT_TOOL_NAME = "submit_report_draft"


SUBMIT_SAVED_TOOL_NAME = "submit_saved_report_draft"


WORKING_DRAFT_KEY = "working_report_draft"


WORKING_DRAFT_VERSION_KEY = "working_report_draft_version"


WORKING_DRAFT_SHA_KEY = "working_report_draft_sha256"


#: Formats this deployment can actually produce. A requested format outside
#: this set becomes a recorded rendering failure, never a silently missing file.
RENDERERS: Mapping[str, tuple[str, str, Any]] = {
    "markdown": ("text/markdown; charset=utf-8", MARKDOWN_RENDERER_VERSION, render_markdown),
    "markdown_bundle": (
        "application/zip", MARKDOWN_BUNDLE_RENDERER_VERSION, render_markdown_bundle,
    ),
    "html": ("text/html; charset=utf-8", HTML_RENDERER_VERSION, render_html),
    "pdf": ("application/pdf", PDF_RENDERER_VERSION, render_pdf),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReportValidationFailed(ToxAgentError):
    """A correctable rejection. Carries the typed violations the next attempt
    must fix, and how many attempts remain."""

    code = "report_validation_failed"
    http_status = 422

    def __init__(
        self, message: str, *, violations: list[Violation], attempts_remaining: int
    ) -> None:
        super().__init__(
            message,
            violations=[v.to_dict() for v in violations],
            attempts_remaining=attempts_remaining,
        )
        self.violations = violations
        self.attempts_remaining = attempts_remaining


@dataclass(frozen=True)
class SubmitReportOutcome:
    artifact: ReportArtifact
    renderings: tuple[ReportRendering, ...]
    rendering_failures: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class DraftCheckpointOutcome:
    """A durable working draft and the validator's current assessment."""

    draft: ReportDraftCandidate
    version: int
    content_sha256: str
    violations: tuple[Violation, ...]


def _draft_sha256(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pointer_parts(path: str) -> list[str]:
    if not path.startswith("/") or path == "/":
        raise Conflict("a draft patch path must be a non-root JSON pointer", path=path)
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _apply_draft_patch(document: dict[str, Any], operations: list[Mapping[str, Any]]) -> None:
    """Apply the small RFC-6902 subset exposed to the report agent.

    The patched document is parsed as ``ReportDraftCandidate`` before it is
    stored, so a patch can never checkpoint a half-shaped report. Supporting
    add/replace/remove is enough to repair validator paths without making the
    model resend an otherwise unchanged 10k-token candidate.
    """
    for operation in operations:
        op = operation.get("op")
        path = str(operation.get("path", ""))
        parts = _pointer_parts(path)
        parent: Any = document
        for part in parts[:-1]:
            if isinstance(parent, list):
                try:
                    parent = parent[int(part)]
                except (ValueError, IndexError) as exc:
                    raise Conflict("a draft patch path does not resolve", path=path) from exc
            elif isinstance(parent, dict) and part in parent:
                parent = parent[part]
            else:
                raise Conflict("a draft patch path does not resolve", path=path)

        leaf = parts[-1]
        if isinstance(parent, list):
            if op == "add" and leaf == "-":
                parent.append(deepcopy(operation.get("value")))
                continue
            try:
                index = int(leaf)
            except ValueError as exc:
                raise Conflict("a list patch path requires an integer index", path=path) from exc
            if op == "add":
                if index < 0 or index > len(parent):
                    raise Conflict("a draft patch index is out of range", path=path)
                parent.insert(index, deepcopy(operation.get("value")))
            elif op == "replace":
                if index < 0 or index >= len(parent):
                    raise Conflict("a draft patch index is out of range", path=path)
                parent[index] = deepcopy(operation.get("value"))
            elif op == "remove":
                if index < 0 or index >= len(parent):
                    raise Conflict("a draft patch index is out of range", path=path)
                parent.pop(index)
            else:
                raise Conflict("unsupported draft patch operation", operation=op)
        elif isinstance(parent, dict):
            if op == "remove":
                if leaf not in parent:
                    raise Conflict("a draft patch path does not resolve", path=path)
                del parent[leaf]
            elif op in {"add", "replace"}:
                if op == "replace" and leaf not in parent:
                    raise Conflict("a draft patch path does not resolve", path=path)
                parent[leaf] = deepcopy(operation.get("value"))
            else:
                raise Conflict("unsupported draft patch operation", operation=op)
        else:
            raise Conflict("a draft patch parent is not a container", path=path)
