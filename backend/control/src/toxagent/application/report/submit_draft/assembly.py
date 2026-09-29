"""Turning an accepted draft into a report: evidence, explanations, figures, provenance, rendering."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from ....domain.evidence import EvidenceRecord
from ....domain.ids import RENDERING, new_id
from ....domain.observation import Observation
from ....domain.report import (
    ExplanationPackage,
    ReportArtifact,
    ReportFigure,
    ReportRendering,
    SubstanceProfile,
)
from ....persistence.object_store import ObjectNotFound, ObjectRef
from ....report.figures import (
    FIGURE_RENDERER_VERSION,
    FigureRejected,
    sanitize_svg,
    store_structure_figure,
)
from ....report.renderers import (
    RendererUnavailable,
)
from ....validation.answer.citations import cited_in_prose
from ....validation.report.draft_wire import ReportDraftCandidate
from ._common import RENDERERS, _now, log


class ReportAssemblyMixin:
    """Assembly of the accepted report artifact."""

    async def _resolve_evidence(
        self, uow, session_id: str, draft: ReportDraftCandidate
    ) -> Mapping[str, EvidenceRecord]:
        ids = {e for c in draft.claims for e in c.citation_ids}
        ids |= {e for s in draft.evidence_synthesis for e in s.evidence_ids}
        # Inline ``[@evd_...]`` markers too: a token in a sentence is a citation
        # that has to resolve, be validated and reach the reference snapshot,
        # exactly like one attached to a claim (REP-01).
        for section in draft.sections:
            ids |= cited_in_prose(section.body_markdown)
        resolved: dict[str, EvidenceRecord] = {}
        for evidence_id in ids:
            record = await uow.evidence.get(evidence_id, session_id=session_id)
            if record is not None:
                resolved[evidence_id] = record
        return resolved

    async def _resolve_read_evidence_ids(self, uow, run_id: str) -> frozenset[str]:
        """Which evidence this run actually opened, not merely saw in a search
        result — the same source of truth ``submit_answer`` uses."""
        calls = await uow.tool_calls.list_for_run(run_id)
        read: set[str] = set()
        for call in calls:
            if call["tool_name"] == "get_evidence_record" and call["status"] == "completed":
                read.update(call.get("observation_ids") or ())
        return frozenset(read)

    @staticmethod
    def _resolve_explanations(
        build, observations_by_id: Mapping[str, Observation]
    ) -> tuple[dict[str, ExplanationPackage], dict[str, ReportFigure]]:
        """Rebuild the packages this build produced, from the observations.

        Read from stored observations rather than from the build's stage state
        so that a figure/observation mismatch is checked against what was
        actually persisted, not against a summary written next to it.
        """
        from ...explanation.service import is_explanation_schema, package_from_observation

        packages: dict[str, ExplanationPackage] = {}
        figures: dict[str, ReportFigure] = {}
        for observation in observations_by_id.values():
            if not is_explanation_schema(observation.schema_version):
                continue
            package = package_from_observation(observation)
            packages[package.explanation_id] = package
            if package.figure is not None:
                figures[package.figure.figure_id] = package.figure
        return packages, figures

    async def _figure_errors(
        self, uow, figures: Mapping[str, ReportFigure], *, owner_id: str, session_id: str
    ) -> dict[str, str]:
        errors: dict[str, str] = {}
        for figure_id, figure in figures.items():
            attachment = await uow.attachments.get(figure.attachment_id, owner_id=owner_id)
            if attachment is None or attachment.session_id != session_id:
                errors[figure_id] = "the attachment does not exist in this session"
                continue
            if attachment.media_type != figure.media_type:
                errors[figure_id] = "the attachment MIME type differs from the figure"
                continue
            if attachment.sha256 != figure.content_sha256:
                errors[figure_id] = "the attachment hash differs from the figure"
                continue
            if self._objects is None:
                errors[figure_id] = "no object store is configured"
                continue
            try:
                data = await self._objects.get(ObjectRef(attachment.object_uri))
            except ObjectNotFound:
                errors[figure_id] = "the stored figure bytes are missing"
                continue
            if hashlib.sha256(data).hexdigest() != figure.content_sha256:
                errors[figure_id] = "the stored figure bytes do not match the content hash"
        return errors

    async def _structure_figure(
        self, uow, snapshot, *, owner_id: str, session_id: str,
        existing_figure_id: str | None,
    ) -> ReportFigure | None:
        """The compound's own picture, drawn once per session and reused.

        Nothing produced this before: ``SubstanceProfile.structure_figure_id``
        existed and was read from a ``stage_state`` key nothing ever wrote, so
        every report's substance profile was pictureless and the only image
        available was the explanation heat map — which is a statement about one
        model on one endpoint, not about what the compound is (REP-02).

        A failure here is never a failed report. The identity of the compound is
        already in the profile as canonical SMILES and, where resolved, a name
        and identifiers; the drawing is an aid, and losing it must not lose the
        document.
        """
        if existing_figure_id:
            row = await uow.reports.get_figure(existing_figure_id, session_id=session_id)
            # A figure written by an older sanitizer is redrawn, not reused:
            # v1 stripped RDKit's paint and stored black boxes.
            if row is not None and row["renderer_version"] == FIGURE_RENDERER_VERSION:
                return ReportFigure(
                    figure_id=row["figure_id"],
                    attachment_id=row["attachment_id"],
                    media_type=row["media_type"],
                    caption=row["caption"],
                    alt_text=row["alt_text"],
                    content_sha256=row["content_sha256"],
                    renderer_version=row["renderer_version"],
                    endpoint=row.get("endpoint"),
                    task=row.get("task"),
                    observation_id=row.get("observation_id"),
                )
        if self._predictor is None or self._objects is None:
            return None
        try:
            response = await self._predictor.depict(snapshot.canonical_smiles)
        except Exception:  # noqa: BLE001 — an absent drawing, never a failed report
            log.warning(
                "could not draw the structure figure for analysis %s", snapshot.id,
                exc_info=True,
            )
            return None
        if not response.depiction_svg:
            return None
        try:
            stored = await store_structure_figure(
                svg=response.depiction_svg,
                object_store=self._objects,
                owner_id=owner_id,
                session_id=session_id,
                canonical_smiles=snapshot.canonical_smiles,
                now=_now(),
            )
        except FigureRejected:
            log.warning("the structure depiction did not survive sanitization", exc_info=True)
            return None
        await uow.attachments.add(stored.attachment)
        await uow.reports.add_figure(
            stored.figure, session_id=session_id, created_at=_now()
        )
        return stored.figure

    @staticmethod
    def _subject_profile(
        build,
        snapshot,
        observations_by_id: Mapping[str, Observation],
        *,
        structure_figure_id: str | None = None,
    ) -> SubstanceProfile:
        """Identity from the compound record when one was resolved, structure
        from the snapshot always. An unresolved field stays ``None``."""
        record: dict[str, Any] | None = None
        for observation in observations_by_id.values():
            if observation.schema_version == "compound-record-v1" and observation.provenance.get(
                "resolved"
            ):
                record = observation.canonical_payload
        if record is None:
            return SubstanceProfile(
                canonical_smiles=snapshot.canonical_smiles,
                structure_figure_id=structure_figure_id,
            )
        provider = record.get("provider", "compound_provider")
        identifiers = dict(record.get("identifiers") or {})
        source_refs = {
            field: f"{provider}:{record.get('canonical_url') or 'record'}"
            for field in ("preferred_name", "synonyms", "identifiers", "properties")
            if record.get(field)
        }
        return SubstanceProfile(
            canonical_smiles=snapshot.canonical_smiles,
            structure_figure_id=structure_figure_id,
            preferred_name=record.get("preferred_name"),
            synonyms=tuple(record.get("synonyms") or ()),
            identifiers=identifiers,
            properties=tuple(record.get("properties") or ()),
            source_refs=source_refs,
        )

    @staticmethod
    def _provenance(
        snapshot, build, run_id: str, *, runtime_manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        provenance = {
            "analysis_content_sha256": snapshot.content_sha256,
            "predictor_service_version": snapshot.provenance.service_version,
            "predictor_git_commit": snapshot.provenance.git_commit,
            "predictor_base_url_id": snapshot.provenance.base_url_id,
            "artifact_hashes": list(snapshot.provenance.artifact_hashes),
            "policy_snapshot": snapshot.policy_snapshot,
            "report_build_id": build.id,
            "run_id": run_id,
            "requested": build.request.to_dict(),
            "instruction_manifest": build.stage_state.get("instruction_manifest"),
        }
        if runtime_manifest is not None:
            provenance["runtime"] = runtime_manifest
        return provenance

    async def _figure_svgs(
        self, uow, artifact: ReportArtifact, *, owner_id: str, session_id: str
    ) -> dict[str, str]:
        """``figure_id -> sanitized SVG source`` for every figure the report shows.

        Re-sanitized on the way out even though the stored bytes were sanitized
        on the way in. The stored bytes are what a renderer inlines into HTML a
        reader opens, and "it was clean when we wrote it" is not a property this
        function can verify about a blob it just fetched — the hash check proves
        the bytes are the ones recorded, not that the recording was safe.

        A figure that cannot be resolved is simply absent from the map, and the
        renderers fall back to its alt text: the report still says what the
        picture would have said.
        """
        if self._objects is None:
            return {}
        resolved: dict[str, str] = {}
        for figure in artifact.figures:
            if figure.media_type != "image/svg+xml":
                continue
            attachment = await uow.attachments.get(figure.attachment_id, owner_id=owner_id)
            if attachment is None or attachment.session_id != session_id:
                continue
            try:
                data = await self._objects.get(ObjectRef(attachment.object_uri))
            except ObjectNotFound:
                continue
            if hashlib.sha256(data).hexdigest() != figure.content_sha256:
                continue
            try:
                resolved[figure.figure_id] = sanitize_svg(data.decode("utf-8"))
            except (FigureRejected, UnicodeDecodeError):
                continue
        return resolved

    async def _render(
        self, artifact: ReportArtifact, build, figure_svgs: Mapping[str, str] | None = None
    ) -> tuple[tuple[ReportRendering, ...], list[dict[str, str]]]:
        """Produce every requested format this deployment can produce.

        A format that fails is recorded as a failure and the report still
        exists: the artifact is the report, and losing a PDF must not lose the
        document (eval scenario 14).
        """
        renderings: list[ReportRendering] = []
        failures: list[dict[str, str]] = []
        for fmt in build.request.output_formats:
            entry = RENDERERS.get(fmt)
            if entry is None:
                failures.append(
                    {
                        "format": fmt,
                        "reason": f"no renderer is pinned for {fmt!r} in this deployment",
                    }
                )
                continue
            media_type, version, renderer = entry
            try:
                rendered = renderer(artifact, figure_svgs=figure_svgs or {})
            except RendererUnavailable as exc:
                failures.append({"format": fmt, "reason": str(exc)})
                continue
            data = rendered if isinstance(rendered, bytes) else rendered.encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
            key = f"reports/{artifact.id}/{fmt}-{digest[:16]}"
            if self._objects is None:
                failures.append(
                    {
                        "format": fmt,
                        "reason": "this deployment has no object store, so no file was written",
                    }
                )
                continue
            await self._objects.put(key, data, content_type=media_type)
            renderings.append(
                ReportRendering(
                    rendering_id=new_id(RENDERING),
                    format=fmt,
                    media_type=media_type,
                    object_uri=key,
                    content_sha256=digest,
                    size_bytes=len(data),
                    renderer_version=version,
                    created_at=_now(),
                )
            )
        return tuple(renderings), failures
