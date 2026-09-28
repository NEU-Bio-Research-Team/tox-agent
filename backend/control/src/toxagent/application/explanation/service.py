"""Explanation packages: one endpoint or assay, one image, one set of highlights.

Spec sections 5.5, 6.2 and workstream R1. This is the deterministic half of
"explain this prediction": the predictor computes the attribution, this module
persists it as an immutable observation, draws and stores the figure, and
extracts the contributor lists. A model never does any of it — it asks for a
target and receives refs.

The invariant the report validator later depends on is established here: the
figure and the highlights are derived from *the same* explanation payload and
carry *the same* observation id, so an explanation narrative and the picture
beside it cannot describe different computations (spec section 11,
"Explanation linkage").

**One pipeline, one cache (XAI-01).** This module used to be the second of two
explanation pipelines. ``create_analysis`` computed eager explanations, wrote
them under ``toxpred-explanation-v2`` and checkpointed them under one cache key;
this module wrote ``toxpred-explanation-v1`` under a different one. A report
builder searching for v1 by that second key could not see a finished v2
explanation, so it called the predictor again — or recorded a gap — with the
answer already in the database. Now both write
``EXPLANATION_SCHEMA_VERSION`` through :func:`explanation_observation`, both key
on :func:`explanation_cache_key`, and this service resolves an existing
explanation by *target identity* rather than by schema string, reading v1 and v2
rows as well. Where a cached payload exists but its figure does not, only the
figure is produced: the backward pass is never repeated to obtain a picture.

Reuse is by cache key, and the key includes the model id for the reason
``get_attribution`` already learned (I11): every other component of the key is
identical between two admitted models for the same endpoint, so a key without
it serves model A's picture for a question about model B.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Final, Sequence

from ...domain.errors import AnalysisNotFound, InvalidRequest
from ...domain.events import EventType
from ...domain.ids import EXPLANATION, new_id
from ...domain.observation import Observation, ObservationKind, Producer
from ...domain.xai_coverage import compute_coverage
from ...domain.report import (
    ExplanationHighlights,
    ExplanationPackage,
    ExplanationStatus,
    ReportFigure,
)
from ...predictor.client import PredictorClient
from ...predictor.contract import TOX21_TASKS
from ...report.figures import (
    FIGURE_RENDERER_VERSION,
    FigureRejected,
    alt_text_for,
    caption_for,
    store_explanation_figure,
)
from .identity import (
    ATTRIBUTION_ALIGNMENT_VERSION,
    EXPLANATION_SCHEMA_VERSION,
    EXPLANATION_SCHEMA_VERSIONS,
    explanation_cache_key,
    is_explanation_schema,
    model_artifact_fingerprint,
)

__all__ = [
    "ATTRIBUTION_ALIGNMENT_VERSION",
    "DEFAULT_TOP_K",
    "EXPLANATION_SCHEMA_VERSION",
    "EXPLANATION_SCHEMA_VERSIONS",
    "ExplanationResult",
    "ExplanationUnavailable",
    "GetOrCreateExplanation",
    "explanation_cache_key",
    "explanation_observation",
    "extract_highlights",
    "is_explanation_schema",
    "package_from_observation",
]

#: How many contributors a package carries per direction. The tail of an
#: attribution is noise, and a model that receives it spends budget reading
#: numbers it must not treat as meaningful.
DEFAULT_TOP_K = 8


class ExplanationUnavailable:
    """Reason codes for "there is no explanation", as a closed vocabulary.

    A caller — and a report gap — needs to distinguish *why*. "The model is not
    loaded", "this provider cannot attribute at all", "the alignment failed",
    "we ran out of budget" and "the numbers are fine but the picture could not
    be stored" lead to five different actions, and a single
    ``explanation_failed`` collapses them into a shrug (spec XAI-01).

    A namespace of plain strings rather than an ``Enum``: these codes are
    written into observation provenance and read back by the report builder,
    where a value that survives a JSON round trip unchanged is worth more than
    one that needs a constructor.
    """

    MODEL_UNAVAILABLE: Final = "model_unavailable"
    ATTRIBUTION_UNSUPPORTED: Final = "attribution_unsupported"
    ALIGNMENT_FAILED: Final = "alignment_failed"
    BUDGET_EXCEEDED: Final = "budget_exceeded"
    FIGURE_STORE_FAILED: Final = "figure_store_failed"
    NO_DEPICTION: Final = "no_depiction_returned"
    NO_OBJECT_STORE: Final = "no_object_store"


#: Predictor-reported reasons, mapped onto the vocabulary above. Anything not
#: listed stays ``None`` and the raw predictor message is carried verbatim —
#: inventing a category for an unrecognised failure would be worse than saying
#: the failure was not one we know how to classify.
_FAILURE_REASONS: Final[dict[str, str]] = {
    "explanation_budget_exceeded": ExplanationUnavailable.BUDGET_EXCEEDED,
    "TimeoutError": ExplanationUnavailable.BUDGET_EXCEEDED,
    "attribution_unsupported": ExplanationUnavailable.ATTRIBUTION_UNSUPPORTED,
    "token_attribution_unsupported": ExplanationUnavailable.ATTRIBUTION_UNSUPPORTED,
    "model_unavailable": ExplanationUnavailable.MODEL_UNAVAILABLE,
    "ModelNotLoaded": ExplanationUnavailable.MODEL_UNAVAILABLE,
    "alignment_failed": ExplanationUnavailable.ALIGNMENT_FAILED,
    "token_offset_alignment_failed": ExplanationUnavailable.ALIGNMENT_FAILED,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def classify_failure(payload: dict[str, Any]) -> str | None:
    """The reason code for a failed payload, or ``None`` when unrecognised."""
    metadata = payload.get("metadata") or {}
    for field in ("reason", "error", "code", "message"):
        raw = metadata.get(field)
        if isinstance(raw, str) and raw in _FAILURE_REASONS:
            return _FAILURE_REASONS[raw]
    return None


def _signed(atom: dict[str, Any]) -> float:
    """The contribution's direction. Falls back to unsigned magnitude only when
    the predictor did not report a signed value, in which case direction is
    unknown and the atom is treated as positive-magnitude, never as negative —
    inventing a negative contribution is worse than declining to split."""
    value = atom.get("signed_contribution")
    if value is None:
        value = atom.get("importance", atom.get("magnitude", 0.0))
    return float(value)


def extract_highlights(
    payload: dict[str, Any], *, top_k: int = DEFAULT_TOP_K
) -> ExplanationHighlights:
    """Split atoms into positive and negative contributors.

    ``unmapped_importance`` is carried through verbatim, including ``0.0``:
    "none of the attribution mass fell outside the structure" and "the
    explainer did not tell us" are different facts, and only the second is
    ``None`` (spec section 6.2).
    """
    atoms: Sequence[dict[str, Any]] = payload.get("atoms") or ()
    scored = [
        {
            "atom_index": atom.get("atom_index"),
            "symbol": atom.get("symbol"),
            "signed_contribution": _signed(atom),
            "relative_importance": atom.get("relative_importance"),
        }
        for atom in atoms
    ]
    positive = sorted(
        (a for a in scored if a["signed_contribution"] > 0),
        key=lambda a: a["signed_contribution"],
        reverse=True,
    )[:top_k]
    negative = sorted(
        (a for a in scored if a["signed_contribution"] < 0),
        key=lambda a: a["signed_contribution"],
    )[:top_k]
    unmapped = payload.get("unmapped_importance")
    coverage = compute_coverage(
        payload,
        positive_contributor_count=len(positive),
        negative_contributor_count=len(negative),
    )
    return ExplanationHighlights(
        positive_contributors=tuple(positive),
        negative_contributors=tuple(negative),
        unmapped_importance=float(unmapped) if unmapped is not None else None,
        coverage=coverage.to_dict(),
    )


def has_numeric_attribution(payload: dict[str, Any]) -> bool:
    """Whether the payload carries attribution worth caching.

    A ``partial`` explanation with real numbers is a useful, reusable result. A
    ``partial`` one with none is a failure wearing a softer word, and caching it
    would pin the failure in place until the weights change (spec XAI-01: "only
    cache completed or partial with valid numeric attribution").
    """
    for collection in ("atoms", "bonds"):
        for item in payload.get(collection) or ():
            for field in ("signed_contribution", "relative_importance", "importance"):
                value = item.get(field)
                if isinstance(value, (int, float)) and value == value:  # not NaN
                    return True
    return False


def explanation_model_projection(
    *,
    payload: dict[str, Any],
    analysis_id: str,
    explanation_id: str,
    model_id: str | None,
    highlights: ExplanationHighlights,
    figure_id: str | None = None,
    figure_unavailable_reason: str | None = None,
    endpoint: str,
    task: str | None,
) -> dict[str, Any]:
    """What a model may read about one explanation. Bounded on purpose."""
    return {
        "analysis_id": analysis_id,
        "explanation_id": explanation_id,
        "endpoint": endpoint,
        "task": task,
        "status": payload.get("status"),
        "method": payload.get("method"),
        "model_id": (payload.get("metadata") or {}).get("model_id") or model_id,
        "probability": payload.get("probability"),
        "positive_contributors": [dict(c) for c in highlights.positive_contributors],
        "negative_contributors": [dict(c) for c in highlights.negative_contributors],
        "unmapped_importance": highlights.unmapped_importance,
        # The model reads the coverage accounting rather than counting
        # contributors for itself: a count it derives is a count it can get
        # wrong, and then two sections of one report disagree (P0-2).
        "coverage": dict(highlights.coverage),
        "figure_id": figure_id,
        "figure_unavailable_reason": figure_unavailable_reason,
        "unavailable_reason": classify_failure(payload)
        if payload.get("status") == "failed" else None,
        "required_limitations": ["attribution_not_causality"],
    }


def explanation_observation(
    *,
    payload: dict[str, Any],
    session_id: str,
    run_id: str,
    analysis_id: str,
    snapshot_provenance: dict[str, Any],
    cache_key: str,
    requested_model_id: str | None,
    endpoint: str,
    task: str | None,
    now: datetime,
    explanation_id: str | None = None,
    figure: ReportFigure | None = None,
    figure_unavailable_reason: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> Observation:
    """The one row shape an explanation is persisted as.

    Both the eager (``create_analysis``) and on-demand (``GetOrCreateExplanation``)
    paths call this, which is the whole of XAI-01's fix: the cache key, the
    explanation id, the target and the figure all live in a single provenance
    shape, so either path's output is resolvable by the other and by the report
    builder.
    """
    explanation_id = explanation_id or new_id(EXPLANATION)
    highlights = extract_highlights(payload, top_k=top_k)
    resolved_model = (payload.get("metadata") or {}).get("model_id") or requested_model_id
    observation = Observation.create(
        session_id=session_id,
        run_id=run_id,
        producer=Producer.ATTRIBUTION,
        kind=ObservationKind.ATTRIBUTION,
        schema_version=EXPLANATION_SCHEMA_VERSION,
        canonical_payload=payload,
        model_projection=explanation_model_projection(
            payload=payload,
            analysis_id=analysis_id,
            explanation_id=explanation_id,
            model_id=requested_model_id,
            highlights=highlights,
            figure_id=figure.figure_id if figure else None,
            figure_unavailable_reason=figure_unavailable_reason,
            endpoint=endpoint,
            task=task,
        ),
        provenance={
            **snapshot_provenance,
            "analysis_id": analysis_id,
            "explanation_id": explanation_id,
            "cache_key": cache_key,
            "alignment_version": ATTRIBUTION_ALIGNMENT_VERSION,
            "method": payload.get("method"),
            "model_id": resolved_model,
            "target": {"endpoint": endpoint, "task": task},
            "atom_order_version": payload.get("atom_order_version"),
            "structure_order_version": payload.get("structure_order_version"),
            "figure_unavailable_reason": figure_unavailable_reason,
            "unavailable_reason": classify_failure(payload)
            if payload.get("status") == "failed" else None,
        },
        now=now,
        required_limitations=("attribution_not_causality",),
    )
    if figure is None:
        return observation
    # The figure travels inside the observation's provenance, so the pair is
    # stored in one row and cannot come apart.
    return replace(
        observation,
        provenance={
            **observation.provenance,
            "figure": replace(figure, observation_id=observation.id).to_dict(),
        },
    )


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    package: ExplanationPackage
    observation: Observation
    reused: bool
    #: True when a cached numeric payload was kept and only its figure was
    #: drawn. The predictor's ``explain`` was not called; ``reused`` is also
    #: True. Recorded because "the picture is new, the numbers are not" is the
    #: one case where an audit could otherwise conclude the attribution was
    #: recomputed.
    figure_only: bool = False


class GetOrCreateExplanation:
    """Create one explanation, or return the one that already exists.

    ``object_store`` may be ``None`` in a deployment that stores no bytes; the
    numeric explanation is still produced and the package simply has no figure,
    which is a visible gap rather than a silent one.
    """

    def __init__(self, database, predictor: PredictorClient, object_store=None) -> None:
        self._db = database
        self._predictor = predictor
        self._objects = object_store

    async def execute(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        analysis_id: str,
        endpoint: str,
        task: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> ExplanationResult:
        async with self._db.unit_of_work() as uow:
            snapshot = await uow.analyses.get(analysis_id, session_id=session_id)
        if snapshot is None:
            raise AnalysisNotFound("no such analysis in this session", analysis_id=analysis_id)

        if endpoint == "tox21" and not task:
            raise InvalidRequest(
                "explaining tox21 requires a task: the twelve assays are independent "
                "measurements and a combined explanation would not mean anything",
                allowed=list(TOX21_TASKS),
            )
        if endpoint != "tox21" and task:
            raise InvalidRequest(f"task is only meaningful for tox21, not {endpoint}")
        if endpoint not in snapshot.served_endpoints:
            raise InvalidRequest(
                f"this analysis has no {endpoint} section",
                served=list(snapshot.served_endpoints),
            )

        model_id = snapshot.model_for(endpoint)
        cache_key = explanation_cache_key(
            canonical_smiles=snapshot.canonical_smiles,
            endpoint=endpoint,
            task=task,
            model_id=model_id,
            artifact_fingerprint=model_artifact_fingerprint(snapshot.provenance, model_id),
        )

        existing = await self._find_existing(
            analysis_id=snapshot.id, cache_key=cache_key, endpoint=endpoint, task=task
        )
        if existing is not None:
            package = package_from_observation(existing, top_k=top_k)
            # A figure stored by an older sanitizer counts as missing. v1
            # dropped RDKit's `style` paint and stored black boxes; reusing one
            # would put that box into every new report of this explanation.
            current_figure = (
                package.figure is not None
                and package.figure.renderer_version == FIGURE_RENDERER_VERSION
            )
            if current_figure or not self._can_draw(existing):
                return ExplanationResult(package=package, observation=existing, reused=True)
            # Numbers cached, picture missing. Draw only the picture: the
            # backward pass is the expensive half and repeating it to obtain an
            # image would also risk a figure that disagrees with the cached
            # numbers it is supposed to depict.
            return await self._add_figure(
                observation=existing,
                package=package,
                owner_id=owner_id,
                session_id=session_id,
                run_id=run_id,
                analysis_id=snapshot.id,
                endpoint=endpoint,
                task=task,
                top_k=top_k,
            )

        # Nothing persisted for this identity. The analysis pipeline may still
        # have checkpointed the payload — that table is the shared cache now,
        # so a checkpoint is a cache hit rather than a private optimisation.
        canonical = await self._checkpointed(cache_key)
        from_checkpoint = canonical is not None
        if canonical is None:
            response = await self._predictor.explain(
                snapshot.canonical_smiles, endpoint, task, model_id=model_id
            )
            canonical = response.model_dump(mode="json")

        status = ExplanationStatus(canonical.get("status", "failed"))
        highlights = extract_highlights(canonical, top_k=top_k)
        figure, figure_error, attachment = await self._draw(
            canonical, status=status, owner_id=owner_id, session_id=session_id,
            endpoint=endpoint, task=task, highlights=highlights,
        )

        explanation_id = new_id(EXPLANATION)
        observation = explanation_observation(
            payload=canonical,
            session_id=session_id,
            run_id=run_id,
            analysis_id=snapshot.id,
            snapshot_provenance=snapshot.provenance.to_dict(),
            cache_key=cache_key,
            requested_model_id=model_id,
            endpoint=endpoint,
            task=task,
            now=_now(),
            explanation_id=explanation_id,
            figure=figure,
            figure_unavailable_reason=figure_error,
            top_k=top_k,
        )
        if figure is not None:
            figure = replace(figure, observation_id=observation.id)

        await self._persist(
            observation=observation, attachment=attachment, figure=figure,
            analysis_id=snapshot.id, session_id=session_id, run_id=run_id,
            endpoint=endpoint, task=task, status=status,
        )
        # Cache only a result worth reusing. A transient failure written to the
        # cache would outlive its cause and make every later request return it.
        if not from_checkpoint and self._cacheable(canonical, status):
            await self._checkpoint(
                cache_key, session_id=session_id, endpoint=endpoint, task=task,
                model_id=model_id, canonical_smiles=snapshot.canonical_smiles,
                payload=canonical,
            )

        package = ExplanationPackage(
            explanation_id=explanation_id,
            observation_id=observation.id,
            endpoint=endpoint,
            task=task,
            method=canonical.get("method"),
            status=status,
            figure=figure,
            highlights=highlights,
            failure_reason=(
                classify_failure(canonical)
                or (canonical.get("metadata") or {}).get("message")
                if status is ExplanationStatus.FAILED
                else figure_error
            ),
        )
        return ExplanationResult(
            package=package, observation=observation, reused=from_checkpoint
        )

    # --- resolution --------------------------------------------------------

    async def _find_existing(
        self, *, analysis_id: str, cache_key: str, endpoint: str, task: str | None
    ) -> Observation | None:
        """The stored explanation for this target, if there is one.

        Resolution is by identity, in two passes. The cache key is the strong
        match. Failing that, an observation written by an older pipeline — v1 or
        v2, before either recorded a key — is matched on the target it names,
        which is the only identity such a row carries. Both passes still require
        the recorded model id to agree where one exists, because an explanation
        by another model is not an answer about this one (I11).
        """
        async with self._db.unit_of_work() as uow:
            items = await uow.observations.list_for_analysis(analysis_id)
        attributions = [
            item
            for item in items
            if item.kind is ObservationKind.ATTRIBUTION and is_explanation_schema(
                item.schema_version
            )
        ]
        keyed = [i for i in attributions if i.provenance.get("cache_key") == cache_key]
        if keyed:
            return keyed[-1]
        legacy = [
            item
            for item in attributions
            if item.provenance.get("cache_key") is None
            and self._targets(item) == (endpoint, task)
        ]
        return legacy[-1] if legacy else None

    @staticmethod
    def _targets(observation: Observation) -> tuple[str | None, str | None]:
        target = observation.provenance.get("target") or {}
        payload = observation.canonical_payload
        projection = observation.model_projection
        endpoint = (
            target.get("endpoint") or payload.get("endpoint") or projection.get("endpoint")
        )
        task = target.get("task") or payload.get("task") or projection.get("task")
        return endpoint, task

    def _can_draw(self, observation: Observation) -> bool:
        if self._objects is None:
            return False
        payload = observation.canonical_payload
        return bool(payload.get("depiction_svg")) and payload.get("status") != "failed"

    @staticmethod
    def _cacheable(payload: dict[str, Any], status: ExplanationStatus) -> bool:
        if status is ExplanationStatus.FAILED:
            return False
        return has_numeric_attribution(payload)

    async def _checkpointed(self, cache_key: str) -> dict[str, Any] | None:
        async with self._db.unit_of_work() as uow:
            found = await uow.explanation_checkpoints.get_many([cache_key])
        payload = found.get(cache_key)
        # A checkpointed failure is not a cache hit: it is a record that an
        # attempt was made, and serving it would make one bad minute permanent.
        if isinstance(payload, dict) and payload.get("status") != "failed":
            return payload
        return None

    async def _checkpoint(
        self, cache_key: str, *, session_id: str, endpoint: str, task: str | None,
        model_id: str | None, canonical_smiles: str, payload: dict[str, Any],
    ) -> None:
        try:
            async with self._db.unit_of_work() as uow:
                await uow.explanation_checkpoints.put(
                    cache_key, session_id=session_id, endpoint=endpoint, task=task,
                    model_id=model_id, canonical_smiles=canonical_smiles,
                    payload=payload, now=_now(),
                )
                await uow.commit()
        except Exception:  # noqa: BLE001 — a lost checkpoint costs a recompute
            import logging

            logging.getLogger("toxagent.explanation").exception(
                "could not checkpoint the %s explanation for session %s", endpoint, session_id
            )

    # --- figure ------------------------------------------------------------

    async def _draw(
        self,
        canonical: dict[str, Any],
        *,
        status: ExplanationStatus,
        owner_id: str,
        session_id: str,
        endpoint: str,
        task: str | None,
        highlights: ExplanationHighlights,
    ) -> tuple[ReportFigure | None, str | None, Any]:
        if status is ExplanationStatus.FAILED:
            return None, None, None
        if not canonical.get("depiction_svg"):
            return None, ExplanationUnavailable.NO_DEPICTION, None
        if self._objects is None:
            return None, ExplanationUnavailable.NO_OBJECT_STORE, None
        try:
            stored = await store_explanation_figure(
                svg=canonical["depiction_svg"],
                object_store=self._objects,
                owner_id=owner_id,
                session_id=session_id,
                endpoint=endpoint,
                task=task,
                # Filled in by the caller: the observation does not exist yet,
                # and a figure claiming an id nothing resolves to is exactly
                # the drift this design prevents.
                observation_id=None,  # type: ignore[arg-type]
                caption=caption_for(
                    endpoint, task, method=canonical.get("method"),
                    predicted_class=canonical.get("target_class"),
                ),
                alt_text=alt_text_for(
                    endpoint, task,
                    top_positive=list(highlights.positive_contributors),
                    top_negative=list(highlights.negative_contributors),
                    predicted_class=canonical.get("target_class"),
                ),
                now=_now(),
            )
        except FigureRejected as exc:
            # A refused depiction is a gap, never a reason to fail the numeric
            # explanation that is otherwise complete.
            return None, f"{ExplanationUnavailable.FIGURE_STORE_FAILED}: {exc}", None
        return stored.figure, None, stored.attachment

    async def _add_figure(
        self,
        *,
        observation: Observation,
        package: ExplanationPackage,
        owner_id: str,
        session_id: str,
        run_id: str,
        analysis_id: str,
        endpoint: str,
        task: str | None,
        top_k: int,
    ) -> ExplanationResult:
        """Give a cached numeric explanation the figure it never got.

        The observation is immutable, so this writes a *new* one carrying the
        same canonical payload, the same cache key and the same explanation id.
        Both rows therefore describe one computation, and the later one is the
        one every reader resolves — an update in place would have rewritten
        history to hide that the first attempt had no picture.
        """
        canonical = observation.canonical_payload
        status = ExplanationStatus(canonical.get("status", "failed"))
        highlights = extract_highlights(canonical, top_k=top_k)
        figure, figure_error, attachment = await self._draw(
            canonical, status=status, owner_id=owner_id, session_id=session_id,
            endpoint=endpoint, task=task, highlights=highlights,
        )
        if figure is None:
            # Nothing was gained; hand back what was already stored rather than
            # writing a second row saying the same thing.
            return ExplanationResult(package=package, observation=observation, reused=True)

        redrawn = explanation_observation(
            payload=canonical,
            session_id=session_id,
            run_id=run_id,
            analysis_id=analysis_id,
            snapshot_provenance={
                k: v
                for k, v in observation.provenance.items()
                if k not in {
                    "analysis_id", "explanation_id", "cache_key", "figure",
                    "figure_unavailable_reason", "target", "unavailable_reason",
                }
            },
            cache_key=observation.provenance.get("cache_key") or "",
            requested_model_id=observation.provenance.get("model_id"),
            endpoint=endpoint,
            task=task,
            now=_now(),
            explanation_id=package.explanation_id,
            figure=figure,
            figure_unavailable_reason=figure_error,
            top_k=top_k,
        )
        figure = replace(figure, observation_id=redrawn.id)
        await self._persist(
            observation=redrawn, attachment=attachment, figure=figure,
            analysis_id=analysis_id, session_id=session_id, run_id=run_id,
            endpoint=endpoint, task=task, status=status,
        )
        return ExplanationResult(
            package=replace(package, figure=figure, observation_id=redrawn.id),
            observation=redrawn,
            reused=True,
            figure_only=True,
        )

    # --- persistence -------------------------------------------------------

    async def _persist(
        self,
        *,
        observation: Observation,
        attachment,
        figure: ReportFigure | None,
        analysis_id: str,
        session_id: str,
        run_id: str,
        endpoint: str,
        task: str | None,
        status: ExplanationStatus,
    ) -> None:
        """Observation, attachment and figure metadata in one transaction.

        All three or none: a figure row whose bytes were never written, or bytes
        with no observation to account for them, is the drift the report
        validator would later have to catch by hashing.
        """
        async with self._db.unit_of_work() as uow:
            if attachment is not None:
                await uow.attachments.add(attachment)
            await uow.observations.add(observation, analysis_id=analysis_id)
            if figure is not None:
                existing = await uow.reports.get_figure(
                    figure.figure_id, session_id=session_id
                )
                if existing is None:
                    await uow.reports.add_figure(
                        figure, session_id=session_id, created_at=observation.created_at
                    )
            uow.emit(
                session_id=session_id, type=EventType.OBSERVATION_CREATED,
                entity_type="observation", entity_id=observation.id, run_id=run_id,
                payload={
                    "kind": "explanation", "endpoint": endpoint, "task": task,
                    "status": status.value,
                    "figure_id": figure.figure_id if figure else None,
                },
            )
            if figure is not None:
                uow.emit(
                    session_id=session_id, type=EventType.REPORT_FIGURE_CREATED,
                    entity_type="report_figure", entity_id=figure.figure_id, run_id=run_id,
                    payload={
                        "figure_id": figure.figure_id,
                        "observation_id": observation.id,
                        "endpoint": endpoint,
                        "task": task,
                    },
                )
            await uow.commit()


def package_from_observation(
    observation: Observation, *, top_k: int = DEFAULT_TOP_K
) -> ExplanationPackage:
    """Rebuild the package from a stored observation.

    Reads the canonical payload rather than the projection: the projection is
    bounded for a prompt, and a reused package must be identical to the one the
    first call produced, not a truncation of it.

    A module function rather than a method because the report validator rebuilds
    packages from persisted observations with no service instance in hand — and
    the rebuild must be the same code in both places, or a report could be
    validated against a package that differs from the one the agent was shown.

    Reads v1, v2 and v3 rows. The differences are all absences: v2 recorded no
    ``explanation_id`` and no ``cache_key``, and neither older version carried a
    figure. Falling back to the target the payload names — rather than refusing
    the row — is what lets a report built today cite an explanation computed
    before XAI-01.
    """
    payload = observation.canonical_payload
    projection = observation.model_projection
    figure_dict = observation.provenance.get("figure")
    figure = None
    if isinstance(figure_dict, dict):
        figure = ReportFigure(
            figure_id=figure_dict["figure_id"],
            attachment_id=figure_dict["attachment_id"],
            media_type=figure_dict["media_type"],
            caption=figure_dict["caption"],
            alt_text=figure_dict["alt_text"],
            content_sha256=figure_dict["content_sha256"],
            renderer_version=figure_dict["renderer_version"],
            endpoint=figure_dict.get("endpoint"),
            task=figure_dict.get("task"),
            observation_id=figure_dict.get("observation_id"),
        )
    status = ExplanationStatus(payload.get("status", "failed"))
    target = observation.provenance.get("target") or {}
    failure_reason = observation.provenance.get("figure_unavailable_reason")
    if status is ExplanationStatus.FAILED:
        failure_reason = (
            observation.provenance.get("unavailable_reason")
            or classify_failure(payload)
            or (payload.get("metadata") or {}).get("message")
            or (payload.get("metadata") or {}).get("error")
        )
    return ExplanationPackage(
        # A v2 row has no explanation id of its own. Deriving one from the
        # observation id keeps it stable across reads — a fresh ``new_id`` per
        # read would give the same explanation a different name every time the
        # report builder looked at it.
        explanation_id=observation.provenance.get("explanation_id")
        or projection.get("explanation_id")
        or f"{EXPLANATION}_{observation.id.split('_', 1)[-1]}",
        observation_id=observation.id,
        endpoint=target.get("endpoint")
        or payload.get("endpoint")
        or projection.get("endpoint"),
        task=target.get("task") or payload.get("task") or projection.get("task"),
        method=payload.get("method"),
        status=status,
        figure=figure,
        highlights=extract_highlights(payload, top_k=top_k),
        failure_reason=failure_reason,
    )
