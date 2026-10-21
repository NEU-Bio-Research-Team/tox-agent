"""Resolve the subject a message is about, at admission (#44, subject_binding_v1).

The request names a subject by reference -- ``{"kind": "activity_screening",
"screening_id": ...}`` or ``{"kind": "analysis", "analysis_id": ...}`` -- and
the server reads what it is from what the session holds. Nothing here takes
an identity from the request: a client can choose among the session's
subjects, never describe one.

``{"kind": "frozen_analysis", "analysis_ref": ...}`` (#50) names a frozen
toxicity analysis the deployment's pinned replay file holds -- the subject of
a toxicity-only study case. The run gets a session-local copy, hash-checked
and made without calling the predictor, exactly as a replayed screening's
toxicity snapshot is; the binding keeps the origin ref and hash beside the
local id, so arms compare what they were given, not the id minted for them.

When the request names none, the session's subjects are its active analysis
and its attached screenings. Exactly one is the subject. None means the run
is subjectless, as before. More than one is ambiguous, and the caller asks
which -- it never takes the first.

Resolution happens in two steps because the run does not exist yet when the
choice is made: :func:`choose_subject` decides (and may say "ask"), and
:func:`bind_subject` builds the frozen binding inside the transaction that
creates the run, making the session-local copy of a replayed screening's
toxicity snapshot under that run.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from ...domain.activity import TargetIdentity
from ...domain.errors import AnalysisNotFound, InvalidRequest, ScreeningNotFound
from ...domain.subject import (
    ArtifactRef,
    RunSubjectBinding,
    SubjectIdentity,
    SubjectKind,
    analysis_binding,
    pending_analysis_binding,
)
from ...platform.flags import is_enabled
from ..activity import replay

#: The wire kind naming a frozen toxicity analysis by its origin id (#50).
FROZEN_ANALYSIS = "frozen_analysis"


@dataclass(frozen=True)
class SubjectRequest:
    """What a client may say about the subject: its kind and its id."""

    kind: str
    subject_id: str

    @classmethod
    def from_wire(cls, value: Mapping[str, Any] | None) -> "SubjectRequest | None":
        if not value:
            return None
        kind = value.get("kind")
        if kind == SubjectKind.ANALYSIS.value:
            identifier = value.get("analysis_id")
        elif kind == SubjectKind.ACTIVITY_SCREENING.value:
            identifier = value.get("screening_id")
        elif kind == FROZEN_ANALYSIS:
            identifier = value.get("analysis_ref")
        else:
            raise InvalidRequest(
                "subject_ref.kind must be analysis, activity_screening or frozen_analysis, "
                f"got {kind!r}"
            )
        if not identifier:
            raise InvalidRequest("subject_ref names no id for its kind")
        return cls(kind=str(kind), subject_id=str(identifier))


@dataclass(frozen=True)
class SubjectChoice:
    kind: SubjectKind | None
    subject_id: str | None
    resolved_by: str
    #: The screening's frozen record, or the analysis snapshot, as read.
    artifact: Any = None
    #: Set when the request named no subject and the session holds several.
    ambiguous_options: tuple[str, ...] = ()
    new_molecule_smiles: str | None = None
    #: A frozen analysis from the replay file, copied into the session when
    #: the run is bound (#50): its origin id, and the replay file's sha256.
    frozen_analysis_ref: str | None = None
    replay_sha256: str | None = None

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.ambiguous_options)

    @property
    def is_screening(self) -> bool:
        return self.kind is SubjectKind.ACTIVITY_SCREENING

    @property
    def is_frozen_analysis(self) -> bool:
        return self.frozen_analysis_ref is not None


NO_SUBJECT = SubjectChoice(kind=None, subject_id=None, resolved_by="none")


async def choose_subject(
    uow, *, session, requested: SubjectRequest | None, analysis_id: str | None,
    new_molecule_smiles: str | None, replay_store=None,
) -> SubjectChoice:
    """Which subject this message is about. Raises a 404 for a subject the
    session does not hold -- including another session's -- and asks rather
    than guesses when the session holds several."""
    screenings_on = is_enabled("activity_replay_v1")
    if requested is not None:
        if new_molecule_smiles:
            raise InvalidRequest(
                "a message names either an existing subject (subject_ref) or a new molecule, "
                "not both"
            )
        if analysis_id and not (
            requested.kind == SubjectKind.ANALYSIS.value and requested.subject_id == analysis_id
        ):
            raise InvalidRequest("subject_ref and analysis_id name different subjects")
        if requested.kind == FROZEN_ANALYSIS:
            frozen = (
                replay_store.get_analysis(requested.subject_id)
                if screenings_on and replay_store is not None else None
            )
            if frozen is None:
                raise AnalysisNotFound(
                    "no such frozen analysis in this deployment's replay file",
                    analysis_ref=requested.subject_id,
                )
            return SubjectChoice(
                SubjectKind.ANALYSIS, None, "explicit", artifact=frozen,
                frozen_analysis_ref=requested.subject_id, replay_sha256=replay_store.sha256,
            )
        if requested.kind == SubjectKind.ANALYSIS.value:
            snapshot = await uow.analyses.get(requested.subject_id, session_id=session.id)
            if snapshot is None:
                raise AnalysisNotFound("no such analysis in this session",
                                       analysis_id=requested.subject_id)
            return SubjectChoice(SubjectKind.ANALYSIS, snapshot.id, "explicit", artifact=snapshot)
        attachment = (
            await uow.screenings.get(requested.subject_id, session_id=session.id)
            if screenings_on else None
        )
        if attachment is None:
            raise ScreeningNotFound("no such screening attached to this session",
                                    screening_id=requested.subject_id)
        return SubjectChoice(SubjectKind.ACTIVITY_SCREENING, attachment.screening_id,
                             "explicit", artifact=attachment)

    if analysis_id:
        snapshot = await uow.analyses.get(analysis_id, session_id=session.id)
        if snapshot is None:
            raise AnalysisNotFound("no such analysis in this session", analysis_id=analysis_id)
        return SubjectChoice(SubjectKind.ANALYSIS, snapshot.id, "analysis_id", artifact=snapshot)

    if new_molecule_smiles:
        return SubjectChoice(SubjectKind.ANALYSIS, None, "new_molecule",
                             new_molecule_smiles=new_molecule_smiles)

    candidates: list[SubjectChoice] = []
    if session.active_analysis_id:
        snapshot = await uow.analyses.get(session.active_analysis_id, session_id=session.id)
        if snapshot is not None:
            candidates.append(
                SubjectChoice(SubjectKind.ANALYSIS, snapshot.id, "only_candidate", artifact=snapshot)
            )
    if screenings_on:
        for attachment in await uow.screenings.list_for_session(session.id):
            candidates.append(SubjectChoice(
                SubjectKind.ACTIVITY_SCREENING, attachment.screening_id, "only_candidate",
                artifact=attachment,
            ))
    if not candidates:
        return NO_SUBJECT
    if len(candidates) == 1:
        return candidates[0]
    return SubjectChoice(
        kind=None, subject_id=None, resolved_by="ambiguous",
        ambiguous_options=tuple(
            f"{'analysis' if c.kind is SubjectKind.ANALYSIS else 'screening'}:{c.subject_id}"
            for c in candidates
        ),
    )


def screening_identity(record: Mapping[str, Any]) -> SubjectIdentity:
    """The subject a frozen screening record states, as the server read it."""
    subject = (record.get("screening_state") or {}).get("subject") or {}
    target = subject.get("target")
    return SubjectIdentity(
        canonical_smiles=str(subject.get("canonical_smiles") or ""),
        target=TargetIdentity.from_dict(target) if target else None,
        assay_context=subject.get("assay_context"),
    )


async def bind_subject(uow, choice: SubjectChoice, *, session_id: str, run_id: str,
                       now: datetime) -> RunSubjectBinding | None:
    """The frozen binding for ``choice``, built inside the run's transaction."""
    if choice.kind is None:
        return None
    if choice.is_frozen_analysis:
        frozen = choice.artifact
        local = await replay.materialise_frozen_analysis(
            uow, frozen, session_id=session_id, run_id=run_id, now=now,
            replayed_from={"replay_sha256": choice.replay_sha256},
        )
        return analysis_binding(
            local.id, canonical_smiles=local.canonical_smiles,
            content_sha256_=local.content_sha256, resolved_by=choice.resolved_by,
            extra_refs=(ArtifactRef(
                kind="frozen_analysis", source_id=str(choice.frozen_analysis_ref),
                content_sha256=str(frozen.get("content_sha256")), local_id=local.id,
            ),),
        )
    if choice.kind is SubjectKind.ANALYSIS:
        if choice.subject_id is None:
            return pending_analysis_binding(choice.new_molecule_smiles or "")
        snapshot = choice.artifact
        return analysis_binding(
            snapshot.id, canonical_smiles=snapshot.canonical_smiles,
            content_sha256_=snapshot.content_sha256, resolved_by=choice.resolved_by,
        )
    attachment = choice.artifact
    record = attachment.record
    refs = [ArtifactRef(kind="screening", source_id=attachment.screening_id,
                        content_sha256=attachment.record_sha256)]
    local = await replay.materialise_toxicity(
        uow, record, session_id=session_id, run_id=run_id, now=now,
    )
    if local is not None:
        link = replay.linked_toxicity(record) or {}
        refs.append(ArtifactRef(
            kind="toxicity_snapshot", source_id=str(link.get("origin_analysis_id")),
            content_sha256=str(link.get("content_sha256")), local_id=local.id,
        ))
    return RunSubjectBinding(
        kind=SubjectKind.ACTIVITY_SCREENING,
        subject_ref={"screening_id": attachment.screening_id},
        identity=screening_identity(record), artifact_refs=tuple(refs),
        resolved_by=choice.resolved_by,
    )


def binding_of(context) -> RunSubjectBinding | None:
    """The run's frozen subject binding, or ``None`` for a run without one."""
    data = getattr(context, "subject", None)
    if not data:
        return None
    return RunSubjectBinding.from_dict(data)


def subject_summary(binding: RunSubjectBinding) -> str:
    """One line for the prompt: what the run is about, and how to read it.
    Pointers only -- the numbers are read through the tools."""
    if binding.kind is SubjectKind.ACTIVITY_SCREENING:
        toxicity = binding.artifact("toxicity_snapshot")
        return (
            f"the subject of this run: {binding.identity.describe()}; read it with "
            "get_screening_state / get_activity_snapshot / get_prioritisation_decision. "
            + (
                f"Its toxicity stage produced analysis {toxicity.local_id} (replayed); read "
                "it with get_analysis_slice. "
                if toxicity is not None and toxicity.local_id else
                "Its toxicity stage produced no analysis in this session; do not compute "
                "one, and do not read a stage that did not run as a negative result. "
            )
            + "Another screening of the same ligand is another subject."
        )
    return f"the subject of this run: analysis of {binding.identity.describe()}"
