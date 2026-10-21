"""Which subject a run is about, fixed when its message is accepted (#44).

A run used to find its subject late: the gateway read the session's active
analysis when it dispatched, and a run about an attached screening had no
subject at all -- its scientific case was keyed ``none``. Both are wrong in the
same way. A researcher who selects another screening while a run is queued
must not have that run answer about the new selection, and one ligand on two
targets is two subjects whose evidence must never share a ledger.

So the subject is resolved once, by the server, when the message is admitted,
and carried frozen in the run's envelope: a :class:`RunSubjectBinding`. It
reuses the identity types the artifacts already carry -- an activity subject is
an :class:`~toxagent.domain.activity.ActivitySubject` -- and never takes an
identity from a model or a request body: the request names a subject by id,
the server reads what that subject is.

Two parts, kept apart on purpose:

* the **identity** (which molecule, which target, which context) decides the
  case a run belongs to. It is what ``subject_key`` digests, so every turn
  about the same subject continues one case;
* the **artifact refs** (which screening record, which toxicity snapshot, with
  their content hashes) say what this run was given. They change from run to
  run as artifacts are added and are hashed into ``binding_sha256``, never into
  the case key -- otherwise each turn would open a new case.

The case key fits the 80-character column: ``analysis:<id>`` for an analysis
(unchanged from before) and ``activity:<40 hex>`` for an activity subject, a
digest of the ligand, the target, its variant and the assay context. The
digest covers only what every screening of that subject states; an optional
enrichment (a UniProt id one screening resolved and another did not) is
checked by :meth:`RunSubjectBinding.mismatches` against the case instead,
under the identity rule the activity domain already has, so it cannot split a
case by itself and cannot silently merge two different proteins either.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .activity import ActivitySubject, TargetIdentity
from .ids import ANALYSIS, SCREENING, is_id
from .provenance import content_sha256

SCHEMA_VERSION = "run-subject-binding-v1"

#: The ``scientific_cases.subject_key`` column width.
MAX_SUBJECT_KEY = 80


class SubjectKind(str, Enum):
    ANALYSIS = "analysis"
    ACTIVITY_SCREENING = "activity_screening"


class InvalidSubject(ValueError):
    """A binding that cannot describe one subject."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """One artifact a run was given about its subject.

    ``source_id`` is the artifact's own id where it was produced; for a
    replayed artifact that is the id from the frozen run, which does not exist
    in this session. ``local_id`` is then the session-local copy the server
    made of it. Fairness between study arms compares ``content_sha256``,
    never a minted id.
    """

    kind: str
    source_id: str
    content_sha256: str
    local_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind, "source_id": self.source_id,
            "content_sha256": self.content_sha256,
        }
        if self.local_id is not None:
            data["local_id"] = self.local_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            kind=str(data["kind"]), source_id=str(data["source_id"]),
            content_sha256=str(data.get("content_sha256") or ""),
            local_id=data.get("local_id"),
        )


@dataclass(frozen=True, slots=True)
class SubjectIdentity:
    """What the subject is. An analysis subject is a molecule; an activity
    subject is a molecule on a target in an assay context."""

    canonical_smiles: str
    target: TargetIdentity | None = None
    assay_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles,
            "target": self.target.to_dict() if self.target is not None else None,
            "assay_context": self.assay_context,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubjectIdentity":
        target = data.get("target")
        return cls(
            canonical_smiles=str(data.get("canonical_smiles") or ""),
            target=TargetIdentity.from_dict(target) if target else None,
            assay_context=data.get("assay_context"),
        )

    def activity_subject(self) -> ActivitySubject | None:
        if self.target is None or not self.assay_context:
            return None
        return ActivitySubject(
            canonical_smiles=self.canonical_smiles, target=self.target,
            assay_context=self.assay_context,
        )

    def describe(self) -> str:
        if self.target is None:
            return self.canonical_smiles or "(molecule pending its snapshot)"
        variant = f" ({self.target.variant})" if self.target.variant else ""
        return (
            f"{self.canonical_smiles} on {self.target.target_id}{variant} "
            f"in {self.assay_context}"
        )


def activity_subject_key(identity: SubjectIdentity | Mapping[str, Any]) -> str:
    """The case key of an activity subject, from its identity alone -- what an
    activity artifact's own ``subject`` block states."""
    if not isinstance(identity, SubjectIdentity):
        identity = SubjectIdentity.from_dict(identity)
    if identity.activity_subject() is None:
        raise InvalidSubject("an activity subject states its ligand, target and assay context")
    return _activity_key(identity)


def _activity_key(identity: SubjectIdentity) -> str:
    target = identity.target
    assert target is not None
    material = json.dumps(
        [identity.canonical_smiles, target.target_id, target.variant or "",
         identity.assay_context or ""],
        ensure_ascii=False, separators=(",", ":"),
    )
    return "activity:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:40]


@dataclass(frozen=True, slots=True)
class RunSubjectBinding:
    kind: SubjectKind
    #: ``{"analysis_id": ...}`` or ``{"screening_id": ...}``. For a run that
    #: snapshots a new molecule first, ``{"analysis_id": None}`` with
    #: ``pending_snapshot`` set: the subject is the analysis this run creates.
    subject_ref: Mapping[str, Any]
    identity: SubjectIdentity
    artifact_refs: tuple[ArtifactRef, ...] = ()
    pending_snapshot: bool = False
    #: How the server chose it: ``explicit`` (the request named it),
    #: ``analysis_id`` (the legacy field), ``only_candidate`` (the session held
    #: exactly one subject) or ``new_molecule``.
    resolved_by: str = "explicit"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind is SubjectKind.ANALYSIS:
            analysis_id = self.subject_ref.get("analysis_id")
            if analysis_id is None and not self.pending_snapshot:
                raise InvalidSubject("an analysis subject names its analysis_id")
            if analysis_id is not None and not is_id(analysis_id, ANALYSIS):
                raise InvalidSubject(f"{analysis_id!r} is not an analysis id")
        else:
            screening_id = self.subject_ref.get("screening_id")
            if not is_id(screening_id, SCREENING):
                raise InvalidSubject(f"{screening_id!r} is not a screening id")
            if self.identity.activity_subject() is None:
                raise InvalidSubject(
                    "an activity subject states its ligand, target and assay context"
                )
            if self.pending_snapshot:
                raise InvalidSubject("only an analysis subject can wait for its snapshot")

    # ------------------------------------------------------------ identity

    @property
    def analysis_id(self) -> str | None:
        return self.subject_ref.get("analysis_id") if self.kind is SubjectKind.ANALYSIS else None

    @property
    def screening_id(self) -> str | None:
        return (
            self.subject_ref.get("screening_id")
            if self.kind is SubjectKind.ACTIVITY_SCREENING else None
        )

    @property
    def subject_key(self) -> str:
        """The scientific case this subject's turns accumulate in."""
        if self.kind is SubjectKind.ANALYSIS:
            if self.analysis_id is None:
                raise InvalidSubject("a pending analysis subject has no case key yet")
            key = f"analysis:{self.analysis_id}"
        else:
            key = _activity_key(self.identity)
        if len(key) > MAX_SUBJECT_KEY:  # pragma: no cover - both forms are fixed width
            raise InvalidSubject(f"subject key {key!r} exceeds {MAX_SUBJECT_KEY} characters")
        return key

    @property
    def subject_ref_string(self) -> str:
        """``analysis:<id>`` / ``screening:<id>``: the form subject_refs use."""
        if self.kind is SubjectKind.ANALYSIS:
            return f"analysis:{self.analysis_id}"
        return f"screening:{self.screening_id}"

    def mismatches(self, other_identity: Mapping[str, Any] | SubjectIdentity) -> tuple[str, ...]:
        """Identity axes on which ``other_identity`` (a case's stored subject)
        describes something else. Empty when they are the same subject."""
        other = (
            other_identity if isinstance(other_identity, SubjectIdentity)
            else SubjectIdentity.from_dict(other_identity)
        )
        mine, theirs = self.identity.activity_subject(), other.activity_subject()
        if mine is not None and theirs is not None:
            return mine.mismatches(theirs)
        if (mine is None) != (theirs is None):
            return ("kind",)
        if (
            self.identity.canonical_smiles and other.canonical_smiles
            and self.identity.canonical_smiles != other.canonical_smiles
        ):
            return ("canonical_smiles",)
        return ()

    # ------------------------------------------------------------ artifacts

    def artifact(self, kind: str) -> ArtifactRef | None:
        return next((ref for ref in self.artifact_refs if ref.kind == kind), None)

    def resolved_with_analysis(self, analysis_id: str, *, canonical_smiles: str,
                               content_sha256_: str) -> "RunSubjectBinding":
        """The pending subject of a new-molecule run, once its snapshot exists."""
        if not self.pending_snapshot:
            raise InvalidSubject("only a pending subject is resolved by its snapshot")
        return RunSubjectBinding(
            kind=SubjectKind.ANALYSIS, subject_ref={"analysis_id": analysis_id},
            identity=SubjectIdentity(canonical_smiles=canonical_smiles),
            artifact_refs=(
                *self.artifact_refs,
                ArtifactRef(kind="analysis", source_id=analysis_id,
                            content_sha256=content_sha256_),
            ),
            pending_snapshot=False, resolved_by=self.resolved_by,
        )

    # ------------------------------------------------------------ wire form

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "subject_ref": dict(self.subject_ref),
            "identity": self.identity.to_dict(),
            "artifact_refs": [ref.to_dict() for ref in self.artifact_refs],
            "pending_snapshot": self.pending_snapshot,
            "resolved_by": self.resolved_by,
        }

    @property
    def binding_sha256(self) -> str:
        return content_sha256(self._body())

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        body["binding_sha256"] = self.binding_sha256
        if not self.pending_snapshot:
            body["subject_key"] = self.subject_key
        return body

    def case_subject(self) -> dict[str, Any]:
        """What a case stores about its subject: the identity, not the run's
        artifacts (those are recorded on each run the case attaches)."""
        return {
            "schema_version": self.schema_version, "kind": self.kind.value,
            "subject_ref": dict(self.subject_ref), "identity": self.identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunSubjectBinding":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise InvalidSubject(f"unknown subject binding version {version!r}")
        try:
            kind = SubjectKind(data["kind"])
        except (KeyError, ValueError) as exc:
            raise InvalidSubject(f"unknown subject kind: {exc}") from exc
        binding = cls(
            kind=kind, subject_ref=dict(data.get("subject_ref") or {}),
            identity=SubjectIdentity.from_dict(data.get("identity") or {}),
            artifact_refs=tuple(ArtifactRef.from_dict(r) for r in data.get("artifact_refs") or ()),
            pending_snapshot=bool(data.get("pending_snapshot")),
            resolved_by=str(data.get("resolved_by") or "explicit"),
        )
        recorded = data.get("binding_sha256")
        if recorded and recorded != binding.binding_sha256:
            raise InvalidSubject("the subject binding does not match its binding_sha256")
        return binding


def analysis_binding(analysis_id: str, *, canonical_smiles: str, content_sha256_: str,
                     resolved_by: str) -> RunSubjectBinding:
    return RunSubjectBinding(
        kind=SubjectKind.ANALYSIS, subject_ref={"analysis_id": analysis_id},
        identity=SubjectIdentity(canonical_smiles=canonical_smiles),
        artifact_refs=(ArtifactRef(kind="analysis", source_id=analysis_id,
                                   content_sha256=content_sha256_),),
        resolved_by=resolved_by,
    )


def pending_analysis_binding(input_smiles: str) -> RunSubjectBinding:
    """A new molecule submitted with a question: the subject is the analysis
    this run will create, and its key is known once the snapshot exists."""
    return RunSubjectBinding(
        kind=SubjectKind.ANALYSIS, subject_ref={"analysis_id": None, "input_smiles": input_smiles},
        identity=SubjectIdentity(canonical_smiles=""), pending_snapshot=True,
        resolved_by="new_molecule",
    )
