"""Activity prediction value objects and immutable snapshot."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .ids import ACTIVITY_SNAPSHOT, RUN, SESSION, new_id, require_id
from .provenance import content_sha256


def finite_number(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return value


class ActivityApplicability(str, Enum):
    IN_DOMAIN = "in_domain"
    LIMITED = "limited"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


class TargetRole(str, Enum):
    """Why this target is being screened.

    A pIC50 threshold only points one way once this is known.  "High pIC50
    deserves more screening" is a statement about an intended target; for an
    anti-target the same number argues the opposite, and for an unspecified
    role it argues nothing.  Prioritisation policies therefore declare the
    role they are valid for instead of assuming one.
    """

    INTENDED_TARGET = "intended_target"
    OFF_TARGET = "off_target"
    UNSPECIFIED = "unspecified"


class ActivityIdentityMismatch(ValueError):
    """An artifact describes a different ligand, target or assay context."""

    def __init__(self, mismatches: tuple[str, ...], detail: str) -> None:
        super().__init__(f"activity identity mismatch on {', '.join(mismatches)}: {detail}")
        self.mismatches = mismatches


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    """A resolved target, not an untraceable free-text protein name.

    ``variant`` names a protein form other than the reference one (a mutant,
    an isoform). It is optional and versioned with this type: left ``None``
    it is not serialised at all, so every artifact hashed before the field
    existed keeps its hash. Unlike ``uniprot_id``, a variant is compared even
    when only one side states it -- a mutant and an unstated form are not
    shown to be the same protein.
    """

    target_id: str
    uniprot_id: str | None = None
    sequence_sha256: str | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id is required")
        if self.sequence_sha256 is not None:
            digest = self.sequence_sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sequence_sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "sequence_sha256", digest)
        if self.variant is not None:
            variant = self.variant.strip()
            if not variant:
                raise ValueError("variant, when given, must not be blank")
            object.__setattr__(self, "variant", variant)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "target_id": self.target_id,
            "uniprot_id": self.uniprot_id,
            "sequence_sha256": self.sequence_sha256,
        }
        if self.variant is not None:
            data["variant"] = self.variant
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetIdentity":
        return cls(
            target_id=str(data["target_id"]),
            uniprot_id=data.get("uniprot_id"),
            sequence_sha256=data.get("sequence_sha256"),
            variant=data.get("variant"),
        )


@dataclass(frozen=True, slots=True)
class ActivitySubject:
    """What one activity artifact is about: one ligand, one target, one context.

    Identity is carried by the canonical SMILES, not by the string a caller
    typed.  Normalisation is owned by the application boundary, so the domain
    compares already-canonical forms and never guesses that two spellings are
    the same molecule.

    ``uniprot_id`` and ``sequence_sha256`` are compared only when both sides
    declare them: a caller may legitimately open a screening knowing only the
    target id and have the predictor resolve the sequence.  Once both sides
    declare a value, a difference is a mismatch.
    """

    canonical_smiles: str
    target: TargetIdentity
    assay_context: str

    def __post_init__(self) -> None:
        if not self.canonical_smiles.strip():
            raise ValueError("subject.canonical_smiles is required")
        if not self.assay_context.strip():
            raise ValueError("subject.assay_context is required")

    def mismatches(self, other: "ActivitySubject") -> tuple[str, ...]:
        found: list[str] = []
        if self.canonical_smiles != other.canonical_smiles:
            found.append("canonical_smiles")
        if self.target.target_id != other.target.target_id:
            found.append("target_id")
        if (
            self.target.uniprot_id is not None
            and other.target.uniprot_id is not None
            and self.target.uniprot_id != other.target.uniprot_id
        ):
            found.append("uniprot_id")
        if (
            self.target.sequence_sha256 is not None
            and other.target.sequence_sha256 is not None
            and self.target.sequence_sha256 != other.target.sequence_sha256
        ):
            found.append("sequence_sha256")
        if self.target.variant != other.target.variant:
            found.append("variant")
        if self.assay_context != other.assay_context:
            found.append("assay_context")
        return tuple(found)

    def require_same(self, other: "ActivitySubject", *, artifact: str) -> None:
        mismatches = self.mismatches(other)
        if mismatches:
            raise ActivityIdentityMismatch(
                mismatches,
                f"{artifact} describes {other.describe()}, screening is {self.describe()}",
            )

    def describe(self) -> str:
        return (
            f"{self.canonical_smiles} on {self.target.target_id} "
            f"in {self.assay_context}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles,
            "target": self.target.to_dict(),
            "assay_context": self.assay_context,
        }


@dataclass(frozen=True, slots=True)
class ActivityEstimate:
    """A regression estimate. It is intentionally not an active/inactive label."""

    predicted_pic50: float
    uncertainty: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None

    def __post_init__(self) -> None:
        prediction = finite_number(self.predicted_pic50, "predicted_pic50")
        object.__setattr__(self, "predicted_pic50", prediction)
        if self.uncertainty is not None:
            uncertainty = finite_number(self.uncertainty, "uncertainty")
            if uncertainty < 0:
                raise ValueError("uncertainty must be non-negative")
            object.__setattr__(self, "uncertainty", uncertainty)
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("prediction interval requires both interval_low and interval_high")
        if self.interval_low is not None and self.interval_high is not None:
            low = finite_number(self.interval_low, "interval_low")
            high = finite_number(self.interval_high, "interval_high")
            if low > high:
                raise ValueError("interval_low must not exceed interval_high")
            if not low <= prediction <= high:
                raise ValueError("prediction interval must contain predicted_pic50")
            object.__setattr__(self, "interval_low", low)
            object.__setattr__(self, "interval_high", high)

    @property
    def has_interval(self) -> bool:
        return self.interval_low is not None and self.interval_high is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_pic50": self.predicted_pic50,
            "uncertainty": self.uncertainty,
            "prediction_interval": (
                [self.interval_low, self.interval_high] if self.has_interval else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ActivityModelProvenance:
    model_id: str
    dataset_id: str
    artifact_hashes: tuple[str, ...] = ()
    split_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id is required")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "artifact_hashes": list(self.artifact_hashes),
            "raw": dict(self.raw),
        }


@dataclass(frozen=True, slots=True)
class ActivitySnapshot:
    id: str
    session_id: str
    run_id: str
    input_smiles: str
    canonical_smiles: str
    target: TargetIdentity
    assay_context: str
    estimate: ActivityEstimate
    ligand_applicability: ActivityApplicability
    target_applicability: ActivityApplicability
    joint_applicability: ActivityApplicability
    provenance: ActivityModelProvenance
    content_sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_id(self.id, ACTIVITY_SNAPSHOT, field="activity_snapshot.id")
        require_id(self.session_id, SESSION, field="activity_snapshot.session_id")
        require_id(self.run_id, RUN, field="activity_snapshot.run_id")
        if not self.input_smiles.strip() or not self.canonical_smiles.strip():
            raise ValueError("input_smiles and canonical_smiles are required")
        if not self.assay_context.strip():
            raise ValueError("assay_context is required")

    @property
    def subject(self) -> ActivitySubject:
        """The identity this observation claims, for comparison against a screening."""
        return ActivitySubject(
            canonical_smiles=self.canonical_smiles,
            target=self.target,
            assay_context=self.assay_context,
        )

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        run_id: str,
        input_smiles: str,
        canonical_smiles: str,
        target: TargetIdentity,
        assay_context: str,
        estimate: ActivityEstimate,
        ligand_applicability: ActivityApplicability,
        target_applicability: ActivityApplicability,
        joint_applicability: ActivityApplicability,
        provenance: ActivityModelProvenance,
        now: datetime,
    ) -> "ActivitySnapshot":
        canonical = {
            "input_smiles": input_smiles,
            "canonical_smiles": canonical_smiles,
            "target": target.to_dict(),
            "assay_context": assay_context,
            "estimate": estimate.to_dict(),
            "applicability": {
                "ligand": ligand_applicability.value,
                "target": target_applicability.value,
                "joint": joint_applicability.value,
            },
            "provenance": provenance.to_dict(),
        }
        return cls(
            id=new_id(ACTIVITY_SNAPSHOT),
            session_id=session_id,
            run_id=run_id,
            input_smiles=input_smiles,
            canonical_smiles=canonical_smiles,
            target=target,
            assay_context=assay_context,
            estimate=estimate,
            ligand_applicability=ligand_applicability,
            target_applicability=target_applicability,
            joint_applicability=joint_applicability,
            provenance=provenance,
            content_sha256=content_sha256(canonical),
            created_at=now,
        )
