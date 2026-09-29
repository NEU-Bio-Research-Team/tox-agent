"""Observation — the only thing a claim is allowed to point at.

Plan section 5.5. An observation holds two payloads with different jobs. The
canonical payload is lossless and is what validation compares against. The
model projection is bounded, keeps the observation id and the field paths, and
is the only version a model ever sees — which is why the projection may drop
detail but may never drop a required limitation, and why binary blobs stay out
of it entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .fieldpath import FieldPathError, exists, resolve
from .ids import OBSERVATION, RUN, SESSION, new_id, require_id
from .provenance import content_sha256


class Producer(str, Enum):
    PREDICTOR = "predictor"
    ATTRIBUTION = "attribution"
    RESEARCH = "research"
    REPORT_PROJECTION = "report_projection"
    VALIDATOR = "validator"
    #: A server-side scientific calculation over cited inputs (W9-13).
    CALCULATOR = "calculator"


class ObservationKind(str, Enum):
    PREDICTION = "prediction"
    ATTRIBUTION = "attribution"
    EVIDENCE_SEARCH = "evidence_search"
    EVIDENCE_RECORD = "evidence_record"
    ANALYSIS_SLICE = "analysis_slice"
    #: The result of a server calculation, e.g. an exposure margin (W9-13).
    CALCULATION = "calculation"


# A model projection that outgrows this is a prompt-budget bug, not a reason to
# raise the cap; the tool returns a slice instead.
MAX_MODEL_PROJECTION_BYTES = 8_192


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    session_id: str
    run_id: str
    producer: Producer
    kind: ObservationKind
    schema_version: str
    canonical_payload: dict[str, Any]
    model_projection: dict[str, Any]
    provenance: dict[str, Any]
    content_sha256: str
    created_at: datetime
    projection_version: str = "projection-v1"
    required_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.id, OBSERVATION, field="observation.id")
        require_id(self.session_id, SESSION, field="observation.session_id")
        require_id(self.run_id, RUN, field="observation.run_id")
        if "observation_id" not in self.model_projection:
            raise ValueError(
                "model_projection must carry its own observation_id; a projection a model "
                "cannot cite is a projection that produces unciteable claims"
            )

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        run_id: str,
        producer: Producer,
        kind: ObservationKind,
        schema_version: str,
        canonical_payload: dict[str, Any],
        model_projection: dict[str, Any],
        provenance: dict[str, Any],
        now: datetime,
        required_limitations: tuple[str, ...] = (),
    ) -> "Observation":
        observation_id = new_id(OBSERVATION)
        projection = {"observation_id": observation_id, **model_projection}
        return cls(
            id=observation_id,
            session_id=session_id,
            run_id=run_id,
            producer=producer,
            kind=kind,
            schema_version=schema_version,
            canonical_payload=canonical_payload,
            model_projection=projection,
            provenance=provenance,
            content_sha256=content_sha256(canonical_payload),
            created_at=now,
            required_limitations=required_limitations,
        )

    def value_at(self, path: str) -> Any:
        """The canonical value a claim must match. Raises if the path is absent.

        A path absent from the canonical payload may still name a *derived*
        field: one the tools show beside this observation and compute from it
        deterministically (``derived_fields``). A model cites what it was shown,
        so what it was shown must resolve.
        """
        try:
            return resolve(self.canonical_payload, path)
        except FieldPathError:
            derived = self.derived_fields()
            if path.split(".", 1)[0] not in derived:
                raise
            return resolve(derived, path)

    def has(self, path: str) -> bool:
        if exists(self.canonical_payload, path):
            return True
        return exists(self.derived_fields(), path)

    def derived_fields(self) -> dict[str, Any]:
        """Fields every tool view of this observation adds to it.

        Only ``explainer_validation`` on an attribution (W1-03): the measured
        verdict on the method behind it, looked up from the pinned measurement
        table rather than stored, so a new measurement needs no data migration.
        A W8-02 run showed the cost of not resolving it: the model cited
        ``explainer_validation.faithfulness_vs_random_control`` twice, both
        drafts were refused as unresolvable, and the turn fell back.
        """
        if self.kind is not ObservationKind.ATTRIBUTION:
            return {}
        from . import explainer_validation

        return {"explainer_validation": explainer_validation.for_observation(self)}
