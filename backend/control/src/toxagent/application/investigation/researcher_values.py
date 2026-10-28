"""Project a researcher-supplied value into an observation a claim can cite (#61).

The grounded-answer resolver only lets a claim point at an
:class:`~toxagent.domain.observation.Observation`, and before this the only
home for a value the researcher stated was a case ``ContextItem`` -- which is
not one. So the margin claim of §5.3, whose inputs are an assay measurement
*and* a researcher exposure, had no way to name its second half.

Three things this projection keeps straight:

* **It is not a measurement.** The producer is ``RESEARCHER``, never
  ``PREDICTOR`` or ``RESEARCH``, and the projection states the verification
  status beside the number. A ``user_asserted`` value is usable, citable and
  not evidence; the required limitation says so in the answer vocabulary.
* **It is a version, not a value.** The payload carries the stable
  ``case_value:<key>@v<n>`` reference, so a claim records *which* exposure it
  was computed from and a later correction does not rewrite what the old
  claim meant (§5.3).
* **The number is canonicalised, with its conversion.** A value stated in µM
  and a record in nM have to meet somewhere; #59 says where, and that the
  conversion travels with the result. A unit this domain cannot place is
  reported as such rather than guessed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ...domain.answer import LimitationCode
from ...domain.measurement_value import NumericConstraint
from ...domain.observation import Observation, ObservationKind, Producer
from ...domain.scientific_case.values import ValueVersion, VerificationStatus

RESEARCHER_VALUE_SCHEMA_VERSION = "researcher-value-v1"

#: What an answer must say when it quotes a value the researcher supplied and
#: nobody has verified. The code is the answer vocabulary's own, so an
#: observation cannot require a limitation an answer has no way to declare.
USER_ASSERTED_LIMITATION = LimitationCode.USER_SUPPLIED_VALUE_NOT_VERIFIED.value


def project_value(
    version: ValueVersion, *, case_id: str, session_id: str, run_id: str, now: datetime,
) -> Observation:
    """One version of one researcher value, as an observation.

    Pure in everything but the minted observation id: the same case version
    projects to the same canonical payload and the same content hash, which is
    what lets a replayed case produce comparable claims across runs.
    """
    constraint = NumericConstraint.parse(version.value, version.unit)
    payload: dict[str, Any] = {
        "ref": version.ref,
        "key": version.key,
        "version": version.version,
        "value": version.value,
        "unit": version.unit,
        "verification": version.verification.value,
        "scope": dict(version.scope) if version.scope else None,
        "stated_by": version.actor,
        "stated_source": version.source,
        "superseded_by": version.superseded_by,
        "constraint": constraint.to_dict(),
    }
    projection = {
        "ref": version.ref,
        "key": version.key,
        "value": version.value,
        "unit": version.unit,
        "verification": version.verification.value,
        "current": version.current,
        "reading": _reading(version),
    }
    limitations = (
        () if version.verification is VerificationStatus.SME_VERIFIED
        else (USER_ASSERTED_LIMITATION,)
    )
    return Observation.create(
        session_id=session_id,
        run_id=run_id,
        producer=Producer.RESEARCHER,
        kind=ObservationKind.RESEARCHER_VALUE,
        schema_version=RESEARCHER_VALUE_SCHEMA_VERSION,
        canonical_payload=payload,
        model_projection=projection,
        provenance={
            "case_id": case_id,
            "context_item_id": version.item_id,
            "value_ref": version.ref,
            "stated_in_run": version.run_id,
        },
        now=now,
        required_limitations=limitations,
    )


def _reading(version: ValueVersion) -> str:
    """One line a model can quote about what the value is, and is not."""
    if version.verification is VerificationStatus.SME_VERIFIED:
        standing = "verified by a subject-matter expert"
    elif version.verification is VerificationStatus.SOURCE_BACKED:
        standing = (
            "attributed by the researcher to a source; the attribution itself has not "
            "been checked against that source"
        )
    else:
        standing = "stated by the researcher and not verified against any source"
    superseded = (
        "" if version.current else
        f" A later version of this value exists ({version.superseded_by}); this one is "
        "kept because claims computed from it were computed from this number."
    )
    return f"A value {standing}. It is an input, not a measurement this product made.{superseded}"
