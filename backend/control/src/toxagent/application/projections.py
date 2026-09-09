"""Projections of an analysis: one for the UI, one for the model, one per slice.

Three views, one canonical payload, and the canonical payload is never edited to
produce them. The distinction that matters is between the *display* projection
(complete, for a human reading a report) and the *model* projection (bounded,
carrying observation ids and required limitations, for a prompt). A model that
receives the display projection would have every number available without a
reason to cite any of them.

The field allowlist below is what plan section 8.4 calls "declared field paths":
a slice may only expose these, and a claim may only cite what a slice exposed.
"""
from __future__ import annotations

from typing import Any, Final, Iterable, Mapping

from ..domain.analysis import AnalysisSnapshot
from ..domain.errors import InvalidRequest
from ..domain.fieldpath import resolve
from ..predictor.contract import TOX21_TASKS
from ..validation.limitations import required_for_analysis

#: section -> path prefix inside the canonical predictor response.
SECTION_PREFIX: Final[dict[str, str]] = {
    "clintox": "predictions.clintox",
    "herg": "predictions.herg",
    "tox21": "predictions.tox21",
    "applicability": "applicability",
    "provenance": "provenance",
}

#: section -> the fields a slice may return. Anything else is a typed error, so
#: a model cannot fish for fields the product has not agreed to expose.
SECTION_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "clintox": ("probability_clinical_toxicity", "label", "threshold", "threshold_source", "model_id"),
    "herg": ("probability_blocker", "label", "threshold", "threshold_source", "model_id"),
    "tox21": ("task_order_version", "model_id"),
    "applicability": ("status", "method", "reasons"),
    "provenance": ("git_commit", "service_version", "artifact_hashes", "model_ids"),
}

TOX21_ASSAY_FIELDS: Final[tuple[str, ...]] = (
    "probability_activity", "active", "threshold", "threshold_source",
)


def display_projection(snapshot: AnalysisSnapshot) -> dict[str, Any]:
    """The report a person reads. Endpoint semantics are preserved exactly:
    three separate sections, no combined score, and unavailable endpoints listed
    as unavailable rather than omitted."""
    response = snapshot.predictor_response
    predictions = response.get("predictions", {})
    sections: dict[str, Any] = {}

    if "herg" in predictions:
        herg = predictions["herg"]
        sections["herg"] = {
            "measurement": "hERG channel blockade liability",
            "probability_blocker": herg["probability_blocker"],
            "label": herg["label"],
            "threshold": herg["threshold"],
            "threshold_source": herg["threshold_source"],
            "model_id": herg["model_id"],
        }
    if "clintox" in predictions:
        clintox = predictions["clintox"]
        sections["clintox"] = {
            "measurement": "Clinical-trial toxicity signal",
            "probability_clinical_toxicity": clintox["probability_clinical_toxicity"],
            "label": clintox["label"],
            "threshold": clintox["threshold"],
            "threshold_source": clintox["threshold_source"],
            "model_id": clintox["model_id"],
        }
    if "tox21" in predictions:
        tox21 = predictions["tox21"]
        sections["tox21"] = {
            "measurement": "Twelve independent Tox21 assay activities",
            "task_order_version": tox21["task_order_version"],
            "model_id": tox21["model_id"],
            # Deliberately a mapping, not a count. SCI-05: the number of active
            # assays across chemically unrelated targets is not a severity.
            "assays": {
                task: {
                    "probability_activity": assay["probability_activity"],
                    "active": assay["active"],
                    "threshold": assay["threshold"],
                    "threshold_source": assay["threshold_source"],
                }
                for task, assay in tox21["assays"].items()
            },
        }

    return {
        "analysis_id": snapshot.id,
        "input_smiles": snapshot.input_smiles,
        "canonical_smiles": snapshot.canonical_smiles,
        "requested_endpoints": list(snapshot.requested_endpoints),
        "served_endpoints": list(snapshot.served_endpoints),
        "unavailable_endpoints": list(snapshot.unavailable_endpoints),
        "sections": sections,
        "applicability": response.get("applicability", {}),
        "provenance": {
            **snapshot.provenance.to_dict(),
            "content_sha256": snapshot.content_sha256,
        },
        "policy_snapshot": snapshot.policy_snapshot,
        "required_limitations": list(required_limitations(snapshot)),
        "created_at": snapshot.created_at.isoformat(),
    }


def model_projection(snapshot: AnalysisSnapshot) -> dict[str, Any]:
    """What a model is told exists, without being handed every value.

    It lists the sections available and the limitations that will be required,
    and stops there. Values arrive through ``get_analysis_slice``, which is what
    ties a number a model writes to a slice the server actually returned.
    """
    return {
        "analysis_id": snapshot.id,
        "canonical_smiles": snapshot.canonical_smiles,
        "available_sections": available_sections(snapshot),
        "unavailable_endpoints": list(snapshot.unavailable_endpoints),
        "required_limitations": list(required_limitations(snapshot)),
        "tox21_tasks": list(TOX21_TASKS) if "tox21" in snapshot.served_endpoints else [],
    }


def available_sections(snapshot: AnalysisSnapshot) -> list[str]:
    sections = [e for e in snapshot.served_endpoints]
    if snapshot.predictor_response.get("applicability"):
        sections.append("applicability")
    if snapshot.predictor_response.get("provenance"):
        sections.append("provenance")
    return sections


def required_limitations(snapshot: AnalysisSnapshot) -> tuple[str, ...]:
    response = snapshot.predictor_response
    predictions = response.get("predictions", {})
    has_probability = any(
        "probability" in key
        for section in predictions.values()
        if isinstance(section, dict)
        for key in section
    ) or "tox21" in predictions
    return required_for_analysis(
        has_probability=has_probability,
        applicability_status=(response.get("applicability") or {}).get("status"),
        unavailable_endpoints=snapshot.unavailable_endpoints,
    )


def slice_analysis(
    snapshot: AnalysisSnapshot,
    section: str,
    fields: Iterable[str] | None = None,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    """One declared slice, with the field path of every value it returns.

    Returning the paths alongside the values is what makes a grounded claim
    possible to write and possible to check: the model cites the path it was
    given, and the validator resolves that same path against the canonical
    payload.
    """
    if section not in SECTION_PREFIX:
        raise InvalidRequest(
            f"unknown section {section!r}", allowed=sorted(SECTION_PREFIX)
        )
    if section in ("herg", "clintox", "tox21") and section not in snapshot.served_endpoints:
        # SCI-06 again, this time at the read path: an unserved endpoint has no
        # slice, and no other endpoint stands in for it.
        raise InvalidRequest(
            f"this analysis has no {section} section",
            served=list(snapshot.served_endpoints),
            unavailable=list(snapshot.unavailable_endpoints),
        )

    if section == "tox21" and task is not None:
        return _tox21_assay_slice(snapshot, task, fields)

    allowed = SECTION_FIELDS[section]
    requested = tuple(fields) if fields else allowed
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise InvalidRequest(
            f"fields {unknown} are not exposed for section {section!r}", allowed=list(allowed)
        )

    prefix = SECTION_PREFIX[section]
    values: dict[str, Any] = {}
    for name in requested:
        path = f"{prefix}.{name}"
        try:
            values[name] = {"value": resolve(snapshot.predictor_response, path), "field_path": path}
        except KeyError:
            continue  # optional provenance keys the predictor did not report
    return {
        "analysis_id": snapshot.id,
        "section": section,
        "values": values,
        "required_limitations": list(required_limitations(snapshot)),
    }


def _tox21_assay_slice(
    snapshot: AnalysisSnapshot, task: str, fields: Iterable[str] | None
) -> dict[str, Any]:
    if task not in TOX21_TASKS:
        raise InvalidRequest(f"unknown Tox21 assay {task!r}", allowed=list(TOX21_TASKS))
    requested = tuple(fields) if fields else TOX21_ASSAY_FIELDS
    unknown = sorted(set(requested) - set(TOX21_ASSAY_FIELDS))
    if unknown:
        raise InvalidRequest(
            f"fields {unknown} are not exposed for a Tox21 assay",
            allowed=list(TOX21_ASSAY_FIELDS),
        )
    prefix = f"predictions.tox21.assays.{task}"
    values = {
        name: {
            "value": resolve(snapshot.predictor_response, f"{prefix}.{name}"),
            "field_path": f"{prefix}.{name}",
        }
        for name in requested
    }
    return {
        "analysis_id": snapshot.id,
        "section": "tox21",
        "task": task,
        "values": values,
        "required_limitations": list(required_limitations(snapshot)),
        "note": (
            "Tox21 assays are independent measurements; the count of active assays is not "
            "a severity score."
        ),
    }
