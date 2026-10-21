"""Activity screenings replayed from frozen records (#44).

No activity model is served yet (#48 has not admitted one), and the agent's
investigation policy still has to be built and measured. Both are possible on
*frozen* screenings: :func:`~toxagent.application.activity.projections.freeze_screening`
records exactly the canonical payloads a live run would have projected, so a
replayed screening reads, cites and validates the way a served one will.

The store is a server-held file pinned by sha256, like the research corpus: a
recorded run is tied to the exact artifacts it read, and a client can only
name a screening the deployment already holds, never post one. Posting would
let anyone hand a session an activity estimate no model produced.

File format (``activity-replay-v1``)::

    {"schema_version": "activity-replay-v1",
     "screenings": [<frozen-screening-v1 record>, ...]}

Each record is re-hashed on load; one whose content does not match its own
``content_sha256`` is refused, as is a file whose bytes do not match the pin.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...domain.analysis import AnalysisSnapshot
from ...domain.errors import CapabilityUnavailable, ScreeningNotFound
from ...domain.observation import Observation
from ...domain.provenance import content_sha256
from ...domain.screening_attachment import ScreeningAttachment
from .projections import (
    TOXICITY_SNAPSHOT_SCHEMA_VERSION,
    activity_observation_from_payload,
    prioritisation_observation_from_payload,
    screening_state_observation_from_payload,
)

SCHEMA_VERSION = "activity-replay-v1"
RECORD_SCHEMA = "frozen-screening-v1"

#: What each read tool returns, by the record field holding its payload.
ARTIFACTS: Mapping[str, str] = {
    "screening_state": "screening_state",
    "activity_estimate": "activity_estimate",
    "prioritisation_decision": "prioritisation_decision",
}


class ReplayError(ValueError):
    """The replay file cannot be trusted as the deployment's frozen screenings."""


def record_sha256(record: Mapping[str, Any]) -> str:
    """The hash freeze_screening wrote, recomputed from the record's content."""
    return content_sha256(
        {k: v for k, v in record.items() if k not in ("frozen_at", "content_sha256")}
    )


def screening_id_of(record: Mapping[str, Any]) -> str:
    return str((record.get("screening_state") or {}).get("screening_id") or "")


def write_replay_file(records: Iterable[Mapping[str, Any]], path: Path, *,
                      analyses: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """Write frozen records as a replay file and return its sha256 -- the pin a
    deployment is configured with (TOXAGENT_ACTIVITY_REPLAY_SHA256).

    ``analyses`` (#50/#51): frozen toxicity analyses that belong to no
    screening -- the toxicity-only cases -- keyed by their origin analysis id,
    each a ``frozen-toxicity-snapshot-v1`` payload. Left out when empty, so a
    file without them hashes as it always did."""
    document = {"schema_version": SCHEMA_VERSION, "screenings": [dict(r) for r in records]}
    if analyses:
        document["analyses"] = {k: dict(v) for k, v in sorted(analyses.items())}
    data = json.dumps(document, sort_keys=True, indent=1, ensure_ascii=False).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def frozen_analysis_problem(analysis_ref: str, frozen: Mapping[str, Any]) -> str | None:
    """Why a frozen toxicity analysis cannot be trusted, or ``None``."""
    if frozen.get("schema_version") != TOXICITY_SNAPSHOT_SCHEMA_VERSION:
        return f"analysis {analysis_ref} is not a {TOXICITY_SNAPSHOT_SCHEMA_VERSION} payload"
    if frozen.get("origin_analysis_id") != analysis_ref:
        return f"analysis {analysis_ref} names origin {frozen.get('origin_analysis_id')!r}"
    if content_sha256(frozen.get("predictor_response") or {}) != frozen.get("content_sha256"):
        return f"analysis {analysis_ref} does not match its content_sha256"
    if (frozen.get("predictor_response") or {}).get("canonical_smiles") != frozen.get(
            "canonical_smiles"):
        return f"analysis {analysis_ref} states two different molecules"
    return None


@dataclass(frozen=True)
class FrozenReplayStore:
    path_name: str
    sha256: str
    _records: Mapping[str, Mapping[str, Any]]
    _analyses: Mapping[str, Mapping[str, Any]] = None  # type: ignore[assignment]

    @classmethod
    def load(cls, path: Path, *, sha256: str) -> "FrozenReplayStore":
        if not sha256:
            raise ReplayError("a replay file must be pinned by its sha256")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise ReplayError(
                f"{path.name} is not the pinned replay file (sha256 {actual}, pinned {sha256})"
            )
        document = json.loads(data)
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ReplayError(f"{path.name} is not an {SCHEMA_VERSION} file")
        records: dict[str, Mapping[str, Any]] = {}
        for record in document.get("screenings") or ():
            if record.get("schema_version") != RECORD_SCHEMA:
                raise ReplayError(f"a screening in {path.name} is not a {RECORD_SCHEMA} record")
            screening_id = screening_id_of(record)
            if not screening_id:
                raise ReplayError(f"a screening in {path.name} names no screening_id")
            if record_sha256(record) != record.get("content_sha256"):
                raise ReplayError(
                    f"screening {screening_id} in {path.name} does not match its content_sha256"
                )
            if screening_id in records:
                raise ReplayError(f"screening {screening_id} appears twice in {path.name}")
            problem = toxicity_snapshot_problem(record)
            if problem:
                raise ReplayError(f"screening {screening_id} in {path.name}: {problem}")
            records[screening_id] = record
        analyses: dict[str, Mapping[str, Any]] = {}
        for analysis_ref, frozen in (document.get("analyses") or {}).items():
            problem = frozen_analysis_problem(analysis_ref, frozen)
            if problem:
                raise ReplayError(f"{path.name}: {problem}")
            analyses[analysis_ref] = frozen
        return cls(path_name=path.name, sha256=sha256, _records=records, _analyses=analyses)

    def get_analysis(self, analysis_ref: str) -> Mapping[str, Any] | None:
        return (self._analyses or {}).get(analysis_ref)

    def get(self, screening_id: str) -> Mapping[str, Any] | None:
        return self._records.get(screening_id)

    def __len__(self) -> int:
        return len(self._records)

    def describe(self) -> dict[str, Any]:
        described = {"path_name": self.path_name, "sha256": self.sha256, "screenings": len(self)}
        if self._analyses:
            # Only when the file holds any, so a file without them describes
            # itself as it always did.
            described["analyses"] = len(self._analyses)
        return described


def toxicity_snapshot_problem(record: Mapping[str, Any]) -> str | None:
    """Why a record's linked toxicity snapshot cannot be trusted, or ``None``.

    The snapshot must be the one the screening's toxicity stage names, of the
    screening's own molecule, with a predictor response that still hashes to
    the content hash it was frozen with.
    """
    frozen = record.get("toxicity_snapshot")
    if frozen is None:
        return None
    if frozen.get("schema_version") != TOXICITY_SNAPSHOT_SCHEMA_VERSION:
        return f"its toxicity snapshot is not a {TOXICITY_SNAPSHOT_SCHEMA_VERSION} payload"
    state = record.get("screening_state") or {}
    named = (state.get("toxicity") or {}).get("analysis_id")
    if frozen.get("origin_analysis_id") != named:
        return (
            f"its toxicity snapshot is analysis {frozen.get('origin_analysis_id')!r}, "
            f"but the screening's toxicity stage names {named!r}"
        )
    subject = state.get("subject") or {}
    if frozen.get("canonical_smiles") != subject.get("canonical_smiles"):
        return "its toxicity snapshot is of a different molecule than the screening"
    if content_sha256(frozen.get("predictor_response") or {}) != frozen.get("content_sha256"):
        return "its toxicity snapshot does not match its content_sha256"
    return None


def linked_toxicity(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The origin id and hash of the toxicity snapshot a record carries."""
    frozen = record.get("toxicity_snapshot")
    if frozen is None:
        return None
    return {
        "origin_analysis_id": frozen["origin_analysis_id"],
        "content_sha256": frozen["content_sha256"],
    }


async def materialise_toxicity(uow, record: Mapping[str, Any], *, session_id: str, run_id: str,
                               now: datetime) -> AnalysisSnapshot | None:
    """The session-local copy of a screening's frozen toxicity snapshot.

    Made once per session (the analysis idempotency key is unique per session)
    and never by calling the predictor: the predictor response is the frozen
    one, byte for byte, and the copy must hash to the same content as the
    origin or it is refused. ``None`` when the screening's toxicity stage did
    not produce one -- not_run is a state, never a gap to fill.
    """
    from ...domain.errors import ArtifactHashMismatch

    frozen = record.get("toxicity_snapshot")
    if frozen is None:
        return None
    problem = toxicity_snapshot_problem(record)
    if problem:
        raise ArtifactHashMismatch(problem, screening_id=screening_id_of(record))
    return await materialise_frozen_analysis(
        uow, frozen, session_id=session_id, run_id=run_id, now=now,
        replayed_from={"record_sha256": record.get("content_sha256")},
    )


async def materialise_frozen_analysis(uow, frozen: Mapping[str, Any], *, session_id: str,
                                      run_id: str, now: datetime,
                                      replayed_from: Mapping[str, Any] | None = None,
                                      ) -> AnalysisSnapshot:
    """A frozen toxicity analysis as a session-local analysis, with the
    prediction observation a served one is stored with -- once per session,
    hash-checked, and never by calling the predictor."""
    from ...domain.analysis import PredictorProvenance, snapshot_from_prediction
    from ...domain.errors import ArtifactHashMismatch

    provenance = frozen.get("provenance") or {}
    candidate = snapshot_from_prediction(
        session_id=session_id, run_id=run_id, input_smiles=frozen["input_smiles"],
        requested_endpoints=tuple(frozen.get("requested_endpoints") or ()),
        predictor_response=dict(frozen["predictor_response"]),
        provenance=PredictorProvenance(
            base_url_id=str(provenance.get("base_url_id") or "replay"),
            service_version=provenance.get("service_version"),
            git_commit=provenance.get("git_commit"),
            artifact_hashes=tuple(provenance.get("artifact_hashes") or ()),
            raw=(frozen["predictor_response"].get("provenance") or {}),
        ),
        policy_snapshot=dict(frozen.get("policy_snapshot") or {}),
        now=now,
    )
    if candidate.content_sha256 != frozen["content_sha256"]:
        raise ArtifactHashMismatch(
            "the replayed toxicity snapshot does not hash to its frozen content",
            analysis_ref=frozen.get("origin_analysis_id"),
        )
    existing = await uow.analyses.find_by_idempotency_key(session_id, candidate.idempotency_key)
    if existing is not None:
        if existing.content_sha256 != candidate.content_sha256:
            raise ArtifactHashMismatch(
                "this session already holds a different analysis under the same key",
                analysis_id=existing.id,
            )
        return existing
    from dataclasses import replace as _replace

    from ..prediction.create_analysis import CreateAnalysis

    await uow.analyses.add(candidate)
    # The prediction observation a served analysis is stored with, so the
    # copy reads and cites exactly as one: get_analysis_slice cites it.
    observation = CreateAnalysis._observation_for(candidate, run_id)
    observation = _replace(observation, provenance={
        **observation.provenance,
        "replayed_from": {
            **dict(replayed_from or {}),
            "origin_analysis_id": frozen["origin_analysis_id"],
        },
    })
    await uow.observations.add(observation, analysis_id=candidate.id)
    return candidate


def build_replay_store(settings) -> FrozenReplayStore | None:
    """The deployment's replay store, or ``None`` when none is configured."""
    if not settings.replay_path:
        return None
    return FrozenReplayStore.load(Path(settings.replay_path), sha256=settings.replay_sha256)


def summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """What an inventory says about one screening: which subject, how far it
    got, and whether toxicity ran -- not the numbers, which a read tool
    returns as a citable observation."""
    state = record.get("screening_state") or {}
    subject = state.get("subject") or {}
    return {
        "screening_id": screening_id_of(record),
        "status": state.get("status"),
        "canonical_smiles": subject.get("canonical_smiles"),
        "target": subject.get("target"),
        "assay_context": subject.get("assay_context"),
        "intended_target_role": state.get("intended_target_role"),
        "toxicity_run": (state.get("toxicity") or {}).get("run"),
        "has_activity_estimate": record.get("activity_estimate") is not None,
        "has_prioritisation_decision": record.get("prioritisation_decision") is not None,
        "model_admitted": ((record.get("activity_estimate") or {}).get("model") or {}).get(
            "admitted"
        ),
        "has_toxicity_snapshot": record.get("toxicity_snapshot") is not None,
        "ligand_names": list(record.get("subject_names") or ()),
        "record_sha256": record.get("content_sha256"),
    }


def observation_for(
    record: Mapping[str, Any],
    artifact: str,
    *,
    session_id: str,
    run_id: str,
    now: datetime,
) -> Observation | None:
    """The citable observation of one artifact of a frozen screening, or
    ``None`` when the screening never produced it (a stage that did not run).
    Provenance names the record it was replayed from."""
    payload = record.get(ARTIFACTS[artifact])
    if payload is None:
        return None
    origin = {
        "replayed_from": {
            "record_sha256": record.get("content_sha256"),
            "frozen_run_id": record.get("run_id"),
        }
    }
    build = {
        "screening_state": screening_state_observation_from_payload,
        "activity_estimate": activity_observation_from_payload,
        "prioritisation_decision": prioritisation_observation_from_payload,
    }[artifact]
    return build(
        dict(payload), session_id=session_id, run_id=run_id, now=now, provenance_extra=origin,
    )


async def attach(
    database, store: FrozenReplayStore | None, *, session_id: str, screening_id: str,
    now: datetime,
) -> ScreeningAttachment:
    """Attach a frozen screening the deployment holds to a session. Naming a
    screening the store does not hold is a 404; a deployment with no store
    cannot attach anything."""
    if store is None:
        raise CapabilityUnavailable(
            "this deployment replays no activity screenings (no replay file is configured)"
        )
    record = store.get(screening_id)
    if record is None:
        raise ScreeningNotFound("no such screening in the replay file", screening_id=screening_id)
    candidate = ScreeningAttachment(
        session_id=session_id, screening_id=screening_id, record=dict(record),
        record_sha256=str(record["content_sha256"]), attached_at=now,
    )
    async with database.unit_of_work() as uow:
        attached = await uow.screenings.add_if_absent(candidate)
        await uow.commit()
    return attached
