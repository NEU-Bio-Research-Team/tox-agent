"""Serialising a run's execution input so a different process can finish it.

I18: `RunScheduler._tasks` is a dict in one process's memory. Everything a
handler needs — who asked, the text, the molecule, the endpoints, the model
binding, the explanation targets — lived only in the `RunContext` held by that
task. A `kill -9` therefore destroyed the *only* copy of the request, which is
why startup reconciliation could do nothing but close the run out: it had a run
id and no way to know what the run was supposed to do.

This is that copy. The envelope is written in the same transaction that creates
the run, so a run that exists is a run that can be executed, by this process or
by whichever one outlives it.

Two constraints shape the format:

**No secrets, ever.** `ai_profile_id` is an opaque connection id, resolved to a
credential at dispatch by the gateway against the owner it re-proves there. A
credential in this row would be a credential in the database, in backups, and
in anything that reads a job queue for operational reasons.

**Additive only.** A worker running the previous release must be able to read
an envelope written by the next one, because that is exactly the situation a
rolling upgrade creates. Unknown keys are ignored on read and every field has a
default, so an envelope missing a field a newer writer would have set decodes
to the same value the older writer meant.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ..domain.run import Intent
from .policy import Actor

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from .run_scheduler import RunContext

#: Bumped only for a change a reader cannot absorb by ignoring a key. Stored so
#: an envelope written by a version this process does not understand is left
#: for a worker that does, rather than decoded into something wrong.
ENVELOPE_VERSION = 1


class UnreadableEnvelope(ValueError):
    """The envelope cannot be decoded by this build."""


def to_envelope(context: "RunContext") -> dict[str, Any]:
    return {
        "v": ENVELOPE_VERSION,
        "actor": {
            "subject_id": context.actor.subject_id,
            "roles": sorted(context.actor.roles),
        },
        "session_id": context.session_id,
        "run_id": context.run_id,
        "intent": context.intent.value,
        "text": context.text,
        "smiles": context.smiles,
        "batch_smiles": list(context.batch_smiles),
        "endpoints": list(context.endpoints) if context.endpoints is not None else None,
        "model_selection": dict(context.model_selection) if context.model_selection else None,
        "ai_profile_id": context.ai_profile_id,
        "threshold_overrides": (
            dict(context.threshold_overrides) if context.threshold_overrides else None
        ),
        "explanation_mode": context.explanation_mode,
        # A tuple of pairs, and JSON has no tuples: pairs become two-element
        # lists and are read back as pairs.
        "explanation_targets": [list(target) for target in context.explanation_targets],
        "analysis_id": context.analysis_id,
        "needs_snapshot_first": context.needs_snapshot_first,
        "language": context.language,
        "attachment_id": context.attachment_id,
    }


def from_envelope(envelope: Mapping[str, Any]) -> "RunContext":
    # Imported here rather than at module scope: `run_scheduler` imports this
    # module for the codec, so a top-level import would be a cycle that only
    # resolves when the scheduler happens to be imported first.
    from .run_scheduler import RunContext

    version = envelope.get("v", 0)
    if not isinstance(version, int) or version > ENVELOPE_VERSION:
        raise UnreadableEnvelope(
            f"envelope version {version!r} was written by a newer build than this one"
        )
    actor = envelope.get("actor") or {}
    subject_id = actor.get("subject_id")
    if not subject_id:
        raise UnreadableEnvelope("envelope has no actor to execute as")
    try:
        intent = Intent(envelope["intent"])
    except (KeyError, ValueError) as exc:
        raise UnreadableEnvelope(f"envelope names no intent this build has: {exc}") from exc

    endpoints = envelope.get("endpoints")
    targets = envelope.get("explanation_targets") or []
    return RunContext(
        actor=Actor(subject_id=subject_id, roles=frozenset(actor.get("roles") or ())),
        session_id=envelope["session_id"],
        run_id=envelope["run_id"],
        intent=intent,
        text=envelope.get("text") or "",
        smiles=envelope.get("smiles"),
        batch_smiles=tuple(envelope.get("batch_smiles") or ()),
        endpoints=tuple(endpoints) if endpoints is not None else None,
        model_selection=envelope.get("model_selection") or None,
        ai_profile_id=envelope.get("ai_profile_id"),
        threshold_overrides=envelope.get("threshold_overrides") or None,
        explanation_mode=envelope.get("explanation_mode") or "on_demand",
        explanation_targets=tuple(
            (target[0], target[1] if len(target) > 1 else None) for target in targets
        ),
        analysis_id=envelope.get("analysis_id"),
        # Carried, unlike in `RunScheduler._terminate`'s recovery contexts.
        # There the snapshot demonstrably already ran (a runtime binding
        # existed); an adopted run crashed at an unknown point and may not
        # have taken it yet. Redoing it is safe because
        # `CreateAnalysis` addresses the snapshot by `snapshot_idempotency_key`
        # and returns the committed one instead of writing a second.
        needs_snapshot_first=bool(envelope.get("needs_snapshot_first")),
        language=envelope.get("language") or "en",
        attachment_id=envelope.get("attachment_id"),
    )
