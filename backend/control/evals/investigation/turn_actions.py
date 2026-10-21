"""A protocol v2 turn, as every arm receives it (#50/#51).

A frozen case's turn is a message and, sometimes, actions a researcher would
take in the product: ``add_context`` (a measured value, an exposure),
``attach_evidence_ref`` (a source they point at) and ``switch_subject`` (they
select another screening or analysis). Section 5.4 of the master plan asks
that B, C and D receive the same factual information at every turn -- D may
organise it in its case, but may not be given a fact the others never see.

So the facts an action carries are rendered into the message text, the same
way for every arm, by one versioned function here; the message is what the
arms share. A case arm additionally files ``add_context`` into its case
through the product's context route (the investigation board's affordance),
carrying the same key, value, unit, source and scope -- nothing more.
``switch_subject`` is not text: it is the subject the next message names,
exactly as a selection in the UI is.

A case's inputs name an analysis subject by its *origin* id in the frozen
replay file (``{"kind": "analysis", "analysis_ref": ...}``); the product
takes it as ``{"kind": "frozen_analysis", "analysis_ref": ...}`` and binds the
run to a session-local copy. :func:`product_subject_ref` is that mapping.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

#: Bumped whenever the rendering changes: two studies whose arms received
#: differently rendered turns are not the same study.
RENDERING_VERSION = "turn-actions-text-v1"

_SCOPE_LABELS = {
    "target_id": "target", "variant": "variant", "experimental_context": "assay context",
    "assay": "assay", "species": "species", "endpoint": "endpoint",
}


def render_context(action: Mapping[str, Any]) -> str:
    """``key = value unit (source: ...; target EGFR, assay context ...)``."""
    text = f"{action['key']} = {action['value']}"
    if action.get("unit"):
        text += f" {action['unit']}"
    qualifiers = []
    if action.get("source"):
        qualifiers.append(f"source: {action['source']}")
    for axis, value in sorted((action.get("scope") or {}).items()):
        qualifiers.append(f"{_SCOPE_LABELS.get(axis, axis)} {value}")
    if action.get("note"):
        qualifiers.append(str(action["note"]))
    return text + (f" ({'; '.join(qualifiers)})" if qualifiers else "")


def sent_text(turn: Mapping[str, Any]) -> str:
    """The message every arm receives for ``turn``: its text, then the facts
    its actions carry, in the order the case states them."""
    lines: list[str] = []
    for action in turn.get("actions") or ():
        kind = action.get("type")
        if kind == "add_context":
            lines.append(f"- {render_context(action)}")
        elif kind == "attach_evidence_ref":
            note = f": {action['note']}" if action.get("note") else ""
            lines.append(f"- the researcher points to source {action['ref']}{note}")
    text = str(turn["text"]).strip()
    if not lines:
        return text
    return text + "\n\nResearcher-supplied context for this turn:\n" + "\n".join(lines)


def context_actions(turn: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(a) for a in turn.get("actions") or () if a.get("type") == "add_context"]


def subject_for_turn(turn: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """The subject this turn is about: a ``switch_subject`` action (or the
    turn's own ``subject_ref``) moves it; otherwise it stays."""
    subject = dict(current)
    for action in turn.get("actions") or ():
        if action.get("type") == "switch_subject":
            subject = dict(action["subject_ref"])
    if turn.get("subject_ref"):
        subject = dict(turn["subject_ref"])
    return subject


def subject_key(ref: Mapping[str, Any]) -> str:
    """``screening:<id>`` / ``analysis:<origin id>``: the input's own name for
    a subject, stable across arms (local ids are not)."""
    if ref.get("kind") == "activity_screening":
        return f"screening:{ref['screening_id']}"
    return f"analysis:{ref.get('analysis_ref') or ref.get('analysis_id')}"


def product_subject_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    """The ``subject_ref`` the product's message route takes for an input's
    subject."""
    if ref.get("kind") == "activity_screening":
        return {"kind": "activity_screening", "screening_id": ref["screening_id"]}
    if ref.get("analysis_ref"):
        return {"kind": "frozen_analysis", "analysis_ref": ref["analysis_ref"]}
    return {"kind": "analysis", "analysis_id": ref["analysis_id"]}


def switched_to(inputs: Mapping[str, Any]) -> set[str]:
    """Subjects a turn switches to: attached when selected, not before."""
    out: set[str] = set()
    for turn in inputs.get("turns") or ():
        for action in turn.get("actions") or ():
            if action.get("type") == "switch_subject":
                out.add(subject_key(action["subject_ref"]))
        if turn.get("subject_ref"):
            out.add(subject_key(turn["subject_ref"]))
    out.discard(subject_key((inputs.get("subject") or {}).get("subject_ref") or {}))
    return out


def declared_digests(inputs: Mapping[str, Any]) -> dict[str, str]:
    """The content hash the inputs declare for each artifact, by subject key."""
    out: dict[str, str] = {}
    for artifact in inputs.get("artifacts") or ():
        if artifact.get("kind") == "screening":
            out[f"screening:{artifact['ref']}"] = str(artifact.get("content_sha256"))
        elif artifact.get("kind") == "frozen_analysis":
            out[f"analysis:{artifact['ref']}"] = str(artifact.get("content_sha256"))
    return out


def rendered_turns_sha256(inputs: Mapping[str, Any]) -> str:
    """What every arm was sent, as one hash: the arms of a study share it."""
    digest = hashlib.sha256(RENDERING_VERSION.encode())
    for turn in inputs.get("turns") or ():
        digest.update(b"\0")
        digest.update(sent_text(turn).encode("utf-8"))
    return digest.hexdigest()
