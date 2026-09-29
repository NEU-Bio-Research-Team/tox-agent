"""An independent reviewer of claim support (RETHINK §4.4 step 5, W9-12).

"A reviewer may be useful for claim support, but as one model call/role that is
measured separately, not a reason to build a swarm. Each new role must beat
the baseline on the same cases, the same budget, without adding scientific
errors." So this role is small and observable: after the answer is accepted,
one runtime turn is handed the answer's claims and, for each, the sources it
cites — the observation field and its value, or the evidence record's title and
abstract — and returns a verdict per claim. The server builds that bundle; the
reviewer has no read tool, so it cannot redo the answering agent's work, and
it never sees that agent's reasoning. Verdicts are recorded on the run's
DecisionSupportStateV1 and never change the answer.
"""
from __future__ import annotations

from typing import Any

from ...domain.fieldpath import FieldPathError

#: Claims a reviewer judges: those that assert something a source must carry.
REVIEWED_KINDS = frozenset({"numeric", "classification", "scientific", "comparison"})

#: What a bundle may quote of an evidence record.
_EXCERPT_CHARS = 1500

REVIEW_INSTRUCTIONS = """\
You are an independent reviewer. You did not write the answer below and you \
do not need to agree with it. For each claim, judge only whether the sources \
listed with it support what the claim says:
- supported: the sources state it, within their scope (compound, endpoint, \
species, assay, dose);
- partially_supported: the sources state part of it, or state it for a \
narrower or different scope than the claim implies;
- not_supported: the sources do not state it, or state something else;
- cannot_judge: the sources given are not enough to tell.
A predictor value supports only the statement that the model produced that \
value, never a real-world outcome. Do not use outside knowledge to fill a \
gap: a claim true in the world but absent from its sources is not_supported. \
Give a one-sentence reason for each verdict. Call submit_claim_review once, \
with a verdict for every claim listed. Do not rewrite the answer."""


async def review_bundle(uow, *, session_id: str, answer) -> dict[str, Any]:
    """The claims under review, each with the sources it cites, from the server."""
    claims: list[dict[str, Any]] = []
    for claim in answer.claims:
        kind = getattr(claim.kind, "value", claim.kind)
        if kind not in REVIEWED_KINDS:
            continue
        sources: list[dict[str, Any]] = []
        if claim.observation_id:
            observation = await uow.observations.get(claim.observation_id, session_id=session_id)
            if observation is not None:
                value: Any = None
                if claim.field_path:
                    try:
                        value = observation.value_at(claim.field_path)
                    except FieldPathError:
                        value = None
                sources.append({
                    "ref": f"observation:{observation.id}", "kind": observation.kind.value,
                    "field_path": claim.field_path, "value": value,
                })
        for evidence_id in claim.citation_ids:
            record = await uow.evidence.get(evidence_id, session_id=session_id)
            if record is None:
                continue
            sources.append({
                "ref": f"evidence:{record.id}", "kind": "evidence_record",
                "title": record.title,
                "excerpt": (record.abstract_or_excerpt or "")[:_EXCERPT_CHARS],
                "published": record.published_at.isoformat() if record.published_at else None,
                "untrusted_external_content": True,
            })
        claims.append({"claim_id": claim.claim_id, "kind": kind, "text": claim.text,
                       "sources": sources})
    return {"answer_id": answer.id, "claims": claims}


def summarize(reviews: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"supported": 0, "partially_supported": 0, "not_supported": 0, "cannot_judge": 0}
    for item in reviews:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    return counts
