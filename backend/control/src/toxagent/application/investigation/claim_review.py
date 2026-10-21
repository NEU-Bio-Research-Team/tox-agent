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
from ...research.measurement import is_measured_value, normalised_record
from ...research.text_access import access_of

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
Where a source lists quoted_passages, those are the passages the claim says \
it quotes: judge the claim against them. Where it lists relations, those are \
the scope the answer declared for that source: a claim that goes beyond it is \
partially_supported at best. A derived claim lists its inputs; judge whether \
it follows from them. Give a one-sentence reason for each verdict. Call \
submit_claim_review once, with a verdict for every claim listed. Do not \
rewrite the answer."""


async def review_bundle(uow, *, session_id: str, answer) -> dict[str, Any]:
    """The claims under review, each with the sources it cites, from the server.

    More than an abstract (#47): for an evidence source, the passages the
    claim quotes by locator, whether its text was read at all, the measured
    value a database record states, and the scope this run's relations
    declared for it; for an observation, what it is about and the caveats it
    carries; for a derived claim, the claims it was computed from.
    """
    by_id = {c.claim_id: c for c in answer.claims}
    relations: dict[str, list[dict[str, Any]]] = {}
    for relation in await uow.evidence_relations.list_for_run(answer.run_id, answer_id=answer.id):
        relations.setdefault(relation.source_ref.source_id, []).append({
            "relation": relation.relation.value,
            "directness": relation.directness.value,
            "scope": relation.scope.declared(),
        })
    texts: dict[str, Any] = {}

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
                source = {
                    "ref": f"observation:{observation.id}", "kind": observation.kind.value,
                    "producer": observation.producer.value,
                    "field_path": claim.field_path, "value": value,
                    "required_limitations": list(observation.required_limitations),
                }
                subject = observation.model_projection.get("subject")
                if subject is not None:
                    source["subject"] = subject
                sources.append(source)
        for evidence_id in claim.citation_ids:
            record = await uow.evidence.get(evidence_id, session_id=session_id)
            if record is None:
                continue
            if evidence_id not in texts:
                texts[evidence_id] = await uow.evidence_text.get(evidence_id, session_id=session_id)
            full_text = texts[evidence_id]
            source = {
                "ref": f"evidence:{record.id}", "kind": "evidence_record",
                "source_type": record.source_type.value,
                "title": record.title,
                "excerpt": (record.abstract_or_excerpt or "")[:_EXCERPT_CHARS],
                "published": record.published_at.isoformat() if record.published_at else None,
                "text_access": (
                    full_text.state if full_text is not None
                    else access_of(record.normalized_facts).value
                ),
                "untrusted_external_content": True,
            }
            if is_measured_value(record.normalized_facts):
                source["measurement"] = normalised_record(record.normalized_facts)
            quoted = [
                full_text.passage(locator) if full_text is not None else None
                for span_evidence, locator in claim.evidence_spans
                if span_evidence == evidence_id
            ]
            if quoted:
                source["quoted_passages"] = [
                    {"locator": p.locator, "section": p.section, "kind": p.kind,
                     "text": p.text}
                    for p in quoted if p is not None
                ]
            if relations.get(evidence_id):
                source["relations"] = relations[evidence_id]
            sources.append(source)
        entry: dict[str, Any] = {
            "claim_id": claim.claim_id, "kind": kind, "text": claim.text, "sources": sources,
        }
        if claim.input_claim_ids:
            entry["inputs"] = [
                {"claim_id": cid, "text": by_id[cid].text,
                 "rendered_value": by_id[cid].rendered_value}
                for cid in claim.input_claim_ids if cid in by_id
            ]
            entry["transform"] = claim.transform
        claims.append(entry)
    return {"answer_id": answer.id, "claims": claims}


def summarize(reviews: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"supported": 0, "partially_supported": 0, "not_supported": 0, "cannot_judge": 0}
    for item in reviews:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    return counts
