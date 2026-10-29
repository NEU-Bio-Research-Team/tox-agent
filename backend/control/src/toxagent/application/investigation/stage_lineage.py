"""How a run's answer came to be what the researcher saw (AG-UI, §10.2, §14.2).

A released answer is the last of several stages: the draft the model first
submitted, the candidate that passed the hard validator (possibly after a
correction), the reviewer's verdicts on it, the repair a blocking verdict
bought, the second review, and then what the release policy did with all of
that. The audit view shows that chain claim by claim, because the two things a
reader needs to judge are invisible in the final answer alone: which claims a
repair removed or narrowed (a repair that only deletes raises precision and
loses information, release_gate.RepairEffect), and which required inputs or
limitations the run never got to.

Everything here is read from what the run recorded -- its drafts, the answers
the server accepted from them, and its run result. Nothing is re-judged: a
verdict is the reviewer's, a disposition the policy's. The candidate's prose is
part of the audit, never of the answer; ``is_public`` says which stage, if any,
is the one the answer route serves.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

LINEAGE_SCHEMA_VERSION = "stage-lineage-v1"

_PHASE_ORDER = {"initial": 0, "hard_correction": 1, "semantic_repair": 2}
_REVIEW_PHASES = ("review_initial", "review_final")


def _value(item: Any) -> str:
    return str(getattr(item, "value", item))


def _claim_key(claim: Any) -> str:
    """A claim's lineage key: its draft label survives a repair; the
    server-issued id does not (release_gate._claim_texts)."""
    if isinstance(claim, Mapping):
        return str(claim.get("local_ref") or claim.get("claim_id") or "")
    return str(getattr(claim, "local_ref", None) or claim.claim_id)


def _claims_of_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": _claim_key(c), "kind": c.get("kind"), "text": str(c.get("text") or "")}
        for c in payload.get("claims") or () if isinstance(c, Mapping)
    ]


def _claims_of_answer(answer: Any) -> list[dict[str, Any]]:
    return [
        {"key": _claim_key(c), "claim_id": c.claim_id, "kind": _value(c.kind), "text": c.text}
        for c in getattr(answer, "claims", ())
    ]


def _limitation_codes(items: Iterable[Any]) -> list[str]:
    codes = []
    for item in items:
        code = item.get("code") if isinstance(item, Mapping) else getattr(item, "code", None)
        if code is not None:
            codes.append(_value(code))
    return sorted(set(codes))


def _violation_codes(violations: Any) -> list[str]:
    return [
        str(v.get("code")) if isinstance(v, Mapping) else str(v)
        for v in violations or ()
    ]


def stage_lineage(
    *,
    run_id: str,
    drafts: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Any],
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The run's stages, review rounds and claim lineage.

    ``drafts`` as the answer-drafts route lists them, in submission order;
    ``answers`` the accepted answers they produced, by id; ``result`` the run
    result's ``to_dict`` (``None`` for a run before a release policy, whose
    review is then unknown rather than passed).
    """
    result = dict(result or {})
    detail = dict(result.get("detail") or {})
    public_id = result.get("public_answer_id")

    stages: list[dict[str, Any]] = []
    for index, draft in enumerate(
        sorted(drafts, key=lambda d: (int(d.get("generation") or 0),
                                      _PHASE_ORDER.get(str(d.get("phase") or "initial"), 0)))
    ):
        payload = dict(draft.get("payload") or {})
        answer_id = draft.get("answer_id")
        answer = answers.get(answer_id) if answer_id else None
        claims = _claims_of_answer(answer) if answer is not None else _claims_of_payload(payload)
        stages.append({
            "index": index,
            "phase": draft.get("phase") or "initial",
            "generation": draft.get("generation"),
            "outcome": draft.get("outcome"),
            "answer_id": answer_id,
            # Hard-valid: the server accepted it as an answer. A rejected
            # draft is kept, with why, because it is what the model first said.
            "hard_valid": answer is not None,
            "is_public": bool(answer_id) and answer_id == public_id,
            "violation_codes": _violation_codes(draft.get("violations")),
            "claim_count": len(claims),
            "limitation_codes": _limitation_codes(
                getattr(answer, "limitations", ()) if answer is not None
                else payload.get("limitations") or ()
            ),
            "answer_markdown": str(payload.get("answer_markdown") or ""),
            "created_at": draft.get("created_at"),
        })

    # Claim texts by key, per accepted answer, so a verdict and a repair can
    # be read against the claim they are about.
    texts_by_answer = {
        answer_id: {c["claim_id"]: c for c in _claims_of_answer(answer)}
        for answer_id, answer in answers.items()
    }
    reviews: list[dict[str, Any]] = []
    for round_index, review in enumerate(detail.get("reviews") or ()):
        claims = texts_by_answer.get(review.get("answer_id"), {})
        reviews.append({
            "round": round_index + 1,
            "phase": _REVIEW_PHASES[round_index] if round_index < len(_REVIEW_PHASES)
            else f"review_{round_index + 1}",
            "status": review.get("status"),
            "reason": review.get("reason"),
            "answer_id": review.get("answer_id"),
            "counts": dict(review.get("counts") or {}),
            "blocking_claim_ids": list(review.get("blocking_claim_ids") or ()),
            "verdicts": [
                {"claim_id": v.get("claim_id"),
                 "key": (claims.get(v.get("claim_id")) or {}).get("key"),
                 "text": (claims.get(v.get("claim_id")) or {}).get("text"),
                 "verdict": v.get("verdict"), "reason": v.get("reason")}
                for v in review.get("verdicts") or ()
            ],
        })

    accepted = [s for s in stages if s["hard_valid"]]
    first = answers.get(accepted[0]["answer_id"]) if accepted else None
    final_id = public_id or result.get("candidate_answer_id") or (
        accepted[-1]["answer_id"] if accepted else None
    )
    final = answers.get(final_id) if final_id else None
    before = {c["key"]: c for c in _claims_of_answer(first)} if first is not None else {}
    after = {c["key"]: c for c in _claims_of_answer(final)} if final is not None else {}
    verdicts_by_key: dict[str, list[dict[str, Any]]] = {}
    for review in reviews:
        for verdict in review["verdicts"]:
            if verdict["key"]:
                verdicts_by_key.setdefault(verdict["key"], []).append(
                    {"round": review["round"], "verdict": verdict["verdict"],
                     "reason": verdict["reason"]}
                )
    claims: list[dict[str, Any]] = []
    for key in list(before) + [k for k in after if k not in before]:
        old, new = before.get(key), after.get(key)
        if old is not None and new is not None:
            change = "kept" if old["text"] == new["text"] else "reworded"
        elif old is not None:
            change = "removed"
        else:
            change = "added"
        claims.append({
            "key": key,
            "kind": (new or old or {}).get("kind"),
            "candidate_text": old["text"] if old is not None else None,
            "final_text": new["text"] if new is not None else None,
            "change": change,
            "verdicts": verdicts_by_key.get(key, []),
        })

    repair = detail.get("repair")
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "recorded": bool(result),
        "release_policy": result.get("release_policy") or "legacy",
        "disposition": result.get("disposition"),
        "review_status": result.get("review_status"),
        "execution_outcome": result.get("execution_outcome"),
        "public_answer_id": public_id,
        "candidate_answer_id": result.get("candidate_answer_id"),
        "stages": stages,
        "reviews": reviews,
        "claims": claims,
        "repair": dict(repair) if isinstance(repair, Mapping) else None,
        # What the reader did not get, and why: inputs the run said it needed,
        # claims still blocked, limitations a repair dropped.
        "missing": {
            "needed_inputs": [str(x) for x in detail.get("needed_inputs") or ()],
            "unresolved_claim_ids": list(detail.get("unresolved_claim_ids") or ()),
            "lost_limitation_codes": list(
                (repair or {}).get("lost_limitation_codes") or ()
            ) if isinstance(repair, Mapping) else [],
            "confirmed_fact_count": len(detail.get("confirmed_facts") or ()),
            "reason": detail.get("reason"),
            "failure_code": detail.get("failure_code"),
        },
    }
