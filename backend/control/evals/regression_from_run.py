"""Production failure -> sanitized regression draft (Wave 5).

A failure seen in production is the most valuable task a suite can gain, and
the most dangerous to copy verbatim: it carries a user's words, identifiers
and possibly personal data. This turns a run, read over the product API with
the operator's own token, into an ``eval-task-v3`` *draft*:

* the conversation keeps only user turns, with emails, phone numbers, URLs and
  ToxAgent ids replaced by placeholders; SMILES are kept (a structure is the
  subject, not personal data);
* ``data_provenance`` records ``production_sanitized`` and a one-way hash of
  the originating run id, never the id itself;
* ``expect`` is seeded from what the run *should* have done (intent, a
  completed run, an accepted non-fallback answer) and marked for review.

Drafts are written to ``evals/regression/drafts/``, which is not a declared
pack: a person reviews the draft, sets the fixture and expectations, and moves
it into ``regression/tasks/``. Nothing enters a gating suite unreviewed.

    python -m evals.regression_from_run --base-url https://… --token … \
        --session ses_… --run run_… --task-id ads-17-short-name
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
DRAFTS_DIR = HERE / "regression" / "drafts"

_SCRUB = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"\b(?:ses|run|ana|obs|evd|ans|rpt|clm|msg)_[0-9a-f]{32}\b"), "<id>"),
    (re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"), "<phone>"),
)


def scrub(text: str) -> str:
    for pattern, placeholder in _SCRUB:
        text = pattern.sub(placeholder, text)
    return text


def draft_from(
    *, task_id: str, run: dict[str, Any], messages: list[dict[str, Any]],
    answer: dict[str, Any] | None,
) -> dict[str, Any]:
    conversation: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = " ".join(
            part.get("content", {}).get("text", "")
            for part in message.get("parts", []) if part.get("type") == "text"
        ).strip()
        turn: dict[str, Any] = {"role": "user", "content": scrub(text)}
        for part in message.get("parts", []):
            smiles = (part.get("content") or {}).get("smiles")
            if smiles:
                turn["molecule"] = {"smiles": smiles}
                turn["intent_hint"] = "analyze"
        conversation.append(turn)
    intent = run.get("intent") or "decision_support"
    observed = {
        "status": run.get("status"), "failure_code": run.get("failure_code"),
        "answer_outcome": (
            None if answer is None else "fallback" if answer.get("is_fallback")
            else "accepted_after_correction" if (answer.get("candidate_generation") or 1) > 1
            else "first_pass"
        ),
        "tool_calls": [c.get("tool_name") for c in run.get("tool_calls") or ()],
    }
    return {
        "task_id": task_id,
        "schema_version": "eval-task-v3",
        "capability_pack": "regression",
        "category": "decision_support" if intent == "decision_support" else "report_qa",
        "title": f"REVIEW: production regression ({intent})",
        "rationale": "Sanitized from a production failure. Observed behaviour: "
                     + json.dumps(observed, sort_keys=True)
                     + ". A reviewer must confirm the fixture, expectations and hard gates.",
        "fixture": "REVIEW-choose-a-fixture",
        "language": "vi" if any(re.search(r"[ăâđêôơưạ-ỹ]", t["content"]) for t in conversation) else "en",
        "risk_tier": "high",
        "runtime_requirement": "agentic_runtime",
        "intent": intent,
        "conversation": conversation or [{"role": "user", "content": "<empty>"}],
        "expect": {
            "run": {"status": "completed", "intent": intent},
            "outcome": {"capability": "accepted"},
        },
        "required_graders": ["outcome_split", "trajectory"],
        "data_provenance": {
            "source": "production_sanitized",
            "sanitized": True,
            "origin_ref": "sha256:" + hashlib.sha256(str(run.get("run_id")).encode()).hexdigest()[:16],
        },
    }


async def _fetch(base_url: str, token: str, session_id: str, run_id: str):
    auth = {"authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30.0) as client:
        run = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=auth)).json()
        messages = (await client.get(f"/v1/sessions/{session_id}/messages", headers=auth)).json()
        answer = None
        for message in messages.get("messages", []):
            for part in message.get("parts", []):
                if part.get("type") == "answer_ref":
                    response = await client.get(
                        f"/v1/sessions/{session_id}/answers/{part['content']['answer_id']}",
                        headers=auth,
                    )
                    if response.status_code == 200 and response.json().get("run_id") == run_id:
                        answer = response.json()
    return run, messages.get("messages", []), answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    run, messages, answer = asyncio.run(_fetch(args.base_url, args.token, args.session, args.run))
    draft = draft_from(task_id=args.task_id, run=run, messages=messages, answer=answer)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    path = DRAFTS_DIR / f"{args.task_id}.json"
    path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote review draft {path.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
