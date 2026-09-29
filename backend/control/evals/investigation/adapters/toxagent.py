"""ToxAgent arms (C, D and D's ablations) through the live product API.

The adapter behaves like the researcher's browser: it submits the structure,
sends each turn, and — only on an arm that keeps a scientific case — also files
a turn's structured context through ``POST /cases/{id}/context``, the way the
investigation board would. The message text is identical on every arm; the
structured copy is the case arm's affordance and is recorded as such.

Afterwards it reads back everything the product recorded and keeps it raw:
runs with tool calls and usage, answers, decision states (including the skill
record), the case, its event log and every dossier, and the evidence records
the answers cite.

The text the graders see is the product's own answer — ``answer_markdown`` —
followed by the sources its claims cite, rendered from the evidence records.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from evals.investigation.adapters import AdapterResult
from evals.investigation.adapters.toxagent_api import ToxAgentAPI
from evals.investigation.record import STATUS_ERROR, STATUS_OK, StudyStore, TurnRecord, now_iso
from evals.investigation.systems import SystemSpec, product_mismatch


def render_sources(answer: dict[str, Any] | None, evidence: list[dict[str, Any]]) -> str:
    if not answer:
        return ""
    by_id = {e.get("evidence_id"): e for e in evidence}
    cited: list[str] = []
    for claim in answer.get("claims") or ():
        for citation in claim.get("citation_ids") or ():
            if citation not in cited:
                cited.append(citation)
    lines = []
    for number, citation in enumerate(cited, 1):
        record = by_id.get(citation) or {}
        identifier = record.get("identifier") or {}
        ids = ", ".join(f"{k.upper()} {v}" for k, v in identifier.items()
                        if v and k in ("doi", "pmid", "pmcid"))
        year = (record.get("published_at") or "")[:4]
        authors = ", ".join((record.get("authors") or [])[:3])
        parts = [p for p in (authors, record.get("title"), year, ids) if p]
        lines.append(f"[{number}] " + ". ".join(parts) if parts else f"[{number}] (unresolved source)")
    return "Sources cited:\n" + "\n".join(lines) if lines else ""


def render_answer(answer: dict[str, Any] | None, evidence: list[dict[str, Any]]) -> str:
    """What the researcher reads: the answer, its limitation notes, its sources."""
    if not answer:
        return ""
    parts = [answer.get("answer_markdown", "").strip()]
    notes = [l.get("text") or l.get("code") for l in answer.get("limitations") or ()]
    notes = [n for n in notes if n]
    if notes:
        parts.append("Limitations noted:\n" + "\n".join(f"- {n}" for n in notes))
    steps = [s.get("text") for s in answer.get("recommended_next_steps") or () if s.get("text")]
    if steps:
        parts.append("Recommended next steps:\n" + "\n".join(f"- {s}" for s in steps))
    sources = render_sources(answer, evidence)
    if sources:
        parts.append(sources)
    return "\n\n".join(p for p in parts if p)


class ToxAgentAdapter:
    def __init__(self, base_url: str, token: str, *,
                 transport: httpx.AsyncBaseTransport | None = None,
                 poll_delay_s: float = 1.0, run_timeout_s: float = 1200.0) -> None:
        self.base_url = base_url
        self._token = token
        self._transport = transport
        self._poll_delay_s = poll_delay_s
        self._run_timeout_s = run_timeout_s
        self._product: dict[str, Any] | None = None

    def _api(self) -> ToxAgentAPI:
        return ToxAgentAPI(self.base_url, self._token, transport=self._transport,
                           poll_delay_s=self._poll_delay_s, run_timeout_s=self._run_timeout_s)

    async def effective_product(self) -> dict[str, Any]:
        if self._product is None:
            async with self._api() as api:
                self._product = await api.effective_product()
        return self._product

    def describe(self) -> dict[str, Any]:
        product = self._product or {}
        return {
            "adapter": "toxagent", "base_url_host": httpx.URL(self.base_url).host,
            "effective_product_hash": product.get("effective_product_hash"),
            "runtime": product.get("runtime"), "scientific_skills": product.get("scientific_skills"),
            "flags": {k: v.get("enabled") for k, v in (product.get("flags") or {}).items()},
        }

    async def run(self, case: dict[str, Any], spec: SystemSpec, *, trial: int,
                  snapshot: dict[str, Any] | None, store: StudyStore) -> AdapterResult:
        del snapshot  # the product computes its own
        product = await self.effective_product()
        problems = product_mismatch(spec, product)
        if problems:
            return AdapterResult(
                status=STATUS_ERROR, turns=[],
                error="deployment is not this arm: " + "; ".join(problems),
            )
        model = {
            "provider_id": (product.get("runtime") or {}).get("provider_id"),
            "model_id_resolved": (product.get("runtime") or {}).get("model_id"),
            "runtime_kind": (product.get("runtime") or {}).get("kind"),
            "runtime_version": (product.get("runtime") or {}).get("runtime_version"),
            "effective_product_hash": product.get("effective_product_hash"),
            "skills_mode": (product.get("scientific_skills") or {}).get("mode"),
            "skill_catalog_sha256": (product.get("scientific_skills") or {}).get("catalog_sha256"),
        }
        turns: list[TurnRecord] = []
        trace: dict[str, Any] = {"runs": [], "context_posts": []}
        async with self._api() as api:
            try:
                session_id = await api.new_session()
                trace["session_id"] = session_id
                analysis = await api.analyse(session_id, case["compound"]["smiles"])
                trace["analysis"] = analysis
                keeps_case = bool(spec.required_flags.get("scientific_case_v1"))
                for index, turn in enumerate(case["turns"]):
                    if turn.get("context") and keeps_case:
                        cases = (await api.get(f"/v1/sessions/{session_id}/cases") or {}).get("cases", [])
                        if cases:
                            posted = await api.post(
                                f"/v1/sessions/{session_id}/cases/{cases[0]['case_id']}/context",
                                {k: v for k, v in turn["context"].items() if k in ("key", "value", "note")},
                            )
                            trace["context_posts"].append({"turn": index, "case_revision": posted.get("revision")})
                    started, clock = now_iso(), time.monotonic()
                    run = await api.send(session_id, turn["text"])
                    ended, duration = now_iso(), round(time.monotonic() - clock, 3)
                    answer = await api.answer_for_run(session_id, run["run_id"])
                    state = await api.get(f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state")
                    dossier = await api.get(f"/v1/sessions/{session_id}/runs/{run['run_id']}/dossier")
                    trace["runs"].append({"run": run, "answer": answer, "decision_state": state,
                                          "dossier": dossier})
                    turns.append(TurnRecord(
                        index=index, user_text=turn["text"], sent_text=turn["text"],
                        started_at=started, ended_at=ended, duration_s=duration,
                        meta={"run_id": run["run_id"], "run_status": run.get("status"),
                              "answer_id": (answer or {}).get("answer_id"),
                              "is_fallback": (answer or {}).get("is_fallback"),
                              "stop_reason": (state or {}).get("stop_reason"),
                              "skills": (state or {}).get("skills"),
                              "tool_calls": [c.get("tool_name") for c in run.get("tool_calls") or ()],
                              "dossier_case_revision": (dossier or {}).get("case_revision")},
                    ))
                evidence = await api.evidence(session_id)
                trace["evidence"] = evidence
                cases = (await api.get(f"/v1/sessions/{session_id}/cases") or {}).get("cases", [])
                trace["cases"] = []
                for summary in cases:
                    case_id = summary["case_id"]
                    trace["cases"].append({
                        "case": await api.get(f"/v1/sessions/{session_id}/cases/{case_id}"),
                        "events": await api.get(f"/v1/sessions/{session_id}/cases/{case_id}/events"),
                    })
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                artifact = store.write_raw(spec.system_id, case["case_id"], trial, "trace.json", trace)
                return AdapterResult(status=STATUS_ERROR, turns=turns, model=model,
                                     artifacts={"trace": artifact},
                                     error=f"{type(exc).__name__}: {exc}")
        for record, run_trace in zip(turns, trace["runs"]):
            record.response_text = render_answer(run_trace["answer"], trace["evidence"])
        artifact = store.write_raw(spec.system_id, case["case_id"], trial, "trace.json", trace)
        usage = _usage(trace["runs"])
        failed = [t.index for t in turns if t.meta.get("run_status") != "completed"]
        return AdapterResult(
            status=STATUS_OK if not failed else STATUS_ERROR,
            turns=turns, final_text=turns[-1].response_text if turns else "",
            model=model, usage=usage, artifacts={"trace": artifact},
            toxagent={
                "session_id": trace.get("session_id"),
                "run_ids": [t.meta["run_id"] for t in turns],
                "case_ids": [c["case"]["case_id"] for c in trace.get("cases", []) if c.get("case")],
                "fallback_turns": [t.index for t in turns if t.meta.get("is_fallback")],
            },
            error=f"turns {failed} did not complete" if failed else None,
        )


def _usage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Provider-reported tokens and cost over the runs, when reported.

    A provider may report one message's usage several times as it grows; only
    the latest revision per message counts, so a sum never double-counts. The
    model ids here are the ones the provider says answered.
    """
    latest: dict[str, dict[str, Any]] = {}
    for item in runs:
        usage = (item.get("run") or {}).get("usage") or {}
        for event in usage.get("events") or ():
            source = event.get("source") or {}
            key = source.get("message_id") or source.get("event_id") or event.get("usage_event_id")
            current = latest.get(key)
            if current is None or (source.get("revision") or 0) >= ((current.get("source") or {}).get("revision") or 0):
                latest[key] = event
    totals: dict[str, Any] = {}
    for event in latest.values():
        for name, value in (event.get("tokens") or {}).items():
            if isinstance(value, (int, float)):
                totals[f"tokens_{name}"] = totals.get(f"tokens_{name}", 0) + value
        cost = event.get("cost") or {}
        if cost.get("amount") is not None:
            key = "cost_" + (cost.get("currency") or "unknown").lower()
            totals[key] = round(totals.get(key, 0.0) + float(cost["amount"]), 8)
    return {
        "status": "reported" if latest else "unknown",
        "provider_reported_models": sorted({e.get("model_id") for e in latest.values() if e.get("model_id")}),
        **totals,
    }
