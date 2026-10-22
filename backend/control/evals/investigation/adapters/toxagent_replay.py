"""ToxAgent arms B, C and D on a protocol v2 case's frozen inputs (#50).

The v1 adapter submitted the case's structure and let the deployment compute
a fresh analysis; on frozen inputs that is exactly what must not happen. This
one puts the case's own inputs to the product the way a researcher's browser
would, and nothing else:

* the session is given the case's frozen artifacts by reference -- a
  screening is attached (``POST /screenings``) when it is first selected, a
  toxicity analysis is named as the message's subject and bound to a
  session-local copy -- so no arm calls the predictor;
* each turn is sent as :func:`turn_actions.sent_text` renders it, the same
  text for every arm, with the turn's subject named explicitly;
* on an arm that keeps a scientific case, a turn's ``add_context`` is also
  filed into the case of that turn's subject, typed (value, unit, source,
  scope) -- the same facts the message carries, organised;
* afterwards everything the product recorded is read back raw: each run with
  its tool calls, phase usage and subject binding, its result (disposition,
  review status, outcomes), the public answer if one was published, every
  draft, the decision state, the dossier, the cases with their events, and
  the evidence records.

What the grader reads for a turn is what the user saw: the published answer
(rendered as the v1 adapter renders it), or -- when the result was withheld
-- the incomplete-result message the product showed instead. A run that ended
without an answer is part of the outcome, never dropped; only a failure to
put the case to the product at all (a deployment that is not the arm, an
artifact whose hash is not the one the inputs declare) makes the record an
error.

Fairness between arms is checked on what each was given: the content hash
of every artifact the run was bound to (screening record, toxicity snapshot,
frozen analysis), compared with the hash the inputs declare. Local ids are
minted per session and never compared.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from evals.investigation import turn_actions
from evals.investigation.adapters import AdapterResult
from evals.investigation.adapters.toxagent import ToxAgentAdapter, _usage, render_answer
from evals.investigation.record import STATUS_ERROR, STATUS_OK, StudyStore, TurnRecord, now_iso
from evals.investigation.systems import SystemSpec, product_mismatch

ADAPTER_VERSION = "toxagent-replay-adapter-v2"


def _observed(binding: Mapping[str, Any] | None) -> dict[str, str]:
    """The content hash of each artifact a run was bound to, keyed the way
    the inputs name it."""
    out: dict[str, str] = {}
    for ref in (binding or {}).get("artifact_refs") or ():
        kind = ref.get("kind")
        if kind == "screening":
            out[f"screening:{ref['source_id']}"] = str(ref.get("content_sha256"))
        elif kind == "frozen_analysis":
            out[f"analysis:{ref['source_id']}"] = str(ref.get("content_sha256"))
        elif kind == "toxicity_snapshot":
            out[f"toxicity_snapshot:{ref['source_id']}"] = str(ref.get("content_sha256"))
    return out


def messages_for_run(messages: list[Mapping[str, Any]], run_id: str,
                     answer_id: str | None = None) -> list[Mapping[str, Any]]:
    """The assistant messages a run posted.

    A message carries no ``run_id`` of its own (the API's ``Message`` has
    none); it names its run through its parts -- ``result_ref`` and
    ``dossier_ref`` carry the run id, ``answer_ref`` the run's answer. Matching
    on a ``run_id`` field matched nothing, so a run whose answer was not
    published was recorded as having said nothing, when the researcher had
    been shown the server's confirmed facts and the reason.
    """
    def belongs(message: Mapping[str, Any]) -> bool:
        for part in message.get("parts") or ():
            content = part.get("content") or {}
            if content.get("run_id") == run_id:
                return True
            if answer_id and part.get("type") == "answer_ref" \
                    and content.get("answer_id") == answer_id:
                return True
        return False

    return [m for m in messages if m.get("role") == "assistant" and belongs(m)]


def visible_text(answer: Mapping[str, Any] | None, messages: list[Mapping[str, Any]],
                 evidence: list[dict[str, Any]], *, published: bool) -> str:
    """What the researcher read for one run."""
    if published and answer:
        return render_answer(dict(answer), evidence)
    texts = [
        str(part.get("content", {}).get("text") or "")
        for message in messages for part in message.get("parts") or ()
        if part.get("type") == "text"
    ]
    text = "\n\n".join(t for t in texts if t.strip())
    if answer and not published:
        # A partial-facts result may point at a public answer of facts only.
        rendered = render_answer(dict(answer), evidence)
        text = "\n\n".join(t for t in (text, rendered) if t)
    return text


class ToxAgentReplayAdapter(ToxAgentAdapter):
    """B, C and D: one deployment per arm, the case's frozen inputs."""

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "adapter_version": ADAPTER_VERSION,
                "turn_rendering": turn_actions.RENDERING_VERSION}

    async def run(self, case: dict[str, Any], spec: SystemSpec, *, trial: int,
                  snapshot: dict[str, Any] | None, store: StudyStore,
                  inputs: dict[str, Any] | None = None,
                  attempt_id: str | None = None) -> AdapterResult:
        del snapshot  # the inputs carry the artifacts; nothing is computed
        if inputs is None:
            return AdapterResult(status=STATUS_ERROR, turns=[],
                                 error="the replay adapter runs a protocol v2 case's inputs")
        product = await self.effective_product()
        problems = product_mismatch(spec, product)
        if problems:
            return AdapterResult(status=STATUS_ERROR, turns=[],
                                 error="deployment is not this arm: " + "; ".join(problems))
        runtime = product.get("runtime") or {}
        model = {
            "provider_id": runtime.get("provider_id"),
            "model_id_resolved": runtime.get("model_id"),
            "runtime_kind": runtime.get("kind"),
            "runtime_version": runtime.get("runtime_version"),
            "effective_product_hash": product.get("effective_product_hash"),
            "skills_mode": (product.get("scientific_skills") or {}).get("mode"),
            "skill_catalog_sha256": (product.get("scientific_skills") or {}).get("catalog_sha256"),
        }
        keeps_case = bool(spec.required_flags.get("scientific_case_v1"))
        declared = turn_actions.declared_digests(inputs)
        later = turn_actions.switched_to(inputs)
        turns: list[TurnRecord] = []
        trace: dict[str, Any] = {
            "adapter_version": ADAPTER_VERSION, "turn_rendering": turn_actions.RENDERING_VERSION,
            "runs": [], "context_posts": [], "attached": [], "observed_sha256": {},
        }
        mismatches: list[str] = []

        def write(name: str, content: Any) -> str:
            return store.write_raw(spec.system_id, case["case_id"], trial, name, content,
                                   attempt_id=attempt_id)

        async with self._api() as api:
            try:
                session_id = await api.new_session()
                trace["session_id"] = session_id
                attached: set[str] = set()

                async def attach(ref: Mapping[str, Any]) -> None:
                    key = turn_actions.subject_key(ref)
                    if ref.get("kind") != "activity_screening" or key in attached:
                        return
                    await api.post(f"/v1/sessions/{session_id}/screenings",
                                   {"screening_id": ref["screening_id"]})
                    attached.add(key)
                    trace["attached"].append(key)

                # The inputs' artifacts are the session's from the start,
                # except a subject a later turn switches to: that one arrives
                # when the researcher selects it.
                for artifact in inputs.get("artifacts") or ():
                    if artifact.get("kind") == "screening" and \
                            f"screening:{artifact['ref']}" not in later:
                        await attach({"kind": "activity_screening",
                                      "screening_id": artifact["ref"]})

                subject = dict((inputs.get("subject") or {}).get("subject_ref") or {})
                local_analysis: dict[str, str] = {}
                for index, turn in enumerate(inputs.get("turns") or ()):
                    subject = turn_actions.subject_for_turn(turn, subject)
                    await attach(subject)
                    key = turn_actions.subject_key(subject)
                    if keeps_case:
                        for action in turn_actions.context_actions(turn):
                            trace["context_posts"].append(await self._post_context(
                                api, session_id, subject, local_analysis, action, index,
                            ))
                    text = turn_actions.sent_text(turn)
                    started, clock = now_iso(), time.monotonic()
                    accepted = await api.post(f"/v1/sessions/{session_id}/messages", {
                        "intent_hint": "auto", "content": [{"type": "text", "text": text}],
                        "subject_ref": turn_actions.product_subject_ref(subject),
                    })
                    run = await api.wait(session_id, accepted["run_id"])
                    ended, duration = now_iso(), round(time.monotonic() - clock, 3)
                    run_id = run["run_id"]
                    base = f"/v1/sessions/{session_id}/runs/{run_id}"
                    binding = (run.get("configuration_snapshot") or {}).get("subject_binding")
                    if binding and binding.get("kind") == "analysis" and subject.get("analysis_ref"):
                        local_analysis[subject["analysis_ref"]] = binding["subject_ref"]["analysis_id"]
                    observed = _observed(binding)
                    trace["observed_sha256"].update(observed)
                    if key in declared and observed.get(key) != declared[key]:
                        mismatches.append(
                            f"turn {index}: {key} was bound with {observed.get(key)!r}, the "
                            f"inputs declare {declared[key]!r}"
                        )
                    result = await api.get(f"{base}/result")
                    answer = await api.answer_for_run(session_id, run_id)
                    messages = messages_for_run(
                        (await api.get(f"/v1/sessions/{session_id}/messages")
                         or {}).get("messages", []),
                        run_id, (answer or {}).get("answer_id"),
                    )
                    trace["runs"].append({
                        "turn": index, "subject": key, "sent_text": text, "run": run,
                        "result": result, "answer": answer, "assistant_messages": messages,
                        "decision_state": await api.get(f"{base}/decision-state"),
                        "dossier": await api.get(f"{base}/dossier"),
                        "answer_drafts": ((await api.get(f"{base}/answer-drafts")) or {})
                        .get("drafts"),
                    })
                    state = trace["runs"][-1]["decision_state"] or {}
                    turns.append(TurnRecord(
                        index=index, user_text=turn["text"], sent_text=text,
                        started_at=started, ended_at=ended, duration_s=duration,
                        meta={"run_id": run_id, "run_status": run.get("status"),
                              "subject": key, "binding_sha256": (binding or {}).get("binding_sha256"),
                              "answer_id": (answer or {}).get("answer_id"),
                              "is_fallback": (answer or {}).get("is_fallback"),
                              "disposition": (result or {}).get("disposition"),
                              "review_status": (result or {}).get("review_status"),
                              "execution_outcome": (result or {}).get("execution_outcome"),
                              "knowledge_outcome": (result or {}).get("knowledge_outcome"),
                              "stop_reason": state.get("stop_reason"),
                              "skills": state.get("skills"),
                              "tool_calls": [c.get("tool_name") for c in run.get("tool_calls") or ()],
                              "phase_usage": run.get("phase_usage"),
                              "dossier_case_revision": (trace["runs"][-1]["dossier"] or {})
                              .get("case_revision")},
                    ))
                trace["evidence"] = await api.evidence(session_id)
                trace["cases"] = []
                for summary in (await api.get(f"/v1/sessions/{session_id}/cases")
                                or {}).get("cases", []):
                    case_id = summary["case_id"]
                    trace["cases"].append({
                        "case": await api.get(f"/v1/sessions/{session_id}/cases/{case_id}"),
                        "events": await api.get(f"/v1/sessions/{session_id}/cases/{case_id}/events"),
                    })
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                artifact = write("trace.json", trace)
                return AdapterResult(status=STATUS_ERROR, turns=turns, model=model,
                                     artifacts={"trace": artifact},
                                     error=f"{type(exc).__name__}: {exc}")
        for record, run_trace in zip(turns, trace["runs"]):
            published = (run_trace.get("result") or {}).get("disposition") == "publish" or (
                not (run_trace.get("result") or {}).get("recorded", True)
            )
            record.response_text = visible_text(
                run_trace["answer"], run_trace["assistant_messages"], trace["evidence"],
                published=published,
            )
        artifact = write("trace.json", trace)
        incomplete = [t.index for t in turns if t.meta.get("run_status") != "completed"]
        return AdapterResult(
            status=STATUS_ERROR if mismatches else STATUS_OK,
            turns=turns, final_text=turns[-1].response_text if turns else "",
            model=model, usage=_usage(trace["runs"]), artifacts={"trace": artifact},
            toxagent={
                "session_id": trace.get("session_id"),
                "run_ids": [t.meta["run_id"] for t in turns],
                "case_ids": [c["case"]["case_id"] for c in trace["cases"] if c.get("case")],
                "fallback_turns": [t.index for t in turns if t.meta.get("is_fallback")],
                "incomplete_turns": incomplete,
                "dispositions": [t.meta.get("disposition") for t in turns],
                "observed_source_sha256": dict(sorted(trace["observed_sha256"].items())),
            },
            error=("artifact_digest_mismatch: " + "; ".join(mismatches)) if mismatches else None,
        )

    async def _post_context(self, api, session_id: str, subject: Mapping[str, Any],
                            local_analysis: Mapping[str, str], action: Mapping[str, Any],
                            index: int) -> dict[str, Any]:
        """File one add_context into the case of ``subject``. A subject with
        no case yet (its first turn) has nowhere to file it; the message
        carries it all the same, and the trace says so."""
        cases = (await api.get(f"/v1/sessions/{session_id}/cases") or {}).get("cases", [])
        wanted = (
            {"screening_id": subject["screening_id"]}
            if subject.get("kind") == "activity_screening"
            else {"analysis_id": local_analysis.get(str(subject.get("analysis_ref")))}
        )
        match = None
        for summary in cases:
            detail = await api.get(f"/v1/sessions/{session_id}/cases/{summary['case_id']}") or {}
            ref = ((detail.get("subject") or {}).get("subject_ref")) or {}
            if all(ref.get(k) == v for k, v in wanted.items() if v) and any(wanted.values()) \
                    and detail.get("status", "open") == "open":
                match = detail
        if match is None:
            return {"turn": index, "key": action.get("key"), "status": "no_case_yet"}
        body = {k: action[k] for k in ("key", "value", "note", "unit", "source", "scope")
                if action.get(k)}
        posted = await api.post(
            f"/v1/sessions/{session_id}/cases/{match['case_id']}/context", body,
        )
        return {"turn": index, "key": action.get("key"), "status": "filed",
                "case_id": match["case_id"], "case_revision": posted.get("revision")}
