# Agentic flow rollout and rollback

WS12 / PR-18 of the 2026-09-13 remediation plan (§13). The order in which the
new paths are switched on, what to watch at each step, and how to go back.
Every step is a flag or a binary — never a schema downgrade.

## Flags involved

Declared in `backend/control/src/toxagent/flags.py`, each with an owner and a
removal date. Environment variable `TOXAGENT_FLAG_<NAME>`.

| Flag | Turns on | Watch | Rollback effect |
|---|---|---|---|
| `external_worker_mode` | API writes jobs, workers execute | `toxagent_run_claims_total`, `toxagent_run_queue_wait_seconds`, `toxagent_worker_handoffs_total` | set off on API first, then drain workers; jobs already written are still claimable by an API process with the flag off |
| `report_orchestrator_v2` | server-driven report stages, one synthesis turn | `toxagent_report_stage_seconds`, `toxagent_report_synthesis_submissions_total` | new builds take the model-driven path; builds in flight keep the handler they started under |
| `answer_draft_v2` | GroundedAnswerDraftV2 | first-pass acceptance (answers table) | v1 wire shape only |
| `evidence_pipeline_v2` | relevance assessment gates citability | promoted precision on the curated set | an accepted payload is a record again; candidates never become citable retroactively |
| `router_v2` | backend IntentDecision router | clarification rate, routing corpus | lexical router |

## Order

1. **Migrate.** `alembic upgrade head` through `0015_concurrency_slots`. All
   additive; old binaries ignore the new columns and table.
2. **Deploy readers** with every flag above off. Confirm `/health/ready` and
   `GET /metrics` on the API; run `python -m evals.runner --runtime scripted --trials 3`.
3. **Workers, internal only.** Start the worker pools from
   `devops/compose/external-workers.yaml` (or the equivalent deployment), then
   set `external_worker_mode` on the API. Run the drills in
   `docs/runbooks/worker-drills.md`. Gate G4.
4. **Orchestrator, internal accounts.** `report_orchestrator_v2` on for an
   internal deployment. Build the minimal hERG report and the golden fixtures;
   a v3 artifact must render in the UI, as Markdown and as HTML.
5. **Canary 10%**, by deployment cohort. Compare against the previous week:
   report p50/p95, synthesis first-pass acceptance, stage failure counts,
   cumulative input tokens from the runtime manifest's `prompt_budget`.
6. **50%, then 100%** only if every hard-gate failure in the canary has been
   read, not just counted.
7. **Alpha, at least seven days.** Export the telemetry summary and run
   `python -m evals.gates --telemetry … --manifest … --signoff …`. A `no_go`
   lists what is missing; `insufficient_data` is a no-go.
8. **Remove** each flag and its old path in its own PR once its removal
   condition holds (see `flags.py`), and no later than its `remove_by` date.

## Stop the line

Any one of these ends the rollout step in progress and flips its flag back:

- a numeric or classification value in a published report that differs from its
  observation;
- a report or answer published after a semantic hard gate refused it;
- a duplicate final answer or report for one run;
- `toxagent_run_claims_total{result="recovery_exhausted"}` increasing outside a drill;
- a cancellation that reports terminal while a worker still commits afterwards;
- any non-zero `toxagent_metrics_label_rejections_total` (an id or prose reached a label).

## What this runbook cannot do

It does not replace the alpha, the SME sign-off on semantic fixtures, or the
fault drills against real infrastructure. `evals/gates.py` refuses to report
`go` without them.
