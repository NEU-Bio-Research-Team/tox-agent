# Metric dictionary

WS12 / PR-18. Every metric the control plane emits, declared once in
`backend/control/src/toxagent/platform/metrics.py` and served at `GET /metrics`
(Prometheus text format). `backend/control/tests/unit/test_metrics.py` fails
when this table and the code disagree.

**Labels never carry** a SMILES, prose, a URL, or an owner, session, run,
report or evidence id. The registry replaces a value that looks like one with
`invalid` and counts it in `toxagent_metrics_label_rejections_total`; a non-zero
value there is a bug at a call site, not a property of traffic.

Metrics are per process. An API replica and a worker each expose their own;
aggregation is the scraper's job.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `toxagent_run_claims_total` | counter | `queue`, `result` | A worker's decision on a claimable job: `claimed`, `deferred` (a concurrency cap was full), `cancelled_before_execution`, `recovery_exhausted`. |
| `toxagent_concurrency_refusals_total` | counter | `scope` | Claims refused a slot, by the scope whose cap was full: `global`, `queue`, `provider`, `tenant`. |
| `toxagent_runs_finished_total` | counter | `intent`, `queue`, `outcome` | Executions that ended on this worker. `outcome` is `completed`, `failed`, `cancelled`, `handed_off` or `taken_over`. |
| `toxagent_run_duration_seconds` | histogram | `queue`, `outcome` | Wall time a worker spent executing a run. Excludes queue wait. |
| `toxagent_run_queue_wait_seconds` | histogram | `queue` | Time from a job being written to its first claim. Adoptions are excluded: their wait includes the previous execution. |
| `toxagent_report_stage_seconds` | histogram | `stage`, `status` | One report stage attempt, from its `running` event to its settled one (`completed`, `skipped`, `failed`). |
| `toxagent_report_synthesis_submissions_total` | counter | `attempt`, `outcome` | Synthesis submissions judged. First-pass acceptance is `attempt="1",outcome="accepted"` over all `attempt="1"`. |
| `toxagent_report_synthesis_violations_total` | counter | `violation_code` | Typed violations in refused submissions. The code set is closed, so this stays low-cardinality. |
| `toxagent_worker_handoffs_total` | counter | `reason` | Runs a worker gave back instead of finishing. `drain` is a shutdown that outlasted its grace window. |

## What the G5 gate reads from these

`backend/control/evals/gates.py` evaluates an alpha telemetry summary. The
summary is exported from the metrics store for the alpha window; these are the
queries behind the fields it needs from this dictionary.

| G5 field | Source |
|---|---|
| `report_latency_p50_s`, `_p95_s` | `histogram_quantile` over `toxagent_run_duration_seconds{queue="report",outcome="completed"}` plus `toxagent_run_queue_wait_seconds{queue="report"}` |
| `qa_latency_p50_s`, `_p95_s` | the same over `queue="interactive"` |
| `first_pass_acceptance` (reports) | `toxagent_report_synthesis_submissions_total` as above |
| `happy_path_report_tool_failures` | `toxagent_report_stage_seconds_count{status="failed"}` on builds that completed |

The Q&A first-pass rate, usage completeness, duplicate usage and evidence
precision come from the database (`answers`, `runtime_usage_events`,
`evidence_records`) and the curated eval, not from these counters: they are
properties of stored rows, and a counter that restated them would be a second
source that could drift.

## Alert thresholds

None are locked. The plan requires at least seven days of internal alpha
telemetry before a threshold is set; an alert tuned before then is tuned to a
guess. Each alert, when added, links its runbook and the rollout flag that
rolls it back (`docs/runbooks/worker-drills.md`, `toxagent.platform.flags`).
