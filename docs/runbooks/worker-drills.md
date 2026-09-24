# External workers: topology, caps and drills

WS08 / PR-14, PR-15 of the 2026-09-13 remediation plan. What the worker split
changes, what the automated suite proves, and the drills that must be run
against a real deployment before gate G4 is signed.

## Topology

| Process | `TOXAGENT_PROCESS_ROLE` | Executes runs | Serves HTTP | Migrates |
|---|---|---|---|---|
| API replica | `api` | no | yes | yes (entrypoint) |
| Worker | `worker` | yes, from `TOXAGENT_WORKER_QUEUES` | no | no (`TOXAGENT_SKIP_MIGRATIONS=1` default) |
| Development | `all` | yes | yes | yes |

Everything above requires `TOXAGENT_FLAG_EXTERNAL_WORKER_MODE=1`. With the flag
off, every process executes the runs it accepts — the behaviour before PR-14 —
and the role is ignored. `python -m toxagent.worker` refuses to start with the
flag off.

Compose: `devops/compose/external-workers.yaml` layers one API, an interactive
pool (`interactive,deterministic`) and a report pool (`report`).

### Queues

| Queue | Intents | Priority |
|---|---|---|
| `interactive` | report_qa, attribution, evidence_research | 0 |
| `deterministic` | analysis, analysis_batch, structure_recognition | 10 |
| `report` | build_report | 20 |

A job written before migration 0014 has no `queue_name`; its queue is derived
from the intent in its envelope.

## Caps

Held in `concurrency_slots`, so they bind across every worker. A run takes all
the slots it needs in a fixed order — global, queue, provider, tenant — or none
of them; a refused run goes back to the queue with `available_at` set
`TOXAGENT_QUOTA_RETRY_S` ahead, and that claim is not counted as an attempt.

| Variable | Scope | 0 means |
|---|---|---|
| `TOXAGENT_GLOBAL_MAX_RUNS` | every run | no cap |
| `TOXAGENT_REPORT_MAX_RUNS` | the report queue | no cap |
| `TOXAGENT_PROVIDER_MAX_RUNS` | one AI profile (not deterministic runs) | no cap |
| `TOXAGENT_TENANT_MAX_RUNS` | one owner | no cap |

Caps are enforced only in external-worker mode. In-process mode keeps the
per-session admission cap and nothing else, rather than a cap half of its runs
would bypass.

`GET /v1/sessions/{session}/runs/{run}/queue` shows a waiting run's queue,
state (`waiting`, `deferred`, `claimed`, `not_queued`), a position estimate,
`retry_after_s` when deferred, and which cap scope it is waiting on.

## Shutdown and recovery

On SIGTERM a worker stops claiming, gives in-flight runs
`TOXAGENT_DRAIN_GRACE_S` to finish, then hands the rest back to the queue.
A handed-off run is **not** cancelled: another worker adopts it and executes it
from its envelope. The first worker's runtime turn is aborted and its binding
recorded as lost; the run is marked `potentially_billed` if the provider had
accepted the turn.

A run may execute `TOXAGENT_MAX_RUN_ATTEMPTS` times (default 2: the first, and
one recovery generation). A job past that bound fails its run with
`recovery_exhausted` instead of circulating.

Set the container stop timeout above the drain grace (`stop_grace_period: 30s`
against a 25 s grace in the overlay), or the orchestrator kills the worker
before it can hand anything off — which is still safe, because the lease
expires, but costs one `LEASE_TTL_S` (30 s) of delay.

## What the automated suite proves

| Property | Test |
|---|---|
| An API process writes unowned jobs and executes nothing | `test_external_workers::test_an_api_scheduler_writes_an_unowned_job_and_executes_nothing` |
| Workers claim only their queues | `…::test_a_worker_claims_only_the_queues_it_serves` |
| Two workers racing: each run executed once | `…::test_two_workers_sweeping_at_once_execute_each_run_exactly_once` |
| A saturated report pool does not hold up a question | `…::test_a_saturated_report_worker_does_not_hold_up_a_question` |
| A queued run cancels without a worker | `…::test_cancelling_a_queued_job_settles_it_without_a_worker` |
| API and worker as two composition roots over one database | `…::test_an_api_process_and_a_worker_process_over_one_database` |
| Global cap across three workers | `test_worker_quotas_and_drain::test_a_global_cap_holds_across_three_workers` |
| Tenant cap; a light tenant is not starved | `…::test_a_tenant_cap_keeps_one_owner_from_occupying_the_fleet` |
| Racing for the last slot grants exactly the limit | `…::test_racing_for_the_last_slots_grants_exactly_the_limit` |
| Drain hands off, does not cancel | `…::test_a_draining_worker_hands_its_run_on_instead_of_cancelling_it` |
| Recovery bounded to one generation | `…::test_recovery_is_bounded_to_one_generation` |
| A killed worker's lease and slot are reclaimed | `…::test_a_dead_workers_lease_and_slot_are_reclaimed_by_the_next` |

These run on SQLite in the default suite and on PostgreSQL when
`TOXAGENT_TEST_DATABASE_URL` is set. SQLite serialises writers, so it cannot
exhibit a PostgreSQL read-committed race; the slot store's mutual exclusion is
the primary key (`ON CONFLICT DO NOTHING`) and a conditional update, neither of
which depends on isolation level, but the PostgreSQL job is the one that
demonstrates it.

## Drills still to run on a real deployment

None of these can be proven in-process. Record date, commit, result and the
run ids examined.

1. **Kill -9 a worker mid-report.** `docker kill -s KILL` the report worker
   after `report.stage_changed synthesizing running`. Expect: the run is adopted
   within ~30 s, completed stages are not re-run (check `stage_checkpoints`
   attempts), `potentially_billed=true` on the run.
2. **Rolling restart of the API.** Restart every API replica while five
   questions are in flight. Expect: no run cancelled; no `run.cancelled` events.
3. **SIGTERM a worker with a long run.** Expect `worker_draining` on the job, the
   run adopted by the other pool member, one `recovery` attempt in `attempts`.
4. **Provider cap under burst.** Submit 30 questions with
   `TOXAGENT_PROVIDER_MAX_RUNS=4`. Expect peak concurrent bindings for that
   profile ≤ 4 and `quota_wait:provider` visible on the queue route.
5. **Cancel through a different replica.** Cancel a claimed run through an API
   replica that did not accept it. Expect `cancellation_relayed_to_owning_worker`
   and a terminal state within `CANCEL_POLL_S` + the runtime abort.
6. **Database failover.** Fail over PostgreSQL during a load of 20 runs. Expect
   sweeps to log and retry, no run left `running` without a job row once the
   database is back.
7. **Connection budget.** With N API replicas and M workers, confirm
   `(N + M) × (pool_size + max_overflow)` is under the server's
   `max_connections` minus the admin reserve.
