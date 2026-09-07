# ToxAgent operations runbook

Covers what `TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md` (schema changes) and
`DOCKER_TEST_RUNBOOK.md` (ToxPred's own container/test flow) do not: deploy,
rollback, secret rotation, dependency outages, stuck runs, orphan cleanup,
and backup/restore for the four agentic-layer boundaries (control plane,
ToxPred, toxocr, frontend). Written against what is actually built as of
2026-09-06 (remaining-plan W6-16) — every procedure below cites the real
mechanism it relies on, and every gap is named as a gap, not implied to be
automated when it is a manual step today.

No production topology exists yet (remaining-plan W6-13/14/15 are open) —
this assumes each boundary's image (`*/deploy/Dockerfile`) runs as one or
more replicas behind a load balancer, reachable at the ports each
`HEALTHCHECK` already probes, with PostgreSQL and an object store as
external managed dependencies. Adjust host/port/orchestrator specifics to
whatever topology W6-13 eventually settles on; the sequencing and mechanism
references below do not change with the orchestrator.

## 1. Deploy

1. Confirm CI is green on the exact commit being deployed: `static`,
   `control-plane`, `postgres-migrations`, `toxocr`, `frontend`, and all
   three `*-container` jobs (`.github/workflows/ci.yml`).
2. Migrate first, deploy second — `TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md`'s
   full procedure, including its backup/restore-drill precondition. Never
   let a new binary's first replica run `alembic upgrade head` as a side
   effect of starting (`toxagent-control/deploy/entrypoint.sh` does this by
   default; set `TOXAGENT_SKIP_MIGRATIONS=1` on every replica once a
   dedicated migration step has already run it once, so N replicas don't
   race the same revision).
3. Roll out control-plane and toxocr replicas behind health checks:
   `/health/live` must be `200` before a replica joins the pool;
   `/health/ready` reflects real dependency state (predictor reachability,
   runtime health) — a load balancer that only checks `/health/live` will
   route traffic to a replica that is up but not ready, so gate on
   `/health/ready` for control-plane specifically.
4. Roll out frontend last (nginx serving the new build) — it has no
   backward-compatibility concern with the API beyond whatever the API
   itself guarantees within the compatibility window the migration runbook
   already requires.
5. Smoke: the three `*-container` CI jobs already prove per-image
   build+boot+one real request; after a real deploy, repeat the same three
   checks (`/health/ready`, one real prediction, one real OCR recognition,
   one real frontend page load) against the actual deployed replicas, not
   just the CI-built image — a deploy step (wrong env var, wrong secret,
   wrong network policy) can break a service CI proved works in isolation.

## 2. Rollback

- **Frontend, toxocr:** stateless — redeploy the previous image tag. No
  coordination needed beyond confirming the previous tag is still pullable.
- **Control plane:** redeploy the previous image tag *only if* no migration
  has run since that version was last live. If a migration ran, forward-only
  policy (migration runbook) means rollback is **not** "redeploy the old
  binary" — it is either a forward-compatible hotfix on the new schema, or a
  restore-from-backup into an isolated environment while the incident is
  triaged. Never point an old binary at a database a newer migration has
  already touched; the migration runbook's expand/migrate/contract
  discipline exists specifically so this is rare, not to make rollback
  itself safe to skip thinking about.
- After any rollback: check `/health/ready` on every replica, and check for
  runs stuck in `queued`/`running` from the moment traffic stopped reaching
  the old version — reconcile per §5 below rather than assuming they will
  resolve on their own.

## 3. Rotate `TOXAGENT_CAPABILITY_SECRET`

**Known limitation, not yet closed:** `tools/capability.py`'s
`CapabilityTokenService` holds exactly one secret per running process —
there is no dual-key/grace-period verification the way OIDC key rotation
usually works. A capability token is short-lived
(`capability_ttl_s` default 900s, `capability_grace_s` default 60s), but
restarting a control-plane replica with a new secret invalidates every
capability token already issued to a runtime for a run still in flight —
the runtime's next MCP call fails auth mid-turn, indistinguishable from a
transport-level auth attack to that run's client.

Until dual-secret verification is implemented, rotate during a maintenance
window:

1. Stop admitting new runs (drain at the load balancer or via a maintenance
   flag — no in-repo mechanism does this today; it must be a deploy-layer
   action).
2. Wait out `capability_ttl_s + capability_grace_s` (default: 16 minutes)
   past the last new run's start, or confirm via `GET /v1/sessions/{id}`
   that no run is `queued`/`running` fleet-wide.
3. Update the secret in the secret manager, restart every control-plane
   replica so all of them pick up the new value simultaneously (a rolling
   restart here means some replicas validate against the old secret and
   some against the new one — a runtime bound to a "new-secret" replica
   whose MCP call round-robins to an "old-secret" replica fails).
4. Resume admitting new runs.

A rotation forced by a suspected leak (not a scheduled rotation) cannot wait
for step 2 — accept that in-flight runs at rotation time fail with a typed
auth error (never silently), and reconcile them per §5.

## 4. Dependency outage

Every dependency outage should already produce a typed, honest answer
instead of a hang or a silent wrong result — this section says which
mechanism to check when triaging one, not new behavior to add.

| Dependency down | What the product already does | Where to look |
|---|---|---|
| ToxPred (predictor) | `/health/ready` reports `predictor.ready: false`; new analysis requests fail typed, not silently degrade | control plane logs, ToxPred's own `/health/ready` |
| OpenCode / runtime host | `AgentRuntimeGateway._probe_health_with_retries` retries bounded, then the run fails `runtime_unavailable`; a run mid-turn when the runtime drops fails `runtime_unavailable` too and a recovery run is created, never a silent reconnect (progress log §3.8/§3.11) | `/health/ready`'s `runtime` block; `runtime_binding`/`recovery_of_run_id` on the affected run |
| EuropePMC (evidence provider) | Circuit breaker (W3-06, `research/providers/europepmc.py`) opens after repeated failures; `search_toxicology_evidence` fails typed rather than hanging or retrying indefinitely against a provider that is down | evidence-related run failures' `failure_code`; provider circuit-breaker state (in-process, resets on control-plane restart) |
| toxocr | `capability_unavailable` if unconfigured; a real outage while configured fails the specific run with `structure_recognition_failed` (`application/recognize_structure.py`), never a queued run nothing ever completes | `structure_recognition` capability in `/health/ready`; the specific run's message |
| PostgreSQL | Everything fails — this is the one true single point of failure; there is no deterministic-lane fallback to SQLite in production | connection pool errors in control-plane logs; PostgreSQL's own monitoring |

None of these outages should be "fixed" by restarting the control plane as a
first response — restarting clears in-memory circuit-breaker state and can
mask a still-down dependency as freshly-recovered for one request before
failing again. Confirm the dependency is actually back via its own health
signal before declaring the incident over.

## 5. Stuck runs

A run stuck in `queued`/`running` after its owning process crashed (control
plane killed mid-run, not a runtime-side failure) is reconciled automatically
on the next control-plane startup:
`application/startup_reconciliation.py::reconcile_orphaned_runs` closes
every run in a non-terminal state to `failed`/`cancelled` with a typed audit
event — a clean shutdown drains all in-flight work first, so anything this
function finds at startup is provably orphaned, not merely slow (progress
log's audit finding A05).

If a run appears stuck **without** a control-plane restart having happened
(the reconciler only runs at startup), that is not yet a case this repo
handles automatically — check the run's `deadline_at`
(`config.py::PolicySettings.run_deadline_s`, default 300s) and whether the
scheduler's task for it is still alive; a `DeadlineExceeded` should fire on
its own once the deadline passes. A run past its deadline that has not
terminated is a bug to investigate (task leaked from the scheduler's
tracking), not something to manually force-close first — closing it without
understanding why the deadline didn't fire hides the underlying defect.

## 6. Orphan process/workspace cleanup

**Not yet automated (remaining-plan W2-11 is open) — manual today.** A
runtime-host crash (OpenCode killed by the OS, container OOM-killed, host
rebooted) can leave a per-run OpenCode workspace directory
(`TOXAGENT_OPENCODE_DIRECTORY`'s child directories) or a runtime process
without a corresponding live control-plane run to reap it.

Interim manual procedure:

1. List workspace child directories under `TOXAGENT_OPENCODE_DIRECTORY`
   older than the longest plausible run (`run_deadline_s` +
   `structure_recognition_deadline_s`'s margin, or simpler: older than a few
   hours).
2. Cross-check each directory's run id (if the naming convention includes
   one) against `GET /v1/sessions/{id}/runs/{run_id}` — a directory whose
   run is already terminal (`completed`/`failed`/`cancelled`) is safe to
   delete.
3. For orphaned OpenCode/DSH processes: match against currently-bound
   `runtime_binding_id`s the control plane reports as active; a process with
   no matching active binding is safe to kill. Do this by PID, never by a
   broad pattern match — progress log §3.11 already hit a real incident
   where `pkill -f "opencode serve"` matched the launcher script's own
   command line and killed the wrong thing.

W2-11's eventual fix should make this a scheduled job cross-referencing the
same data this manual procedure uses, with a soak test asserting the orphan
count converges to zero — not a new mechanism, just automating this exact
check.

## 7. Backup and restore

- **PostgreSQL:** managed provider's own point-in-time backup, on the
  schedule DEC-04 (retention policy) settles once decided. Restore drill
  procedure is in `TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md`'s own
  precondition section — run it into an isolated environment, verify
  Alembic's revision matches expectations, and only then consider the
  backup validated.
- **Object store (W4-06 `ObjectStore` interface, attachments/raw evidence):**
  whatever the eventual production adapter's own backend backup mechanism
  is (versioning/replication on the object storage service) — this repo's
  `InMemoryObjectStore`/filesystem implementations are test-only and carry
  no backup story of their own by design.
- **Restoring both together:** a database restore and an object-store
  restore must target the **same point in time** — a `raw_payload_ref`
  pointing at an object that a database restore resurrects but the object
  store's own restore does not (or vice versa) produces a dangling
  reference. No automated consistency check across the two exists yet;
  after any combined restore, spot-check a sample of `raw_payload_ref`s
  actually resolve before declaring the restore complete.

## 8. What this runbook does not cover

Container build/publish itself (`.github/workflows/ci.yml`'s `*-container`
jobs already do this — this runbook starts from "an image exists and passed
its container smoke"), the real deploy topology and its secrets/network
policy (W6-13/14/15, blocked on infrastructure decisions this repo cannot
make on its own), and SLO/alerting thresholds (W9-08, needs alpha telemetry
data that does not exist yet).
