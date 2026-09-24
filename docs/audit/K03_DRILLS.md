# Drills that were run, and what they establish — at `b3361cd`

[K01_BASELINE.md](K01_BASELINE.md) and [K02_MEASURED_AFTER_FIXES.md](K02_MEASURED_AFTER_FIXES.md)
both say, in their own words, that they measured test suites and nothing else:
no container was started, so the deployment-shaped closing criteria of I16,
I17 and I18 stayed open however green the unit tests were. This records the
drills that have now been run, on this machine, with real containers.

It is still not a release sign-off. Everything here is local Docker on one
host; none of it is a hosted deployment, a staging environment, or a
multi-host cluster.

## Provenance

| | |
|---|---|
| Commit | `b3361cd` (plus the drill scripts this document describes) |
| Docker | Engine on WSL2, `postgres:16` (resolved `16.15`, Debian) |
| Python | 3.10.20 (`drug-tox-env`), with `asyncpg` 0.31.0 and `psycopg` 3.3.5 |
| Ports | 55432 (suite), 55440/55441 (drill) — deliberately not the stack's |

## 1. Forward-only migration from an empty database

```
alembic upgrade head
  0001_baseline → 0002_runtime_usage_events → 0003_session_titles
  → 0004_investigation_kernel → 0005_session_configuration
  → 0006_model_conn_name → 0007_run_job_leases → 0008_explanation_ckpt
```

Ran clean on an empty PostgreSQL 16.15. This is I34's closing criterion, and
it now covers 0007 and 0008 as well — both were added after the issue was
written and both carry the same guard.

## 2. The whole control suite against migrated PostgreSQL

`TOXAGENT_TEST_DATABASE_URL` points every test at the migrated database rather
than at a temporary SQLite file, which is what the CI job intended and never
did.

| | Before this session | Now |
|---|---|---|
| integration + e2e on PostgreSQL | never ran — both `postgres`-marked files skipped | **200 passed, 0 skipped** |
| integration + e2e on SQLite | 197 passed | 200 passed |
| unit + contract + integration + e2e (SQLite) | 818 passed, 17 skipped | 833 passed, 17 skipped |

The four two-replica admission tests and the seven lease/fencing/adoption
tests of I17/I18 are inside that 200. They now run against real PostgreSQL,
which is the closing criterion the audit recorded as outstanding.

**It found a real defect.** `try_reserve` reserved a tool call with one
`INSERT ... SELECT ... WHERE count < budget` and claimed that was atomic
enough. It is on SQLite, whose database-level write lock serializes writers;
under READ COMMITTED it is not, and five concurrent calls against a budget of
two were all admitted. Fixed by locking the parent run first, as session
admission already did.

## 3. Backup, restore onto a different instance, and container recreate

`devops/scripts/restore_drill.sh`. Two PostgreSQL containers: a source
migrated from empty and populated through the application's own repositories,
and a target that has never seen this application.

```
drill: backup is 5941 bytes
drill: restoring into an instance that was never the source
verify: session=ses_670128c8… owner=drill-owner messages=1 runs=1 events=1
drill: state volume 'toxagent-state' at /var/lib/toxagent covers
       /var/lib/toxagent/attachments and /var/lib/toxagent/model-secrets
drill: PASSED
```

What each part is for:

- **A different instance.** `toxagent backup | toxagent restore` on one
  database proves neither that the dump is complete nor that it can be read by
  an instance that was never the source. The target here is a separate
  container.
- **Through the repositories.** The seed writes a session, a message with a
  text part, a run and an outbox event using the application's own mapping,
  and reads them back the same way after the restore. Raw `INSERT`s would have
  proved PostgreSQL can copy rows, which was not in doubt.
- **Row counts per table, plus `alembic_version`.** A restore that loses the
  schema version leaves an instance that would migrate itself again.
- **Container recreate.** A second container mounts the same named volume and
  finds the attachment bytes, the credential, and the credential's `0600`
  mode. The volume name, its mount path and both directories are read out of
  `compose.yaml`, so renaming one in the deployment and not in the drill fails
  the drill rather than testing a path nothing uses.

The comparison was checked against a deliberately damaged target — one row
deleted after the restore — and the drill exits 1 naming the table:

```
drill: FAILED — restored contents differ
22c22
< runs=1
---
> runs=0
```

## What these drills still do not establish

- No hosted deployment, no staging environment, no rollback to a previous
  image digest. I30's remaining half needs a cloud project.
- Two schedulers against one PostgreSQL is not two hosts. Network partition,
  clock skew between replicas and the W2-09 failure-injection orchestrator are
  untested.
- The control image was not built or started here. The recreate drill exercises
  the volume contract read from `compose.yaml`, not the running service writing
  through it.
- No paid model was called, and no live provider matrix was run.
