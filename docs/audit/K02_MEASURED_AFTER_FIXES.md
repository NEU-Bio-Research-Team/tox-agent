# Measured after the I07–I31 fixes — at `f830e3b`

A second measurement in the same shape as [K01_BASELINE.md](K01_BASELINE.md),
taken after the remaining issues in
[SYSTEM_ISSUES_VI.md](SYSTEM_ISSUES_VI.md) were worked through. Like that one,
this is a record of one measurement on one machine, not a release sign-off.

## Provenance

| | |
|---|---|
| Commit | `f830e3b` |
| Working tree | **dirty** — three files deleted by the owner and not committed (`ToxAgent_02_Harness_Architecture.pptx`, `ToxAgent_03_Harness_Master_Plan.pptx`, `audit_5_9.md`); `new_plan.md` untracked |
| Python | 3.10.20 (`drug-tox-env`); OCR uses `toxocr-env` |
| Node | v22.23.2 |
| Predictor manifest | `sha256:9518b2b728b50eff4b8d1182a0b0f58d6f665e8f9d1b35d3b396049ba4608d8e` (unchanged from K01) |
| Predictor contract snapshot | `sha256:128ee4431644b3c50f3f7d815b18fdf8f3da04d78bb19effb34e2db7b93c6016` (changed at `d18b8f4`, after the K01 baseline) |
| Model weights | local `.data/models`, not re-provisioned from a clean download |

No container was started, no paid model called, nothing deployed.

## Results

| Suite | Command (from the service directory) | K01 at `7a32b27` | Here |
|---|---|---|---|
| Predictor unit + contract | `pytest tests/unit tests/contract` | 178 passed, 5 skipped | **190 passed, 5 skipped** |
| Predictor golden | `pytest tests/golden` | 7 passed | **7 passed** |
| Control unit + contract | `pytest tests/unit tests/contract` | 542 passed, 12 skipped | **626 passed, 12 skipped** |
| Control non-live full | `pytest tests/unit tests/integration tests/e2e -m 'not postgres and not live_'` | 462 passed, 4 skipped | **743 passed, 5 deselected, 113s** |
| OCR | `PYTHONPATH=src pytest tests` | 6 passed | **6 passed** |
| Frontend unit | `npm test` | 57 passed | **102 passed, 21 files** |
| devops wrapper + deploy targets | `pytest devops/tests` | (8) | **14 passed** |
| Docs | `python devops/scripts/check_docs.py` | OK | **OK** |

The 546 golden values across 42 molecules at tolerance 1e-6 are unchanged; no
commit in this range touched numerical policy.

## Every skip, and why

| Count | Suite | Reason | Resolved by |
|---|---|---|---|
| 12 | control contract | No OpenCode contract snapshot; needs the pinned server | K09 |
| 5 | predictor unit | Split manifest not built | K10 |
| 5 (deselected) | control | `postgres` / `live_*` markers | K07 gate, CI's `control-integration` job |

## What these numbers do not establish

Unchanged from K01, and worth restating because more code now depends on it:

- No clean clone, wheel install, image build, release or live provider. K02,
  K12, K13.
- The golden pass used artifacts already on this machine. Clean provisioning is
  unproven until CI's `scientific-regression` job runs with
  `MODEL_ARTIFACTS_URI` set.
- The lease, adoption and cross-instance cancel behaviour (I17/I18) is tested
  against SQLite with two schedulers in one process. That models two workers
  faithfully — their task maps are genuinely separate — but it is not a
  two-replica PostgreSQL run, which is the `postgres`-marked gate.
- No deployment ran from the new `deploy.yml`, no staging smoke, no rollback
  drill. I30's remaining acceptance needs the cloud project.
- ClinTox stays blocked, and remains blocked on an owner decision: recover the
  original tokenizer with its hashes, or fund a `clintox-smilesgnn-v2` retrain.
