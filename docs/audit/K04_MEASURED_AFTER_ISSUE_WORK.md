# Measured after the K05–K13 issue work

A third measurement in the shape of [K01_BASELINE.md](K01_BASELINE.md) and
[K02_MEASURED_AFTER_FIXES.md](K02_MEASURED_AFTER_FIXES.md), taken after working
through the open GitHub issues. The drills that back it are in
[K03_DRILLS.md](K03_DRILLS.md).

Like the other two, this is one measurement on one machine, not a release
sign-off.

## Provenance

| | |
|---|---|
| Working tree | dirty in the same way as K02 — three files deleted by the owner and not committed; `new_plan.md` untracked |
| Python | 3.10.20 (`drug-tox-env`), now with `asyncpg` 0.31.0 and `psycopg` 3.3.5 |
| Node | v22.23.2 |
| PostgreSQL | 16.15 in Docker, migrated from empty by `alembic upgrade head` |
| Model weights | local `.data/models`, not re-provisioned from a clean download |

## Results

| Suite | K01 | K02 | Here |
|---|---|---|---|
| Predictor unit + contract | 178 passed, 5 skipped | 190 passed, 5 skipped | **202 passed, 5 skipped** |
| Predictor golden | 7 passed | 7 passed | **7 passed** |
| Control unit + contract | 542 passed, 12 skipped | 626 passed, 12 skipped | **660 passed, 12 skipped** |
| Control full non-live (SQLite) | 462 passed, 4 skipped | 743 passed | **900 passed, 17 skipped** |
| **Control integration + e2e on PostgreSQL** | never ran | never ran | **212 passed, 0 skipped** |
| OCR | 6 passed | 6 passed | **6 passed** |
| Frontend unit | 57 passed | 102 passed | **133 passed, 24 files** |
| devops | (8) | 14 passed | **96 passed** |
| Docs | OK | OK | **OK** |
| Handoff surface | — | — | **757 included, 40 withheld, 0 undecided** |

The 546 golden values across 42 molecules at tolerance 1e-6 are unchanged. The
`.` fix in `token_structure_align.py` touches the explanation path, not the
numerical one, and the golden suite confirms it.

## What changed about what these numbers mean

The PostgreSQL row is the significant one. Both K01 and K02 said the
`postgres`-marked tests were a gate that had not run. It turns out they had
never run **in CI either**: the job started a PostgreSQL service and then ran
every test against a temporary SQLite file beside it, because the suite reads
`TOXAGENT_TEST_DATABASE_URL` and the job set two other names.

Turning it on found a real defect immediately — `try_reserve` admitted five
concurrent tool calls against a budget of two, because one
`INSERT ... SELECT ... WHERE count < budget` is a serialization point on
SQLite's database-level write lock and not under READ COMMITTED.

## Defects found by running things rather than by reading them

| Found by | Defect |
|---|---|
| The PostgreSQL gate | Tool-call budget admitted 5 concurrent calls against a limit of 2 |
| Reviewing the auth path | `build_auth` used the capability signing key to decide user identity, checking neither issuer nor audience |
| The XAI benchmark | `/v1/explanations` failed for every salt: the SMILES walk had no case for `.` |
| The release manifest | The predictor image, alone of the three, was built from a moving tag |
| The handoff checker | 14 shipped documents linked to withheld ones; `backend/ocr/README.md` had been broken since the relocation |
| The guard registry | I13 was closed with no test guarding it; K09's "not promoted" was prose only |
| The compare admission check | The existing test compared two models the catalogue had never heard of and asserted success |

## What these numbers still do not establish

- No clean clone, no wheel install into a fresh environment, no image build for
  control or OCR, no release, no live provider.
- Two schedulers against one PostgreSQL is not two hosts. Network partition,
  clock skew and the W2-09 failure-injection orchestrator are untested.
- No hosted deployment, no staging smoke, no rollback to a previous digest, and
  no OIDC flow against a real identity provider.
- The eval suite executes 6 of 50 tasks. The other 44 need a provider
  credential; the manifest says so rather than reporting a pass rate over the
  six as if it were the suite.
- ClinTox stays blocked on the owner's decision about the tokenizer.
- XAI faithfulness measured at 15/34 and 14/35 against a random control, which
  is no better than chance. That is a recorded negative result, not a gate.
