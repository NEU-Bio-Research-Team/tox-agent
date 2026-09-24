# K01 baseline — measured at `7a32b27`

The baseline the audit could not produce, because at `ce49f5d` two control
unit tests failed on a path, the contract suite stopped at fixture setup, and
a combined CI collection ended in 18 errors. Those are fixed (I22–I26, I33);
what follows is what this tree actually does now.

Read this as a record of one measurement, not a release sign-off. A count
belongs to the commit, environment and configuration below and to nothing else.

## Provenance

| | |
|---|---|
| Commit | `7a32b2717f9738e3adb133a3946e880e1777bc58` |
| Working tree | **dirty** — three files deleted by the owner and not committed (`ToxAgent_02_Harness_Architecture.pptx`, `ToxAgent_03_Harness_Master_Plan.pptx`, `audit_5_9.md`); `new_plan.md` untracked |
| Python | 3.10.20 (`drug-tox-env`); OCR uses `toxocr-env` |
| Node | v22.23.2 |
| Predictor manifest | `sha256:9518b2b728b50eff4b8d1182a0b0f58d6f665e8f9d1b35d3b396049ba4608d8e` |
| Predictor contract snapshot | `sha256:ead2ad5ce7bf1d160e76536b1daaa99c8f2031cc27ceb7148585a495b6dbc8d8` |
| Model weights | local `.data/models`, not re-provisioned from a clean download |

No container was started, no paid model called, nothing deployed.

## Results

| Suite | Command (from the service directory) | Result |
|---|---|---|
| Predictor unit + contract | `pytest tests/unit tests/contract` | **178 passed, 5 skipped** |
| Predictor golden | `pytest tests/golden` | **7 passed** |
| Control unit + contract | `pytest tests/unit tests/contract` | **542 passed, 12 skipped** |
| Control non-live full | `pytest tests/unit tests/integration tests/e2e -m 'not postgres and not live_*'` | **462 passed, 4 skipped, 5 deselected, 365s** |
| OCR | `PYTHONPATH=src pytest tests` | **6 passed** |
| Frontend unit | `npm test` | **57 passed, 18 files** |
| Docs | `python devops/scripts/check_docs.py` | **OK** |

The control non-live full run counts differently from unit+contract because
the boundary tests it shares were still empty-parametrized when it ran; both
figures are from real runs and neither should be added to the other.

## Every skip, and why

No skip is counted as a pass. The four that were not real skips at all are now
collection failures (`empty_parameter_set_mark = fail_at_collect`).

| Count | Suite | Reason | Resolved by |
|---|---|---|---|
| 12 | control contract | No OpenCode contract snapshot; needs the pinned server | K09 |
| 5 | predictor unit | Split manifest not built | K10 |
| 5 (deselected) | control | `postgres` / `live_*` markers | K07 gate, CI's `control-integration` job |

## What changed against the audit's numbers

| Measure | Audit at `ce49f5d` | Here | Why |
|---|---|---|---|
| Predictor unit+contract | 151 passed, 26 skipped | 178 passed, 5 skipped | I23: the manifest resolves, so 26 registry-dependent tests run |
| Control unit | 302 passed, 4 skipped, **2 failed** | included in 542 passed | I22: profiles resolve from the package |
| Control contract | Stopped at fixture setup | 12 skipped, rest pass | I24: snapshot located through its package |
| Combined collection | 382 collected, **17 errors** | Not run as one command | I25: per-service rootdirs, 183 + 400 collected |
| Control full non-live | exit 124, incomplete | 462 passed in 365s | Not a hang — 180s was too short; per-test DB setup dominates (19.9s slowest) |
| ADR 0001 boundary guard | reported "skipped" | 158 assertions, all pass | I33: the guard had been inert since the relocation |

Two things the audit predicted, confirmed by making the checks real:

- The predictor contract had genuinely drifted — `model_selection` on
  `PredictionRequest` and `BatchPredictionRequest`. The check had been
  skipping itself. Re-pinned; the surface assertion now names the field.
- The attribution method id in the test doubles (`grad_x_embedding_l2_v1`)
  described unsigned L2 maths the provider no longer does. The served id is
  `grad_x_input_v2`, signed and logit-targeted.

## Still not established

Nothing here is evidence for a clean clone, a wheel install, an image, a
release, or any live provider. Those are K02, K12 and K13. In particular the
golden pass above used artifacts already present on this machine; the
clean-provisioning path is unproven until CI's `scientific-regression` job
runs with `MODEL_ARTIFACTS_URI` set.
