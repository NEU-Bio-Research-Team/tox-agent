# Workspace layout

Product source is organised around deployable boundaries (ADR 0001, 0006); the
layout inside each is described in that service's README.

```text
frontend/                 browser product — grouped by feature (frontend/README.md)
backend/control/          control plane and agent harness — layered, features inside
                          each layer (backend/control/README.md)
backend/predictor/        predictor service, registry, benchmark, research
backend/ocr/              structure-recognition service (backend/ocr/README.md)
devops/
  compose/                Docker Compose topology and its overlays
  cloud/                  Cloud Build
  checks/                 docs and workspace checks CI runs
  release/                handoff surface, release manifest, deploy-target resolution
  ops/                    backup restore drill
  agent-runtime/          local OpenCode runtime and its Docker bridge (bin/toxagent)
  tests/                  tests for all of the above and for bin/toxagent
docs/                     tutorials, how-to, reference, explanation, ADRs, results,
                          internal records (docs/README.md)
bin/toxagent              operator CLI: setup, doctor, up, smoke, backup, restore
Makefile                  developer entry point: lint, typecheck, test, check, lock
ruff.toml                 one lint configuration for every Python tree
```

Each Python service declares its dependencies once, in its `pyproject.toml`,
and pins the resolved set in `requirements.lock` (`make lock`); the images
install from the lock. torch is the exception: each Dockerfile installs it from
the PyTorch index for its `TORCH_VARIANT` before the lock.

## Evaluation code, evidence and scratch output

`backend/control/evals/` holds the agent evaluation: code and inputs (runner,
graders, packs, tasks, fixtures, schemas), **committed evidence** that a
document or the lab grading cites (`manifests/<run>/`, `investigation/runs/`,
`experiments/runs/`, `external/scifact/runs/`), and `results/`, the git-ignored
default output of every run. `evals/scripts/` holds the TAB-Suite lane scripts.

`backend/predictor/evals/benchmark/` holds the frozen benchmark assets and the
runner CI uses.

## Model binaries

Model binaries are deliberately not source files. Local development uses
`.data/models/` (or `TOXPRED_MODELS_HOST_PATH`); deployed environments provide
the same content through an immutable volume or object-store provisioning.
The reviewed manifests in `backend/predictor/registry/` remain the authority
for identity, checksum and admission status.

## Research code

Offline training and legacy recovery code lives under
`backend/predictor/research/`, with its own environment
(`research/environment.yml`). It is not packaged into, or imported by, the
predictor service, and nothing outside it imports `research.legacy_backend`:

```text
research/legacy_backend/  the pre-refactor model code the training scripts use
research/scripts/         training and prediction scripts
research/configs/         training configurations (the two the service reads,
                          smilesgnn_config.yaml and workspace_mode.yaml, stay in
                          backend/predictor/configs/)
research/evals/           evaluation scripts for research models
research/benchmark/       generators of the frozen benchmark assets: the split
                          manifest and the golden baseline
```

Run a research module with:

```bash
PYTHONPATH=backend/predictor python -m research.scripts.train_tox21_gatv2
```
