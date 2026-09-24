# Workspace layout

Product source is organised around deployable boundaries:

```text
frontend/                 browser product
backend/control/          control plane and agent harness
backend/predictor/        predictor service, registry, evaluations, research
backend/ocr/              structure-recognition service
devops/                   Compose, cloud and operational scripts
docs/                     product and engineering documentation
```

Model binaries are deliberately not source files. Local development uses
`.data/models/` (or `TOXPRED_MODELS_HOST_PATH`); deployed environments provide
the same content through an immutable volume or object-store provisioning.
The reviewed manifests in `backend/predictor/registry/` remain the authority
for identity, checksum and admission status.

Offline training and legacy recovery code lives under
`backend/predictor/research/`. It is not packaged into, or imported by, the
predictor service. Run a research module with:

```bash
PYTHONPATH=backend/predictor python -m research.scripts.train_tox21_gatv2
```

Frozen benchmark assets and runners live at
`backend/predictor/evals/benchmark/`.
