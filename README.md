# ToxPred

Headless toxicity prediction over SMILES: hERG channel blockade and twelve Tox21
assay activities, with ClinTox declared and awaiting its tokenizer.

Each endpoint is a separate measurement with its own field name and its own
threshold. The service never merges them into one verdict, and every probability
arrives with the threshold that labelled it, where that threshold came from, and
the SHA-256 of the weights that produced it.

## Quick start

```bash
conda env create -f environment.yml && conda activate toxpred
uvicorn toxpred.api.app:app --port 8080
```

```bash
curl -s localhost:8080/health/ready

curl -s -X POST localhost:8080/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
```

```json
{
  "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "predictions": {
    "herg": {
      "probability_blocker": 0.0315,
      "label": "non_blocker",
      "threshold": 0.4133453071117401,
      "threshold_source": "artifact",
      "model_id": "herg-tox21-chemberta-v1"
    },
    "tox21": { "task_order_version": "tox21-12task-v1", "assays": { "...": {} } }
  },
  "applicability": { "status": "ok", "method": "element_rules_v1" },
  "provenance": { "artifacts": [{ "weights_sha256": "c851e815…" }] }
}
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | Process liveness. Says nothing about the models |
| GET | `/health/ready` | Every required model loaded and verified |
| GET | `/v1/models` | Model inventory, including why a model is unavailable |
| POST | `/v1/predictions` | One molecule, one or more endpoints |
| POST | `/v1/predictions:batch` | Up to 256 molecules; order preserved, errors per item |
| POST | `/v1/attributions` | Token importance for one endpoint |

Errors are typed: `400 invalid_smiles`, `422` for an unknown request field,
`503 model_not_ready` for an endpoint this build does not serve. An invalid
molecule never comes back as a zero probability.

## Docker

```bash
docker build -f deploy/Dockerfile --build-arg TORCH_VARIANT=cpu -t toxpred .
docker run --rm -p 8080:8080 -v "$PWD/models:/app/models:ro" toxpred
```

The image carries no weights. `deploy/entrypoint.sh` fetches them from
`MODEL_ARTIFACTS_URI` at container start; a request never triggers a download.
Mount `models/` to work offline.

## Tests and benchmark

```bash
pytest                                 # 141 tests
python benchmarks/run_benchmark.py     # frozen split, full metrics
python scripts/check_no_agent_deps.py  # runtime carries no agent/LLM dependency
```

Measured on the frozen scaffold split: hERG AUROC **0.837** [0.821, 0.851] over
2 690 molecules, Tox21 macro AUROC **0.759** over 12 tasks. These reproduce the
metrics recorded when the artifact was trained, to within 7e-5.

**hERG ECE is 0.12.** The probabilities rank molecules well; they are not
calibrated risks. See the [model card](docs/model-card.md).

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, dependency rule, request flow, artifact registry |
| [docs/model-card.md](docs/model-card.md) | Measured performance, intended use, limitations |
| [docs/benchmark-protocol.md](docs/benchmark-protocol.md) | How the numbers are produced |
| [docs/refactor/](docs/refactor/) | What changed from the agent-era codebase and why |

## Scope

This repository is the predictor. The agent layer — orchestration, report chat,
research tools, MolRAG and the web frontend — was removed in the predictor-only
refactor and is recoverable in full from the `archive/agent-layer-*` tag. It is
planned to return in a later phase, built on this scientific kernel rather than
around it.

Model checkpoints and the training code under `backend/` and `scripts/` are kept.
