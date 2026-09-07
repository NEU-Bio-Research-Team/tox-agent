# ToxPred Docker Test Runbook

## 1) Scope and history

This runbook builds and smoke-tests **ToxPred**, the predictor service, in
Docker. It replaces an earlier version of this file (remaining-plan W6-17)
that documented `model_server/main.py`'s `/analyze` endpoint and a
`final_verdict` field — both removed by the predictor-only rebuild
(`docs/refactor/PREDICTOR_ONLY_STATUS_VI.md`; ADR 0002 in
`toxagent-control/docs/adr/` is the standing rule that no layer computes an
aggregate toxicity/safety verdict). Every command and response shape below
was run against this repository's real, current service before being written
down, not carried over from the old version.

Three other deployables now have their own Dockerfiles and CI smoke jobs
(remaining-plan W6-10): `toxagent-control/deploy/Dockerfile` +
`control-plane-container`, `toxocr/deploy/Dockerfile` + `toxocr-container`,
and `frontend/deploy/Dockerfile` + `frontend-container`. They deliberately
have different build contexts, dependencies, readiness semantics and smoke
criteria. This runbook remains intentionally scoped to ToxPred; do not
combine the boundaries into one image or copy ToxPred's model-mount procedure
to the other services.

Repository root: wherever you cloned `tox-agent` — no absolute path is
assumed.

## 2) Prerequisites

- Docker installed and running.
- Model artifacts. Two ways to get them into the container:
  - **Mounted** (fastest for local iteration): `models/` already populated in
    your checkout, mounted read-only at `/app/models`.
  - **Fetched at startup**: set `MODEL_ARTIFACTS_URI` to a source
    `deploy/entrypoint.sh` can pull from; the image itself carries no
    weights (`.dockerignore` excludes `models/`).
- For a GPU build: an NVIDIA driver (`nvidia-smi` works) and the NVIDIA
  Container Toolkit configured for Docker.

## 3) Build the image

From the repository root:

```bash
# CPU
docker build -f deploy/Dockerfile --build-arg TORCH_VARIANT=cpu -t toxpred:cpu .

# GPU (CUDA 12.1 wheels)
docker build -f deploy/Dockerfile --build-arg TORCH_VARIANT=cu121 -t toxpred:cu121 .
```

## 4) Run the container

```bash
# CPU, model artifacts mounted from the checkout
docker run --rm -p 8080:8080 --name toxpred-cpu \
  -e MODEL_ARTIFACTS_URI="" \
  -v "$PWD/models:/app/models:ro" \
  toxpred:cpu

# GPU
docker run --rm --gpus all -p 8080:8080 --name toxpred-gpu \
  -v "$PWD/models:/app/models:ro" \
  toxpred:cu121
```

Notes:
- Service binds `0.0.0.0:${PORT}`, default `8080`.
- The image's own `HEALTHCHECK` polls `GET /health/live` — `docker ps` shows
  `(healthy)` once that starts answering; startup can take a while while
  models load.

## 5) Health checks

Two endpoints, deliberately different questions (`api/routes.py`):
`/health/live` answers "is the process up", `/health/ready` answers "can it
actually predict".

```bash
curl -s http://localhost:8080/health/live
curl -s http://localhost:8080/health/ready | python -m json.tool
```

A real `/health/ready` response, artifacts loaded:

```json
{
  "ready": true,
  "reasons": [],
  "served_endpoints": ["herg", "tox21"]
}
```

`served_endpoints` lists exactly what this build can answer for — never
assume `clintox` is present; check this field. `reasons` is non-empty (and
`ready` is `false`) when something declared in `artifacts/predictor-manifest.yaml`
failed to load; the message names which one.

## 6) API smoke tests

### 6.1 Single prediction

```bash
curl -s -X POST http://localhost:8080/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}' | python -m json.tool
```

Real response shape (aspirin, this repo's pinned artifact) — trimmed to one
Tox21 assay for brevity, the rest follow the same shape:

```json
{
  "input_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "predictions": {
    "herg": {
      "probability_blocker": 0.0315106064081192,
      "label": "non_blocker",
      "threshold": 0.4133453071117401,
      "threshold_source": "artifact",
      "model_id": "herg-tox21-chemberta-v1"
    },
    "tox21": {
      "task_order_version": "tox21-12task-v1",
      "assays": {
        "SR-MMP": {
          "probability_activity": 0.023642102256417274,
          "active": false,
          "threshold": 0.5399999022483826,
          "threshold_source": "artifact"
        }
      },
      "model_id": "herg-tox21-chemberta-v1"
    }
  },
  "applicability": {
    "status": "ok",
    "method": "element_rules_v1",
    "reasons": ["all elements are common in the training sets; this rule cannot confirm distributional similarity beyond element composition"]
  },
  "provenance": {
    "request_id": "...",
    "predictor_version": "0.1.0.dev0",
    "artifacts": [{"model_id": "herg-tox21-chemberta-v1", "weights_sha256": "...", "..."}]
  }
}
```

**What to actually check, not just eyeball:** `hergTox21` are separate keys
under `predictions` (never merged); there is no `verdict`/`final_verdict`/
`clinical` key anywhere in the payload — if one appears, something has
regressed toward the removed aggregate-score design (ADR 0002). Every number
under `predictions` traces to `provenance.artifacts` — that is the
claim-source link the whole agentic layer is built to trust.

### 6.2 Batch prediction

```bash
curl -s -X POST http://localhost:8080/v1/predictions:batch \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["CCO", "c1ccccc1"]}' | python -m json.tool
```

Response: `{"count": 2, "results": [<same per-molecule shape as 6.1>, ...]}`.

### 6.3 A small test suite

```bash
python - <<'PY'
import requests

URL = "http://localhost:8080/v1/predictions"
TEST_SUITE = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Thalidomide", "O=C1CCC(=O)N1C1CCCc2ccccc21"),
    ("Ethanol", "CCO"),
]

for name, smiles in TEST_SUITE:
    r = requests.post(URL, json={"smiles": smiles}, timeout=60)
    d = r.json()
    herg = d.get("predictions", {}).get("herg", {})
    print(name, "status=", r.status_code,
          "herg_label=", herg.get("label"),
          "herg_probability=", herg.get("probability_blocker"),
          "applicability=", d.get("applicability", {}).get("status"))
PY
```

## 7) Error contract validation

An invalid molecule is a typed `400`, never a fabricated prediction:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/v1/predictions \
  -H "Content-Type: application/json" -d '{"smiles":"not-a-smiles"}'
curl -s -X POST http://localhost:8080/v1/predictions \
  -H "Content-Type: application/json" -d '{"smiles":"not-a-smiles"}'
```

Real response:

```
400
{"error":"invalid_smiles","message":"invalid SMILES 'not-a-smiles': RDKit could not parse the structure","detail":{"smiles":"not-a-smiles","reason":"RDKit could not parse the structure"}}
```

Other typed error codes to expect from this contract (`README.md`): `422`
for an unknown request field, `503 model_not_ready` when asking for an
endpoint this build does not serve (check `/health/ready`'s
`served_endpoints` first — this is not a bug, it is an honest "not this
build").

## 8) Logs and debugging

```bash
docker logs -f toxpred-cpu   # or toxpred-gpu
```

- `/health/ready` with `ready: false`: read `reasons` — it names the exact
  artifact `artifacts/predictor-manifest.yaml` declares and could not load.
- GPU run failing: test the CPU image first, to separate an infrastructure
  problem (driver, toolkit) from a model-loading problem.

## 9) Stop and cleanup

```bash
docker stop toxpred-cpu   # or toxpred-gpu, or Ctrl+C if running in the foreground
docker image rm toxpred:cpu toxpred:cu121
```

## 10) References in repo

- Docker image definition: `deploy/Dockerfile`
- Entrypoint (artifact fetch/startup): `deploy/entrypoint.sh`
- API server: `toxpred/api/app.py`, routes in `toxpred/api/routes.py`
- Request/response schemas: `toxpred/api/schemas.py`
- Artifact manifest: `artifacts/predictor-manifest.yaml`
- The same build → run → health → predict → batch sequence runs in CI:
  `.github/workflows/ci.yml`'s `container` job. Its sibling
  `control-plane-container`, `toxocr-container`, and `frontend-container`
  jobs cover their respective images. These jobs, not this document, certify
  a given commit; treat a difference between them as this runbook being stale,
  not the other way around.
