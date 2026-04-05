# ToxAgent Docker Setup and Test Runbook

## 1) Scope

This runbook helps you build and test ToxAgent in Docker for:
- local CPU testing
- local GPU testing (optional)
- API smoke tests (`/health`, `/analyze`)

Repository root assumed:
- `/home/mluser/BRT-FDA/MinhQuang/tox-agent`


## 2) Prerequisites

Required:
- Docker installed and running
- Model artifacts present in workspace:
  - `models/smilesgnn_model/best_model.pt`
  - `models/smilesgnn_model/tokenizer.pkl`
  - `models/tox21_gatv2_model/best_model.pt`

Optional (for calibrated mechanism thresholds):
- `models/tox21_gatv2_model/tox21_task_thresholds.json`
  - fallback filename also supported: `task_thresholds.json`

For GPU container test:
- NVIDIA driver installed (`nvidia-smi` works)
- NVIDIA Container Toolkit configured for Docker


## 3) Build Image

From repo root.

### CPU image
```bash
docker build \
  --build-arg TORCH_VARIANT=cpu \
  -t toxagent:cpu \
  -f model_server/Dockerfile .
```

### GPU image (CUDA 12.1 wheels)
```bash
docker build \
  --build-arg TORCH_VARIANT=cu121 \
  -t toxagent:cu121 \
  -f model_server/Dockerfile .
```


## 4) Run Container

### CPU run
```bash
docker run --rm -p 8080:8080 --name toxagent-cpu toxagent:cpu
```

### GPU run
```bash
docker run --rm --gpus all -p 8080:8080 --name toxagent-gpu toxagent:cu121
```

Notes:
- Service binds on port `8080`.
- Startup can take time while loading models.


## 5) Health Check

In a second terminal:
```bash
curl -s http://localhost:8080/health | python -m json.tool
```

Expected key fields:
- `status`: `healthy`
- `xsmiles_loaded`: `true`
- `tox21_loaded`: `true`
- `tox21_thresholds_loaded`: `true` if threshold JSON exists
- `tox21_threshold_count`: `12` when thresholds are loaded


## 6) API Smoke Tests

### 6.1 Single analyze request
```bash
curl -s -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "return_all_scores": false,
    "explain_only_if_alert": true,
    "explainer_epochs": 50
  }' | python -m json.tool
```

### 6.2 Test suite (3 molecules)
```bash
python - <<'PY'
import requests, json

URL = "http://localhost:8080/analyze"
TEST_SUITE = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Thalidomide", "O=C1CCC(=O)N1C1CCCc2ccccc21"),
    ("Doxorubicin", "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(=O)CO)Oc4c(O)c3C(=O)c12"),
]

for name, smiles in TEST_SUITE:
    payload = {
        "smiles": smiles,
        "return_all_scores": False,
        "explain_only_if_alert": True,
        "explainer_epochs": 50,
    }
    r = requests.post(URL, json=payload, timeout=120)
    d = r.json()
    print(name)
    print(" status=", r.status_code)
    print(" verdict=", d.get("final_verdict"))
    print(" clinical=", d.get("clinical", {}).get("label"), d.get("clinical", {}).get("p_toxic"))
    print(" hits=", d.get("mechanism", {}).get("assay_hits"))
    print(" explanation_present=", d.get("explanation") is not None)
    print("-" * 80)
PY
```


## 7) Timeout and Error Contract Validation

### Invalid SMILES
```bash
curl -s -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"smiles":"not-a-smiles"}' | python -m json.tool
```

Expected:
- HTTP `400`
- JSON includes `error: "invalid_smiles"`

### Explainer timeout behavior
```bash
curl -s -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(=O)CO)Oc4c(O)c3C(=O)c12",
    "explain_only_if_alert": false,
    "target_task": "SR-p53",
    "explainer_epochs": 500,
    "explainer_timeout_ms": 1000
  }' | python -m json.tool
```

Expected:
- HTTP `504`
- JSON includes `error: "explainer_timeout"`


## 8) Logs and Debugging

Container logs:
```bash
docker logs -f toxagent-cpu
# or
# docker logs -f toxagent-gpu
```

Common checks:
- If `/health` is `starting`, wait until model load finishes.
- If thresholds are not loaded, ensure one of these files exists:
  - `models/tox21_gatv2_model/tox21_task_thresholds.json`
  - `models/tox21_gatv2_model/task_thresholds.json`
- If GPU run fails, test CPU image first to isolate infra from model logic.


## 9) Stop and Cleanup

If running foreground, use `Ctrl+C`.

If running detached:
```bash
docker stop toxagent-cpu
# or
# docker stop toxagent-gpu
```

Optional image cleanup:
```bash
docker image rm toxagent:cpu toxagent:cu121
```


## 10) References in Repo

- Docker image definition: `model_server/Dockerfile`
- API server: `model_server/main.py`
- API schemas: `model_server/schemas.py`
- Function-calling flow: `FUNCTION_CALLING_FLOW.md`
