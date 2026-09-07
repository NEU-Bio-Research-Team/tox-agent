# toxocr

Optical chemical structure recognition: an image of a 2D structure in, a
SMILES string out. One endpoint, one job.

A **separate deployable** from both `toxpred` and `toxagent-control` — see
[`toxagent-control/docs/adr/0006-ocr-fourth-boundary.md`](../toxagent-control/docs/adr/0006-ocr-fourth-boundary.md)
for why. It imports [MolScribe](https://github.com/thomas0809/MolScribe) and
nothing else knows that dependency exists.

## Layout

```
toxocr/
  scientific/   MolScribePredictor — model load, checkpoint resolution, inference
  api/          FastAPI app, routes, schemas, error mapping
```

## Why its own environment

MolScribe pins `torch>=1.11.0,<2.0`. Neither `toxpred` nor `toxagent-control`
can host that constraint (the live predictor runs a current torch). Verified
working setup, CPU only:

```bash
conda create -n toxocr-env python=3.10
conda activate toxocr-env
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r toxocr/requirements.txt
pip install "numpy<2"   # see requirements.txt's comment — molscribe's pinned
                         # deps (timm, torchvision 0.14, albumentations 1.1.0)
                         # predate NumPy 2.0's ABI break
```

## Running it

```bash
PYTHONPATH=/path/to/tox-agent \
  uvicorn toxocr.api.app:app --host 127.0.0.1 --port 8090
```

First request (or startup, since `TOXOCR_EAGER_LOAD` defaults on) downloads
the checkpoint from `yujieq/MolScribe` on HuggingFace to the local HF cache —
a few hundred MB, one-time. Point `toxagent-control` at it with
`TOXAGENT_OCR_URL=http://127.0.0.1:8090`; leaving it unset is a supported,
tested state (`capability_unavailable`, see ADR 0006), not a missing feature.

## Contract

`POST /v1/structure-recognition`

```json
{"mime_type": "image/png", "data_base64": "<...>"}
```

→ `200 {"smiles": "...", "canonical_smiles": "...", "confidence": 0.89}`,
`422` (`smiles_not_detected`) if no structure was recognised, `415`
(`unsupported_image_format`) if the bytes don't decode as an image, `400`
(`invalid_request`) for a malformed request.

`GET /health/ready` → `{"ready": bool}` — whether the model has finished
loading.

## Tests

```bash
PYTHONPATH=/path/to/tox-agent pytest toxocr/tests -q
```

Fast — a fake predictor stands in for MolScribe, so this exercises the FastAPI
wiring and error mapping, not the model. There is no CI job for a real
inference run; that was verified manually (ADR 0006) and is expensive to
repeat (checkpoint download, ~1-2s/image on CPU once loaded).
