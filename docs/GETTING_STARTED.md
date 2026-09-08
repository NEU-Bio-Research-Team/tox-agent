# Getting started

The ordered procedure lives in the root [README](../README.md#setup-in-order):
prerequisites, the model weights a clean clone has to be given, then `setup`,
`doctor`, `up`, `smoke` and the browser. This page is the background to those
steps, not a second copy of them.

## What setup actually does

It creates `.env` from `.env.example`, generates the PostgreSQL password, the
capability secret and a local access token, and provisions two separate sets of
model artifacts:

- The pinned MolScribe OCR checkpoint, downloaded into the ignored
  `.artifacts/` directory and checked against its SHA-256.
- The predictor artifacts. These are downloaded only when `MODEL_ARTIFACTS_URI`
  names a bundle this deployment may use; otherwise they must already be under
  `.data/models` (or `TOXPRED_MODELS_HOST_PATH`). Either way every file is
  verified against `backend/predictor/registry/models/`.

Those two are the wrapper's only network downloads outside Docker image builds.
The predictor check runs even when nothing is downloaded, because a stack whose
weights are missing builds every image and then never becomes ready — failing
in `setup`, by name, is the point.

Nothing here overwrites an existing secret or an artifact that is already
present, so re-running `setup` is safe.

## When up or smoke fails

`up` waits for Compose health checks. When it fails, run `./bin/toxagent logs`
and use the service named in the error. A successful `smoke` proves the browser
route, control plane and predictor route are connected; it sends one small
SMILES request on purpose and is not a model-heavy benchmark.

## About the token

The token printed by `setup` is a development-only local credential. Paste it
into the frontend gate; do not send it to other users and do not deploy this
stack to a public host with it.
