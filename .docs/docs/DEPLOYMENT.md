# Tox-Agent Deployment & Operations

This document outlines the deployment lifecycle for the **Tox-Agent** platform, ensuring consistent delivery from local research to production-scale inference.

## Supported Environments

| Environment | Purpose | Infrastructure |
| --- | --- | --- |
| `dev` | Local development and GNN model training | Docker Compose / Local Firebase Emulator |
| `staging` | Integration testing and beta-user evaluation | Google Cloud Run (Staging Project) |
| `production` | Live molecular analysis for research partners | Google Cloud Run (Production Project) |

## Environment Variables

All secrets must be stored in **Google Cloud Secret Manager** or a local `.env` file. Never commit secrets to the repository.

### Infrastructure Config

| Variable | Required | Purpose |
| --- | --- | --- |
| `NODE_ENV` | Yes | `development`, `staging`, or `production` |
| `PORT` | No | Listening port, defaults to `3000` |
| `DATABASE_URL` | Yes | PostgreSQL connection string that supports RLS |
| `FIREBASE_PROJECT_ID` | Yes | Identifier used by Firebase Auth and the Admin SDK |

### AI Engine Config

| Variable | Required | Purpose |
| --- | --- | --- |
| `GNN_MODEL_PATH` | Yes | Path to the serialized `.pt` or `.h5` model file |
| `INFERENCE_DEVICE` | No | `cpu` or `cuda`, defaults to `cpu` |
| `MAX_SMILES_LENGTH` | No | Guardrail for computational complexity |

## Containerization Guidance

The Tox-Agent service requires a dual-runtime environment with Node.js and Python for the GNN engine.

- **Base image**: Use a Debian-based Python slim image and install Node.js so scientific libraries such as RDKit and DGL remain compatible.
- **Security**: Run the process as a non-root user such as `node`. Do not run production containers as `root`.
- **Liveness probe**: `GET /health`
- **Readiness probe**: `GET /ready`
  Readiness should verify database connectivity and successful GNN model loading.

## Deployment Workflow

### 1. Pre-deploy Checklist

- [ ] `npm test` passes and Python unit tests for the GNN engine pass
- [ ] Database migrations are verified as backward compatible
- [ ] Tox-Agent beta evaluation survey results have been reviewed for major releases
- [ ] API contract in `openapi.yaml` matches the implementation

### 2. Staging Deployment

1. Trigger **Google Cloud Build** on the `develop` branch.
2. Deploy to the staging service.
3. Run smoke tests:
   - execute `POST /v1/predict` with a known compound such as Aspirin
   - verify RLS by ensuring test users cannot see each other's history

### 3. Production Deployment

1. Deploy using **Blue/Green** or **Rolling Update** to avoid downtime.
2. Monitor golden signals, especially latency and `5xx` rates, for at least 15 minutes.
3. Confirm that the GNN engine successfully loaded model weights into memory.

## Rollback Procedure

Trigger rollback immediately if:

- the `/ready` probe fails, indicating database or model-loading errors
- prediction latency exceeds 10 seconds for standard molecules
- user data leakage is suspected, especially from RLS misconfiguration

### Steps

1. Revert the service artifact:

```bash
gcloud run deploy tox-agent-api \
  --image gcr.io/[PROJECT_ID]/tox-agent:[PREVIOUS_STABLE_TAG]
```

2. If a migration caused the failure, run `npm run migrate:rollback` only when data loss is not a risk.
3. Verify that `/health` returns `200` and that the previous model version is active.

## Operations & Maintenance

### Health Check Endpoints

- **`GET /health`**: Returns `200 OK` with `uptime` and `version`
- **`GET /ready`**: Returns `200 OK` only if:
  - PostgreSQL connectivity is healthy
  - the GATv2 model is loaded and verified
  - Firebase Auth handshake is successful

### Log Management

All logs are piped to **Cloud Logging**. Filter by `jsonPayload.severity >= "ERROR"` for immediate incident investigation.

### Escalation

For infrastructure-level failures, refer to the [Runbook](./RUNBOOK.md).
