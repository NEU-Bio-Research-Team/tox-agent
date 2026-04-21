# Tox-Agent Runbook

This document serves as the "break glass in case of emergency" guide for the Tox-Agent platform. It provides the protocol for responding to system failures, model degradation, or security breaches.

## Service Overview

Tox-Agent consists of:

- a Node.js API
- a Python-based GNN inference engine
- a PostgreSQL database

Critical failure usually presents as:

- **Inference Failure**: Requests to `POST /v1/predict` return `5xx` or time out
- **Data Leakage**: Users can see molecular history that does not belong to them, indicating a potential RLS failure
- **Service Unavailability**: The API or Firebase Auth layer is unreachable

## Unhealthy Service Indicators (Golden Signals)

Monitor these metrics to identify a service-down or degraded state:

- **Inference Latency**: p95 latency for `/v1/predict` exceeds 5 seconds
- **Error Rate**: Any spike in `500 Internal Server Error` or `403 Forbidden`, which may indicate an RLS or auth issue
- **Model Accuracy Drift**: Sudden drop in confidence scores for known control compounds
- **GPU/CPU Saturation**: High utilization on inference nodes causing request queuing
- **Database Connectivity**: Failure in the `/ready` probe due to PostgreSQL connection pool exhaustion

## Immediate Checks (First Response)

1. Check API health by verifying `GET /health` and `GET /ready`.
2. Verify the auth layer and ensure Firebase Emulator or production Firebase is accepting tokens.
3. Inspect RLS status by running a manual PostgreSQL query to confirm Row-Level Security policies are active.
4. Review inference engine logs for OOM (Out of Memory) conditions or `CudaError`.
5. Check Google Cloud Build for any successful deployments in the last 60 minutes.

## Common Failure Scenarios

### 1. AI Inference Engine Timeout

**Symptoms**

- API returns `504 Gateway Timeout` on prediction requests

**Cause**

- The GNN model is hanging
- The worker queue is backed up

**Action**

- Restart the Python inference worker
- Check whether the SMILES string being processed is abnormally large or complex

### 2. Unauthorized Access (RLS Failure)

**Symptoms**

- User A can see User B's molecular search history

**Cause**

- PostgreSQL `session_user` or `current_setting` for RLS is not being set correctly by the controller

**Action**

- Disable the service immediately
- Audit the database controller logic where the tenant ID is passed to the connection

### 3. Database Migration Failure

**Symptoms**

- `GET /v1/history` returns `400` or `500`

**Cause**

- The latest schema change broke existing molecular records

**Action**

- Roll back the database migration
- Revert the API version

## Rollback Procedures

### API and Engine Rollback

To revert to a previous stable version:

- redeploy the previous known-good API release
- roll back the Python inference engine to the last validated model-serving version
- verify `GET /health`, `GET /ready`, and a manual `/v1/predict` smoke test before reopening traffic

### Database Rollback

If using a migration tool such as Knex or Sequelize:

- roll back the latest migration
- validate that old application code remains compatible with the reverted schema
- verify data integrity before and after the rollback

**Note**: Always verify data integrity before rolling back migrations in production.

## Escalation Path

If the incident is not resolved within 15 minutes, notify the following in order:

1. **Primary Lead**: Teddy (System and Security)
2. **Infrastructure**: Nghia Nguyen (Database and Deployment)
3. **UI/UX**: Nhat Minh (Frontend and Dashboard State)

**Communication Channel**: `#tox-agent-ops` on Slack or Discord

## Actions to Avoid (Strict)

- Do not disable Row-Level Security on the production database to debug a connection issue
- Do not manually edit molecular records in the database without an audit trail
- Do not bypass Firebase Auth for local testing against the production endpoint
- Do not restart the entire database cluster without checking connection pool statistics first

## Incident Closure

Before closing the incident:

1. Verify the `5xx` error rate has returned to less than `0.1%`.
2. Perform a manual smoke test of the `/v1/predict` endpoint.
3. Update the changelog or an ADR if the incident led to a permanent architecture change.
4. Conduct a post-mortem and document findings in `docs/post-mortems/`.
