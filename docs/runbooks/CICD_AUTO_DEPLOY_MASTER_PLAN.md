# CI/CD Auto Deploy Master Plan (main + agent_test)

## 1. Goal

Build a reliable CI/CD pipeline so that pushing to main can automatically deploy:
- Backend (Cloud Run)
- Frontend (Firebase Hosting live)
- Database config (Firestore rules and indexes)

And pushing to agent_test can automatically deploy a safe staging stack for testing.

## 2. Current State (as of 2026-04-11)

Already in repo:
- Frontend auto deploy on merge to main via GitHub Actions.
- Frontend preview deploy on pull request.
- Manual backend deployment via Cloud Build + Cloud Run runbook.
- Firestore rules and indexes are present in firebase.json but not auto deployed by workflow.

Missing for full automation:
- Backend auto deploy workflow
- Firestore rules/indexes auto deploy workflow
- Branch environment separation strategy (main vs agent_test)
- Post-deploy smoke tests and rollback process

## 3. Target Deployment Model

### 3.1 main (production)

Trigger: push to main

Pipeline order:
1. Pre-deploy checks (frontend build, backend import checks, lint/static checks)
2. Build and push backend image to Artifact Registry
3. Deploy backend to Cloud Run service tox-agent-cpu
4. Deploy Firestore rules/indexes
5. Deploy frontend to Firebase Hosting live
6. Run smoke tests on hosting URL and Cloud Run URL
7. Report deployment summary in GitHub Actions output

### 3.2 agent_test (staging)

Trigger: push to agent_test

Pipeline order:
1. Same pre-deploy checks
2. Build and push backend image with agent-test tag
3. Deploy backend to Cloud Run service tox-agent-cpu-agent-test
4. Deploy Firestore rules/indexes only if explicitly enabled (recommended: manual gate)
5. Deploy frontend to Firebase Hosting preview channel agent-test, with rewrite pointing to tox-agent-cpu-agent-test
6. Run smoke tests and post preview URL

## 4. Security and Identity Model

Use GitHub OIDC (recommended), avoid long-lived JSON keys.

Required setup:
- Workload Identity Pool + Provider for GitHub repo
- Deploy service account for CI
- Firebase deploy identity

Minimum IAM roles for CI deploy service account:
- Cloud Build Editor
- Artifact Registry Writer
- Cloud Run Admin
- Service Account User
- Firebase Hosting Admin
- Firebase Rules Admin (or equivalent Firestore deployment permission)

## 5. Required Repository Inputs

### 5.1 GitHub Secrets / Variables

Repository variables:
- GCP_PROJECT_ID
- GCP_REGION
- ARTIFACT_REPO
- CLOUD_RUN_SERVICE_PROD (tox-agent-cpu)
- CLOUD_RUN_SERVICE_STAGING (tox-agent-cpu-agent-test)
- FIREBASE_PROJECT_ID (tox-agent)

Repository secrets:
- WIF_PROVIDER
- WIF_SERVICE_ACCOUNT
- FIREBASE_SERVICE_ACCOUNT_TOX_AGENT (only if still using key-based Firebase action)

### 5.2 Files to Add

Planned workflow files:
- .github/workflows/ci.yml
- .github/workflows/deploy-backend-prod.yml
- .github/workflows/deploy-backend-staging.yml
- .github/workflows/deploy-firestore.yml
- .github/workflows/deploy-frontend-staging.yml

Planned support files:
- deploy/firebase.agent_test.json (or generate dynamically in workflow)
- scripts/smoke_test_deploy.sh

## 6. Implementation Phases

### Phase A: Foundation

Deliverables:
- CI workflow for build checks
- OIDC auth wired and validated
- Shared environment variables in GitHub repo settings

Exit criteria:
- CI workflow green on PR
- OIDC token can authenticate and read project metadata

### Phase B: Backend Auto Deploy

Deliverables:
- Automatic Cloud Build image build and push
- Cloud Run deployment on main and agent_test
- Health endpoint smoke test

Exit criteria:
- main push updates tox-agent-cpu revision
- agent_test push updates tox-agent-cpu-agent-test revision
- /health returns healthy in both environments

### Phase C: Database Auto Deploy

Deliverables:
- Firestore rules/indexes deploy workflow
- Optional manual approval gate for production rules deploy

Exit criteria:
- firestore.rules and firestore.indexes.json deploy reliably from workflow
- Rule deployment logs captured in action summary

### Phase D: Frontend Environment Split

Deliverables:
- main -> Firebase live deploy
- agent_test -> Firebase preview channel deploy with rewrite to staging backend

Exit criteria:
- Production URL serves latest main
- Staging preview URL serves latest agent_test and calls staging backend

### Phase E: Guardrails and Rollback

Deliverables:
- Smoke tests for /health, /predict, /analyze
- Rollback commands documented in workflow summary
- Optional concurrency lock to avoid parallel deploy race conditions

Exit criteria:
- Failed deploy does not silently pass
- Last known good image/tag is always visible in logs

## 7. Database Strategy Recommendation

Recommended:
- Keep production Firestore in project tox-agent
- For agent_test, prefer separate Firebase project (tox-agent-staging) if possible
- If staging project is not available yet, keep agent_test backend isolated at service level and avoid destructive schema/rules changes without manual gate

## 8. Rollback Policy

Backend rollback:
- gcloud run services update-traffic with previous revision

Frontend rollback:
- Firebase Hosting: release previous version from console or CLI

Database rollback:
- Re-deploy previous firestore.rules and firestore.indexes.json from tagged commit

## 9. Definition of Done

Done when:
- Push to main automatically deploys backend, frontend live, and firestore rules/indexes
- Push to agent_test automatically deploys staging backend and staging frontend preview
- Smoke tests pass and publish URLs in GitHub Action summary
- Runbook reflects emergency rollback and branch-specific deploy instructions

## 10. Notes for Execution Phase

This file is the approved planning baseline only.
Implementation should be executed in a dedicated change set and validated in agent_test before enabling production auto deploy gates.
