# Tox-Agent End-to-End Deployment Runbook (Docker + Cloud Run + Firebase Hosting)

This guide is written so a new contributor can clone the repo and deploy successfully.

## 1) Scope

This runbook deploys:
- Backend API container to Google Cloud Run (`model_server`)
- Frontend app to Firebase Hosting (`tox-agent.web.app`)
- Hosting rewrites `/health`, `/predict`, `/predict/**`, `/explain`, `/analyze` to Cloud Run

It also reproduces required model checkpoints used by API endpoints.

## 2) Prerequisites

Required tools:
- `git`
- `conda` (or miniconda)
- `node` + `npm`
- `gcloud` CLI (authenticated)
- Firebase CLI via `npx -y firebase-tools@latest`

Required cloud permissions (minimum):
- Cloud Build Editor
- Artifact Registry Writer
- Cloud Run Admin
- Service Account User
- Firebase Hosting Admin

## 3) Clone and Enter Repo

```bash
git clone https://github.com/NEU-Bio-Research-Team/tox-agent.git
cd tox-agent
```

## 4) Python Environment (drug-tox-env)

Create and activate environment:

```bash
conda env create -f environment.yml
conda activate drug-tox-env
```

If environment already exists:

```bash
conda activate drug-tox-env
```

## 5) Reproduce Checkpoints for API

Required by API logic:
- Clinical branch (`/predict`, clinical block in `/analyze`):
  - `models/smilesgnn_model/best_model.pt`
  - `models/smilesgnn_model/tokenizer.pkl`
- Mechanism branch (`/analyze`):
  - `models/tox21_gatv2_model/best_model.pt`
  - `models/tox21_gatv2_model/tox21_task_thresholds.json`

### 5.1 Train Tox21 GATv2 checkpoint

```bash
conda activate drug-tox-env
mkdir -p logs
PYTHONUNBUFFERED=1 python scripts/train_tox21_gatv2.py \
  --device cuda \
  --config config/tox21_gatv2_config.yaml \
  2>&1 | tee logs/train_tox21_gatv2.log
```

Notes:
- If CUDA is unavailable, script falls back to CPU.
- Some DeepChem warnings about optional packages (tensorflow, dgl, jax, lightning) can be ignored for this workflow.

### 5.2 Train SMILESGNN (ClinTox) checkpoint

`train_hybrid.py` is guarded by workspace mode. Temporarily enable ClinTox:

```bash
cat > config/workspace_mode.yaml <<'YAML'
workspace:
  mode: dual_train
  primary_dataset: tox21
  clintox_enabled: true
  tox21_enabled: true
  message: "ClinTox and Tox21 workflows are enabled for checkpoint reproduction."
YAML
```

Run training:

```bash
conda activate drug-tox-env
mkdir -p logs
PYTHONUNBUFFERED=1 python scripts/train_hybrid.py \
  --device cuda \
  --config config/smilesgnn_config.yaml \
  2>&1 | tee logs/train_smilesgnn_hybrid.log
```

Restore workspace mode after training:

```bash
cat > config/workspace_mode.yaml <<'YAML'
workspace:
  mode: tox21_only
  primary_dataset: tox21
  clintox_enabled: false
  tox21_enabled: true
  message: "ClinTox workflows are disabled in this workspace mode."
YAML
```

### 5.3 Verify artifacts exist

```bash
ls -lh models/smilesgnn_model
ls -lh models/tox21_gatv2_model
```

Expected key files:
- `models/smilesgnn_model/best_model.pt`
- `models/smilesgnn_model/tokenizer.pkl`
- `models/tox21_gatv2_model/best_model.pt`
- `models/tox21_gatv2_model/tox21_task_thresholds.json`
- `models/tox21_gatv2_model/task_thresholds.json`

## 6) Configure Cloud and Firebase Variables

```bash
export PROJECT_ID=tox-agent
export REGION=asia-southeast1
export REPO=tox-agent-repo
export SERVICE=tox-agent-cpu
export SITE_ID=tox-agent
```

Set active gcloud project:

```bash
gcloud config set project "$PROJECT_ID"
```

Confirm Firebase project:

```bash
npx -y firebase-tools@latest use
```

## 7) Ensure Artifact Registry Repository Exists

```bash
gcloud artifacts repositories describe "$REPO" \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  >/dev/null 2>&1 || \
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID"
```

## 8) Build Backend Docker Image with Cloud Build

Important:
- `model_server/Dockerfile` copies repo into `/app`
- Ensure `models/` is present locally before this step
- `.gcloudignore` must not exclude `models/`

Create temporary Cloud Build config:

```bash
cat > /tmp/cloudbuild-tox-agent.yaml <<'YAML'
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build", "-f", "model_server/Dockerfile", "-t", "${_IMAGE}", "."]
images:
  - "${_IMAGE}"
YAML
```

Build and push:

```bash
TAG="cpu-$(date +%Y%m%d-%H%M%S)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/tox-agent:${TAG}"

gcloud builds submit \
  --project="$PROJECT_ID" \
  --config=/tmp/cloudbuild-tox-agent.yaml \
  --substitutions="_IMAGE=${IMAGE}" \
  .

echo "IMAGE=${IMAGE}"
```

## 9) Deploy Backend to Cloud Run

```bash
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --cpu=4 \
  --memory=8Gi \
  --concurrency=1 \
  --timeout=600 \
  --min-instances=1 \
  --startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=0,timeoutSeconds=5,periodSeconds=10,failureThreshold=30 \
  --allow-unauthenticated \
  --project="$PROJECT_ID"
```

Get URL and smoke test:

```bash
RUN_URL=$(gcloud run services describe "$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(status.url)')

echo "RUN_URL=$RUN_URL"

curl -sS "$RUN_URL/health" | head -c 2000; echo
curl -sS -X POST "$RUN_URL/predict" \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO","threshold":0.5}' | head -c 1200; echo
curl -sS -X POST "$RUN_URL/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO","clinical_threshold":0.5,"mechanism_threshold":0.5,"return_all_scores":true,"explain_only_if_alert":true,"explainer_epochs":80,"explainer_timeout_ms":30000}' | head -c 2000; echo
```

## 10) Build and Deploy Frontend to Firebase Hosting

```bash
npm ci
npm run build
npx -y firebase-tools@latest deploy --only hosting --project "$PROJECT_ID"
```

Expected URL:
- `https://tox-agent.web.app`

## 11) Verify Public Firebase App and API Rewrites

```bash
APP_URL="https://${SITE_ID}.web.app"

curl -I -sS "$APP_URL" | head -n 5
curl -sS "$APP_URL/health" | head -c 1500; echo
curl -sS -X POST "$APP_URL/predict" \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO","threshold":0.5}' | head -c 1200; echo
curl -sS -X POST "$APP_URL/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO","clinical_threshold":0.5,"mechanism_threshold":0.5,"return_all_scores":true,"explain_only_if_alert":true,"explainer_epochs":80,"explainer_timeout_ms":30000}' | head -c 2000; echo
```

## 12) Localhost API Testing (Optional)

Run local frontend with API proxy:

```bash
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:
- `http://127.0.0.1:5173`

## 13) Troubleshooting

### A) Cloud Run deploy fails: container did not listen on PORT

Check latest revision logs:

```bash
REV=$(gcloud run revisions list \
  --service="$SERVICE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(metadata.name)' \
  --limit=1)

gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND resource.labels.revision_name=\"$REV\"" \
  --project="$PROJECT_ID" \
  --limit=120 \
  --format='value(timestamp,severity,textPayload)'
```

Most common cause:
- Missing model files in image (`/app/models/...`)

### B) `/health` returns degraded

Inspect payload fields:
- `xsmiles_loaded`
- `tox21_loaded`
- `startup_errors`

### C) `/analyze` explanation is null

This is normal when:
- `explain_only_if_alert=true`
- No mechanism alert (`assay_hits == 0`)

Force explanation for testing:
- Set `explain_only_if_alert=false`

### D) Firebase deploy fails with Hosting rewrite errors

Re-check:
- `firebase.json` rewrite `serviceId` and `region`
- Cloud Run service exists and is serving traffic

## 14) Quick Re-Deploy (after code change)

```bash
# 1) Retrain only if model code/data changed
# 2) Rebuild backend image + redeploy Cloud Run
# 3) npm run build + firebase hosting deploy
```

For frontend-only changes:
```bash
npm run build
npx -y firebase-tools@latest deploy --only hosting --project "$PROJECT_ID"
```

For backend-only changes (no model changes):
- Rebuild image and deploy Cloud Run; Firebase Hosting usually does not need redeploy.
