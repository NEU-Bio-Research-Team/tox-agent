#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-tox-agent}"
REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-tox-agent-cpu}"
ENV_FILE="${1:-deploy/cloudrun-env.yaml}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE"
  echo "Create it from deploy/cloudrun-env.yaml and paste your keys first."
  exit 1
fi

if grep -E "REPLACE_ME_|PASTE_YOUR_" "$ENV_FILE" >/dev/null 2>&1; then
  echo "Please replace placeholder values in $ENV_FILE before deploying."
  exit 1
fi

echo "Applying env vars from $ENV_FILE to service $SERVICE..."
gcloud run services update "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --quiet \
  --env-vars-file "$ENV_FILE"

echo "Done."
echo "Service URL:"
gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)'
