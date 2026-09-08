#!/usr/bin/env bash
# Verify that the browser-facing topology reaches the same control-plane
# revision that exposes the Quick Predict and XAI routes. Run after every
# coordinated rollout, before sending traffic to a new frontend.
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8088}"
TOKEN="${TOXAGENT_SMOKE_TOKEN:-}"
if [[ -z "$TOKEN" ]]; then
  echo "Set TOXAGENT_SMOKE_TOKEN to a valid bearer token." >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"
headers=(-H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json')

check_route() {
  local label=$1 method=$2 path=$3 data=${4:-}
  local status
  if [[ -n "$data" ]]; then
    status="$(curl -sS -o /tmp/toxagent-smoke-response.json -w '%{http_code}' -X "$method" "${headers[@]}" --data "$data" "$BASE_URL$path")"
  else
    status="$(curl -sS -o /tmp/toxagent-smoke-response.json -w '%{http_code}' -X "$method" "${headers[@]}" "$BASE_URL$path")"
  fi
  if [[ "$status" == "404" ]]; then
    echo "$label returned 404 through frontend proxy: incompatible deployment." >&2
    exit 1
  fi
  if [[ "$status" -ge 500 ]]; then
    echo "$label returned HTTP $status" >&2
    exit 1
  fi
  echo "$label: HTTP $status"
}

check_route "predict capabilities" GET /v1/predict/capabilities
check_route "quick predict" POST /v1/predict '{"smiles":"C1CCCCCC1","endpoints":["herg"]}'
check_route "quick explain" POST /v1/predict/explain '{"smiles":"C1CCCCCC1","endpoint":"herg"}'

echo "Browser proxy and control-plane route surface are compatible."
