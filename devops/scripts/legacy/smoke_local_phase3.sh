#!/usr/bin/env bash
# Exercise localhost Predictor -> ToxAgent -> OpenCode with one SMILES.
# The final report-Q&A uses a configured LLM provider and can therefore incur
# provider usage. It never sends a capability token in the model prompt.
set -euo pipefail

PYTHON_BIN="${TOXAGENT_PYTHON:-$(command -v python)}"
PREDICTOR_URL="${TOXPRED_URL:-http://127.0.0.1:8080}"
CONTROL_URL="${TOXAGENT_URL:-http://127.0.0.1:8000}"
TOKEN="${TOXAGENT_TEST_TOKEN:-dev-local}"
SMILES="${TOXAGENT_TEST_SMILES:-CC(=O)Oc1ccccc1C(=O)O}"

json_field() {
  local expression=$1
  "$PYTHON_BIN" -c "import json, sys; print($expression)"
}

request() {
  curl --fail-with-body --silent --show-error "$@"
}

echo '== ToxPred checkpoint inventory =='
request "$PREDICTOR_URL/health/ready"
echo
request "$PREDICTOR_URL/v1/models"
echo

echo '== Direct prediction for the supplied SMILES =='
request -X POST "$PREDICTOR_URL/v1/predictions" \
  -H 'content-type: application/json' \
  -d "{\"smiles\":\"$SMILES\",\"endpoints\":[\"herg\",\"tox21\"]}"
echo

SESSION_BODY=$(request -X POST "$CONTROL_URL/v1/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"preferred_language":"vi","title":"localhost Phase 3 smoke"}')
SESSION_ID=$(printf '%s' "$SESSION_BODY" | json_field 'json.load(sys.stdin)["session_id"]')
echo "Created session: $SESSION_ID"

ANALYSIS_BODY=$(request -X POST "$CONTROL_URL/v1/sessions/$SESSION_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d "{\"intent_hint\":\"analyze\",\"molecule\":{\"smiles\":\"$SMILES\"},\"analysis_options\":{\"endpoints\":[\"herg\",\"tox21\"]}}")
ANALYSIS_RUN_ID=$(printf '%s' "$ANALYSIS_BODY" | json_field 'json.load(sys.stdin)["run_id"]')

for _ in $(seq 1 90); do
  RUN_BODY=$(request -H "Authorization: Bearer $TOKEN" "$CONTROL_URL/v1/sessions/$SESSION_ID/runs/$ANALYSIS_RUN_ID")
  RUN_STATUS=$(printf '%s' "$RUN_BODY" | json_field 'json.load(sys.stdin)["status"]')
  if [[ "$RUN_STATUS" == "completed" ]]; then
    break
  fi
  if [[ "$RUN_STATUS" == "failed" || "$RUN_STATUS" == "cancelled" ]]; then
    echo "Analysis run did not complete: $RUN_BODY" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${RUN_STATUS:-}" != "completed" ]]; then
  echo "Timed out waiting for analysis run $ANALYSIS_RUN_ID" >&2
  exit 1
fi

SESSION_VIEW=$(request -H "Authorization: Bearer $TOKEN" "$CONTROL_URL/v1/sessions/$SESSION_ID")
ANALYSIS_ID=$(printf '%s' "$SESSION_VIEW" | json_field 'json.load(sys.stdin)["active_analysis"]["analysis_id"]')
echo "Analysis snapshot: $ANALYSIS_ID"

echo '== Grounded report Q&A through OpenCode/MCP =='
QA_BODY=$(request -X POST "$CONTROL_URL/v1/sessions/$SESSION_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d "{\"intent_hint\":\"ask_report\",\"analysis_id\":\"$ANALYSIS_ID\",\"content\":[{\"type\":\"text\",\"text\":\"Giải thích kết quả hERG và các giới hạn của dự đoán này.\"}]}")
QA_RUN_ID=$(printf '%s' "$QA_BODY" | json_field 'json.load(sys.stdin)["run_id"]')

for _ in $(seq 1 180); do
  QA_RUN=$(request -H "Authorization: Bearer $TOKEN" "$CONTROL_URL/v1/sessions/$SESSION_ID/runs/$QA_RUN_ID")
  QA_STATUS=$(printf '%s' "$QA_RUN" | json_field 'json.load(sys.stdin)["status"]')
  if [[ "$QA_STATUS" == "completed" || "$QA_STATUS" == "failed" || "$QA_STATUS" == "cancelled" ]]; then
    break
  fi
  sleep 1
done

echo "Run result: $QA_RUN"
echo '== Persisted product messages =='
request -H "Authorization: Bearer $TOKEN" "$CONTROL_URL/v1/sessions/$SESSION_ID/messages"
echo

if [[ "${QA_STATUS:-}" != "completed" ]]; then
  cat >&2 <<EOF
The Q&A run did not complete. Check the control and OpenCode logs. A failed run
is expected if the selected model cannot call submit_grounded_answer, if the
OpenCode server is unavailable, or if the model/provider is not configured.
EOF
  exit 1
fi

echo "Phase 3 local smoke passed for session $SESSION_ID."
