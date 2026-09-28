#!/usr/bin/env bash
# One TAB-Suite live batch: restart the control plane under a configuration,
# wait for readiness, run the runner, write everything under
# backend/control/evals/manifests/<label>/.
#
#   backend/control/evals/scripts/run_tab_suite.sh <label> <packs> <trials> [KEY=VALUE ...]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
label="$1"; packs="$2"; trials="$3"; shift 3
cd "$ROOT"
env "$@" docker compose --project-directory "$ROOT" -f devops/compose/compose.yaml \
  -f devops/compose/local-opencode-host.yaml -f devops/compose/benchmark.yaml \
  up -d --no-deps toxagent-control >/dev/null
for _ in $(seq 1 90); do
  curl -fsS --max-time 2 http://127.0.0.1:8000/health/ready >/dev/null 2>&1 && break
  sleep 2
done
token="$(grep '^TOXAGENT_STATIC_TOKENS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f1)"
out="backend/control/evals/manifests/$label"
mkdir -p "$out"
printf '%s\n' "$@" > "$out/config.env"
cd backend/control
set +e
.venv-test/bin/python -m evals.runner --runtime opencode --packs "$packs" --trials "$trials" \
  --base-url http://127.0.0.1:8000 --token "$token" --out "evals/manifests/$label" \
  >"evals/manifests/$label/stdout.json" 2>"evals/manifests/$label/stderr.log"
echo "batch $label exit=$?"
