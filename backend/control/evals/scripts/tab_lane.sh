#!/usr/bin/env bash
# One TAB-Suite lane: a dedicated control-plane container on its own loopback
# port, running a list of batches one after another. Several lanes run in
# parallel against the same postgres, toxpred and OpenCode host runtime; each
# run tells OpenCode its own lane's MCP URL, so lanes never cross.
#
#   backend/control/evals/scripts/tab_lane.sh <lane> <port> <batches-file>
#
# batches-file lines:  label|packs|trials|KEY=VALUE KEY=VALUE ...
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
lane="$1"; port="$2"; batches="$3"
base="tox-agent-toxagent-control-1"
name="tox-agent-tab-$lane"
cd "$ROOT"
envfile="$(mktemp)"; chmod 600 "$envfile"
trap 'rm -f "$envfile"' EXIT

while IFS='|' read -r label packs trials overrides; do
  [[ -z "$label" || "$label" == \#* ]] && continue
  docker inspect "$base" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep -E '^TOXAGENT_|^PORT=' \
    | grep -vE '^TOXAGENT_(MCP_RUNTIME_URL|FLAG_|RESEARCH_)' > "$envfile"
  echo "TOXAGENT_MCP_RUNTIME_URL=http://127.0.0.1:$port/internal/mcp" >> "$envfile"
  for kv in $overrides; do echo "$kv" >> "$envfile"; done
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --network tox-agent_default --memory 1g \
    --add-host host.docker.internal:host-gateway \
    -p "127.0.0.1:$port:8000" --env-file "$envfile" \
    -v tox-agent_toxagent-state:/var/lib/toxagent \
    -v "$ROOT/.data/opencode-workspaces:$ROOT/.data/opencode-workspaces" \
    -v "$ROOT/backend/control/evals/fixtures:/opt/toxagent-evals/fixtures:ro" \
    tox-agent-toxagent-control >/dev/null
  for _ in $(seq 1 120); do
    curl -fsS --max-time 2 "http://127.0.0.1:$port/health/ready" >/dev/null 2>&1 && break
    sleep 2
  done
  token="$(grep '^TOXAGENT_STATIC_TOKENS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f1)"
  out="backend/control/evals/manifests/$label"
  mkdir -p "$out"; printf '%s\n' $overrides > "$out/config.env"
  echo "$(date -Is) lane=$lane start $label"
  ( cd backend/control && .venv-test/bin/python -m evals.runner --runtime opencode \
      --packs "$packs" --trials "$trials" --base-url "http://127.0.0.1:$port" --token "$token" \
      --out "evals/manifests/$label" >"evals/manifests/$label/stdout.json" 2>"evals/manifests/$label/stderr.log" ) \
    && rc=0 || rc=$?
  echo "$(date -Is) lane=$lane done $label exit=$rc"
done < "$batches"
docker rm -f "$name" >/dev/null 2>&1 || true
echo "$(date -Is) lane=$lane ALL-DONE"
