#!/usr/bin/env bash
# One ToxAgent arm of the comparison study (backend/control/evals/investigation):
# a dedicated control-plane container on its own loopback port, sharing
# postgres, toxpred and the OpenCode host runtime with the base stack, with the
# flag overrides that define the arm. Modelled on tab_lane.sh.
#
#   backend/control/evals/scripts/study_arm.sh <name> <port> [KEY=VALUE ...]
#   backend/control/evals/scripts/study_arm.sh D 8012 TOXAGENT_FLAG_ANSWER_DRAFT_V2=1 \
#       TOXAGENT_FLAG_SCIENTIFIC_CASE_V1=1 TOXAGENT_FLAG_SCIENTIFIC_SKILLS_V1=1
#   backend/control/evals/scripts/study_arm.sh --stop D
#
# The base container (tox-agent-toxagent-control-1, started with compose.yaml +
# local-opencode-host.yaml + benchmark.yaml) supplies every other setting, so
# arms differ only by their overrides. The study runner checks each arm's
# /v1/system/effective-product before recording anything under its name.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
base="tox-agent-toxagent-control-1"

if [[ "${1:-}" == "--stop" ]]; then
  docker rm -f "tox-agent-study-$2" >/dev/null 2>&1 || true
  exit 0
fi

arm="$1"; port="$2"; shift 2
name="tox-agent-study-$arm"
cd "$ROOT"
envfile="$(mktemp)"; chmod 600 "$envfile"
trap 'rm -f "$envfile"' EXIT

docker inspect "$base" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^TOXAGENT_|^PORT=' \
  | grep -vE '^TOXAGENT_(MCP_RUNTIME_URL|FLAG_)' > "$envfile"
echo "TOXAGENT_MCP_RUNTIME_URL=http://127.0.0.1:$port/internal/mcp" >> "$envfile"
# The base container already migrated the shared database.
echo "TOXAGENT_SKIP_MIGRATIONS=1" >> "$envfile"
for kv in "$@"; do echo "$kv" >> "$envfile"; done

docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --network tox-agent_default --memory 1g \
  --add-host host.docker.internal:host-gateway \
  -p "127.0.0.1:$port:8000" --env-file "$envfile" \
  -v tox-agent_toxagent-state:/var/lib/toxagent \
  -v "$ROOT/.data/opencode-workspaces:$ROOT/.data/opencode-workspaces" \
  -v "$ROOT/backend/control/evals/fixtures:/opt/toxagent-evals/fixtures:ro" \
  tox-agent-toxagent-control >/dev/null
for _ in $(seq 1 120); do
  curl -fsS --max-time 2 "http://127.0.0.1:$port/health/ready" >/dev/null 2>&1 && { echo "arm $arm ready on $port"; exit 0; }
  sleep 2
done
docker logs --tail 40 "$name" >&2
echo "arm $arm did not become ready" >&2
exit 1
