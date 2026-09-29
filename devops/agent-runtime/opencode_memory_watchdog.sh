#!/usr/bin/env bash
# Restart the local OpenCode runtime when its RSS passes a ceiling. Observed
# during TAB-Suite lanes: RSS grows with every session (4.3 GB after ~80 runs on
# an 8 GB WSL host) until the host runs out of memory. A restart drops the
# turns in flight; the eval runner records those as infra_error, not as fails.
#   opencode_memory_watchdog.sh [limit_mb=2500] [interval_s=60]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
limit_mb="${1:-2500}"; interval="${2:-60}"
while true; do
  rss_kb="$(ps -o rss= -C opencode | sort -rn | head -1 | tr -d ' ')"
  if [[ -n "$rss_kb" && "$rss_kb" -gt $((limit_mb * 1024)) ]]; then
    echo "$(date -Is) opencode rss=$((rss_kb / 1024))MB > ${limit_mb}MB; restarting"
    "$ROOT/devops/agent-runtime/opencode_local_runtime.sh" stop
    "$ROOT/devops/agent-runtime/opencode_local_runtime.sh" start | tail -1
  fi
  pgrep -f tab_lane.sh >/dev/null || { echo "$(date -Is) no lanes running; watchdog exits"; exit 0; }
  sleep "$interval"
done
