#!/usr/bin/env bash
# Supervisor for the local, user-owned OpenCode runtime used by `toxagent up --agent`.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCODE_BIN="${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
PIN="1.17.11"
AUTH_ROOT="$ROOT/.data/opencode-auth"
RUNTIME_ROOT="$ROOT/.data/opencode-runtime"
WORKSPACES="$ROOT/.data/opencode-workspaces"
LOG_DIR="$ROOT/.data/logs"
PROFILE="$ROOT/toxagent-control/agent_profiles/opencode/toxagent.json"
SERVER_PID="$RUNTIME_ROOT/server.pid"
BRIDGE_PID="$RUNTIME_ROOT/bridge.pid"
SERVER_LOG="$LOG_DIR/opencode-runtime.log"
BRIDGE_LOG="$LOG_DIR/opencode-bridge.log"

fail() { printf 'toxagent agent: %s\n' "$*" >&2; exit 1; }
running() { [[ -f "$1" ]] && kill -0 "$(<"$1")" 2>/dev/null; }
stop_pid() { [[ -f "$1" ]] || return 0; local pid; pid="$(<"$1")"; kill "$pid" 2>/dev/null || true; rm -f "$1"; }

start() {
  [[ -x "$OPENCODE_BIN" ]] || fail "OpenCode $PIN was not found at $OPENCODE_BIN"
  [[ "$("$OPENCODE_BIN" --version)" == "$PIN" ]] || fail "OpenCode $PIN is required"
  [[ -f "$AUTH_ROOT/data/opencode/auth.json" ]] || fail "no isolated provider auth; run ./bin/toxagent setup --agent"
  mkdir -p "$RUNTIME_ROOT"/{home,config,state,cache} "$WORKSPACES" "$LOG_DIR"
  chmod 700 "$AUTH_ROOT" "$AUTH_ROOT/data" "$AUTH_ROOT/data/opencode" "$RUNTIME_ROOT" "$WORKSPACES"

  if ! running "$SERVER_PID"; then
    rm -f "$SERVER_PID"
    setsid env -i \
      "PATH=$PATH" \
      "HOME=$RUNTIME_ROOT/home" \
      "XDG_DATA_HOME=$AUTH_ROOT/data" \
      "XDG_CONFIG_HOME=$RUNTIME_ROOT/config" \
      "XDG_STATE_HOME=$RUNTIME_ROOT/state" \
      "XDG_CACHE_HOME=$RUNTIME_ROOT/cache" \
      "OPENCODE_CONFIG=$PROFILE" \
      "TERM=${TERM:-xterm}" \
      "$OPENCODE_BIN" serve --pure --hostname 127.0.0.1 --port 4096 >"$SERVER_LOG" 2>&1 < /dev/null &
    echo $! > "$SERVER_PID"
  fi
  for _ in $(seq 1 20); do
    curl -fsS --max-time 2 "http://127.0.0.1:4096/agent?directory=$WORKSPACES" >/dev/null 2>&1 && break
    running "$SERVER_PID" || { tail -80 "$SERVER_LOG" >&2 || true; fail "OpenCode exited during startup"; }
    sleep 1
  done
  python3 "$ROOT/toxagent-control/scripts/assert_opencode_surface.py" \
    --url http://127.0.0.1:4096 --agent toxagent --directory "$WORKSPACES"

  local gateway
  gateway="$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')"
  if ! running "$BRIDGE_PID"; then
    rm -f "$BRIDGE_PID"
    setsid python3 "$ROOT/scripts/opencode_docker_bridge.py" --bind "$gateway" >"$BRIDGE_LOG" 2>&1 < /dev/null &
    echo $! > "$BRIDGE_PID"
  fi
  sleep 1
  running "$BRIDGE_PID" || { tail -80 "$BRIDGE_LOG" >&2 || true; fail "Docker bridge proxy exited during startup"; }
  printf 'OpenCode runtime ready on loopback; Docker bridge is private.\n'
}

case "${1:-}" in
  start) start ;;
  stop) stop_pid "$BRIDGE_PID"; stop_pid "$SERVER_PID" ;;
  status)
    running "$SERVER_PID" && echo 'OpenCode: running' || echo 'OpenCode: stopped'
    running "$BRIDGE_PID" && echo 'Docker bridge: running' || echo 'Docker bridge: stopped'
    ;;
  logs) tail -n "${2:-200}" "$SERVER_LOG" "$BRIDGE_LOG" 2>/dev/null || true ;;
  *) fail 'usage: opencode_local_runtime.sh {start,stop,status,logs [lines]}' ;;
esac
