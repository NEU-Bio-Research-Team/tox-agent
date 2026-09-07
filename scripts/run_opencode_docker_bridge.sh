#!/usr/bin/env bash
# Serve the pinned OpenCode runtime only to containers on the current Docker
# bridge. Credentials are copied into an isolated XDG home; global OpenCode
# configuration and its unrelated MCP servers never leak into ToxAgent.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
opencode_bin="${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
auth_file="${OPENCODE_AUTH_FILE:-$HOME/.local/share/opencode/auth.json}"
runtime_home="$repo_root/.data/opencode-home"
run_directory="$repo_root/.data/opencode-runs"
profile="$repo_root/toxagent-control/agent_profiles/opencode/toxagent.json"
gateway="${TOXAGENT_OPENCODE_BRIDGE_HOST:-$(docker network inspect tox-agent_default --format '{{(index .IPAM.Config 0).Gateway}}')}"

test -x "$opencode_bin"
test -f "$auth_file"
test -f "$profile"
mkdir -p "$run_directory" "$runtime_home/.config" "$runtime_home/.local/share/opencode" "$runtime_home/.local/state" "$runtime_home/.cache"
cp "$auth_file" "$runtime_home/.local/share/opencode/auth.json"

exec env -i \
  "PATH=$PATH" "HOME=$runtime_home" \
  "XDG_CONFIG_HOME=$runtime_home/.config" \
  "XDG_DATA_HOME=$runtime_home/.local/share" \
  "XDG_STATE_HOME=$runtime_home/.local/state" \
  "XDG_CACHE_HOME=$runtime_home/.cache" \
  "OPENCODE_CONFIG=$profile" \
  "$opencode_bin" serve --pure --hostname "$gateway" --port 4096
