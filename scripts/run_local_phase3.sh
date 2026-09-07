#!/usr/bin/env bash
# Start the complete, localhost-only Phase 3 stack.
#
# This deliberately binds every service to 127.0.0.1. It is a development
# harness, not a production launcher. Ctrl-C stops every child process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_DIR="$REPO_ROOT/toxagent-control"
LOCAL_DATA_DIR="$REPO_ROOT/.data"
LOG_DIR="$LOCAL_DATA_DIR/logs"
PYTHON_BIN="${TOXAGENT_PYTHON:-$(command -v python)}"
OPENCODE_BIN="${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
PROFILE="$CONTROL_DIR/agent_profiles/opencode/toxagent.json"
# An isolated HOME/XDG root for the OpenCode worker. `serve --pure` alone did
# not stop the machine's ~/.opencode and ~/.config/opencode from leaking
# `read: allow` and foreign MCP servers into the resolved agent (progress §4.2),
# so the server is launched with HOME and every XDG dir pointed in here.
OPENCODE_HOME="$LOCAL_DATA_DIR/opencode-home"
# Provider credentials still have to reach the isolated home. Point this at the
# real auth file OpenCode wrote (`opencode auth login`); the default is the
# usual XDG location.
OPENCODE_AUTH_FILE="${OPENCODE_AUTH_FILE:-$HOME/.local/share/opencode/auth.json}"
REPLACE_LISTENERS=0
if [[ "${1:-}" == "--replace" ]]; then
  REPLACE_LISTENERS=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--replace]" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "TOXAGENT_PYTHON is not an executable Python: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$OPENCODE_BIN" ]]; then
  echo "OpenCode 1.17.11 was not found at $OPENCODE_BIN; set OPENCODE_BIN." >&2
  exit 2
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "ToxAgent OpenCode profile is missing: $PROFILE" >&2
  exit 2
fi
OCR_PYTHON="${TOXOCR_PYTHON:-}"
if [[ -z "$OCR_PYTHON" || ! -x "$OCR_PYTHON" ]]; then
  echo "TOXOCR_PYTHON is required and must point at the dedicated toxocr environment." >&2
  echo "Provision MolScribe/checkpoint first, then set TOXOCR_PYTHON and optionally TOXOCR_CHECKPOINT_PATH." >&2
  exit 2
fi

# An OpenCode provider/model must be a real configured pair, for example
# `openai/gpt-5.6-luna`. Do not let the control-plane's scripted defaults leak
# into a real OpenCode prompt.
if [[ -z "${TOXAGENT_OPENCODE_MODEL:-}" || "$TOXAGENT_OPENCODE_MODEL" != */* ]]; then
  cat >&2 <<'EOF'
TOXAGENT_OPENCODE_MODEL is required in provider/model form.
Discover configured choices with:  opencode models
Example: TOXAGENT_OPENCODE_MODEL=openai/gpt-5.6-luna ./scripts/run_local_phase3.sh
EOF
  exit 2
fi
MODEL_PROVIDER="${TOXAGENT_OPENCODE_MODEL%%/*}"
MODEL_ID="${TOXAGENT_OPENCODE_MODEL#*/}"
if [[ -z "$MODEL_PROVIDER" || -z "$MODEL_ID" || "$MODEL_PROVIDER" == "$MODEL_ID" ]]; then
  echo "TOXAGENT_OPENCODE_MODEL must be provider/model, got $TOXAGENT_OPENCODE_MODEL" >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

missing = [name for name in ("sqlalchemy", "fastapi", "uvicorn", "aiosqlite", "torch", "rdkit")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        f"{sys.executable} is missing: {', '.join(missing)}. "
        "Activate drug-tox-env, or set TOXAGENT_PYTHON to its bin/python."
    )
PY

if [[ "$($OPENCODE_BIN --version)" != "1.17.11" ]]; then
  echo "This Phase 3 adapter is pinned to OpenCode 1.17.11." >&2
  echo "Found: $($OPENCODE_BIN --version)" >&2
  exit 2
fi

mkdir -p "$LOCAL_DATA_DIR/opencode-runs" "$LOG_DIR" \
  "$OPENCODE_HOME/.config" "$OPENCODE_HOME/.local/share/opencode" \
  "$OPENCODE_HOME/.local/state" "$OPENCODE_HOME/.cache"

# Carry only the provider credentials into the isolated home — never the global
# opencode.json (MCP servers, permission grants) that sits beside it.
if [[ -f "$OPENCODE_AUTH_FILE" ]]; then
  cp "$OPENCODE_AUTH_FILE" "$OPENCODE_HOME/.local/share/opencode/auth.json"
  echo "Copied OpenCode provider auth into the isolated home."
else
  echo "No OpenCode auth file at $OPENCODE_AUTH_FILE." >&2
  echo "If the model call needs one, run 'opencode auth login' or set OPENCODE_AUTH_FILE." >&2
fi

export TOXPRED_MANIFEST="${TOXPRED_MANIFEST:-$REPO_ROOT/artifacts/predictor-manifest.yaml}"
export TOXPRED_DEVICE="${TOXPRED_DEVICE:-cpu}"
export TOXAGENT_DATABASE_URL="${TOXAGENT_DATABASE_URL:-sqlite+aiosqlite:///$LOCAL_DATA_DIR/toxagent-local.db}"
export TOXAGENT_PREDICTOR_URL="${TOXAGENT_PREDICTOR_URL:-http://127.0.0.1:8080}"
export TOXAGENT_OCR_URL="${TOXAGENT_OCR_URL:-http://127.0.0.1:8090}"
export TOXAGENT_RUNTIME_KIND="opencode"
export TOXAGENT_OPENCODE_URL="http://127.0.0.1:4096"
export TOXAGENT_OPENCODE_VERSION="1.17.11"
export TOXAGENT_OPENCODE_DIRECTORY="$LOCAL_DATA_DIR/opencode-runs"
export TOXAGENT_OPENCODE_CREATE_RUN_DIRECTORIES=1
export TOXAGENT_MCP_RUNTIME_URL="http://127.0.0.1:8000/internal/mcp"
export TOXAGENT_PROVIDER_ID="$MODEL_PROVIDER"
export TOXAGENT_MODEL_ID="$MODEL_ID"
export TOXAGENT_ENV="development"
export TOXAGENT_STATIC_TOKENS="${TOXAGENT_STATIC_TOKENS:-dev-local:dev-user:expert}"
# This signs per-run MCP capabilities only. It is deliberately local-only and
# must be replaced by a secret-manager value outside this development harness.
export TOXAGENT_CAPABILITY_SECRET="${TOXAGENT_CAPABILITY_SECRET:-local-phase3-capability-secret-change-me}"
# The frontend's .env.local points at this control plane by absolute URL
# (http://127.0.0.1:8000), which the browser treats as cross-origin from the
# Vite dev server (http://localhost:5173) even though both are loopback — a
# different port is a different origin. Without this, every fetch() carrying
# the Authorization header fails its CORS preflight (a bare 405, no
# Access-Control-Allow-Origin) before the real request is ever sent, and the
# UI silently shows no sessions at all. `api/app.py` only installs the CORS
# middleware when this is non-empty.
export TOXAGENT_CORS_ALLOW_ORIGINS="${TOXAGENT_CORS_ALLOW_ORIGINS:-http://localhost:5173,http://127.0.0.1:5173}"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${FRONTEND_PID:-}" "${CONTROL_PID:-}" "${OCR_PID:-}" "${OPENCODE_PID:-}" "${PREDICTOR_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

start() {
  local name=$1
  shift
  "$@" >"$LOG_DIR/$name.log" 2>&1 &
  LAST_PID=$!
}

# Both local Python applications are source-tree packages rather than installed
# distributions.  Keep their import roots explicit when spawning in the
# background; the migration above happens in a subshell and does not change
# this launcher's working directory.
start_in_dir() {
  local name=$1
  local workdir=$2
  shift 2
  (
    cd "$workdir"
    exec "$@"
  ) >"$LOG_DIR/$name.log" 2>&1 &
  LAST_PID=$!
}

wait_for() {
  local label=$1
  local url=$2
  local pid=$3
  for _ in $(seq 1 45); do
    # Check the PID before probing the socket. Otherwise a failed new process
    # can be mistaken for an older listener that already owns the port.
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited before becoming ready. Last log lines:" >&2
      tail -n 80 "$LOG_DIR/$label.log" >&2 || true
      return 1
    fi
    if curl --fail --silent --show-error --output /dev/null "$url"; then
      return 0
    fi
    sleep 1
  done
  echo "$label did not become ready. See $LOG_DIR/$label.log" >&2
  return 1
}

ensure_port_free() {
  local port=$1
  local listener
  listener="$(fuser -n tcp "$port" 2>/dev/null || true)"
  if [[ -z "$listener" ]]; then
    return 0
  fi
  if [[ "$REPLACE_LISTENERS" != "1" ]]; then
    echo "TCP port $port is already owned by PID(s): $listener" >&2
    echo "Refusing to probe or start against an unknown/old service. Inspect it, or rerun with --replace to stop listeners on the local stack ports." >&2
    exit 1
  fi
  echo "Stopping existing listener(s) on TCP port $port: $listener" >&2
  fuser -k -TERM -n tcp "$port" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    [[ -z "$(fuser -n tcp "$port" 2>/dev/null || true)" ]] && return 0
    sleep 1
  done
  echo "TCP port $port did not become free after TERM." >&2
  exit 1
}

for port in 5173 8000 8080 8090 4096; do
  ensure_port_free "$port"
done

echo "Migrating local SQLite database with $PYTHON_BIN"
(
  cd "$CONTROL_DIR"
  "$PYTHON_BIN" -m alembic upgrade head
)

start_in_dir predictor "$REPO_ROOT" "$PYTHON_BIN" -m uvicorn toxpred.api.app:app --host 127.0.0.1 --port 8080
PREDICTOR_PID=$LAST_PID
wait_for predictor http://127.0.0.1:8080/health/ready "$PREDICTOR_PID"

# OCR is a separate environment because MolScribe pins an older torch. The
# launcher never claims image upload is available unless its service has
# actually loaded a checkpoint. Set TOXOCR_PYTHON to that environment's Python.
start_in_dir ocr "$REPO_ROOT" env PYTHONPATH="$REPO_ROOT" "$OCR_PYTHON" -m uvicorn toxocr.api.app:app --host 127.0.0.1 --port 8090
OCR_PID=$LAST_PID
wait_for ocr http://127.0.0.1:8090/health/ready "$OCR_PID"

# The adapter currently has no password-authentication support. An unsecured
# OpenCode server is acceptable here only because it is bound to loopback.
# `env -i` starts from an empty environment so nothing on this shell (least of
# all a stray OPENCODE_*) reaches the worker; HOME and the XDG dirs are the
# isolated root, and only an explicit allowlist of provider-credential vars is
# forwarded.
opencode_env=(
  "PATH=$PATH"
  "HOME=$OPENCODE_HOME"
  "XDG_CONFIG_HOME=$OPENCODE_HOME/.config"
  "XDG_DATA_HOME=$OPENCODE_HOME/.local/share"
  "XDG_STATE_HOME=$OPENCODE_HOME/.local/state"
  "XDG_CACHE_HOME=$OPENCODE_HOME/.cache"
  "OPENCODE_CONFIG=$PROFILE"
)
for var in OPENAI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY GEMINI_API_KEY \
           GOOGLE_GENERATIVE_AI_API_KEY GROQ_API_KEY DEEPSEEK_API_KEY \
           AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION TERM; do
  if [[ -n "${!var:-}" ]]; then
    opencode_env+=("$var=${!var}")
  fi
done
start_in_dir opencode "$REPO_ROOT" env -i "${opencode_env[@]}" \
  "$OPENCODE_BIN" serve --pure --hostname 127.0.0.1 --port 4096
OPENCODE_PID=$LAST_PID
wait_for opencode "http://127.0.0.1:4096/agent?directory=$LOCAL_DATA_DIR/opencode-runs" "$OPENCODE_PID"

# Gate the run on the *live* resolved surface, not the checked-in profile. If
# anything but the ToxAgent MCP namespace resolves to `allow`, stop here.
if ! "$PYTHON_BIN" "$CONTROL_DIR/scripts/assert_opencode_surface.py" \
      --url http://127.0.0.1:4096 --agent toxagent \
      --directory "$LOCAL_DATA_DIR/opencode-runs"; then
  echo "OpenCode captured surface failed isolation check; not starting the control plane." >&2
  exit 1
fi

start_in_dir control "$CONTROL_DIR" "$PYTHON_BIN" -m uvicorn toxagent.api.app:create_app --factory --host 127.0.0.1 --port 8000
CONTROL_PID=$LAST_PID
wait_for control http://127.0.0.1:8000/health/live "$CONTROL_PID"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to start the frontend." >&2
  exit 2
fi
start_in_dir frontend "$REPO_ROOT/frontend" npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
FRONTEND_PID=$LAST_PID
wait_for frontend http://127.0.0.1:5173 "$FRONTEND_PID"

cat <<EOF

Local Phase 3 stack is ready:
  ToxPred:  http://127.0.0.1:8080  (checkpoint-backed hERG + Tox21)
  ToxOCR:   http://127.0.0.1:8090  (MolScribe image -> SMILES)
  OpenCode: http://127.0.0.1:4096  (private runtime API)
  ToxAgent: http://127.0.0.1:8000  (Bearer token: dev-local)
  Frontend: http://127.0.0.1:5173

Run ./scripts/smoke_local_phase3.sh in a second terminal to execute an
actual SMILES prediction followed by a grounded report-Q&A attempt.
Logs: $LOG_DIR
EOF

# Return as soon as one component exits so the trap tears down the other two.
wait -n "$PREDICTOR_PID" "$OCR_PID" "$OPENCODE_PID" "$CONTROL_PID" "$FRONTEND_PID"
