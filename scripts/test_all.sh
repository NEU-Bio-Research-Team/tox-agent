#!/usr/bin/env bash
# Run every package's test suite with an explicit interpreter/environment for
# each — never "whichever python happens to be on PATH". This exists because
# toxocr/tests was once reported as hanging past ~90s (remaining-implementation
# -plan section 2.2); the actual cause was invoking it with the wrong python
# (missing PYTHONPATH, drug-tox-env's torch instead of toxocr-env's), not a
# product hang — confirmed fast (<1s) once invoked as below. See PROGRESS.md
# section 13 (W0-04).
#
# Live-dependent suites (predictor/runtime/evidence marked tests, OpenCode
# eval sweeps) are deliberately out of scope here — this is the fast,
# no-credential, no-network gate every PR should pass, not a release gate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRUG_TOX_PY="${DRUG_TOX_PY:-$HOME/miniconda3/envs/drug-tox-env/bin/python}"
TOXOCR_PY="${TOXOCR_PY:-$HOME/miniconda3/envs/toxocr-env/bin/python}"

run() {
  local label=$1
  shift
  echo "== $label =="
  "$@"
  echo
}

if [[ ! -x "$DRUG_TOX_PY" ]]; then
  echo "DRUG_TOX_PY is not an executable python: $DRUG_TOX_PY" >&2
  echo "Set DRUG_TOX_PY to the interpreter that has sqlalchemy/fastapi/torch/rdkit." >&2
  exit 2
fi
if [[ ! -x "$TOXOCR_PY" ]]; then
  echo "TOXOCR_PY is not an executable python: $TOXOCR_PY" >&2
  echo "Set TOXOCR_PY to toxocr-env's interpreter (torch<2.0, separate from drug-tox-env)." >&2
  exit 2
fi

cd "$REPO_ROOT"

run "toxpred (root package)" "$DRUG_TOX_PY" -m pytest tests -q
run "toxagent-control" "$DRUG_TOX_PY" -m pytest toxagent-control/tests -q
PYTHONPATH="$REPO_ROOT" run "toxocr" "$TOXOCR_PY" -m pytest toxocr/tests -q

if command -v npm >/dev/null 2>&1; then
  (
    cd "$REPO_ROOT/frontend"
    run "frontend typecheck" npm run typecheck
    run "frontend policy lint" npm run lint:policy
    run "frontend build" npm run build
  )
else
  echo "npm not found on PATH; skipping frontend checks." >&2
fi

echo "All suites passed."
