#!/usr/bin/env bash
# Container entrypoint: fetch artifacts if this deployment names a source,
# verify them, then serve.
set -euo pipefail

# MODEL_ARTIFACTS_URI was passed through Compose and Cloud Run config but
# nothing ever executed it, so a deployment that supplied a bundle URI still
# started with no weights and failed verification below with a message about
# a missing directory (I27). Fetching happens here, once, before the port is
# bound — never during a request.
if [[ -n "${MODEL_ARTIFACTS_URI:-}" ]]; then
  MODELS_ROOT="${MODELS_ROOT:-/app/models}" python /app/deploy/download_model_artifacts.py
fi

python - <<'PY'
from pathlib import Path
from toxpred.scientific.artifacts import load_manifest

manifest = Path("/app/registry/predictor-manifest.yaml")
for specification in load_manifest(manifest, Path("/app/models")).values():
    if specification.required:
        specification.verify()
print("required predictor artifacts verified", flush=True)
PY

exec uvicorn toxpred.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers "${WEB_CONCURRENCY:-1}" \
  --timeout-keep-alive 75
