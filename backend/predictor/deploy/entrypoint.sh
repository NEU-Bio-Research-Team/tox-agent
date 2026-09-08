#!/usr/bin/env bash
# Container entrypoint: verify immutable, image-bundled artifacts, then serve.
set -euo pipefail

python - <<'PY'
from pathlib import Path
from toxpred.scientific.artifacts import load_manifest

manifest = Path("/app/registry/predictor-manifest.yaml")
for specification in load_manifest(manifest).values():
    if specification.required:
        specification.verify()
print("required predictor artifacts verified", flush=True)
PY

exec uvicorn toxpred.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers "${WEB_CONCURRENCY:-1}" \
  --timeout-keep-alive 75
