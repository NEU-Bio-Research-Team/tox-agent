#!/usr/bin/env bash
set -euo pipefail

python /app/model_server/scripts/download_model_artifacts.py

exec uvicorn model_server.main:app \
  --host 0.0.0.0 \
  --port "${AIP_HTTP_PORT:-8080}" \
  --workers 1 \
  --timeout-keep-alive 75