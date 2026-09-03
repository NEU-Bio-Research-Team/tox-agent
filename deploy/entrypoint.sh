#!/usr/bin/env bash
# Container entrypoint: provision artifacts, then serve.
#
# Artifact download happens here and only here. A request must never trigger a
# download — that is what turns a cold instance into a 600-second stall.
set -euo pipefail

python /app/deploy/download_model_artifacts.py

exec uvicorn toxpred.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers "${WEB_CONCURRENCY:-1}" \
  --timeout-keep-alive 75
