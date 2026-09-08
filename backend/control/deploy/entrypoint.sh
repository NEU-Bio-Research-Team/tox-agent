#!/usr/bin/env bash
# Container entrypoint: migrate, then serve.
#
# ADR: forward-only production migration, run once before the app binds a
# port — a request must never trigger a schema change, and a container that
# starts serving before its own schema is current would silently corrupt the
# audit trail's ordering guarantees. Set TOXAGENT_SKIP_MIGRATIONS=1 only for
# a deliberately read-only replica of an already-migrated database.
set -euo pipefail
cd /app

if [ "${TOXAGENT_SKIP_MIGRATIONS:-0}" != "1" ]; then
  alembic -c /app/alembic.ini upgrade head
fi

exec uvicorn toxagent.api.app:create_app \
  --factory \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-1}" \
  --timeout-keep-alive 75
