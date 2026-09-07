# ToxAgent control plane.
#
# Build: docker build -f toxagent-control/deploy/Dockerfile -t toxagent-control toxagent-control
#
# Build context is this package's own directory, not the repo root — ADR
# 0001's three-boundary topology means this image must not be able to import
# toxpred or toxocr even by accident, and the surest way to guarantee that is
# for their source to never be inside the build context at all.
# Pinned by digest (2026-09-06) — W6-11: a floating tag can change under a
# rebuild with no diff in this repo to review. Re-pin deliberately, not by
# accident, when a base image needs to move.
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2

ENV PORT=8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# curl is for the health probe; nothing here needs rdkit/torch (this package
# imports neither — see tests/unit/test_import_boundaries.py).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
# The production image talks to PostgreSQL only (asyncpg/psycopg) — the
# aiosqlite dev path exists for local/CI convenience, not for a deployable
# image (pyproject.toml's own comment on the `postgres` extra).
RUN pip install --no-cache-dir -e '.[postgres]'

COPY . /app/

# Fail the build rather than the first request if the package cannot import.
RUN python -c "from toxagent.api.app import create_app; print('import OK')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health/live" || exit 1

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
