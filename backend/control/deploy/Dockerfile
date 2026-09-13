# ToxAgent control plane.
#
# Build: docker build -f deploy/Dockerfile -t toxagent-control .
# Test:  docker build --target test -f deploy/Dockerfile -t toxagent-control:test . && docker run --rm toxagent-control:test
#
# Build context is this package's own directory, not the repo root — ADR
# 0001's three-boundary topology means this image must not be able to import
# toxpred or toxocr even by accident, and the surest way to guarantee that is
# for their source to never be inside the build context at all.
#
# Three stages. `base` is the application and its production dependencies;
# `test` adds the dev extras and the suites; `runtime` is what ships. The split
# is not cosmetic — before it, `COPY . /app/` put the tests, the eval corpus and
# the fixtures into the deployed image, and there was no documented way to run
# the suite in a container at all (P2-3, P2-4 of the 2026-09-13 audit).
#
# Pinned by digest (2026-09-06) — W6-11: a floating tag can change under a
# rebuild with no diff in this repo to review. Re-pin deliberately, not by
# accident, when a base image needs to move.
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS base

ENV PORT=8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# curl is for the health probe; nothing here needs rdkit/torch (this package
# imports neither — see tests/unit/test_import_boundaries.py).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
# Editable install needs to discover the src layout, but dependencies must not
# be invalidated by every application-source edit. The real package replaces
# this minimal marker in the next COPY layer.
RUN mkdir -p /app/src/toxagent && touch /app/src/toxagent/__init__.py
# The production image talks to PostgreSQL only (asyncpg/psycopg) — the
# aiosqlite dev path exists for local/CI convenience, not for a deployable
# image (pyproject.toml's own comment on the `postgres` extra).
RUN pip install --no-cache-dir -e '.[postgres]'

# Named explicitly rather than `COPY . /app/`: what the application needs to
# run is a short list, and anything not on it has no business in a deployed
# image.
COPY src /app/src
COPY migrations /app/migrations
COPY alembic.ini /app/alembic.ini
COPY deploy /app/deploy

# Fail the build rather than the first request if the package cannot import.
RUN python -c "from toxagent.api.app import create_app; print('import OK')"


# --- test -------------------------------------------------------------------
# Everything `make test` runs, in a container, with no Python on the host.
FROM base AS test

RUN pip install --no-cache-dir -e '.[postgres,dev]'
COPY tests /app/tests
COPY evals /app/evals
COPY scripts /app/scripts
# The regeneration half of the predictor contract test skips here, by design:
# the predictor source is not in this build context and must not be.
ENV TOXAGENT_REQUIRE_PREDICTOR_CONTRACT=0
CMD ["sh", "-c", "python -m pytest tests -q && python -m evals.runner --runtime scripted --trials 3"]


# --- runtime ----------------------------------------------------------------
# The default target, and the only one that ships. No pytest, no aiosqlite, no
# tests, no eval fixtures.
FROM base AS runtime

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health/live" || exit 1

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
