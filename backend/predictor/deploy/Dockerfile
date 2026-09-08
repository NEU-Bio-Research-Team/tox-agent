# syntax=docker/dockerfile:1.7
# ToxPred prediction service.
#
# Build CPU: docker build -f backend/predictor/deploy/Dockerfile --build-arg TORCH_VARIANT=cpu -t toxpred .
#
# Model artifacts are deployment-owned and mounted or fetched before startup.
# The image carries manifests and code only; a missing/corrupt required artifact
# fails startup rather than being silently downloaded during a request.
ARG TORCH_VARIANT=cpu

FROM python:3.10-slim

ENV PORT=8080 \
    MODELS_ROOT=/app/models \
    TOXPRED_MANIFEST=/app/registry/predictor-manifest.yaml \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

WORKDIR /app

# RDKit needs the cairo/freetype stack; curl is for the health probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxrender1 libxext6 libcairo2 libfontconfig1 libfreetype6 curl \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VARIANT=cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 600 --retries 5 torch==2.6.0+${TORCH_VARIANT} \
      --extra-index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

COPY backend/predictor/deploy/requirements.txt /app/deploy/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 600 --retries 5 -r /app/deploy/requirements.txt

COPY backend/predictor/src/toxpred /app/toxpred
COPY backend/predictor/registry /app/registry
COPY backend/predictor/configs /app/configs
COPY backend/predictor/deploy/entrypoint.sh /app/deploy/entrypoint.sh

# Fail the build rather than the first request if the package cannot import.
RUN python -c "import toxpred; from toxpred.api.app import create_app; print('import OK')"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health/live" || exit 1

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
