# ToxPred prediction service.
#
# Build CPU:  docker build -f deploy/Dockerfile --build-arg TORCH_VARIANT=cpu -t toxpred .
# Build GPU:  docker build -f deploy/Dockerfile --build-arg TORCH_VARIANT=cu121 -t toxpred .
#
# The image carries no model weights: .dockerignore excludes models/, and
# deploy/entrypoint.sh fetches them from MODEL_ARTIFACTS_URI at container start.
ARG TORCH_VARIANT=cpu

FROM python:3.10-slim

ENV PORT=8080 \
    MODELS_ROOT=/app/models \
    MODEL_ARTIFACTS_URI="" \
    TOXPRED_MANIFEST=/app/artifacts/manifest.yaml \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# RDKit needs the cairo/freetype stack; curl is for the health probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxrender1 libxext6 libcairo2 libfontconfig1 libfreetype6 curl \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VARIANT=cpu
RUN pip install --no-cache-dir \
      torch==2.6.0+${TORCH_VARIANT} \
      --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# torch-geometric is required only by the ClinTox graph model. It is kept so
# that endpoint returns the moment its tokenizer is restored.
RUN pip install --no-cache-dir \
      torch-scatter torch-sparse torch-geometric \
      -f https://data.pyg.org/whl/torch-2.6.0+${TORCH_VARIANT}.html

COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

COPY . /app/

# Fail the build rather than the first request if the package cannot import.
RUN python -c "import toxpred; from toxpred.api.app import create_app; print('import OK')"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health/live" || exit 1

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
