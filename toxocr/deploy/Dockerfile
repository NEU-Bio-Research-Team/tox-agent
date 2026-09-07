# syntax=docker/dockerfile:1.7
# toxocr: optical chemical structure recognition (image -> SMILES).
#
# Build: docker build -f toxocr/deploy/Dockerfile -t toxocr toxocr
#
# Build context is this package's own directory — same three/four-boundary
# reasoning as toxagent-control's Dockerfile (ADR 0006): this image must not
# be able to import toxpred or toxagent-control even by accident.
#
# MolScribe pins torch<2.0 (requirements.txt) — this image never shares a
# base with toxpred's (torch 2.6) or toxagent-control's (no torch at all).
# Pinned by digest (2026-09-06) — W6-11, same reasoning as
# toxagent-control/deploy/Dockerfile.
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2

ENV PORT=8090 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

WORKDIR /app

# RDKit needs the cairo/freetype stack; MolScribe's opencv-python import
# needs libGL even headless (the pure "-headless" opencv package is not what
# molscribe's own dependency pins in); curl is for the health probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxrender1 libxext6 libcairo2 libfontconfig1 libfreetype6 \
      libgl1 libglib2.0-0 libsm6 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# --extra-index-url, not --index-url: the latter replaces PyPI entirely, and
# torch's own wheel index does not host its transitive build dependencies
# (e.g. typing-extensions) — a clean image (unlike a conda env that already
# has them) then fails trying to build one from source.
ARG TORCH_VARIANT=cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==1.13.1+${TORCH_VARIANT} --extra-index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

COPY requirements.txt /app/requirements.txt
# molscribe's pinned deps (timm, torchvision 0.14, albumentations 1.1.0)
# predate NumPy 2.0's ABI break — requirements.txt already pins numpy<2, but
# a second explicit pass after molscribe resolves it the same way the
# verified local install sequence does (requirements.txt's own comment).
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt && pip install "numpy<2"

# The build context (toxocr/) *is* the package's own contents — there is no
# nested toxocr/toxocr/ the way a packaged project would have — so the
# import root has to be one level above where the source lands, exactly like
# every other place in this repo runs it with PYTHONPATH pointed at the
# monorepo root. Recreate that shape inside the image instead.
COPY . /app/toxocr/
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu

# Fail the build rather than the first request if the package cannot import.
# This never touches molscribe/torch's model-loading path — that import is
# lazy, inside MolScribePredictor.load() (scientific/molscribe_predictor.py).
RUN python -c "from toxocr.api.app import create_app; print('import OK')"

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health/ready" || exit 1

# CUDA 11.7/cuDNN resolves the unversioned driver name, while the NVIDIA
# container runtime commonly mounts only libcuda.so.1. Create the compatible
# link after the runtime mount is present, at container startup.
CMD ["sh", "-c", "driver_lib=$(ldconfig -p | awk '/libcuda\\.so\\.1/{print $NF; exit}'); if [ -n \"$driver_lib\" ] && [ ! -e /usr/local/lib/libcuda.so ]; then ln -s \"$driver_lib\" /usr/local/lib/libcuda.so; fi; exec uvicorn toxocr.api.app:app --host 0.0.0.0 --port ${PORT}"]
