# model_server/Dockerfile
# Supports: Cloud Run (CPU), Vertex AI Endpoint (CPU/GPU), Vertex AI training

#-------------------------------------------------------------------
# Build arg để switch CPU <-> không cần sửa file
# Build CPU:    docker build --build-arg TORCH_VARIANT=cpu
# Build GPU:    docker build --build-arg TORCH_VARIANT=cu121
#-------------------------------------------------------------------
ARG TORCH_VARIANT=cpu

# Use python 3.10 slim as base 
FROM python:3.10-slim

# Metadata cho Vertex AI Prediction
# Vertex AI sẽ gọi GET /health và POST /predict
# Có thể override qua env vars khi deploy
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

# Google Cloud project config (inject khi deploy)
# Không hardcore ở đây - truyền qua --set-env-vars khi gcloud deploy
ENV GOOGLE_CLOUD_PROJECT=""
ENV GOOGLE_CLOUD_LOCATION="us-central1"
ENV MOLSCRIBE_PRELOAD_ON_STARTUP="true"
ENV MOLSCRIBE_AUTO_DOWNLOAD="true"
ENV MOLSCRIBE_REPO_ID="yujieq/MolScribe"
ENV MOLSCRIBE_CHECKPOINT_NAME="swin_base_char_aux_1m.pth"

# Python behaviour
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working dir
WORKDIR /app

# System dependencies for RDKit and PyTorch Geometric
# Thêm curl để health probe từ bên ngoài container
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libxrender1 libxext6 libcairo2 libfontconfig1 libfreetype6 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy XSmiles source (entire project, since model_server import src/)
COPY . /app/

# PyTorch (CPU default, GPU khi có quota)
# TORCH_VARIANT được truyền từ build arg
ARG TORCH_VARIANT=cpu
RUN pip install --no-cache-dir \
    torch==2.6.0+${TORCH_VARIANT} \
    torchvision==0.21.0+${TORCH_VARIANT} \
    --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    --extra-index-url https://pypi.org/simple

# PyG: torch-scatter, torch-geometric
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.6.0+${TORCH_VARIANT}.html

# Google Cloud SDK (Vertex AI, Gemini, Firebase)
RUN pip install --no-cache-dir \
    google-cloud-aiplatform>=1.60.0 \
    google-generativeai>=0.7.0 \
    firebase-admin>=6.5.0 \
    google-auth>=2.29.0

# Install model server dependencies (lean runtime set)
RUN pip install --no-cache-dir -r /app/model_server/requirements.txt

# MolScribe depends on OpenNMT (module `onmt`), but both MolScribe/OpenNMT pin
# torch<2.0 in metadata. Install without deps to preserve the pre-installed
# torch runtime used by the toxicity pipeline.
RUN pip install --no-cache-dir --no-deps OpenNMT-py==2.2.0

# OpenNMT-py 2.2.0 imports legacy torchtext (torchtext.data) from its package
# __init__, which is not needed for MolScribe inference and does not have
# reliable wheels on modern Python. Patch the package init to avoid importing
# training/data-loading modules at import time.
RUN python -c "import site; from pathlib import Path; init_path=next((Path(base) / 'onmt' / '__init__.py' for base in site.getsitepackages() if (Path(base) / 'onmt' / '__init__.py').exists()), None); assert init_path is not None, 'Could not locate onmt/__init__.py in site-packages'; init_path.write_text('\"\"\"Trimmed OpenNMT package init for MolScribe inference.\n\"\"\"\n__version__ = \"2.2.0\"\n'); print(f'Patched {init_path}')"

# MolScribe currently pins torch<2.0 in metadata, so install without deps to
# preserve the pre-installed torch runtime used by the toxicity pipeline.
RUN pip install --no-cache-dir --no-deps molscribe==1.1.1

# Fail the image build early if the optional image-to-SMILES stack cannot import.
RUN python -c "from molscribe import MolScribe; print('MolScribe import OK')"

# Export port (Cloud Run uses 8080)
EXPOSE 8080

# Health probe (Cloud Run sẽ dùng cái này)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${AIP_HTTP_PORT}${AIP_HEALTH_ROUTE}" || exit 1

# Start FastAPI server
CMD ["uvicorn", "model_server.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--workers", "1", \
    "--timeout-keep-alive", "75"]


