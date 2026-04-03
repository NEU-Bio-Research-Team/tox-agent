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
    torch==2.4.0+${TORCH_VARIANT} \
    --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# PyG: torch-scatter, torch-geometric
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.4.0+${TORCH_VARIANT}.html

# Google Cloud SDK (Vertex AI, Gemini, Firebase)
RUN pip install --no-cache-dir \
    google-cloud-aiplatform>=1.60.0 \
    google-generativeai>=0.7.0 \
    firebase-admin>=6.5.0 \
    google-auth>=2.29.0

# Install model server dependencies (lean runtime set)
RUN pip install --no-cache-dir -r /app/model_server/requirements.txt

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


