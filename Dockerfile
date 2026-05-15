# ============================================================
# Stage 1 — Builder
# ============================================================
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Virtualenv
# ------------------------------------------------------------
RUN python -m venv $VIRTUAL_ENV

WORKDIR /app

COPY backend/requirements.txt .

# ------------------------------------------------------------
# Python packages
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# CPU-only torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Main dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Application files
# ------------------------------------------------------------
COPY backend/main.py .
COPY backend/processing.py .

# ------------------------------------------------------------
# Download AI models during build
# ------------------------------------------------------------
ENV U2NET_HOME=/opt/models/u2net

RUN python -c "\
from rembg import new_session; \
new_session('isnet-general-use')"

RUN python -c "\
from iopaint.model.lama import LaMa, LAMA_MODEL_URL, LAMA_MODEL_MD5; \
from iopaint.helper import download_model; \
download_model(LAMA_MODEL_URL, LAMA_MODEL_MD5)"

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
RUN find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /opt/venv -type d -name 'tests' -exec rm -rf {} + && \
    find /opt/venv -type d -name 'test' -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.pyc' -delete && \
    find /opt/venv -type f -name '*.pyo' -delete && \
    find /opt/venv -type f -name '*.a' -delete

# Strip native binaries
RUN find /opt/venv -name "*.so" -exec strip --strip-unneeded {} + || true

# ============================================================
# Stage 2 — Distroless Runtime
# ============================================================
FROM gcr.io/distroless/python3-debian12:nonroot

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    U2NET_HOME=/opt/models/u2net \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    OPENCV_OPENCL_RUNTIME=disabled \
    OPENCV_IO_ENABLE_OPENEXR=0 \
    ORT_DISABLE_TELEMETRY=1

WORKDIR /app

# ------------------------------------------------------------
# Copy Python environment
# ------------------------------------------------------------
COPY --from=builder /opt/venv /opt/venv

# ------------------------------------------------------------
# Copy models
# ------------------------------------------------------------
COPY --from=builder /opt/models /opt/models

# ------------------------------------------------------------
# Copy app
# ------------------------------------------------------------
COPY backend/main.py .
COPY backend/processing.py .

# Static assets
COPY index.html ./static/index.html
COPY style.css ./static/style.css
COPY script.js ./static/script.js
COPY favicon.ico ./static/favicon.ico

# ------------------------------------------------------------
# Expose
# ------------------------------------------------------------
EXPOSE 7860

# ============================================================
# Runtime
# ============================================================
CMD ["/opt/venv/bin/uvicorn", \
     "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]
