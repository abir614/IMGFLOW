# syntax=docker/dockerfile:1.7-labs

# ─────────────────────────────────────────────────────────────
# ULTIMATE AI PRODUCTION DOCKERFILE
#
# Optimized for:
# - FastAPI
# - rembg
# - ONNX Runtime
# - OpenCV
# - HuggingFace Spaces
# - Railway / Render / VPS
#
# Goals:
# - minimal cold starts
# - maximum runtime stability
# - low RAM usage
# - fast builds
# - deterministic environment
# - secure runtime
# - production hardened
# ─────────────────────────────────────────────────────────────

# ============================================================
# Stage 1 — Builder
# ============================================================
FROM python:3.11-slim-bookworm AS builder

# ---------- Environment ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ---------- System Dependencies ----------
# Minimal required native libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Virtual Environment ----------
RUN python -m venv $VIRTUAL_ENV

WORKDIR /app

# ---------- Install Dependencies ----------
COPY backend/requirements.txt .

# Install base tooling first
RUN pip install --upgrade pip setuptools wheel packaging

# Install Python requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---------- Copy Backend ----------
COPY backend/main.py .
COPY backend/processing.py .

# ---------- Pre-download AI Models ----------
ENV U2NET_HOME=/opt/models/u2net

RUN python -c "\
from rembg import new_session; \
new_session('isnet-general-use'); \
print('✓ ISNet model downloaded successfully')"

# ---------- Cleanup ----------
# Removes unnecessary files to shrink final image
RUN find $VIRTUAL_ENV -type d -name '__pycache__' -exec rm -rf {} + && \
    find $VIRTUAL_ENV -type d -name 'tests' -exec rm -rf {} + && \
    find $VIRTUAL_ENV -type f -name '*.pyc' -delete && \
    find $VIRTUAL_ENV -type f -name '*.pyo' -delete

# ============================================================
# Stage 2 — Runtime
# ============================================================
FROM python:3.11-slim-bookworm

LABEL maintainer="Ultimate-AI-Container" \
      description="Ultra-optimized FastAPI + rembg + ONNX Runtime container"

# ---------- Runtime Environment ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    U2NET_HOME=/opt/models/u2net \
    # ---------- AI / NumPy / OpenBLAS Optimizations ----------
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    # ---------- Memory Optimizations ----------
    MALLOC_ARENA_MAX=2 \
    # ---------- OpenCV Optimizations ----------
    OPENCV_OPENCL_RUNTIME=disabled \
    # ---------- ONNX Runtime ----------
    ORT_DISABLE_TELEMETRY=1

# ---------- Runtime Libraries ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- Security ----------
# Non-root runtime user
RUN useradd -m -u 1001 appuser

# ---------- Application Directory ----------
WORKDIR /app

# ---------- Copy Python Environment ----------
COPY --from=builder /opt/venv /opt/venv

# ---------- Copy AI Models ----------
COPY --from=builder /opt/models /opt/models

# ---------- Copy Backend ----------
COPY backend/main.py .
COPY backend/processing.py .

# ---------- Copy Frontend ----------
COPY index.html ./static/index.html
COPY style.css ./static/style.css
COPY script.js ./static/script.js

# ---------- Permissions ----------
RUN mkdir -p /app/static && \
    chown -R appuser:appuser /app /opt/models

USER appuser

# ---------- Healthcheck ----------
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

EXPOSE 7860

# ============================================================
# Production Runtime
#
# IMPORTANT:
# Multiple workers duplicate ONNX/rembg models in RAM.
#
# For AI inference:
#   workers=1 is usually fastest and most stable.
#
# uvloop + httptools significantly improve FastAPI throughput.
# ============================================================
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]
