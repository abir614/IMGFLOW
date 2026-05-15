# ============================================================
# Stage 1 — Builder
# Ultra-optimized wheel + model builder
# ============================================================
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Only absolute minimum build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv $VIRTUAL_ENV

WORKDIR /app

COPY backend/requirements.txt .

# Faster pip + smaller install
RUN pip install --upgrade pip wheel setuptools

# Install CPU-only torch first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy backend only
COPY backend/main.py .
COPY backend/processing.py .

# ============================================================
# Pre-download models
# ============================================================

ENV U2NET_HOME=/opt/models/u2net

RUN python -c "\
from rembg import new_session; \
new_session('isnet-general-use')"

RUN python -c "\
from iopaint.model.lama import LaMa, LAMA_MODEL_URL, LAMA_MODEL_MD5; \
from iopaint.helper import download_model; \
download_model(LAMA_MODEL_URL, LAMA_MODEL_MD5)"

# ============================================================
# Aggressive cleanup
# ============================================================

RUN find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /opt/venv -type d -name 'tests' -exec rm -rf {} + && \
    find /opt/venv -type d -name 'test' -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.pyc' -delete && \
    find /opt/venv -type f -name '*.pyo' -delete && \
    find /opt/venv -type f -name '*.a' -delete && \
    find /opt/venv -type f -name '*.so.debug' -delete && \
    strip --strip-unneeded /opt/venv/lib/python3.11/site-packages/**/*.so 2>/dev/null || true

# ============================================================
# Stage 2 — Runtime
# Distroless-style slim runtime
# ============================================================
FROM python:3.11-slim-bookworm

LABEL maintainer="Ultimate-AI-Container" \
      description="Extreme optimized FastAPI + rembg + LaMa container"

# ============================================================
# Runtime environment optimizations
# ============================================================

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
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

# Runtime-only libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -r -u 1001 -g users appuser

WORKDIR /app

# Copy only final runtime artifacts
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

# Backend
COPY backend/main.py .
COPY backend/processing.py .

# Static assets
COPY index.html ./static/index.html
COPY style.css ./static/style.css
COPY script.js ./static/script.js
RUN wget -O ./static/favicon.ico "https://raw.githubusercontent.com/abir614/IMGFLOW/refs/heads/main/favicon.ico"

# Permissions
RUN mkdir -p /app/static && \
    chown -R appuser:users /app /opt/models

USER appuser

EXPOSE 7860

# ============================================================
# Healthcheck
# ============================================================

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

# ============================================================
# Launch
# ============================================================

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]
