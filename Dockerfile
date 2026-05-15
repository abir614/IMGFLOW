# syntax=docker/dockerfile:1.7-labs

# ─────────────────────────────────────────────────────────────
# IMGFLOW — Production Dockerfile
# Target: HuggingFace Spaces (CPU, x86_64, port 7860)
#
# Base image rationale — python:3.11-slim-bookworm:
#   • Alpine/musl: INCOMPATIBLE — torch, numpy, scipy, onnxruntime
#     all ship manylinux (glibc) wheels only; musl libc breaks them
#   • distroless: no shell → can't run healthcheck curl or venv activate
#   • slim-bookworm: smallest Debian 12 (glibc 2.36) image that runs
#     the full stack; libstdc++6 pre-installed; verified compatible
#     with every wheel in requirements.txt
#   • bullseye (Debian 11): glibc 2.31 also works but older security
#     patches and HF build runners default to bookworm
# ─────────────────────────────────────────────────────────────

# ============================================================
# Stage 1 — Builder
# ============================================================
FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Leave pip cache enabled here — mount cache handles it efficiently
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME=/opt/models/torch \
    U2NET_HOME=/opt/models/u2net

# Build-time deps only — no g++ needed (all packages have pre-built wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VIRTUAL_ENV

WORKDIR /app
COPY backend/requirements.txt .

RUN pip install --upgrade pip setuptools wheel packaging

# PyTorch CPU wheel must be installed BEFORE iopaint to avoid torch version
# conflicts. Explicit CPU index prevents pip pulling a 3 GB CUDA build.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY backend/main.py backend/processing.py ./

# ── Pre-download AI models at build time (zero cold-start delay) ──────────

# ISNet background removal model (~170 MB)
RUN python -c "\
from rembg import new_session; \
new_session('isnet-general-use'); \
print('✓ ISNet ready')"

# LaMa inpainting model (~200 MB) — curl avoids importing iopaint.model
# which would chain-import diffusers/anytext and fail before torch is ready
RUN mkdir -p /opt/models/torch/hub/checkpoints && \
    curl -fsSL --retry 3 --retry-delay 2 \
        -o /opt/models/torch/hub/checkpoints/big-lama.pt \
        https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt && \
    echo "✓ LaMa ready"

# ── Aggressive venv cleanup (~40–80 MB saved) ─────────────────────────────
RUN find $VIRTUAL_ENV \( \
        -type d -name '__pycache__' -o \
        -type d -name 'tests'       -o \
        -type d -name 'test'        -o \
        -type d -name '*.dist-info' -o \
        -type f -name '*.pyc'       -o \
        -type f -name '*.pyo'       -o \
        -type f -name '*.pyd' \
    \) -exec rm -rf {} + 2>/dev/null || true


# ============================================================
# Stage 2 — Runtime  (inherits nothing from builder except COPY)
# ============================================================
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.description="IMGFLOW — FastAPI image processing for HuggingFace Spaces"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    TORCH_HOME=/opt/models/torch \
    U2NET_HOME=/opt/models/u2net \
    # Thread pinning: prevents CPU oversubscription under single-worker uvicorn
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2 \
    # Limit glibc malloc arena growth (saves ~50–150 MB RSS on long-running AI workloads)
    MALLOC_ARENA_MAX=2 \
    OPENCV_OPENCL_RUNTIME=disabled \
    ORT_DISABLE_TELEMETRY=1

# Runtime native libs — deliberately minimal:
#   libglib2.0-0  : cv2 links libglib-2.0.so.0 (not bundled in headless wheel)
#   libgomp1      : OpenMP used by numpy, scipy, torch CPU kernels
#   curl          : healthcheck only — wget not in slim-bookworm by default
# NOT needed (already in slim-bookworm or bundled by wheel):
#   libstdc++6    : pre-installed in every slim image
#   libgl1        : cv2 >= 4.6 headless bundles its own libGL stub
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser

WORKDIR /app

COPY --from=builder /opt/venv    /opt/venv
COPY --from=builder /opt/models  /opt/models
COPY backend/main.py backend/processing.py ./
COPY index.html style.css script.js ./static/

RUN chown -R appuser:appuser /app /opt/models

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
