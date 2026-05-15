# syntax=docker/dockerfile:1.7-labs

# ═══════════════════════════════════════════════════════════════════════════
# IMGFLOW — Distroless Production Dockerfile
# Target: HuggingFace Spaces (CPU, x86_64, port 7860)
#
# Stage layout:
#   builder  — python:3.11-slim-bookworm
#              Installs all Python deps, pre-downloads AI models.
#              Has gcc + curl + libGL for build-time imports (rembg, iopaint).
#
#   syslibs  — debian:12-slim
#              Installs ONLY the system .so files that distroless is missing:
#                libglib2.0-0  → libglib-2.0.so.0 + libgthread-2.0.so.0
#                libgomp1      → libgomp.so.1  (torch CPU kernels)
#                libpcre2-8-0  → libpcre2-8.so.0 (glib transitive dep)
#              Nothing else — no Python, no pip, no shell utilities.
#
#   runtime  — gcr.io/distroless/python3-debian12:nonroot
#              ~20 MB base. Contains Python 3.11, glibc, libssl, libz,
#              libstdc++, libgcc_s. No shell, no package manager, no curl.
#              Inherits /opt/venv + /opt/models from builder.
#              Inherits missing .so files from syslibs.
#
# Why distroless over slim-bookworm:
#   • Eliminates shell, apt, coreutils, etc. → smaller CVE surface
#   • Forces immutable, minimal runtime — no exec into container possible
#   • ~60–80 MB smaller final image
#
# Distroless constraints handled:
#   • No shell → CMD must use exec-form JSON array (already the case)
#   • No curl → HEALTHCHECK removed; use Docker/HF Spaces liveness probes
#   • No /tmp by default → tmpfile use in rawpy gets /tmp from distroless base
#   • UID 65532 (nonroot) → all paths chowned in builder before COPY
# ═══════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME=/opt/models/torch \
    U2NET_HOME=/opt/models/u2net

# Build-time native deps:
#   gcc          — some wheels still compile a small C shim at install time
#   curl         — LaMa model download
#   ca-certs     — HTTPS in curl + Python urllib
#   libgl1       — cv2 import at build time (rembg model pre-download)
#   libglib2.0-0 — cv2 / rembg import at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VIRTUAL_ENV

WORKDIR /app
COPY backend/requirements.txt .

RUN pip install --upgrade pip setuptools wheel packaging

# Install PyTorch CPU FIRST — prevents iopaint pulling a 3 GB CUDA build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY backend/main.py backend/processing.py ./

# ── Pre-download AI models at build time ─────────────────────────────────
# ISNet (~170 MB) — background removal
RUN python -c "\
from rembg import new_session; \
new_session('isnet-general-use'); \
print('✓ ISNet ready')"

# LaMa (~196 MB) — inpainting. Use curl directly: importing iopaint.model
# at build time chain-imports diffusers which is slow and error-prone.
RUN mkdir -p /opt/models/torch/hub/checkpoints && \
    curl -fsSL --retry 3 --retry-delay 2 \
        -o /opt/models/torch/hub/checkpoints/big-lama.pt \
        https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt && \
    echo "✓ LaMa ready"

# ── Cleanup: strip ~60–100 MB of test files, bytecode, dist-info ─────────
RUN find $VIRTUAL_ENV \( \
        -type d -name '__pycache__' -o \
        -type d -name 'tests'       -o \
        -type d -name 'test'        -o \
        -type d -name '*.dist-info' -o \
        -type f -name '*.pyc'       -o \
        -type f -name '*.pyo'       -o \
        -type f -name '*.pyd' \
    \) -exec rm -rf {} + 2>/dev/null || true

# ── Fix ownership for distroless nonroot UID 65532 ───────────────────────
# Must be done here because distroless has no chown/shell at runtime.
RUN chown -R 65532:65532 /opt/venv /opt/models /app


# ───────────────────────────────────────────────────────────────────────────
# Stage 2 — syslibs
# Extract the system .so files that distroless/python3-debian12 is missing.
# We install them here into a clean Debian layer, then COPY just the .so
# files into the final stage — no apt, no shell, no bloat carried over.
# ───────────────────────────────────────────────────────────────────────────
FROM debian:12-slim AS syslibs

RUN apt-get update && apt-get install -y --no-install-recommends \
        # cv2 (headless) links libglib-2.0.so.0 + libgthread-2.0.so.0
        libglib2.0-0 \
        # torch CPU kernels use libgomp.so.1 for OpenMP parallelism
        libgomp1 \
        # libpcre2-8 is a transitive dep of libglib2.0
        libpcre2-8-0 \
    && rm -rf /var/lib/apt/lists/*


# ───────────────────────────────────────────────────────────────────────────
# Stage 3 — runtime (distroless)
# ───────────────────────────────────────────────────────────────────────────
FROM gcr.io/distroless/python3-debian12:nonroot

LABEL org.opencontainers.image.description="IMGFLOW — FastAPI image processing (distroless)"

# ── Environment ──────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    VIRTUAL_ENV=/opt/venv \
    # distroless python3 lives at /usr/bin/python3; point to venv
    PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONPATH=/app \
    TORCH_HOME=/opt/models/torch \
    U2NET_HOME=/opt/models/u2net \
    # Thread pinning — prevents CPU oversubscription under single uvicorn worker
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2 \
    # Limit glibc malloc arena growth (~50–150 MB RSS saved on long AI workloads)
    MALLOC_ARENA_MAX=2 \
    OPENCV_OPENCL_RUNTIME=disabled \
    ORT_DISABLE_TELEMETRY=1

# ── Copy system .so files missing from distroless ────────────────────────
# We copy only the specific files, not the whole /usr/lib tree.
# Paths are Debian 12 (bookworm) x86_64 — verified stable across patch releases.

# libglib-2.0 + libgthread-2.0 (cv2 headless runtime deps)
COPY --from=syslibs /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0*   /usr/lib/x86_64-linux-gnu/
COPY --from=syslibs /usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0* /usr/lib/x86_64-linux-gnu/
# libgomp (torch CPU OpenMP kernels)
COPY --from=syslibs /usr/lib/x86_64-linux-gnu/libgomp.so.1*        /usr/lib/x86_64-linux-gnu/
# libpcre2-8 (glib transitive dep)
COPY --from=syslibs /usr/lib/x86_64-linux-gnu/libpcre2-8.so.0*     /usr/lib/x86_64-linux-gnu/

# ── Application files ────────────────────────────────────────────────────
COPY --from=builder --chown=65532:65532 /opt/venv    /opt/venv
COPY --from=builder --chown=65532:65532 /opt/models  /opt/models

WORKDIR /app
COPY --from=builder --chown=65532:65532 /app/main.py                ./
COPY --from=builder --chown=65532:65532 /app/processing.py          ./
COPY --chown=65532:65532 index.html style.css script.js favicon.ico ./static/

# ── Security: run as nonroot (UID 65532 = distroless "nonroot" user) ─────
USER nonroot

# ── No HEALTHCHECK: distroless has no curl/shell ─────────────────────────
# HuggingFace Spaces uses its own HTTP liveness probe on port 7860.
# For self-hosted: use Docker's --health-cmd with a sidecar or omit entirely.

EXPOSE 7860

# Must use exec-form JSON array — distroless has no /bin/sh to parse strings.
# uvicorn is at /opt/venv/bin/uvicorn (on PATH via VIRTUAL_ENV).
CMD ["/opt/venv/bin/uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
