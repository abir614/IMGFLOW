# ─────────────────────────────────────────────────────────────
# Stage 1 — Builder
# Installs Python dependencies + pre-downloads ISNet model
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies required for:
# - opencv-python-headless
# - scipy
# - numpy
# - onnxruntime
# - rembg
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install packaging explicitly before everything else —
# rembg's dep chain (pooch) needs it at import time but doesn't declare it,
# causing ModuleNotFoundError in lean images regardless of requirements.txt.
RUN pip install --no-cache-dir --prefix=/install packaging

# Install Python dependencies
COPY backend/requirements.txt .

RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    -r requirements.txt

# Copy backend files
COPY backend/main.py .
COPY backend/processing.py .

# Pre-download rembg ISNet model
ENV U2NET_HOME=/app/.u2net

RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "\
from rembg import new_session; \
new_session('isnet-general-use'); \
print('ISNet model cached successfully.')"

# ─────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# Small clean production image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Runtime libraries only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# packaging is required by pooch (rembg dep) but not declared as a dep.
# Install it directly in the runtime stage so it is always present.
RUN pip install --no-cache-dir packaging

# Copy cached rembg models
COPY --from=builder /app/.u2net /app/.u2net

# Backend
COPY backend/main.py .
COPY backend/processing.py .

# Frontend static files
COPY index.html ./static/index.html
COPY style.css ./static/style.css
COPY script.js ./static/script.js

# Runtime optimizations
ENV U2NET_HOME=/app/.u2net \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

EXPOSE 7860

# Single worker — rembg/onnxruntime/iopaint are not thread-safe across workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
