FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

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

RUN python -m venv $VIRTUAL_ENV

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --upgrade pip setuptools wheel packaging

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY backend/main.py .
COPY backend/processing.py .

ENV U2NET_HOME=/opt/models/u2net

RUN python -c "\
from rembg import new_session; \
new_session('isnet-general-use'); \
print('✓ ISNet model downloaded successfully')"

RUN python -c "\
from iopaint.model.lama import LaMa, LAMA_MODEL_URL, LAMA_MODEL_MD5; \
from iopaint.helper import download_model; \
download_model(LAMA_MODEL_URL, LAMA_MODEL_MD5); \
print('✓ LaMa model downloaded successfully')"

RUN find $VIRTUAL_ENV -type d -name '__pycache__' -exec rm -rf {} + && \
    find $VIRTUAL_ENV -type d -name 'tests' -exec rm -rf {} + && \
    find $VIRTUAL_ENV -type f -name '*.pyc' -delete && \
    find $VIRTUAL_ENV -type f -name '*.pyo' -delete

FROM python:3.11-slim-bookworm

LABEL maintainer="Ultimate-AI-Container" \
      description="Ultra-optimized FastAPI + rembg + ONNX Runtime container"
      
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
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    OPENCV_OPENCL_RUNTIME=disabled \
    ORT_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --from=builder /opt/models /opt/models

COPY backend/main.py .
COPY backend/processing.py .

COPY index.html ./static/index.html
COPY style.css ./static/style.css
COPY script.js ./static/script.js

RUN mkdir -p /app/static && \
    chown -R appuser:appuser /app /opt/models

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
