FROM python:3.11-slim

# System libs needed by opencv + rembg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/main.py       .
COPY backend/processing.py .

# Copy frontend into the static/ directory that FastAPI serves
COPY index.html   static/index.html
COPY style.css    static/style.css
COPY script.js    static/script.js

# Pre-download rembg ISNet model weights at build time
# so the first user request isn't slow
RUN python -c "\
import os; os.environ['U2NET_HOME']='/app/.u2net'; \
from rembg import new_session; \
new_session('isnet-general-use'); \
print('ISNet model cached.')"

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
