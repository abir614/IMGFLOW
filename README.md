---
title: IMGFLOW Pro Pipeline
emoji: 🖼️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# IMGFLOW — Pro Pipeline (Server-Side Edition)

All image processing now runs **on the server** via a Python FastAPI backend.
No ONNX downloads to the browser. No GPU required in the visitor's device.

## Architecture

```
Browser (UI only)              Server (FastAPI + Python)
─────────────────              ─────────────────────────────────
Drop images                →   POST /api/process  (multipart form)
Show progress bar          ←   Processing runs in Python:
Receive blob + headers     ←     • Pillow  — Lanczos upscale, resize, WebP
Download / ZIP             ←     • rembg   — ISNet BG removal
                           ←     • opencv  — edge erosion, Sobel saliency
                           ←     • scipy   — Gaussian feather / blur
                           ←     • iopaint — LaMa AI fill (optional)
```

## Flows

| Flow | Pipeline |
|---|---|
| 1 — Standard | Lanczos upscale → Shopify resize → WebP encode |
| 2 — No Background | rembg ISNet BG removal → edge refine → upscale → WebP/PNG |
| 3 — Smart Resize | Saliency crop/extend → fill (edge/AI/solid/blur) → upscale → WebP |

## Repo structure

```
├── Dockerfile
├── README.md
├── index.html
├── style.css
├── script.js
└── backend/
    ├── main.py
    ├── processing.py
    └── requirements.txt
```

## Local dev

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 7860
# open http://localhost:7860
```

Or with Docker:

```bash
docker build -t imgflow .
docker run -p 7860:7860 imgflow
```
