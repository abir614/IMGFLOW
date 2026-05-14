"""
IMGFLOW — FastAPI server
Exposes three pipeline endpoints plus SSE for progress streaming.

POST /api/process          — single image, returns processed file
POST /api/process/batch    — multiple images, returns ZIP
GET  /api/health           — health check
"""

import io
import json
import time
import zipfile
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from processing import run_flow1, run_flow2, run_flow3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("imgflow")

app = FastAPI(title="IMGFLOW API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════
# SINGLE IMAGE PROCESS
# ═══════════════════════════════════════

@app.post("/api/process")
async def process_image(
    file:              UploadFile = File(...),
    flow:              int        = Form(1),

    # Flow 1+2+3 shared
    upscale_factor:    float      = Form(2.0),
    upscale_method:    str        = Form("lanczos"),
    shopify_size:      int        = Form(2048),
    webp_quality:      int        = Form(85),
    max_kb:            int        = Form(500),

    # Flow 2
    bg_model:          str        = Form("isnet"),
    alpha_threshold:   int        = Form(80),
    feather:           int        = Form(1),
    output_format:     str        = Form("webp"),

    # Flow 3
    resize_w:          int        = Form(1200),
    resize_h:          int        = Form(1200),
    resize_mode:       str        = Form("smart-crop-extend"),
    resize_focus:      str        = Form("smart"),
    resize_align:      str        = Form("center"),
    resize_blend:      int        = Form(40),
    resize_fill:       str        = Form("ai-extend"),
    fill_color:        str        = Form("#ffffff"),

    # Rename
    rename_pattern:    str        = Form(""),
):
    # Read image
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")

    orig_name = file.filename or "image"
    base_name = orig_name.rsplit(".", 1)[0]

    cfg = {
        "factor":          upscale_factor,
        "method":          upscale_method,
        "shopify":         shopify_size,
        "quality":         webp_quality / 100.0,
        "max_kb":          max_kb,
        "bg_model":        _map_bg_model(bg_model),
        "alpha_threshold": alpha_threshold,
        "feather":         feather,
        "output_format":   output_format,
        "resize_w":        resize_w,
        "resize_h":        resize_h,
        "resize_mode":     resize_mode,
        "resize_focus":    resize_focus,
        "resize_align":    resize_align,
        "resize_blend":    resize_blend,
        "resize_fill":     resize_fill,
        "fill_color":      fill_color,
    }

    logger.info(f"[flow={flow}] Processing {orig_name} ({img.width}×{img.height})")
    t0 = time.time()

    try:
        if flow == 1:
            result = run_flow1(img, cfg)
        elif flow == 2:
            result = run_flow2(img, cfg)
        elif flow == 3:
            result = run_flow3(img, cfg)
        else:
            raise HTTPException(400, f"Unknown flow: {flow}")
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        raise HTTPException(500, f"Processing error: {e}")

    elapsed = time.time() - t0
    logger.info(f"[flow={flow}] Done {orig_name} in {elapsed:.1f}s → {len(result['blob'])//1024} KB")

    # Output filename
    out_ext = result["ext"]
    if rename_pattern:
        out_name = rename_pattern.replace("{name}", base_name) + f".{out_ext}"
    else:
        out_name = f"{result['prefix']}_{base_name}.{out_ext}"

    mime = "image/png" if out_ext == "png" else "image/webp"

    def _ascii(s: str) -> str:
        """Strip non-latin-1 characters so HTTP headers never crash."""
        return s.encode("latin-1", errors="replace").decode("latin-1")

    return Response(
        content=result["blob"],
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="{_ascii(out_name)}"',
            "X-IMGFLOW-Dims":      _ascii(result["dims"]),
            "X-IMGFLOW-Log":       _ascii(" | ".join(result["log"])),
            "X-IMGFLOW-Time":      f"{elapsed:.2f}s",
            "X-IMGFLOW-Name":      _ascii(out_name),
        },
    )


# ═══════════════════════════════════════
# BATCH PROCESS → ZIP
# ═══════════════════════════════════════

@app.post("/api/process/batch")
async def process_batch(
    files:             list[UploadFile] = File(...),
    flow:              int        = Form(1),

    upscale_factor:    float      = Form(2.0),
    upscale_method:    str        = Form("lanczos"),
    shopify_size:      int        = Form(2048),
    webp_quality:      int        = Form(85),
    max_kb:            int        = Form(500),

    bg_model:          str        = Form("isnet"),
    alpha_threshold:   int        = Form(80),
    feather:           int        = Form(1),
    output_format:     str        = Form("webp"),

    resize_w:          int        = Form(1200),
    resize_h:          int        = Form(1200),
    resize_mode:       str        = Form("smart-crop-extend"),
    resize_focus:      str        = Form("smart"),
    resize_align:      str        = Form("center"),
    resize_blend:      int        = Form(40),
    resize_fill:       str        = Form("ai-extend"),
    fill_color:        str        = Form("#ffffff"),

    rename_pattern:    str        = Form(""),
):
    cfg = {
        "factor":          upscale_factor,
        "method":          upscale_method,
        "shopify":         shopify_size,
        "quality":         webp_quality / 100.0,
        "max_kb":          max_kb,
        "bg_model":        _map_bg_model(bg_model),
        "alpha_threshold": alpha_threshold,
        "feather":         feather,
        "output_format":   output_format,
        "resize_w":        resize_w,
        "resize_h":        resize_h,
        "resize_mode":     resize_mode,
        "resize_focus":    resize_focus,
        "resize_align":    resize_align,
        "resize_blend":    resize_blend,
        "resize_fill":     resize_fill,
        "fill_color":      fill_color,
    }

    zip_buf = io.BytesIO()
    errors = []

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, upload in enumerate(files):
            raw = await upload.read()
            orig_name = upload.filename or f"image_{i}"
            base_name = orig_name.rsplit(".", 1)[0]

            try:
                img = Image.open(io.BytesIO(raw))
                img.load()
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

                if flow == 1:
                    result = run_flow1(img, cfg)
                elif flow == 2:
                    result = run_flow2(img, cfg)
                elif flow == 3:
                    result = run_flow3(img, cfg)
                else:
                    raise ValueError(f"Unknown flow {flow}")

                out_ext = result["ext"]
                if rename_pattern:
                    n = rename_pattern.replace("{name}", base_name).replace("{n}", str(i + 1))
                    out_name = f"{n}.{out_ext}"
                else:
                    out_name = f"{result['prefix']}_{base_name}.{out_ext}"

                zf.writestr(out_name, result["blob"])
                logger.info(f"  [{i+1}/{len(files)}] ✓ {out_name}")

            except Exception as e:
                errors.append({"file": orig_name, "error": str(e)})
                logger.error(f"  [{i+1}/{len(files)}] ✗ {orig_name}: {e}")

        if errors:
            zf.writestr("errors.json", json.dumps(errors, indent=2))

    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="imgflow_pipeline.zip"'},
    )


# ═══════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ═══════════════════════════════════════
# STATIC FILES (frontend)
# ═══════════════════════════════════════

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def _map_bg_model(name: str) -> str:
    """Map frontend model name to rembg model name."""
    return {
        "isnet":        "isnet-general-use",
        "isnet_fp16":   "isnet-general-use",   # rembg uses same model; FP16 handled internally
        "isnet_quint8": "isnet-general-use",
        "u2net":        "u2net",
        "u2netp":       "u2netp",
    }.get(name, "isnet-general-use")
