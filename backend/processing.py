"""
IMGFLOW — Server-side image processing
Mirrors all three browser pipeline flows from script.js

Flow 1 — Standard:   Lanczos upscale → Shopify resize → WebP encode
Flow 2 — No BG:      rembg ISNet remove → edge refine → upscale → WebP/PNG
Flow 3 — Smart Resize: auto-detect crop/extend → fill → upscale → WebP
"""

import io
import math
import time
import numpy as np
from PIL import Image, ImageFilter
import cv2
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════
# FLOW 1 — STANDARD PIPELINE
# ═══════════════════════════════════════

def run_flow1(img: Image.Image, cfg: dict) -> dict:
    """Upscale → Shopify resize → WebP encode"""
    t0 = time.time()
    orig_size = _img_size(img)

    # 1. Upscale
    img = upscale(img, cfg["factor"], cfg["method"])
    after_up = _img_size(img)

    # 2. Shopify resize (cap longest side)
    img = shopify_resize(img, cfg["shopify"])
    after_sh = _img_size(img)

    # 3. Encode WebP
    blob = encode_webp(img, cfg["quality"], cfg["max_kb"])

    return {
        "blob": blob,
        "ext": "webp",
        "prefix": "shopify",
        "dims": f"{img.width}×{img.height}",
        "log": [
            f"upscaled {orig_size} → {after_up}",
            f"shopify resize → {after_sh}",
            f"webp encode → {len(blob)//1024} KB  ({time.time()-t0:.1f}s)",
        ],
    }


# ═══════════════════════════════════════
# FLOW 2 — NO BACKGROUND
# ═══════════════════════════════════════

def run_flow2(img: Image.Image, cfg: dict) -> dict:
    """rembg ISNet BG removal → edge refine → upscale → WebP / PNG"""
    t0 = time.time()
    orig_size = _img_size(img)

    # 1. Background removal (lazy import so startup is fast when not used)
    from rembg import remove, new_session
    session = new_session(cfg["bg_model"])
    img = remove(img, session=session)                    # returns RGBA PNG
    img = img.convert("RGBA")

    # 2. Edge refinement: alpha threshold + feathering
    img = refine_edges(img, cfg["alpha_threshold"], cfg["feather"])
    after_bg = _img_size(img)

    # 3. Upscale (preserve RGBA)
    img = upscale(img, cfg["factor"], cfg["method"])
    after_up = _img_size(img)

    # 4. Encode
    use_png = cfg.get("output_format", "webp") == "png"
    if use_png:
        blob = encode_png(img)
        ext = "png"
    else:
        blob = encode_webp(img, cfg["quality"], cfg["max_kb"])
        ext = "webp"

    return {
        "blob": blob,
        "ext": ext,
        "prefix": "nobg",
        "dims": f"{img.width}×{img.height}",
        "log": [
            f"BG removed → {after_bg}",
            f"upscaled → {after_up}",
            f"{ext} encode → {len(blob)//1024} KB  ({time.time()-t0:.1f}s)",
        ],
    }


# ═══════════════════════════════════════
# FLOW 3 — SMART RESIZE
# ═══════════════════════════════════════

def run_flow3(img: Image.Image, cfg: dict) -> dict:
    """Smart Resize: detect → crop/extend → upscale → WebP"""
    t0 = time.time()
    orig_size = _img_size(img)
    tw, th = cfg["resize_w"], cfg["resize_h"]
    mode    = cfg.get("resize_mode", "smart-crop-extend")

    if mode == "proportional":
        img, decision = proportional_resize(img, tw, th, cfg)
    else:
        img, decision = smart_resize(img, tw, th, cfg)

    after_resize = _img_size(img)

    # Upscale (now at target resolution so factor is 1 unless user wants more quality)
    factor = cfg.get("factor", 1.0)
    if factor > 1.0:
        img = upscale(img, factor, cfg["method"])

    # Encode
    blob = encode_webp(img, cfg["quality"], cfg["max_kb"])
    prefix = "fit" if mode == "proportional" else "resize"

    return {
        "blob": blob,
        "ext": "webp",
        "prefix": prefix,
        "dims": f"{img.width}×{img.height}",
        "log": [
            f"decision: {decision}  {orig_size} → {after_resize}",
            f"webp encode → {len(blob)//1024} KB  ({time.time()-t0:.1f}s)",
        ],
    }


# ═══════════════════════════════════════
# UPSCALE
# ═══════════════════════════════════════

def upscale(img: Image.Image, factor: float, method: str) -> Image.Image:
    """Lanczos-3 or bicubic upscale by factor."""
    if factor <= 1.0:
        return img
    nw = round(img.width  * factor)
    nh = round(img.height * factor)
    resample = Image.LANCZOS if method == "lanczos" else Image.BICUBIC
    return img.resize((nw, nh), resample=resample)


def shopify_resize(img: Image.Image, max_dim: int) -> Image.Image:
    """Cap longest side to max_dim, preserve aspect ratio."""
    r = min(max_dim / img.width, max_dim / img.height, 1.0)
    if r >= 1.0:
        return img
    return img.resize((round(img.width * r), round(img.height * r)), Image.LANCZOS)


# ═══════════════════════════════════════
# EDGE REFINEMENT (Flow 2)
# ═══════════════════════════════════════

def refine_edges(img: Image.Image, alpha_threshold: int, feather: int) -> Image.Image:
    """Apply alpha threshold, erosion at boundary, and optional Gaussian feather."""
    arr = np.array(img)                          # H×W×4 uint8

    # 1. Hard threshold
    alpha = arr[:, :, 3].astype(np.float32)
    lo, hi = alpha_threshold, 255 - alpha_threshold
    alpha[alpha <= lo] = 0
    alpha[alpha >= hi] = 255

    # 2. Boundary erosion: shrink semi-transparent fringe
    binary = (alpha > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    fringe = (binary > 0) & (eroded == 0)
    alpha[fringe] = np.maximum(0, alpha[fringe] - 80)

    # 3. Optional Gaussian feather
    if feather > 0:
        alpha = gaussian_filter(alpha, sigma=feather * 0.45 + 0.5)

    arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# ═══════════════════════════════════════
# SMART RESIZE — crop + extend
# ═══════════════════════════════════════

def smart_resize(img: Image.Image, tw: int, th: int, cfg: dict):
    """
    Per-axis smart crop + extend.
    Mirrors smartResize() from script.js exactly.
    """
    sw, sh = img.width, img.height
    t_ar = tw / th
    s_ar = sw / sh
    focus  = cfg.get("resize_focus",  "smart")
    align  = cfg.get("resize_align",  "center")
    fill   = cfg.get("resize_fill",   "extend")
    blend  = cfg.get("resize_blend",  40)
    color  = cfg.get("fill_color",    "#ffffff")

    # Detect focal point
    fx, fy = 0.5, 0.4
    if focus == "smart":
        fx, fy = pixel_saliency_center(img)
    else:
        fm = {"center": (.5, .5), "top": (.5, .15), "bottom": (.5, .85),
              "left": (.15, .5), "right": (.85, .5)}
        fx, fy = fm.get(focus, (.5, .5))

    # Determine crop region
    crop_w, crop_h = min(sw, tw), min(sh, th)
    crop_x, crop_y = 0, 0

    if sw > tw or sh > th:
        if s_ar > t_ar:
            crop_h = min(sh, th)
            crop_w = round(crop_h * t_ar)
        else:
            crop_w = min(sw, tw)
            crop_h = round(crop_w / t_ar)
        crop_w = min(crop_w, sw)
        crop_h = min(crop_h, sh)
        crop_x = round(fx * sw - crop_w / 2)
        crop_y = round(fy * sh - crop_h / 2)
        crop_x = max(0, min(sw - crop_w, crop_x))
        crop_y = max(0, min(sh - crop_h, crop_y))

    placed = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

    ox, oy = get_anchor_offset(crop_w, crop_h, tw, th, align)
    needs_fill = crop_w < tw or crop_h < th

    # Decision string for log
    if sw < tw and sh < th:
        decision = f"extend both axes → {tw}×{th}"
    elif sw >= tw and sh >= th:
        if abs(s_ar - t_ar) < 0.005:
            decision = f"scale → {tw}×{th}"
        elif s_ar > t_ar:
            decision = f"crop width (source wider) → {tw}×{th}"
        else:
            decision = f"crop height (source taller) → {tw}×{th}"
    else:
        decision = f"mixed crop+extend → {tw}×{th}"

    if not needs_fill:
        out = placed.resize((tw, th), Image.LANCZOS) if placed.size != (tw, th) else placed
        return out, decision

    # Build output canvas
    has_alpha = img.mode == "RGBA"
    mode = "RGBA" if (has_alpha or fill == "transparent") else "RGB"
    out = Image.new(mode, (tw, th))

    if fill == "extend":
        out = fill_seamless_pil(placed, ox, oy, tw, th, blend)
    elif fill == "white":
        out = Image.new(mode, (tw, th), (255, 255, 255, 255) if mode == "RGBA" else (255, 255, 255))
        out.paste(placed, (ox, oy))
    elif fill == "black":
        out = Image.new(mode, (tw, th), (0, 0, 0, 255) if mode == "RGBA" else (0, 0, 0))
        out.paste(placed, (ox, oy))
    elif fill == "transparent":
        out = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        out.paste(placed, (ox, oy))
    elif fill == "color":
        rgb = _hex_to_rgb(color)
        out = Image.new(mode, (tw, th), rgb)
        out.paste(placed, (ox, oy))
    elif fill == "ai-extend":
        out = fill_lama(placed, ox, oy, tw, th, blend)
    else:
        # fallback: edge extend
        out = fill_seamless_pil(placed, ox, oy, tw, th, blend)

    return out, decision


def proportional_resize(img: Image.Image, tw: int, th: int, cfg: dict):
    """Scale to fit within target, then pad. Mirrors proportionalResize()."""
    sw, sh = img.width, img.height
    ratio = min(tw / sw, th / sh)
    fit_w = round(sw * ratio)
    fit_h = round(sh * ratio)
    scaled = img.resize((fit_w, fit_h), Image.LANCZOS)

    fill   = cfg.get("resize_fill",  "extend")
    align  = cfg.get("resize_align", "center")
    color  = cfg.get("fill_color",   "#ffffff")
    blend  = cfg.get("resize_blend", 40)

    ox, oy = get_anchor_offset(fit_w, fit_h, tw, th, align)
    mode = "RGBA" if (img.mode == "RGBA" or fill == "transparent") else "RGB"

    if fill == "blur":
        out = _blurred_background(img, tw, th)
        out.paste(scaled, (ox, oy))
    elif fill == "white":
        out = Image.new(mode, (tw, th), (255, 255, 255))
        out.paste(scaled, (ox, oy))
    elif fill == "black":
        out = Image.new(mode, (tw, th), (0, 0, 0))
        out.paste(scaled, (ox, oy))
    elif fill == "transparent":
        out = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        out.paste(scaled, (ox, oy))
    elif fill == "color":
        out = Image.new(mode, (tw, th), _hex_to_rgb(color))
        out.paste(scaled, (ox, oy))
    elif fill == "extend":
        out = fill_seamless_pil(scaled, ox, oy, tw, th, blend)
    else:
        out = fill_seamless_pil(scaled, ox, oy, tw, th, blend)

    decision = f"proportional fit: {fit_w}×{fit_h} + padding → {tw}×{th}"
    return out, decision


def get_anchor_offset(sw: int, sh: int, W: int, H: int, align: str):
    cx = (W - sw) // 2
    cy = (H - sh) // 2
    bx, by = W - sw, H - sh
    return {
        "center":        (cx, cy),
        "top-left":      (0, 0),
        "top-center":    (cx, 0),
        "top-right":     (bx, 0),
        "middle-left":   (0, cy),
        "middle-right":  (bx, cy),
        "bottom-left":   (0, by),
        "bottom-center": (cx, by),
        "bottom-right":  (bx, by),
    }.get(align, (cx, cy))


# ═══════════════════════════════════════
# SEAMLESS EXTENSION (edge pixel fill)
# Mirrors fillSeamless() from script.js
# ═══════════════════════════════════════

def fill_seamless_pil(src: Image.Image, ox: int, oy: int, W: int, H: int, blend_radius: int) -> Image.Image:
    """
    Place src at (ox,oy) on a W×H canvas.
    Fill extension zones by sampling nearby edge pixels of src (weighted average).
    Then apply seam blending.
    """
    sw, sh = src.width, src.height
    has_alpha = src.mode == "RGBA"
    channels = 4 if has_alpha else 3
    src_arr = np.array(src.convert("RGBA") if not has_alpha else src, dtype=np.float32)

    out_arr = np.zeros((H, W, 4), dtype=np.float32)
    STRIP = max(6, min(blend_radius, int(min(sw, sh) * 0.18)))

    for y in range(H):
        for x in range(W):
            rx, ry = x - ox, y - oy
            in_x = 0 <= rx < sw
            in_y = 0 <= ry < sh

            if in_x and in_y:
                out_arr[y, x] = src_arr[ry, rx]
                continue

            # Weighted average of STRIP edge samples
            r = g = b = a = wt = 0.0
            for k in range(STRIP):
                w = ((STRIP - k) / STRIP) ** 1.5
                if not in_x and not in_y:
                    sx = min(k, sw - 1) if rx < 0 else max(sw - 1 - k, 0)
                    sy = min(k, sh - 1) if ry < 0 else max(sh - 1 - k, 0)
                elif not in_x:
                    sx = min(k, sw - 1) if rx < 0 else max(sw - 1 - k, 0)
                    sy = max(0, min(sh - 1, ry))
                else:
                    sy = min(k, sh - 1) if ry < 0 else max(sh - 1 - k, 0)
                    sx = max(0, min(sw - 1, rx))
                p = src_arr[sy, sx]
                r += p[0] * w; g += p[1] * w; b += p[2] * w; a += (p[3] / 255.0) * w
                wt += w

            if wt > 0:
                r /= wt; g /= wt; b /= wt; a /= wt

            out_arr[y, x] = [
                np.clip(r, 0, 255),
                np.clip(g, 0, 255),
                np.clip(b, 0, 255),
                np.clip(a * 255, 0, 255),
            ]

    if blend_radius > 0:
        _blend_seam(out_arr, ox, oy, sw, sh, W, H, blend_radius)

    out = Image.fromarray(out_arr.astype(np.uint8), "RGBA")
    return out if has_alpha else out.convert("RGB")


def _blend_seam(arr: np.ndarray, ox: int, oy: int, sw: int, sh: int, W: int, H: int, radius: int):
    """Smooth the seam between placed image and fill zone. Mirrors blendSeam()."""
    for y in range(oy, min(oy + sh, H)):
        for x in range(ox, min(ox + sw, W)):
            dx = min(x - ox, ox + sw - 1 - x)
            dy = min(y - oy, oy + sh - 1 - y)
            d = min(dx, dy)
            if d >= radius:
                continue
            t = d / radius
            smooth = t * t * (3 - 2 * t)

            if dx <= dy:
                nx = ox - 1 if x < ox + sw // 2 else ox + sw
                ny = y
            else:
                ny = oy - 1 if y < oy + sh // 2 else oy + sh
                nx = x

            nx = max(0, min(W - 1, nx))
            ny = max(0, min(H - 1, ny))

            for c in range(3):
                arr[y, x, c] = arr[ny, nx, c] * (1 - smooth) + arr[y, x, c] * smooth


# ═══════════════════════════════════════
# AI FILL — LaMa via iopaint / lama-cleaner
# ═══════════════════════════════════════

def fill_lama(src: Image.Image, ox: int, oy: int, W: int, H: int, blend_radius: int) -> Image.Image:
    """
    Use LaMa inpainting for seamless extension.
    Falls back to edge-pixel fill if lama-cleaner is not installed.

    To enable: pip install iopaint
    """
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy, LDMSampler
    except ImportError:
        # Graceful fallback
        return fill_seamless_pil(src, ox, oy, W, H, blend_radius)

    sw, sh = src.width, src.height
    # Build extended canvas with edge fill first (context for LaMa)
    base = fill_seamless_pil(src, ox, oy, W, H, blend_radius)
    base_rgb = base.convert("RGB")

    # Build mask: white = extension zones (areas to inpaint), black = known
    mask = Image.new("L", (W, H), 255)
    mask.paste(Image.new("L", (sw, sh), 0), (ox, oy))

    try:
        model = ModelManager(name="lama", device="cpu")
        result = model(
            base_rgb,
            mask,
            InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)
        )
        result_rgba = result.convert("RGBA")
        # Re-stamp original src so it isn't altered
        result_rgba.paste(src, (ox, oy))
        # Seam blend
        arr = np.array(result_rgba, dtype=np.float32)
        _blend_seam(arr, ox, oy, sw, sh, W, H, max(8, blend_radius))
        _blend_seam(arr, ox, oy, sw, sh, W, H, blend_radius)
        _blend_seam(arr, ox, oy, sw, sh, W, H, max(4, blend_radius // 2))
        return Image.fromarray(arr.astype(np.uint8), "RGBA")
    except Exception as e:
        print(f"[WARN] LaMa failed ({e}), falling back to edge fill")
        return fill_seamless_pil(src, ox, oy, W, H, blend_radius)


# ═══════════════════════════════════════
# PIXEL SALIENCY CENTER
# Mirrors pixelSaliencyCenter() from script.js
# ═══════════════════════════════════════

def pixel_saliency_center(img: Image.Image) -> tuple:
    """Return (fx, fy) normalised focal point via pixel saliency."""
    TW = 80
    TH = max(1, round(img.height / img.width * 80))
    small = img.resize((TW, TH), Image.LANCZOS).convert("RGB")
    arr = np.array(small, dtype=np.float32)

    r_ch = arr[:, :, 0]; g_ch = arr[:, :, 1]; b_ch = arr[:, :, 2]
    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch

    # Colour distance from mean
    mr, mg, mb = r_ch.mean(), g_ch.mean(), b_ch.mean()
    col_dist = np.sqrt((r_ch - mr)**2 + (g_ch - mg)**2 + (b_ch - mb)**2)

    # Edge magnitude (Sobel)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx**2 + gy**2)

    # Local contrast (std in 3×3)
    local_c = np.zeros_like(lum)
    for y in range(1, TH - 1):
        for x in range(1, TW - 1):
            patch = lum[y-1:y+2, x-1:x+2]
            local_c[y, x] = patch.std()

    def norm(a):
        mx = a.max()
        return a / mx if mx > 1e-6 else a

    sal = norm(col_dist) * 0.45 + norm(edges) * 0.30 + norm(local_c) * 0.25

    # Centre bias
    ys, xs = np.mgrid[0:TH, 0:TW]
    cx = np.abs(xs / TW - 0.5) * 2
    cy = np.abs(ys / TH - 0.5) * 2
    sal *= (1 - np.maximum(cx, cy) * 0.20)

    # Gaussian blur
    blurred = gaussian_filter(sal, sigma=6 * 0.45 + 0.5)
    thresh = blurred.max() * 0.60
    mask = blurred >= thresh

    if mask.sum() < 1:
        return 0.5, 0.4

    sw_sum = blurred[mask].sum()
    fy_val = (np.where(mask)[0] * blurred[mask]).sum() / sw_sum / TH
    fx_val = (np.where(mask)[1] * blurred[mask]).sum() / sw_sum / TW
    return float(fx_val), float(fy_val)


# ═══════════════════════════════════════
# ENCODERS
# ═══════════════════════════════════════

def encode_webp(img: Image.Image, quality: float, max_kb: int) -> bytes:
    """
    Encode to WebP, iteratively reducing quality if > max_kb.
    Mirrors encodeWebP() from script.js.
    """
    q = int(quality * 100) if quality <= 1.0 else int(quality)
    q = max(35, min(100, q))
    max_bytes = max_kb * 1024

    for _ in range(20):
        buf = io.BytesIO()
        save_img = img.convert("RGB") if img.mode == "RGBA" else img
        save_img.save(buf, format="WEBP", quality=q, method=4)
        data = buf.getvalue()
        if len(data) <= max_bytes or q <= 35:
            return data
        q = max(35, q - 5)

    return data


def encode_png(img: Image.Image) -> bytes:
    """Lossless PNG encode (for RGBA transparency)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def _img_size(img: Image.Image) -> str:
    return f"{img.width}×{img.height}"

def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _blurred_background(img: Image.Image, W: int, H: int) -> Image.Image:
    """Scale-to-cover, then heavy blur + darken. Mirrors drawBlurredBackground()."""
    sw, sh = img.width, img.height
    cover = max(W / sw, H / sh)
    cw, ch = round(sw * cover), round(sh * cover)
    big = img.resize((cw, ch), Image.LANCZOS).convert("RGB")
    ox, oy = (cw - W) // 2, (ch - H) // 2
    bg = big.crop((ox, oy, ox + W, oy + H))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=24))
    # Darken
    arr = np.array(bg, dtype=np.float32)
    arr = arr * 0.6
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
