import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# --------- Robust conversion to uint8 (for stable enhancement + easy viewing) ---------
def robust_to_uint8_channelwise(img, low_q=0.5, high_q=99.5):
    """
    Convert grayscale or multi-channel image (uint16/float/uint8) to uint8 using percentile clipping.
    - If img is uint8 already, returns it directly.
    - For multi-channel, applies percentile clipping per channel (preserves channel contrast).
    """
    if img.dtype == np.uint8:
        return img

    x = img.astype(np.float32)

    if x.ndim == 2:  # grayscale
        lo = np.percentile(x, low_q)
        hi = np.percentile(x, high_q)
        if hi <= lo + 1e-6:
            lo, hi = float(x.min()), float(x.max() + 1e-6)
        y = np.clip((x - lo) / (hi - lo), 0, 1)
        return (y * 255.0 + 0.5).astype(np.uint8)

    if x.ndim == 3:  # color
        out = np.zeros_like(x, dtype=np.uint8)
        for c in range(x.shape[2]):
            ch = x[:, :, c]
            lo = np.percentile(ch, low_q)
            hi = np.percentile(ch, high_q)
            if hi <= lo + 1e-6:
                lo, hi = float(ch.min()), float(ch.max() + 1e-6)
            ch_n = np.clip((ch - lo) / (hi - lo), 0, 1)
            out[:, :, c] = (ch_n * 255.0 + 0.5).astype(np.uint8)
        return out

    raise ValueError(f"Unsupported image dim: {x.ndim}")

# --------- Unsharp mask for RGB/BGR ---------
def unsharp_mask_color(img_u8, amount=0.8, sigma=1.0, threshold=2):
    """
    Unsharp mask applied per-channel (stable and simple).
    """
    out = np.empty_like(img_u8)
    for c in range(img_u8.shape[2]):
        ch = img_u8[:, :, c]
        blur = cv2.GaussianBlur(ch, (0, 0), sigma)
        sharp = cv2.addWeighted(ch, 1.0 + amount, blur, -amount, 0)

        if threshold > 0:
            low_contrast = np.abs(ch.astype(np.int16) - blur.astype(np.int16)) < threshold
            sharp = sharp.copy()
            sharp[low_contrast] = ch[low_contrast]

        out[:, :, c] = np.clip(sharp, 0, 255).astype(np.uint8)
    return out

# --------- Main RGB enhancement pipeline ---------
def enhance_rgb_pipeline(
    img_bgr_u8,
    scale=4,
    interp="lanczos",
    denoise_h=6,
    clahe_clip=2.0,
    clahe_grid=(8, 8),
    sharpen_amount=0.8,
    sharpen_sigma=1.0,
):
    inter_map = {
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interp not in inter_map:
        raise ValueError("interp must be 'bicubic' or 'lanczos'")

    # 1) upscale (keep aspect ratio)
    up = cv2.resize(img_bgr_u8, None, fx=scale, fy=scale, interpolation=inter_map[interp])

    # 2) light denoise per channel
    den = np.empty_like(up)
    for c in range(3):
        den[:, :, c] = cv2.fastNlMeansDenoising(
            up[:, :, c], None,
            h=denoise_h,
            templateWindowSize=7,
            searchWindowSize=21
        )

    # 3) CLAHE on V channel only (avoid color shift)
    hsv = cv2.cvtColor(den, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    hsv[:, :, 2] = clahe.apply(v)
    eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4) unsharp
    sharp = unsharp_mask_color(eq, amount=sharpen_amount, sigma=sharpen_sigma, threshold=2)

    return sharp

def ensure_bgr(img_any):
    """
    Ensure 3-channel BGR uint8 input for the pipeline.
    - If grayscale, replicate to 3 channels.
    - If 4-channel (BGRA/RGBA), drop alpha.
    """
    if img_any.ndim == 2:
        return cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)
    if img_any.ndim == 3 and img_any.shape[2] == 1:
        return cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)
    if img_any.ndim == 3 and img_any.shape[2] == 3:
        return img_any
    if img_any.ndim == 3 and img_any.shape[2] == 4:
        return img_any[:, :, :3]
    raise ValueError(f"Unsupported image shape: {img_any.shape}")

def is_image_file(p: Path):
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def batch_enhance_folder(
    input_dir,
    output_dir,
    scale=4,
    interp="lanczos",
    denoise_h=6,
    clahe_clip=2.0,
    clahe_grid=(8, 8),
    sharpen_amount=0.8,
    sharpen_sigma=1.0,
    overwrite=True,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not files:
        raise RuntimeError(f"No image files found in: {input_dir}")

    for p in tqdm(sorted(files), desc="Enhancing images", unit="img"):
        out_path = output_dir / p.name
        if out_path.exists() and not overwrite:
            continue

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[Skip] Failed to read: {p}")
            continue

        # Robust convert to uint8 for enhancement (keeps RGB/BGR, handles high dynamic range)
        img_u8 = robust_to_uint8_channelwise(img, low_q=0.5, high_q=99.5)

        # Ensure 3-channel BGR
        bgr = ensure_bgr(img_u8)

        # Enhance
        out = enhance_rgb_pipeline(
            bgr,
            scale=scale,
            interp=interp,
            denoise_h=denoise_h,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
            sharpen_amount=sharpen_amount,
            sharpen_sigma=sharpen_sigma,
        )

        # Force-save as 8-bit (PyCharm-friendly)
        out = np.clip(out, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            print(f"[Fail] Save failed: {out_path}")
        else:
            # print(f"[OK] {p.name} -> {out_path}")
            pass

# --------------------------- 使用示例（改这两行） ---------------------------
if __name__ == "__main__":
    input_folder = r"dataset/with_film/data_film_lighted"    # 例如 r"D:\data\wafer\raw"
    output_folder = r"dataset/with_film_ligted_enhanced/data_film_lighted_enhanced"  # 例如 r"D:\data\wafer\enhanced"

    batch_enhance_folder(
        input_folder,
        output_folder,
        scale=4,              # 2 或 4
        interp="lanczos",      # "lanczos" 或 "bicubic"
        denoise_h=6,           # 4~12
        clahe_clip=2.0,        # 1.2~3.0
        clahe_grid=(8, 8),
        sharpen_amount=0.8,    # 0.6~1.2
        sharpen_sigma=1.0,     # 0.8~1.4
        overwrite=True,
    )
