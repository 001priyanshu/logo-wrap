#!/usr/bin/env python3
"""
Logo Wrap - Wrap a logo around the subject of an image so that part of the
logo sits in front of the subject and part sits behind, producing a realistic
"embracing" composition.

Pipeline:
    1. Segment the subject with BiRefNet (via fal.ai).
    2. Clean the mask with morphology + connected-components filtering.
    3. Compute adaptive placement parameters from the subject's geometry.
    4. Partition the logo into front / behind regions using a rotated rectangle
       and a quadrant rule, then alpha-composite both halves around the subject.
"""

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import requests
import fal_client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


INPUT_IMAGE_NAME = "input_image"
MASKED_IMAGE_NAME = "masked_image"
OUTPUT_IMAGE_NAME = "output_image"
JSON_PARAMS_NAME = "json_params.json"

STAGE1_MASK_RAW = "stage1_mask_raw.png"
STAGE2_MASK_CLEAN = "stage2_mask_clean.png"
STAGE3_PLACEMENT = "stage3_placement.png"
STAGE4_FRONT_BEHIND = "stage4_front_behind.png"


@dataclass
class ProcessingResult:
    input_image: str
    masked_image: str
    width_factor_used: float
    vertical_offset_used: float
    horizontal_offset_used: float
    output_image_path: str


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def convert_to_bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[2] == 4:
        return img
    raise ValueError("Invalid image format")


def overlay_rgba(base: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-composite an RGBA overlay onto a base image at (x, y)."""
    base = base.copy()
    H, W = base.shape[:2]
    h, w = overlay.shape[:2]

    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return base

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)

    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    base_roi = base[y0:y1, x0:x1].astype(np.float32)
    overlay_roi = overlay[oy0:oy1, ox0:ox1].astype(np.float32)

    alpha = overlay_roi[..., 3:4] / 255.0
    base_roi[..., :3] = alpha * overlay_roi[..., :3] + (1 - alpha) * base_roi[..., :3]
    base_roi[..., 3] = alpha[..., 0] * 255 + (1 - alpha[..., 0]) * base_roi[..., 3]

    base[y0:y1, x0:x1] = base_roi.astype(np.uint8)
    return base


def rotate_and_scale_rgba(img: np.ndarray, angle: float, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)

    cos_val, sin_val = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)

    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2

    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def apply_opacity(img: np.ndarray, opacity: float) -> np.ndarray:
    out = img.copy()
    out[..., 3] = (out[..., 3] * opacity).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

FAL_ENDPOINT = "fal-ai/birefnet/v2"


def download_image(url: str, output_path: Path) -> Path:
    resp = requests.get(url)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def segment_subject(image_url: str, model: str = "General Use (Heavy)") -> str:
    handle = fal_client.submit(
        FAL_ENDPOINT,
        arguments={
            "image_url": image_url,
            "model": model,
            "operating_resolution": "2048x2048",
            "output_mask": True,
            "output_format": "png",
            "refine_foreground": True,
        },
    )
    request_id = handle.request_id

    while True:
        status = fal_client.status(FAL_ENDPOINT, request_id)
        if isinstance(status, fal_client.InProgress):
            time.sleep(1)
        elif isinstance(status, fal_client.Completed):
            break

    return fal_client.result(FAL_ENDPOINT, request_id)["mask_image"]["url"]


def apply_noise_filter(mask: np.ndarray) -> np.ndarray:
    """Close small holes, keep the largest connected component plus any
    component whose area is >= 25% of the largest (handles multi-body subjects
    like two people or a person holding an object)."""
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return cleaned

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)
    largest_area = areas.max()

    threshold = largest_area * 0.25
    kept_labels = {largest_idx}
    for i in range(1, num_labels):
        if i != largest_idx and stats[i, cv2.CC_STAT_AREA] >= threshold:
            kept_labels.add(i)

    return np.isin(labels, list(kept_labels)).astype(np.uint8) * 255


def get_file_extension(url: str) -> str:
    ext = Path(url).suffix.lower()
    return ext if ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif'] else '.png'


def process_segmentation(
        image_url: str,
        output_dir: Path,
        mask_image_url: Optional[str] = None,
        model: str = "General Use (Heavy)",
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    url_ext = get_file_extension(image_url)
    original_path = output_dir / f"{INPUT_IMAGE_NAME}{url_ext}"
    download_image(image_url, original_path)
    base_bgr = cv2.imread(str(original_path))

    mask_url = mask_image_url or segment_subject(image_url, model=model)

    mask_output_path = output_dir / f"{MASKED_IMAGE_NAME}.png"
    download_image(mask_url, mask_output_path)

    mask_img = Image.open(mask_output_path).convert("L")
    mask = np.array(mask_img.resize((base_bgr.shape[1], base_bgr.shape[0])))
    cleaned_mask = apply_noise_filter(mask)

    cv2.imwrite(str(output_dir / STAGE1_MASK_RAW), mask)
    cv2.imwrite(str(output_dir / STAGE2_MASK_CLEAN), cleaned_mask)

    return base_bgr, cleaned_mask, mask_output_path, original_path


# ---------------------------------------------------------------------------
# Adaptive placement
# ---------------------------------------------------------------------------

def calculate_adjusted_parameters(
        mask: np.ndarray,
        width_factor: float,
        image_width: int,
) -> tuple[float, float]:
    """Shrink width_factor if the subject is too close to a frame edge, and
    pick a vertical offset from a fit over (aspect_ratio, width_factor).

    Width clamp:
        w_adj = min(w, left_space + 1.15, right_space + 1.15)

    Vertical offset (empirical fit, see README):
        v = 0.905 - 0.133 * aspect_ratio - 0.0623 * w_adj
    """
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    subj_w = x_max - x_min
    subj_h = y_max - y_min
    aspect_ratio = subj_h / subj_w

    left_space = x_min / float(subj_w)
    right_space = (image_width - x_max) / float(subj_w)

    adjusted_width_factor = min(width_factor, left_space + 1.15, right_space + 1.15)
    vertical_offset = 0.905 - 0.133 * aspect_ratio - 0.0623 * adjusted_width_factor

    return adjusted_width_factor, vertical_offset


# ---------------------------------------------------------------------------
# Logo wrapping
# ---------------------------------------------------------------------------

def wrap_logo(
        base_bgr: np.ndarray,
        logo_bgra: np.ndarray,
        person_mask: np.ndarray,
        width_factor: float,
        height_factor: float,
        vertical_offset: float,
        horizontal_offset: float = 0.5,
        strict_mode: bool = False,
        angle_deg: float = 0.0,
        logo_opacity: float = 0.9,
        rect_width_frac: float = 0.20,
        rect_height_frac: float = 0.45,
        rect_angle_deg: float = 40.0,
        stage_output_dir: Optional[Path] = None,
) -> tuple[np.ndarray, float, float, float]:
    H, W = base_bgr.shape[:2]
    if person_mask.shape != (H, W):
        person_mask = cv2.resize(person_mask, (W, H))

    mask_clean = apply_noise_filter(person_mask)
    mask_eroded = cv2.erode(mask_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    subj01_behind = mask_eroded.astype(np.float32) / 255.0

    ys, xs = np.where(mask_clean > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    subj_w = x_max - x_min
    subj_h = y_max - y_min

    if strict_mode:
        width_factor_used = width_factor
        vertical_offset_used = vertical_offset
    else:
        width_factor_used, vertical_offset_used = calculate_adjusted_parameters(
            mask_clean, width_factor, W
        )
    horizontal_offset_used = horizontal_offset

    cx = int(x_min + subj_w * horizontal_offset_used)
    cy = int(y_min + subj_h * vertical_offset_used)

    h0, w0 = logo_bgra.shape[:2]
    desired_w = width_factor_used * subj_w
    desired_h = height_factor * subj_h
    scale = min(desired_w / w0, desired_h / h0)

    logo_transformed = rotate_and_scale_rgba(logo_bgra, angle_deg, scale)
    logo_transformed = apply_opacity(logo_transformed, logo_opacity)
    hL, wL = logo_transformed.shape[:2]

    x0 = cx - wL // 2
    y0 = cy - hL // 2

    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    canvas = overlay_rgba(canvas, logo_transformed, x0, y0)

    logo_alpha = canvas[..., 3] / 255.0
    logo_rgb = canvas[..., :3].astype(np.float32)
    logo_mask = logo_alpha > 0

    cx_logo = x0 + wL / 2.0
    cy_logo = y0 + hL / 2.0

    # Rotated rectangle in the logo's front zone
    rect_w = rect_width_frac * wL
    rect_h = rect_height_frac * hL
    rect_offset_frac = 0.005
    L = max(wL, hL)
    cx_rect = cx_logo
    cy_rect = cy_logo - rect_offset_frac * L

    Y, X = np.indices((H, W))
    Xc, Yc = X - cx_rect, Y - cy_rect

    theta = np.deg2rad(rect_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xr = Xc * cos_t + Yc * sin_t
    Yr = -Xc * sin_t + Yc * cos_t

    half_w, half_h = rect_w / 2.0, rect_h / 2.0
    rect_front = (np.abs(Xr) <= half_w) & (np.abs(Yr) <= half_h)
    rect_front = logo_mask & rect_front

    # Top-right and bottom-left quadrants also read as "front" (outside the
    # central rectangle) — this produces the diagonal wrap look.
    is_left = X < cx_logo
    is_top = Y < cy_logo
    quad_tr = logo_mask & (~is_left) & is_top
    quad_bl = logo_mask & is_left & (~is_top)
    outside = logo_mask & (~rect_front)

    front_mask = rect_front | (quad_tr & outside) | (quad_bl & outside)
    behind_mask = logo_mask & (~front_mask)

    front_alpha = logo_alpha * front_mask
    behind_alpha = logo_alpha * behind_mask * (1 - subj01_behind)

    front_rgba = np.dstack([logo_rgb * front_mask[..., None], front_alpha * 255]).astype(np.uint8)
    behind_rgba = np.dstack([logo_rgb * behind_mask[..., None], behind_alpha * 255]).astype(np.uint8)

    out = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2BGRA)
    out = overlay_rgba(out, behind_rgba, 0, 0)
    out = overlay_rgba(out, front_rgba, 0, 0)

    if stage_output_dir is not None:
        placement_vis = base_bgr.copy()
        cv2.rectangle(placement_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.rectangle(placement_vis, (x0, y0), (x0 + wL, y0 + hL), (255, 0, 0), 3)
        cv2.drawMarker(placement_vis, (cx, cy), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
        cv2.imwrite(str(stage_output_dir / STAGE3_PLACEMENT), placement_vis)

        partition_vis = base_bgr.copy()
        partition_vis[front_mask] = (
            0.5 * partition_vis[front_mask] + 0.5 * np.array([0, 0, 255])
        ).astype(np.uint8)
        partition_vis[behind_mask] = (
            0.5 * partition_vis[behind_mask] + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)
        cv2.imwrite(str(stage_output_dir / STAGE4_FRONT_BEHIND), partition_vis)

    return out[..., :3], width_factor_used, vertical_offset_used, horizontal_offset_used


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(
        input_image_url: str,
        logo_path: str,
        output_dir: str,
        width_factor: float = 1.4,
        height_factor: float = 1.0,
        vertical_offset: float = 0.55,
        horizontal_offset: float = 0.5,
        strict_mode: bool = False,
        fal_key: Optional[str] = None,
        mask_image_url: Optional[str] = None,
        model: str = "General Use (Heavy)",
        rect_width_frac: float = 0.20,
        rect_height_frac: float = 0.45,
        rect_angle_deg: float = 40.0,
        logo_opacity: float = 0.9,
) -> ProcessingResult:
    if fal_key:
        os.environ["FAL_KEY"] = fal_key

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_bgr, cleaned_mask, mask_path, input_path = process_segmentation(
        input_image_url, output_path, mask_image_url, model
    )

    logo_rgba = Image.open(logo_path).convert("RGBA")
    logo_bgra = cv2.cvtColor(np.array(logo_rgba), cv2.COLOR_RGBA2BGRA)

    result_image, w_used, v_used, h_used = wrap_logo(
        base_bgr=base_bgr,
        logo_bgra=logo_bgra,
        person_mask=cleaned_mask,
        width_factor=width_factor,
        height_factor=height_factor,
        vertical_offset=vertical_offset,
        horizontal_offset=horizontal_offset,
        strict_mode=strict_mode,
        rect_width_frac=rect_width_frac,
        rect_height_frac=rect_height_frac,
        rect_angle_deg=rect_angle_deg,
        logo_opacity=logo_opacity,
        stage_output_dir=output_path,
    )

    output_image_path = output_path / f"{OUTPUT_IMAGE_NAME}.jpeg"
    cv2.imwrite(str(output_image_path), result_image)

    result = ProcessingResult(
        input_image=str(input_path),
        masked_image=str(mask_path),
        width_factor_used=w_used,
        vertical_offset_used=v_used,
        horizontal_offset_used=h_used,
        output_image_path=str(output_image_path),
    )

    with open(output_path / JSON_PARAMS_NAME, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"Results saved to: {output_path / JSON_PARAMS_NAME}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Logo Wrap - segment subject and wrap a logo around it")

    parser.add_argument("--input", default=os.environ.get("INPUT_IMAGE_URL"),
                        help="Input image URL (or set INPUT_IMAGE_URL in .env)")
    parser.add_argument("--logo", default=os.environ.get("LOGO_PATH"),
                        help="Path to logo PNG (or set LOGO_PATH in .env)")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./out"),
                        help="Output directory (or set OUTPUT_DIR in .env)")

    parser.add_argument("--fal-key", help="FAL API key (or set FAL_KEY env var)")
    parser.add_argument("--mask-image-url", help="Pre-existing mask URL (skips BiRefNet)")
    parser.add_argument("--model", default="General Use (Heavy)", help="BiRefNet model")

    parser.add_argument("--width-factor", type=float, default=1.4, help="Logo width as multiple of subject width")
    parser.add_argument("--vertical-offset", type=float, default=0.55, help="Vertical placement fraction (used only with --strict)")
    parser.add_argument("--horizontal-offset", type=float, default=0.5, help="Horizontal placement fraction (0=left, 1=right)")

    parser.add_argument("--strict", type=str, nargs='?', const='true', default='false',
                        help="Use provided width/vertical values directly instead of adaptive fit")

    parser.add_argument("--rect-width-frac", type=float, default=0.20)
    parser.add_argument("--rect-height-frac", type=float, default=0.45)
    parser.add_argument("--rect-angle-deg", type=float, default=40)
    parser.add_argument("--logo-opacity", type=float, default=0.9)

    args = parser.parse_args()

    if not args.input or not args.logo:
        parser.error("--input and --logo are required (either via flags or .env)")

    process_image(
        input_image_url=args.input,
        logo_path=args.logo,
        output_dir=args.output_dir,
        width_factor=args.width_factor,
        height_factor=1.0,
        vertical_offset=args.vertical_offset,
        horizontal_offset=args.horizontal_offset,
        strict_mode=args.strict == 'true',
        fal_key=args.fal_key,
        mask_image_url=args.mask_image_url,
        model=args.model,
        rect_width_frac=args.rect_width_frac,
        rect_height_frac=args.rect_height_frac,
        rect_angle_deg=args.rect_angle_deg,
        logo_opacity=args.logo_opacity,
    )


if __name__ == "__main__":
    main()
