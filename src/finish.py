"""Finishing and post-processing module for CirklaGen.

Implements a two-stage finishing pipeline:
Stage A: Repair (Optional)
- composite_tray(generated_img, tray_cutout, bbox): perspective-warp the real tray
  cut-out onto the detected region, feather the mask, match brightness and color
  temperature to the surrounding area, add a soft contact shadow.
- fix_region(image, bbox, instruction, client=None): re-edit only that region (e.g. hands)
  through client.image_edit and paste it back with a feathered blend.

Stage B: Phone Finishing (each strength 0-1, seeded, defaults from config)
- Light grain in shadow areas (luminance-dependent noise)
- Slight vignette (radial corner falloff)
- Slight lens softness (Gaussian diffusion)
- Mild chromatic aberration (radial red/blue lateral channel dispersion)
- Mild highlight bloom on film glare (diffused specular glare from MAP shrink-wrap)
- JPEG recompression at quality 82-90
- Tiny tilt under 1.5 degrees (handheld camera micro-tilt)
- Slight off-center crop (human candid framing)
- Metadata preservation with "AI-generated mockup" provenance field.

CLI & Demo:
  python -m src.finish img.png --demo
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

from src.client import GenAIClient
from src.config import Config, load_config
from src.logger import logger


# ==============================================================================
# HELPER UTILITIES FOR BOUNDING BOXES & METADATA
# ==============================================================================

def parse_bbox_to_pixels(
    bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]],
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """Parses various bounding box formats into integer pixel coordinates [left, top, right, bottom].

    Supports:
    - [ymin, xmin, ymax, xmax] on 0.0-1.0 float scale
    - [ymin, xmin, ymax, xmax] on 0-1000 integer scale
    - [left, top, right, bottom] in absolute pixel coordinates
    - dict with 'box_2d' or 'xmin', 'ymin', 'xmax', 'ymax'
    """
    if isinstance(bbox, dict):
        if "box_2d" in bbox:
            raw_box = bbox["box_2d"]
            ymin, xmin, ymax, xmax = [float(v) for v in raw_box]
        else:
            xmin = float(bbox.get("xmin", bbox.get("left", 0)))
            ymin = float(bbox.get("ymin", bbox.get("top", 0)))
            xmax = float(bbox.get("xmax", bbox.get("right", img_width)))
            ymax = float(bbox.get("ymax", bbox.get("bottom", img_height)))
            return int(xmin), int(ymin), int(xmax), int(ymax)
    else:
        coords = [float(v) for v in bbox]
        if len(coords) != 4:
            raise ValueError(f"Invalid bounding box coordinate length {len(coords)}: {bbox}")
        ymin, xmin, ymax, xmax = coords

    # Check if normalized on 0.0-1.0 scale
    if ymax <= 1.0 and xmax <= 1.0 and (ymax > 0 or xmax > 0):
        left = int(xmin * img_width)
        top = int(ymin * img_height)
        right = int(xmax * img_width)
        bottom = int(ymax * img_height)
    # Check if normalized on 0-1000 scale
    elif (ymax > 1.0 or xmax > 1.0) and (ymax <= 1000 and xmax <= 1000) and (img_height > 1000 or img_width > 1000 or ymax > img_height):
        left = int((xmin / 1000.0) * img_width)
        top = int((ymin / 1000.0) * img_height)
        right = int((xmax / 1000.0) * img_width)
        bottom = int((ymax / 1000.0) * img_height)
    else:
        # Already pixel coordinates [ymin, xmin, ymax, xmax]
        left = int(xmin)
        top = int(ymin)
        right = int(xmax)
        bottom = int(ymax)

    # Ensure valid coordinate bounds
    left = max(0, min(img_width - 1, left))
    top = max(0, min(img_height - 1, top))
    right = max(left + 1, min(img_width, right))
    bottom = max(top + 1, min(img_height, bottom))

    return left, top, right, bottom


def attach_mockup_metadata(img: Image.Image, tag: str = "AI-generated mockup") -> Image.Image:
    """Attaches required provenance fields to Image.info."""
    if not hasattr(img, "info") or img.info is None:
        img.info = {}
    img.info["Comment"] = tag
    img.info["Description"] = tag
    img.info["Software"] = "CirklaGen Candid Smartphone Synthesizer"
    img.info["Provenance"] = tag
    return img


# ==============================================================================
# STAGE A: REPAIR (OPTIONAL)
# ==============================================================================

def composite_tray(
    generated_img: Union[Image.Image, str, Path],
    tray_cutout: Union[Image.Image, str, Path],
    bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]],
    feather_radius: int = 5,
    shadow_strength: float = 0.45,
) -> Image.Image:
    """Perspective-warps a real tray cut-out onto the detected region.

    Steps:
    1. Perspective-warp cutout onto detected bbox.
    2. Feather the alpha mask for seamless perimeter transition.
    3. Match brightness and color temperature to surrounding ambient background.
    4. Add a soft contact shadow underneath the tray.
    """
    gen_pil = Image.open(generated_img) if isinstance(generated_img, (str, Path)) else generated_img.copy()
    if gen_pil.mode != "RGB":
        gen_pil = gen_pil.convert("RGB")

    cutout_pil = Image.open(tray_cutout) if isinstance(tray_cutout, (str, Path)) else tray_cutout.copy()
    if cutout_pil.mode != "RGBA":
        cutout_pil = cutout_pil.convert("RGBA")

    w, h = gen_pil.size
    left, top, right, bottom = parse_bbox_to_pixels(bbox, w, h)
    box_w = max(10, right - left)
    box_h = max(10, bottom - top)

    logger.info("Stage A composite_tray: target box [%d, %d, %d, %d] (%dx%d)", left, top, right, bottom, box_w, box_h)

    cutout_w, cutout_h = cutout_pil.size
    cutout_np = np.array(cutout_pil)

    # 1. Perspective Transform Matrix
    # Source corners of the cutout
    src_pts = np.float32([
        [0, 0],
        [cutout_w, 0],
        [cutout_w, cutout_h],
        [0, cutout_h],
    ])

    # Destination quad in the generated image with subtle natural perspective convergence
    # Slightly taper top width by 3% to reflect retail handheld viewing angle
    inset = box_w * 0.02
    dst_pts = np.float32([
        [left + inset, top],
        [right - inset, top],
        [right, bottom],
        [left, bottom],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_rgba = cv2.warpPerspective(cutout_np, M, (w, h), flags=cv2.INTER_LANCZOS4)

    warped_rgb = warped_rgba[..., :3].astype(np.float32)
    warped_alpha = warped_rgba[..., 3].astype(np.float32) / 255.0

    if np.sum(warped_alpha) < 1.0:
        logger.warning("composite_tray: Warped alpha mask is empty.")
        return gen_pil

    # 2. Color Temperature and Brightness Matching
    gen_np = np.array(gen_pil).astype(np.float32)

    # Extract surrounding background margin (15px halo around bbox)
    halo_top = max(0, top - 20)
    halo_bottom = min(h, bottom + 20)
    halo_left = max(0, left - 20)
    halo_right = min(w, right + 20)
    surrounding_roi = gen_np[halo_top:halo_bottom, halo_left:halo_right]

    ambient_mean = np.mean(surrounding_roi, axis=(0, 1))
    cutout_solid_mask = warped_alpha > 0.5
    if np.any(cutout_solid_mask):
        cutout_mean = np.mean(warped_rgb[cutout_solid_mask], axis=0)
    else:
        cutout_mean = np.mean(warped_rgb, axis=(0, 1))

    # Calculate per-channel gain ratio with safe bounding [0.65, 1.35]
    color_gains = np.clip(ambient_mean / (cutout_mean + 1e-5), 0.65, 1.35)
    # Blend 60% ambient color balance, 40% authentic product color
    blended_gains = 0.60 * color_gains + 0.40 * np.ones_like(color_gains)

    color_matched_rgb = np.clip(warped_rgb * blended_gains, 0.0, 255.0)

    # 3. Soft Contact Shadow underneath the tray
    # Shift alpha downward and blur to simulate ground/hand ambient occlusion
    shadow_shift_y = int(max(3, box_h * 0.035))
    shadow_mask = np.zeros_like(warped_alpha)
    if shadow_shift_y < h:
        shadow_mask[shadow_shift_y:, :] = warped_alpha[:-shadow_shift_y, :]

    # Dilate and Gaussian blur shadow
    kernel_size = max(7, (feather_radius * 4) | 1)
    shadow_blurred = cv2.GaussianBlur(shadow_mask, (kernel_size, kernel_size), sigmaX=feather_radius * 1.5)
    # Attenuate shadow so it doesn't blotch
    shadow_factor = np.clip(shadow_blurred * shadow_strength, 0.0, 0.85)

    # Apply shadow to underlying generated image
    gen_np = gen_np * (1.0 - shadow_factor[..., None])

    # 4. Feathered Alpha Mask for Smooth Perimeter Integration
    feather_ksize = max(3, (feather_radius * 2) | 1)
    feathered_alpha = cv2.GaussianBlur(warped_alpha, (feather_ksize, feather_ksize), sigmaX=feather_radius * 0.6)
    feathered_alpha = np.clip(feathered_alpha, 0.0, 1.0)[..., None]

    # Composite color-matched cutout over shadowed background
    composited_np = color_matched_rgb * feathered_alpha + gen_np * (1.0 - feathered_alpha)
    composited_np = np.clip(composited_np, 0, 255).astype(np.uint8)

    result_pil = Image.fromarray(composited_np)
    setattr(result_pil, "_is_composite", True)
    result_pil.info["is_composite"] = True
    attach_mockup_metadata(result_pil)
    return result_pil


# Alias for backward and external compatibility
composite_real_tray = composite_tray


# ==============================================================================
# STAGE A.1: MANDATORY PACKAGE REPLACEMENT WITH REAL CUTOUT & HAND PRESERVATION
# ==============================================================================

LOCATE_TRAY_CORNERS_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "found": {
            "type": "BOOLEAN",
            "description": "True if the meat tray corners are detected in the image",
        },
        "corners": {
            "type": "ARRAY",
            "items": {
                "type": "ARRAY",
                "items": {"type": "NUMBER"},
            },
            "description": "4 corner points [[x, y], [x, y], [x, y], [x, y]] in clockwise order: top_left, top_right, bottom_right, bottom_left (pixel coordinates or 0-1000 scale)",
        },
        "box_2d": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
            "description": "[ymin, xmin, ymax, xmax] normalized on a 0 to 1000 integer scale",
        },
    },
    "required": ["found"],
}


def locate_tray_corners(
    image: Union[Image.Image, str, Path],
    client: Optional[GenAIClient] = None,
    placement_region: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
) -> List[Tuple[float, float]]:
    """Detects or derives the 4 corners of the meat tray [top_left, top_right, bottom_right, bottom_left].

    If client.vision provides explicit corners, they are scaled to pixel coordinates.
    Otherwise, derives corners from detected bounding box or placement_region with realistic
    handheld perspective taper (approx 2% convergence).
    """
    pil_img = Image.open(image) if isinstance(image, (str, Path)) else image
    w, h = pil_img.size

    # 1. Try vision model detection if client is provided
    if client is not None:
        try:
            raw_res = client.vision(
                images=[pil_img],
                prompt=(
                    "Locate the Cirkla molded-fiber meat tray in this photo. "
                    "Return strict JSON with 'found': true/false and the 4 outer corner points "
                    "'corners': [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]] in clockwise order "
                    "(top_left, top_right, bottom_right, bottom_left) on a 0-1000 integer scale, "
                    "and the bounding box 'box_2d': [ymin, xmin, ymax, xmax] on a 0-1000 integer scale."
                ),
                json_schema=LOCATE_TRAY_CORNERS_SCHEMA,
            )
            data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
            if data and data.get("found", False):
                raw_corners = data.get("corners")
                if raw_corners and len(raw_corners) == 4:
                    pts: List[Tuple[float, float]] = []
                    for pt in raw_corners:
                        px, py = float(pt[0]), float(pt[1])
                        if px > 1.0 or py > 1.0:
                            pts.append(((px / 1000.0) * w, (py / 1000.0) * h))
                        else:
                            pts.append((px * w, py * h))
                    if len(pts) == 4:
                        if placement_region is not None:
                            p_left, p_top, p_right, p_bottom = parse_bbox_to_pixels(placement_region, w, h)
                            inside = all(p_left - 15 <= pt[0] <= p_right + 15 and p_top - 15 <= pt[1] <= p_bottom + 15 for pt in pts)
                            if inside:
                                return pts
                        else:
                            return pts

                if "box_2d" in data:
                    left, top, right, bottom = parse_bbox_to_pixels(data["box_2d"], w, h)
                    if placement_region is not None:
                        p_left, p_top, p_right, p_bottom = parse_bbox_to_pixels(placement_region, w, h)
                        if p_left - 15 <= left and right <= p_right + 15 and p_top - 15 <= top and bottom <= p_bottom + 15:
                            box_w = max(10, right - left)
                            inset = box_w * 0.02
                            return [
                                (float(left + inset), float(top)),
                                (float(right - inset), float(top)),
                                (float(right), float(bottom)),
                                (float(left), float(bottom)),
                            ]
                    else:
                        box_w = max(10, right - left)
                        inset = box_w * 0.02
                        return [
                            (float(left + inset), float(top)),
                            (float(right - inset), float(top)),
                            (float(right), float(bottom)),
                            (float(left), float(bottom)),
                        ]
        except Exception as e:
            logger.debug("locate_tray_corners vision call failed (%s). Using geometric fallback.", e)

    # 2. Geometric fallback from placement_region or default center quad
    if placement_region is not None:
        p_left, p_top, p_right, p_bottom = parse_bbox_to_pixels(placement_region, w, h)
        reg_w = max(10, p_right - p_left)
        reg_h = max(10, p_bottom - p_top)

        # If placement_region covers a large vertical expanse (e.g. shopper torso/body > 38% height),
        # position the tray proportionally in front of the lower chest/stomach/hands,
        # NEVER spanning the person's head, face, or entire height.
        if reg_h > int(h * 0.38):
            cx = p_left + int(reg_w * 0.50)
            cy = p_top + int(reg_h * 0.62)
            tw = min(int(reg_w * 0.55), int(w * 0.35))
            th = max(10, int(tw / 1.25))
            left = max(p_left + int(reg_w * 0.05), cx - tw // 2)
            right = min(p_right - int(reg_w * 0.05), left + tw)
            top = max(p_top + int(reg_h * 0.30), cy - th // 2)
            bottom = min(p_bottom - int(reg_h * 0.05), top + th)
        else:
            left, top, right, bottom = p_left, p_top, p_right, p_bottom
    else:
        left = int(w * 0.35)
        top = int(h * 0.44)
        right = int(w * 0.58)
        bottom = int(h * 0.62)

    box_w = max(10, right - left)
    inset = box_w * 0.02
    return [
        (float(left + inset), float(top)),
        (float(right - inset), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    ]


def extract_hand_occlusion_mask(
    image: Image.Image,
    tray_corners: List[Tuple[float, float]],
    client: Optional[GenAIClient] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts pre-replacement shopper hands and fingers gripping over the tray rim.

    Uses skin-tone chrominance/luminance segmentation (YCrCb and HSV color spaces)
    within the tray rim perimeter and hand contact regions.

    Returns:
        (hand_rgb, hand_alpha): RGB float32 array [0-255] and soft float32 alpha mask [0.0-1.0].
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    img_np = np.array(img_rgb)

    # 1. Skin-tone color segmentation
    # YCrCb color space
    ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)

    # HSV color space
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    skin_hsv = (h_ch >= 0) & (h_ch <= 50) & (s_ch >= 30) & (s_ch <= 180) & (v_ch >= 45)

    skin_mask = (skin_ycrcb & skin_hsv).astype(np.uint8) * 255

    # 2. Construct Tray Polygon and Rim Perimeter Band
    tray_poly = np.array(tray_corners, dtype=np.int32).reshape((-1, 1, 2))
    tray_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tray_mask, [tray_poly], 255)

    xs = [pt[0] for pt in tray_corners]
    ys = [pt[1] for pt in tray_corners]
    box_w = max(10, max(xs) - min(xs))
    box_h = max(10, max(ys) - min(ys))
    rim_thickness = max(8, int(min(box_w, box_h) * 0.18))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_thickness, rim_thickness))
    tray_inner = cv2.erode(tray_mask, kernel)
    tray_rim = cv2.subtract(tray_mask, tray_inner)

    # Allow slight expansion outside tray for fingers wrapping around
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tray_outer = cv2.dilate(tray_mask, outer_kernel)
    rim_search_zone = cv2.subtract(tray_outer, tray_inner)

    # 3. Intersect skin mask with rim search zone
    hand_over_rim = cv2.bitwise_and(skin_mask, rim_search_zone)

    # Remove small speckles with morphological opening
    clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hand_clean = cv2.morphologyEx(hand_over_rim, cv2.MORPH_OPEN, clean_k)

    # Dilate slightly and soften with Gaussian blur for natural edge transition
    hand_soft = cv2.GaussianBlur(hand_clean.astype(np.float32) / 255.0, (5, 5), sigmaX=1.2)
    hand_alpha = np.clip(hand_soft, 0.0, 1.0)

    hand_rgb = img_np.astype(np.float32)
    return hand_rgb, hand_alpha


def replace_packaged_tray(
    image: Union[Image.Image, str, Path],
    package_cutout: Optional[Union[Image.Image, str, Path]] = None,
    corners: Optional[List[Tuple[float, float]]] = None,
    client: Optional[GenAIClient] = None,
    placement_region: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
    feather_radius: int = 5,
    shadow_strength: float = 0.35,
    config: Optional[Config] = None,
) -> Image.Image:
    """Always replaces the generated meat package with the real cut-out from assets/packaged/.

    Directives:
    1. Perspective-warps the real package cut-out to the detected tray corners.
    2. Re-lights the package to match the scene ambient temperature and illumination.
    3. Pastes hands/fingers back over the rim so the shopper grip remains authentic.
    4. The model-generated package must NEVER appear in the final output.
    5. Confines modifications within placement_region if specified to preserve region-lock.
    """
    cfg = config or load_config(require_api_key=False)
    cli = client or GenAIClient(config=cfg)

    base_pil = Image.open(image) if isinstance(image, (str, Path)) else image.copy()
    if base_pil.mode != "RGB":
        base_pil = base_pil.convert("RGB")
    w, h = base_pil.size

    if getattr(image, "_package_already_authentic", False):
        base_pil.info["package_replaced"] = True
        return base_pil

    # 1. Resolve Package Cutout from assets/packaged/
    cutout_img: Optional[Image.Image] = None
    if package_cutout is not None:
        cutout_img = Image.open(package_cutout) if isinstance(package_cutout, (str, Path)) else package_cutout.copy()
    else:
        pkg_dir = Path(getattr(cfg.paths, "packaged_dir", "assets/packaged"))
        if pkg_dir.is_dir():
            candidates = sorted([p for p in pkg_dir.glob("*.png") if not p.name.startswith("sleeve_")])
            if candidates:
                cutout_img = Image.open(candidates[0])
            else:
                all_pngs = sorted([p for p in pkg_dir.glob("*.png")])
                if all_pngs:
                    cutout_img = Image.open(all_pngs[0])

        if cutout_img is None:
            trays_dir = Path(cfg.paths.trays_dir)
            tray_cutouts = sorted(list(trays_dir.glob("*removebg*.png")))
            if tray_cutouts:
                cutout_img = Image.open(tray_cutouts[0])

    if cutout_img is None:
        logger.warning("replace_packaged_tray: No package cutout found in assets/packaged/. Returning base image.")
        return base_pil

    if cutout_img.mode != "RGBA":
        cutout_img = cutout_img.convert("RGBA")

    cutout_w, cutout_h = cutout_img.size
    cutout_np = np.array(cutout_img)

    # 2. Detect Tray Corners
    tray_pts = corners
    if tray_pts is None:
        tray_pts = locate_tray_corners(base_pil, client=cli, placement_region=placement_region)

    clamped_pts = [
        (max(0.0, min(float(w - 1), float(pt[0]))), max(0.0, min(float(h - 1), float(pt[1]))))
        for pt in tray_pts
    ]

    xs = [pt[0] for pt in clamped_pts]
    ys = [pt[1] for pt in clamped_pts]
    tray_left = int(min(xs))
    tray_top = int(min(ys))
    tray_right = int(max(xs))
    tray_bottom = int(max(ys))
    tray_w = max(10, tray_right - tray_left)
    tray_h = max(10, tray_bottom - tray_top)

    # 3. Extract Hand/Finger Occlusion over Rim BEFORE replacement
    hand_rgb, hand_alpha = extract_hand_occlusion_mask(base_pil, clamped_pts, client=cli)

    # 4. Perspective Warping
    src_pts = np.float32([
        [0, 0],
        [cutout_w, 0],
        [cutout_w, cutout_h],
        [0, cutout_h],
    ])
    dst_pts = np.float32(clamped_pts)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_rgba = cv2.warpPerspective(cutout_np, M, (w, h), flags=cv2.INTER_LANCZOS4)

    warped_rgb = warped_rgba[..., :3].astype(np.float32)
    warped_alpha = warped_rgba[..., 3].astype(np.float32) / 255.0

    if np.sum(warped_alpha) < 1.0:
        logger.warning("replace_packaged_tray: Warped package alpha mask is empty.")
        return base_pil

    # 5. Re-lighting: Color Temperature and Ambient Matching
    base_np = np.array(base_pil).astype(np.float32)

    halo_top = max(0, tray_top - 25)
    halo_bottom = min(h, tray_bottom + 25)
    halo_left = max(0, tray_left - 25)
    halo_right = min(w, tray_right + 25)
    surrounding_roi = base_np[halo_top:halo_bottom, halo_left:halo_right]

    ambient_mean = np.mean(surrounding_roi, axis=(0, 1))
    cutout_solid = warped_alpha > 0.4
    if np.any(cutout_solid):
        cutout_mean = np.mean(warped_rgb[cutout_solid], axis=0)
    else:
        cutout_mean = np.mean(warped_rgb, axis=(0, 1))

    color_gains = np.clip(ambient_mean / (cutout_mean + 1e-5), 0.65, 1.35)
    blended_gains = 0.55 * color_gains + 0.45 * np.ones_like(color_gains)
    relit_rgb = np.clip(warped_rgb * blended_gains, 0.0, 255.0)

    # 6. Contact Shadow Underneath Tray
    shadow_shift_y = int(max(3, tray_h * 0.035))
    shadow_mask = np.zeros_like(warped_alpha)
    if shadow_shift_y < h:
        shadow_mask[shadow_shift_y:, :] = warped_alpha[:-shadow_shift_y, :]

    ksize = max(7, (feather_radius * 4) | 1)
    shadow_blurred = cv2.GaussianBlur(shadow_mask, (ksize, ksize), sigmaX=feather_radius * 1.5)
    shadow_factor = np.clip(shadow_blurred * shadow_strength, 0.0, 0.70)
    scene_np = base_np * (1.0 - shadow_factor[..., None])

    # 7. Feathered Composite of Re-lit Package over Scene
    feather_k = max(3, (feather_radius * 2) | 1)
    feathered_alpha = cv2.GaussianBlur(warped_alpha, (feather_k, feather_k), sigmaX=feather_radius * 0.6)
    feathered_alpha = np.clip(feathered_alpha, 0.0, 1.0)[..., None]

    package_composite = relit_rgb * feathered_alpha + scene_np * (1.0 - feathered_alpha)

    # 8. Paste Hands Back Over the Rim
    hand_alpha_3d = hand_alpha[..., None]
    final_composite = hand_rgb * hand_alpha_3d + package_composite * (1.0 - hand_alpha_3d)

    # 9. Strictly Enforce Placement Region if Provided (Preserves Region-Lock Invariant)
    if placement_region is not None:
        p_left, p_top, p_right, p_bottom = parse_bbox_to_pixels(placement_region, w, h)
        outside_mask = np.ones((h, w), dtype=bool)
        outside_mask[p_top:p_bottom, p_left:p_right] = False
        final_composite[outside_mask] = base_np[outside_mask]

    result_np = np.clip(final_composite, 0, 255).astype(np.uint8)
    result_pil = Image.fromarray(result_np)
    if hasattr(image, "info") and isinstance(image.info, dict):
        result_pil.info.update(image.info)
    for attr in dir(image):
        if attr.startswith("_") and not attr.startswith("__") and not hasattr(result_pil, attr):
            try:
                setattr(result_pil, attr, getattr(image, attr))
            except Exception:
                pass
    result_pil.info["package_replaced"] = True
    result_pil.info["is_composite"] = True
    attach_mockup_metadata(result_pil)
    return result_pil


def fix_region(
    image: Union[Image.Image, str, Path],
    bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]],
    instruction: str,
    client: Optional[GenAIClient] = None,
    padding_ratio: float = 0.20,
    feather_radius: int = 15,
) -> Image.Image:
    """Re-edits a specific target region (e.g. malformed hands) through client.image_edit and blends it back."""
    cli = client or GenAIClient()
    base_pil = Image.open(image) if isinstance(image, (str, Path)) else image.copy()
    if base_pil.mode != "RGB":
        base_pil = base_pil.convert("RGB")

    w, h = base_pil.size
    left, top, right, bottom = parse_bbox_to_pixels(bbox, w, h)

    # Add surrounding context padding
    pad_w = int((right - left) * padding_ratio)
    pad_h = int((bottom - top) * padding_ratio)
    crop_l = max(0, left - pad_w)
    crop_t = max(0, top - pad_h)
    crop_r = min(w, right + pad_w)
    crop_b = min(h, bottom + pad_h)
    crop_w = crop_r - crop_l
    crop_h = crop_b - crop_t

    logger.info("Stage A fix_region: cropping [%d, %d, %d, %d] with prompt: %s", crop_l, crop_t, crop_r, crop_b, instruction)

    cropped_region = base_pil.crop((crop_l, crop_t, crop_r, crop_b))

    try:
        edited_list = cli.image_edit(
            images=[cropped_region],
            prompt=f"Repair and refine this specific cropped detail: {instruction}. Keep overall lighting and composition seamless.",
            n=1,
        )
        edited_crop = edited_list[0] if edited_list else cropped_region
    except Exception as e:
        logger.warning("Stage A fix_region: client.image_edit failed (%s). Returning original crop.", e)
        edited_crop = cropped_region

    # Resize edited crop back to target crop dimensions if needed
    if edited_crop.size != (crop_w, crop_h):
        edited_crop = edited_crop.resize((crop_w, crop_h), Image.Resampling.LANCZOS)

    # Create feathered elliptical/rectangular blend mask
    mask_np = np.ones((crop_h, crop_w), dtype=np.float32)
    border_x = max(2, int(crop_w * 0.15))
    border_y = max(2, int(crop_h * 0.15))

    # Inner rectangle is 1.0, outer border slopes to 0.0
    for y in range(crop_h):
        for x in range(crop_w):
            dist_x = min(x, crop_w - 1 - x) / float(border_x)
            dist_y = min(y, crop_h - 1 - y) / float(border_y)
            dist = min(dist_x, dist_y)
            mask_np[y, x] = np.clip(dist, 0.0, 1.0)

    # Gaussian blur the blend mask
    ksize = max(5, (feather_radius * 2) | 1)
    mask_blurred = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=feather_radius * 0.7)[..., None]

    base_np = np.array(base_pil).astype(np.float32)
    edited_np = np.array(edited_crop).astype(np.float32)

    # Blend edited crop into base image
    roi = base_np[crop_t:crop_b, crop_l:crop_r]
    blended_roi = edited_np * mask_blurred + roi * (1.0 - mask_blurred)
    base_np[crop_t:crop_b, crop_l:crop_r] = np.clip(blended_roi, 0, 255)

    result_pil = Image.fromarray(base_np.astype(np.uint8))
    attach_mockup_metadata(result_pil)
    return result_pil


# ==============================================================================
# STAGE A.2: OPTIONAL LIGHTING MATCH STEP (PLATE AMBIENT HARMONIZATION)
# ==============================================================================

LOCATE_PERSON_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "found": {
            "type": "BOOLEAN",
            "description": "True if a human person, shopper, or customer holding or interacting with the meat tray is detected in the image",
        },
        "box_2d": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
            "description": "[ymin, xmin, ymax, xmax] normalized on a 0 to 1000 integer scale encompassing the person (torso, arms, head)",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Detection confidence from 0.0 to 1.0",
        },
        "notes": {
            "type": "STRING",
            "description": "Short description of detected person framing, pose, and location",
        },
    },
    "required": ["found", "box_2d"],
}


def locate_person(
    image: Union[Image.Image, str, Path],
    client: Optional[GenAIClient] = None,
) -> Optional[List[float]]:
    """Detects normalized [ymin, xmin, ymax, xmax] bounding box around the person.

    Returns normalized coordinates (0.0 to 1.0), or fallback [0.08, 0.10, 0.92, 0.90].
    """
    cli = client or GenAIClient()
    pil_img = Image.open(image) if isinstance(image, (str, Path)) else image

    prompt = (
        "Locate the human shopper, customer, or person holding or interacting with the meat tray in this image.\n"
        "Detect the bounding box around the person including their head, shoulders, torso, and arms.\n"
        "Return JSON with 'found' (boolean) and 'box_2d' as [ymin, xmin, ymax, xmax] on a 0-1000 scale.\n"
        "If no person is visible, return found=false and box_2d=[0,0,0,0]."
    )

    try:
        raw_res = cli.vision(
            images=pil_img,
            prompt=prompt,
            json_schema=LOCATE_PERSON_SCHEMA,
        )
        data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
        if not data or not data.get("found"):
            logger.info("locate_person: Person not detected via vision; using default person framing.")
            return [0.08, 0.10, 0.92, 0.90]

        box = data.get("box_2d")
        if not box or len(box) != 4:
            return [0.08, 0.10, 0.92, 0.90]

        ymin, xmin, ymax, xmax = [float(v) for v in box]
        if ymax > 1.0 or xmax > 1.0:
            ymin, xmin, ymax, xmax = ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0

        if ymax <= ymin or xmax <= xmin:
            return [0.08, 0.10, 0.92, 0.90]

        logger.info("locate_person: Detected person bounding box [%.3f, %.3f, %.3f, %.3f]", ymin, xmin, ymax, xmax)
        return [ymin, xmin, ymax, xmax]
    except Exception as e:
        logger.warning("locate_person failed: %s. Using default person framing.", e)
        return [0.08, 0.10, 0.92, 0.90]


def sample_plate_lighting(
    plate: Union[Image.Image, str, Path],
    person_bbox: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    sample_margin_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Samples color temperature, chromaticity, and brightness of the plate near the person region.

    Extracts ambient lighting context from the background plate where the person is situated.
    Returns:
    - 'mean_rgb': np.ndarray of shape (3,) with mean [R, G, B]
    - 'brightness': float luminance (0.299*R + 0.587*G + 0.114*B) in [0, 255]
    - 'color_temp_ratio': float R / B ratio (correlated color temperature proxy)
    - 'lab_mean': np.ndarray of shape (3,) with [L*, a*, b*] in CIE-LAB space
    - 'sample_box': (left, top, right, bottom) pixel coordinates sampled on plate
    """
    plate_pil = Image.open(plate) if isinstance(plate, (str, Path)) else plate.copy()
    if plate_pil.mode != "RGB":
        plate_pil = plate_pil.convert("RGB")

    if target_size is not None and plate_pil.size != target_size:
        plate_pil = plate_pil.resize(target_size, Image.Resampling.LANCZOS)

    w, h = plate_pil.size
    p_box = person_bbox if person_bbox is not None else [0.08, 0.10, 0.92, 0.90]
    left, top, right, bottom = parse_bbox_to_pixels(p_box, w, h)

    # Pad around person box to sample surrounding background context
    pad_x = int((right - left) * sample_margin_ratio)
    pad_y = int((bottom - top) * sample_margin_ratio)
    halo_l = max(0, left - pad_x)
    halo_r = min(w, right + pad_x)
    halo_t = max(0, top - pad_y)
    halo_b = min(h, bottom + pad_y)

    plate_np = np.array(plate_pil).astype(np.float32)

    # Sample context in halo region
    sample_roi = plate_np[halo_t:halo_b, halo_l:halo_r]
    if sample_roi.size == 0:
        sample_roi = plate_np

    mean_rgb = np.mean(sample_roi, axis=(0, 1))
    brightness = float(0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2])
    color_temp_ratio = float(mean_rgb[0] / (mean_rgb[2] + 1e-5))

    roi_u8 = np.clip(sample_roi, 0, 255).astype(np.uint8)
    roi_lab = cv2.cvtColor(roi_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_mean = np.mean(roi_lab, axis=(0, 1))

    logger.info(
        "sample_plate_lighting: halo=[%d, %d, %d, %d], mean_rgb=[%.1f, %.1f, %.1f], "
        "brightness=%.1f, color_temp_ratio (R/B)=%.3f, lab=[%.1f, %.1f, %.1f]",
        halo_l, halo_t, halo_r, halo_b,
        mean_rgb[0], mean_rgb[1], mean_rgb[2],
        brightness, color_temp_ratio, lab_mean[0], lab_mean[1], lab_mean[2],
    )

    return {
        "mean_rgb": mean_rgb,
        "brightness": brightness,
        "color_temp_ratio": color_temp_ratio,
        "lab_mean": lab_mean,
        "sample_box": (halo_l, halo_t, halo_r, halo_b),
    }


def match_plate_lighting(
    image: Union[Image.Image, str, Path],
    plate: Union[Image.Image, str, Path],
    person_bbox: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
    relight_strength: float = 0.35,
    shadow_strength: float = 0.25,
    feather_radius: int = 25,
    client: Optional[GenAIClient] = None,
) -> Image.Image:
    """Samples plate color temperature and brightness near person region and relights person region only.

    Applies:
    1. Color grade: shifts person region color temperature and brightness toward the plate's ambient aisle lighting.
    2. Soft shadow: simulates overhead cold supermarket fluorescent falloff and ambient contact shadow.
    3. Seamless blend: feathered alpha mask ensures background outside the person region is 100% untouched.
    """
    gen_pil = Image.open(image) if isinstance(image, (str, Path)) else image.copy()
    if gen_pil.mode != "RGB":
        gen_pil = gen_pil.convert("RGB")

    w, h = gen_pil.size

    # 1. Resolve Person Bounding Box
    if person_bbox is not None:
        p_box = person_bbox
    else:
        p_box = locate_person(gen_pil, client=client) or [0.08, 0.10, 0.92, 0.90]

    left, top, right, bottom = parse_bbox_to_pixels(p_box, w, h)
    box_w = max(10, right - left)
    box_h = max(10, bottom - top)

    # 2. Sample Plate Ambient Lighting Near Person Region
    plate_stats = sample_plate_lighting(plate, person_bbox=p_box, target_size=(w, h))
    plate_rgb = plate_stats["mean_rgb"]
    plate_bright = plate_stats["brightness"]
    plate_ct = plate_stats["color_temp_ratio"]

    # 3. Sample Current Person Region in Image
    gen_np = np.array(gen_pil).astype(np.float32)
    person_roi = gen_np[top:bottom, left:right].copy()
    person_rgb = np.mean(person_roi, axis=(0, 1))
    person_bright = float(0.299 * person_rgb[0] + 0.587 * person_rgb[1] + 0.114 * person_rgb[2])
    person_ct = float(person_rgb[0] / (person_rgb[2] + 1e-5))

    # 4. Color Grade Gains (Color Temperature + Exposure Harmonization)
    raw_color_gains = plate_rgb / (person_rgb + 1e-5)
    mean_gain = np.mean(raw_color_gains)
    normalized_color_gains = raw_color_gains / (mean_gain + 1e-5)

    brightness_ratio = plate_bright / (person_bright + 1e-5)
    bounded_brightness_gain = float(np.clip(brightness_ratio, 0.85, 1.15))

    target_gains = normalized_color_gains * bounded_brightness_gain
    relight_str = float(np.clip(relight_strength, 0.0, 1.0))
    effective_gains = (1.0 - relight_str) * 1.0 + relight_str * target_gains
    effective_gains = np.clip(effective_gains, 0.70, 1.30)

    # Apply color grade to person ROI
    person_graded = np.clip(person_roi * effective_gains, 0.0, 255.0)

    # 5. Soft Overhead Shadow and Directional Ambient Occlusion
    # Downward falloff from overhead fluorescent lights
    y_coords = np.linspace(0.0, 1.0, box_h, dtype=np.float32)[:, None]
    overhead_shadow = 1.0 - (float(shadow_strength) * 0.20 * (y_coords ** 1.3))

    # Lateral ambient contact occlusion
    x_coords = np.linspace(-1.0, 1.0, box_w, dtype=np.float32)[None, :]
    lateral_ao = 1.0 - (float(shadow_strength) * 0.10 * (x_coords ** 2))

    shadow_map = np.clip(overhead_shadow * lateral_ao, 0.75, 1.0)[..., None]
    person_relit = np.clip(person_graded * shadow_map, 0.0, 255.0)

    # 6. Feathered Person Region Mask (Restricted to Person Region Only)
    mask = np.ones((box_h, box_w), dtype=np.float32)
    border_x = max(3, int(box_w * 0.12))
    border_y = max(3, int(box_h * 0.12))

    for y in range(box_h):
        for x in range(box_w):
            dx = min(x, box_w - 1 - x) / float(border_x)
            dy = min(y, box_h - 1 - y) / float(border_y)
            mask[y, x] = np.clip(min(dx, dy), 0.0, 1.0)

    ksize = max(5, (feather_radius * 2) | 1)
    feathered_mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=feather_radius * 0.6)[..., None]

    # Blend relit person over original ROI
    blended_person = person_relit * feathered_mask + person_roi * (1.0 - feathered_mask)

    # 7. Reconstruct Full Image (Background Outside Person Box is Completely Untouched)
    out_np = gen_np.copy()
    out_np[top:bottom, left:right] = blended_person
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)

    result_pil = Image.fromarray(out_np)

    # 8. Record Metrics in Image Info
    result_pil.info = dict(getattr(gen_pil, "info", {}))
    result_pil.info["lighting_matched"] = True
    result_pil.info["plate_color_temp"] = round(plate_ct, 3)
    result_pil.info["plate_brightness"] = round(plate_bright, 1)
    result_pil.info["person_color_temp_before"] = round(person_ct, 3)
    result_pil.info["person_brightness_before"] = round(person_bright, 1)

    person_after_rgb = np.mean(blended_person, axis=(0, 1))
    person_after_ct = float(person_after_rgb[0] / (person_after_rgb[2] + 1e-5))
    person_after_bright = float(0.299 * person_after_rgb[0] + 0.587 * person_after_rgb[1] + 0.114 * person_after_rgb[2])

    result_pil.info["person_color_temp_after"] = round(person_after_ct, 3)
    result_pil.info["person_brightness_after"] = round(person_after_bright, 1)
    setattr(result_pil, "_lighting_matched", True)

    logger.info(
        "match_plate_lighting complete: box=[%d, %d, %d, %d], plate_ct=%.3f, "
        "person_ct_before=%.3f -> person_ct_after=%.3f (gain=%.2f, shadow=%.2f)",
        left, top, right, bottom, plate_ct, person_ct, person_after_ct, relight_str, shadow_strength,
    )

    attach_mockup_metadata(result_pil)
    return result_pil


relight_person = match_plate_lighting


# ==============================================================================
# STAGE A.3: PLATE-MATCHED FINISHING PASS FOR EDITED REGION
# ==============================================================================

def measure_plate_noise(
    plate: Union[Image.Image, np.ndarray],
    edited_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """Measures the sensor noise standard deviation from the plate background.

    Samples outside the edited placement region to establish the plate's natural sensor noise floor.
    Uses high-pass residual: I - GaussianBlur(I, (3, 3), 0).
    """
    if isinstance(plate, Image.Image):
        plate_gray = np.array(plate.convert("L"), dtype=np.float32)
    elif plate.ndim == 3:
        plate_gray = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        plate_gray = plate.astype(np.float32)

    h, w = plate_gray.shape[:2]
    outside_mask = np.ones((h, w), dtype=bool)
    if edited_bbox is not None:
        l, t, r, b = edited_bbox
        outside_mask[t:b, l:r] = False

    smoothed = cv2.GaussianBlur(plate_gray, (3, 3), 0)
    residual = plate_gray - smoothed

    sample = residual[outside_mask] if np.any(outside_mask) else residual
    sigma = float(np.std(sample))
    return max(1.0, sigma)


def measure_sharpness(
    image: Union[Image.Image, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> float:
    """Measures image sharpness using the variance of the Laplacian (gradient energy)."""
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.float32)
    elif image.ndim == 3:
        gray = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    if mask is not None and np.any(mask):
        val = float(np.var(lap[mask]))
    else:
        val = float(np.var(lap))
    return max(0.1, val)


def sample_surrounding_plate_pixels(
    plate_np: np.ndarray,
    bbox: Tuple[int, int, int, int],
    collar_width: int = 30,
) -> np.ndarray:
    """Samples an ambient collar of pixels from the plate immediately surrounding the bbox."""
    h, w = plate_np.shape[:2]
    left, top, right, bottom = bbox

    pad_l = max(0, left - collar_width)
    pad_t = max(0, top - collar_width)
    pad_r = min(w, right + collar_width)
    pad_b = min(h, bottom + collar_width)

    # Collar mask: inside expanded box but outside inner box
    collar_mask = np.zeros((h, w), dtype=bool)
    collar_mask[pad_t:pad_b, pad_l:pad_r] = True
    collar_mask[top:bottom, left:right] = False

    if np.any(collar_mask):
        pixels = plate_np[collar_mask]
    else:
        pixels = plate_np.reshape(-1, 3)
    return pixels


def match_lab_color_to_surrounding(
    roi_rgb: np.ndarray,
    surround_rgb: np.ndarray,
    strength: float = 0.55,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Performs Reinhard color transfer in CIELAB color space against surrounding pixels.

    Harmonizes L*, a*, b* channel statistics between the edited region and surrounding plate collar.
    """
    roi_u8 = np.clip(roi_rgb, 0, 255).astype(np.uint8)
    surround_u8 = np.clip(surround_rgb, 0, 255).astype(np.uint8)

    # Convert ROI to LAB
    roi_lab = cv2.cvtColor(roi_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Convert surrounding pixels to LAB
    if surround_u8.ndim == 2:
        surround_u8 = surround_u8.reshape(1, -1, 3)
    surround_lab = cv2.cvtColor(surround_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    roi_mean = np.mean(roi_lab, axis=(0, 1))
    roi_std = np.std(roi_lab, axis=(0, 1))

    surround_mean = np.mean(surround_lab, axis=(0, 1))
    surround_std = np.std(surround_lab, axis=(0, 1))

    # Scale and shift in LAB space
    matched_lab = np.zeros_like(roi_lab)
    for c in range(3):
        scale = surround_std[c] / (roi_std[c] + 1e-4)
        scale = np.clip(scale, 0.70, 1.40)
        channel_val = (roi_lab[..., c] - roi_mean[c]) * scale + surround_mean[c]
        matched_lab[..., c] = (1.0 - strength) * roi_lab[..., c] + strength * channel_val

    matched_lab[..., 0] = np.clip(matched_lab[..., 0], 0, 255)
    matched_lab[..., 1] = np.clip(matched_lab[..., 1], 0, 255)
    matched_lab[..., 2] = np.clip(matched_lab[..., 2], 0, 255)

    matched_rgb = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    stats = {
        "surround_lab_mean": [round(float(v), 2) for v in surround_mean],
        "surround_lab_std": [round(float(v), 2) for v in surround_std],
        "roi_lab_mean_before": [round(float(v), 2) for v in roi_mean],
        "roi_lab_mean_after": [round(float(v), 2) for v in np.mean(matched_lab, axis=(0, 1))],
    }
    return matched_rgb, stats


def soften_face_region(
    roi_rgb: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
    strength: float = 0.50,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Applies a slight softening to the face region using bilateral filtering and feathered masking.

    Emulates authentic smartphone lens softening on skin and eliminates hyper-sharp AI facial artifacts.
    """
    h, w = roi_rgb.shape[:2]
    roi_u8 = np.clip(roi_rgb, 0, 255).astype(np.uint8)

    # 1. Detect face using Haar cascade if not explicitly provided
    detected_face: Optional[Tuple[int, int, int, int]] = face_bbox
    if detected_face is None:
        try:
            gray = cv2.cvtColor(roi_u8, cv2.COLOR_RGB2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cascade_path).is_file():
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(int(w * 0.12), int(h * 0.08)),
                )
                if len(faces) > 0:
                    best_f = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = best_f
                    detected_face = (int(fx), int(fy), int(fx + fw), int(fy + fh))
        except Exception as e:
            logger.debug("Haar cascade face detection bypassed: %s", e)
            detected_face = None

    # Fallback heuristic: upper-middle region of subject ROI
    if detected_face is None:
        fx1 = int(w * 0.22)
        fx2 = int(w * 0.78)
        fy1 = int(h * 0.08)
        fy2 = int(h * 0.40)
        detected_face = (fx1, fy1, fx2, fy2)

    fx1, fy1, fx2, fy2 = detected_face
    fw = max(4, fx2 - fx1)
    fh = max(4, fy2 - fy1)

    face_crop = roi_u8[fy1:fy2, fx1:fx2]
    if face_crop.size == 0:
        return roi_rgb, detected_face

    # Bilateral filter gently softens flat skin textures while preserving structural eye/lip boundaries
    softened_crop = cv2.bilateralFilter(face_crop, d=7, sigmaColor=30, sigmaSpace=7).astype(np.float32)

    # Elliptical feathered mask for organic transition
    y_idx, x_idx = np.indices((fh, fw), dtype=np.float32)
    cx, cy = fw / 2.0, fh / 2.0
    norm_dist = ((x_idx - cx) / (cx + 1e-5)) ** 2 + ((y_idx - cy) / (cy + 1e-5)) ** 2
    face_mask = np.clip(1.0 - norm_dist, 0.0, 1.0)
    k_blur = max(5, (min(fw, fh) // 4) | 1)
    face_mask = cv2.GaussianBlur(face_mask, (k_blur, k_blur), sigmaX=k_blur * 0.4)[..., None]

    blended_face = (1.0 - strength * face_mask) * face_crop.astype(np.float32) + (strength * face_mask) * softened_crop

    out_roi = roi_rgb.copy()
    out_roi[fy1:fy2, fx1:fx2] = blended_face
    return out_roi, detected_face


def match_sharpness_to_plate(
    roi_rgb: np.ndarray,
    sharpness_plate: float,
    strength: float = 0.60,
) -> Tuple[np.ndarray, float, float]:
    """Adjusts edited ROI sharpness to match the background plate's sharpness profile."""
    sharpness_roi = measure_sharpness(roi_rgb)
    out_roi = roi_rgb.copy()

    ratio = sharpness_roi / (sharpness_plate + 1e-4)

    if ratio > 1.15:
        # Overly sharp AI rendering: apply subtle Gaussian softening
        sigma = min(1.2, 0.35 + 0.25 * (ratio - 1.0))
        softened = cv2.GaussianBlur(roi_rgb, (3, 3), sigmaX=sigma)
        alpha = min(0.65, strength * 0.70)
        out_roi = (1.0 - alpha) * out_roi + alpha * softened
    elif ratio < 0.75:
        # Softer than plate: subtle unsharp mask
        blurred = cv2.GaussianBlur(roi_rgb, (3, 3), 0)
        high_pass = roi_rgb - blurred
        out_roi = np.clip(roi_rgb + 0.25 * high_pass, 0.0, 255.0)

    sharpness_after = measure_sharpness(out_roi)
    return out_roi, sharpness_roi, sharpness_after


def match_noise_to_plate(
    roi_rgb: np.ndarray,
    plate_noise_sigma: float,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, float, float]:
    """Synthesizes high-frequency noise in the edited region to match the plate's sensor noise floor."""
    if rng is None:
        rng = np.random.RandomState()

    roi_noise_sigma = measure_plate_noise(roi_rgb)
    out_roi = roi_rgb.copy()

    if plate_noise_sigma > roi_noise_sigma:
        delta_sigma = math.sqrt(max(0.0, plate_noise_sigma ** 2 - roi_noise_sigma ** 2))
        noise = rng.normal(0.0, delta_sigma, roi_rgb.shape).astype(np.float32)
        out_roi = np.clip(out_roi + noise, 0.0, 255.0)

    noise_after = measure_plate_noise(out_roi)
    return out_roi, roi_noise_sigma, noise_after


def apply_mild_shadow_grain(
    roi_rgb: np.ndarray,
    noise_sigma: float = 9.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Adds fine sensor grain concentrated in dark shadow areas (under chin, under tray)."""
    if rng is None:
        rng = np.random.RandomState()

    gray = 0.299 * roi_rgb[..., 0] + 0.587 * roi_rgb[..., 1] + 0.114 * roi_rgb[..., 2]
    # Shadow weight: higher in dark areas (< 110 luminance)
    shadow_weight = (np.clip((110.0 - gray) / 110.0, 0.0, 1.0) ** 1.4)[..., None]

    grain = rng.normal(0.0, noise_sigma, roi_rgb.shape).astype(np.float32)
    out_roi = np.clip(roi_rgb + grain * shadow_weight * 0.75, 0.0, 255.0)
    return out_roi


def apply_plate_matched_finishing(
    image: Union[Image.Image, str, Path],
    plate: Union[Image.Image, str, Path],
    edited_region: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
    face_region: Optional[Tuple[int, int, int, int]] = None,
    jpeg_quality: int = 86,
    feather_radius: int = 20,
    seed: Optional[int] = None,
    client: Optional[GenAIClient] = None,
) -> Image.Image:
    """Applies authentic plate-matched finishing pass to the edited region:

    1. Noise measured from the plate -> synthesized into edited region to match sensor noise floor.
    2. Sharpness measured from the plate -> harmonizes edge gradient energy with plate optics.
    3. Slight softening of the face region -> authentic smartphone skin reproduction (not hyper-sharp).
    4. LAB color matching against surrounding pixels -> aligns ambient color cast, white balance, and tone.
    5. Mild shadow grain -> sensor shot noise in low-luminance crevices (under chin, under tray).
    6. Feathered blend back into image -> preserves seamless transition.
    7. One global JPEG pass -> unifies whole-image 8x8 DCT quantization artifacts.
    """
    img_pil = Image.open(image) if isinstance(image, (str, Path)) else image.copy()
    plate_pil = Image.open(plate) if isinstance(plate, (str, Path)) else plate.copy()

    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    if plate_pil.mode != "RGB":
        plate_pil = plate_pil.convert("RGB")

    w, h = img_pil.size
    if plate_pil.size != (w, h):
        plate_pil = plate_pil.resize((w, h), Image.Resampling.LANCZOS)

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # 1. Resolve Edited Placement Bounding Box
    if edited_region is not None:
        p_box = edited_region
    else:
        info_region = getattr(img_pil, "info", {}).get("placement_region")
        if info_region:
            p_box = info_region
        else:
            p_box = [0.15, 0.15, 0.95, 0.85]

    left, top, right, bottom = parse_bbox_to_pixels(p_box, w, h)
    box_w = max(10, right - left)
    box_h = max(10, bottom - top)

    img_np = np.array(img_pil).astype(np.float32)
    plate_np = np.array(plate_pil).astype(np.float32)

    # 2. Measure Noise and Sharpness from Plate Background (Outside Edited Region)
    plate_noise_sigma = measure_plate_noise(plate_np, edited_bbox=(left, top, right, bottom))
    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[top:bottom, left:right] = False
    plate_sharpness = measure_sharpness(plate_np, mask=outside_mask)

    # Extract target edited ROI
    roi_rgb = img_np[top:bottom, left:right].copy()

    # 3. LAB Color Matching Against Surrounding Pixels
    surround_pixels = sample_surrounding_plate_pixels(plate_np, (left, top, right, bottom), collar_width=35)
    matched_roi, lab_stats = match_lab_color_to_surrounding(roi_rgb, surround_pixels, strength=0.55)

    # 4. Slight Softening of the Face Region (Phone cameras are less sharp than AI renders)
    softened_roi, detected_face = soften_face_region(matched_roi, face_bbox=face_region, strength=0.45)

    # 5. Sharpness Matching to Plate Profile
    sharp_matched_roi, sharpness_before, sharpness_after = match_sharpness_to_plate(
        softened_roi, plate_sharpness, strength=0.60
    )

    # 6. Noise Matching to Plate Noise Floor
    noise_matched_roi, noise_before, noise_after = match_noise_to_plate(
        sharp_matched_roi, plate_noise_sigma, rng=rng
    )

    # 7. Mild Shadow Grain in Low-Luminance Crevices (Under Chin, Under Tray)
    finished_roi = apply_mild_shadow_grain(noise_matched_roi, noise_sigma=8.5, rng=rng)

    # 8. Feathered Blend Back into Composite Image
    mask = np.ones((box_h, box_w), dtype=np.float32)
    border_x = max(3, int(box_w * 0.10))
    border_y = max(3, int(box_h * 0.10))
    for y in range(box_h):
        for x in range(box_w):
            dx = min(x, box_w - 1 - x) / float(border_x)
            dy = min(y, box_h - 1 - y) / float(border_y)
            mask[y, x] = np.clip(min(dx, dy), 0.0, 1.0)

    ksize = max(5, (feather_radius * 2) | 1)
    feathered_mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=feather_radius * 0.5)[..., None]

    blended_roi = finished_roi * feathered_mask + roi_rgb * (1.0 - feathered_mask)
    blended_img_np = img_np.copy()
    blended_img_np[top:bottom, left:right] = np.clip(blended_roi, 0, 255)
    blended_pil = Image.fromarray(blended_img_np.astype(np.uint8))

    # 9. One Global JPEG Pass Across the Entire Image
    q = int(np.clip(jpeg_quality, 80, 92))
    buf = io.BytesIO()
    blended_pil.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    final_pil = Image.open(buf).convert("RGB")

    # 10. Record Provenance and Metrics in Metadata
    final_pil.info = dict(getattr(img_pil, "info", {}))
    final_pil.info["plate_matched_finishing"] = {
        "plate_noise_sigma": round(plate_noise_sigma, 2),
        "plate_sharpness": round(plate_sharpness, 1),
        "roi_noise_before": round(noise_before, 2),
        "roi_noise_after": round(noise_after, 2),
        "roi_sharpness_before": round(sharpness_before, 1),
        "roi_sharpness_after": round(sharpness_after, 1),
        "face_softening_applied": True,
        "face_bbox": list(detected_face) if detected_face else None,
        "lab_matching": lab_stats,
        "global_jpeg_quality": q,
    }
    setattr(final_pil, "_plate_matched", True)
    attach_mockup_metadata(final_pil)

    logger.info(
        "apply_plate_matched_finishing: box=[%d, %d, %d, %d], plate_noise=%.2f, "
        "roi_noise=%.2f->%.2f, plate_sharpness=%.1f, roi_sharpness=%.1f->%.1f, jpeg_q=%d",
        left, top, right, bottom, plate_noise_sigma, noise_before, noise_after,
        plate_sharpness, sharpness_before, sharpness_after, q,
    )
    return final_pil


# ==============================================================================
# STAGE B: CANDID SMARTPHONE OPTICAL FINISHING
# ==============================================================================

def apply_phone_finishing(
    image: Union[Image.Image, str, Path],
    grain: float = 0.15,
    vignette: float = 0.20,
    softness: float = 0.10,
    chromatic_aberration: float = 0.15,
    highlight_bloom: float = 0.20,
    jpeg_quality: int = 85,
    tilt_deg: float = 1.0,
    off_center_crop: float = 0.15,
    saturation: float = 0.90,
    uneven_white_balance: float = 0.15,
    motion_blur: float = 0.15,
    realism_pass: bool = True,
    plate: Optional[Union[Image.Image, str, Path]] = None,
    person_bbox: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
    match_plate_lighting: bool = False,
    relight_strength: float = 0.35,
    shadow_strength: float = 0.25,
    seed: Optional[int] = None,
) -> Image.Image:
    """Applies authentic candid smartphone camera optical imperfections.

    Effects:
    0. Optional Lighting Match: samples plate color temp & brightness, relights person only with soft shadow
    1. Tiny handheld micro-tilt under 1.5 degrees
    2. Slight off-center crop (asymmetric handheld framing)
    3. Mild chromatic aberration (radial lateral lens dispersion)
    4. Mild highlight bloom on film glare (specular diffusion)
    5. Light sensor grain in shadow areas (luminance-dependent noise)
    6. Realism Pass: slightly lower saturation (-10%), uneven white balance gradient, small directional motion blur
    7. Slight lens softness (optical diffraction)
    8. Slight vignette (corner radial falloff)
    9. Realistic JPEG compression re-encoding (quality 82-90)
    10. Provenance metadata attachment ("AI-generated mockup")
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    else:
        pil_img = image.copy()

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    orig_info = dict(pil_img.info) if hasattr(pil_img, "info") and pil_img.info else {}

    # 0. Optional Lighting Match to Clean Plate (Person Region Relight + Soft Shadow)
    if match_plate_lighting and plate is not None:
        pil_img = relight_person(
            image=pil_img,
            plate=plate,
            person_bbox=person_bbox,
            relight_strength=relight_strength,
            shadow_strength=shadow_strength,
        )

    # Initialize deterministic or pseudorandom RNG
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    w, h = pil_img.size

    # 1. Handheld Camera Micro-Tilt (clamped strictly under 1.5 degrees)
    effective_tilt = float(np.clip(tilt_deg, 0.0, 1.5))
    if effective_tilt > 0.05:
        angle = float(rng.uniform(-effective_tilt, effective_tilt))
        # Rotate with bicubic interpolation without black border corners
        # Zoom slightly to eliminate empty corner pixels: scale = 1 / cos(theta)
        rad = math.radians(abs(angle))
        zoom_factor = 1.0 + math.sin(rad) + 0.005
        rotated = pil_img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)
        cl = (w - crop_w) // 2
        ct = (h - crop_h) // 2
        pil_img = rotated.crop((cl, ct, cl + crop_w, ct + crop_h)).resize((w, h), Image.Resampling.LANCZOS)

    # 2. Slight Off-Center Crop (Natural human framing imperfection)
    if off_center_crop > 0.01:
        # Crop 1.0% to 2.5% of margins with off-center bias
        margin_pct = 0.01 + 0.015 * min(1.0, off_center_crop)
        crop_w = int(w * (1.0 - margin_pct))
        crop_h = int(h * (1.0 - margin_pct))
        max_shift_x = w - crop_w
        max_shift_y = h - crop_h
        # Random off-center offset
        off_x = int(rng.uniform(0.1, 0.9) * max_shift_x)
        off_y = int(rng.uniform(0.1, 0.9) * max_shift_y)
        pil_img = pil_img.crop((off_x, off_y, off_x + crop_w, off_y + crop_h)).resize((w, h), Image.Resampling.LANCZOS)

    img_np = np.array(pil_img).astype(np.float32)

    # Realism Pass 1: Slightly lower saturation (removes synthetic AI hypersaturation)
    if realism_pass and saturation < 1.0:
        hsv = cv2.cvtColor(np.clip(img_np, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_gain = float(np.clip(saturation, 0.65, 1.0))
        hsv[..., 1] *= sat_gain
        np.clip(hsv[..., 1], 0.0, 255.0, out=hsv[..., 1])
        img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # Realism Pass 2: Uneven white balance (spatial fluorescent / ambient lighting gradient)
    if realism_pass and uneven_white_balance > 0.005:
        wb_str = float(min(1.0, uneven_white_balance)) * 0.045
        y_ramp = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        x_ramp = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :]
        red_gain = 1.0 + wb_str * y_ramp + (wb_str * 0.4) * x_ramp
        blue_gain = 1.0 - wb_str * y_ramp - (wb_str * 0.4) * x_ramp
        img_np[..., 0] = np.clip(img_np[..., 0] * red_gain, 0.0, 255.0)
        img_np[..., 2] = np.clip(img_np[..., 2] * blue_gain, 0.0, 255.0)

    # Realism Pass 3: Small directional motion blur (handheld camera micro-motion at 1/45s)
    if realism_pass and motion_blur > 0.005:
        mb_ksize = int(max(3, round(3 + min(1.0, motion_blur) * 3)))
        if mb_ksize % 2 == 0:
            mb_ksize += 1
        mb_kernel = np.zeros((mb_ksize, mb_ksize), dtype=np.float32)
        mb_angle_rad = math.radians(18.0)
        cx_mb, cy_mb = mb_ksize // 2, mb_ksize // 2
        for i in range(mb_ksize):
            offset = i - cx_mb
            kx = int(round(cx_mb + offset * math.cos(mb_angle_rad)))
            ky = int(round(cy_mb + offset * math.sin(mb_angle_rad)))
            if 0 <= kx < mb_ksize and 0 <= ky < mb_ksize:
                mb_kernel[ky, kx] = 1.0
        mb_sum = np.sum(mb_kernel)
        if mb_sum > 0:
            mb_kernel /= mb_sum
            blurred_np = cv2.filter2D(img_np, -1, mb_kernel)
            blend_w = float(min(1.0, motion_blur)) * 0.35
            img_np = (1.0 - blend_w) * img_np + blend_w * blurred_np
            np.clip(img_np, 0.0, 255.0, out=img_np)

    # 3. Mild Chromatic Aberration (Radial lateral color fringing away from optical center)
    if chromatic_aberration > 0.01:
        # Scale red outwards by 0.15-0.35%, scale blue inwards
        ca_strength = min(1.0, chromatic_aberration) * 0.003
        cx = w / 2.0
        cy = h / 2.0
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        norm_dx = (x_coords - cx) / cx
        norm_dy = (y_coords - cy) / cy
        r_sq = norm_dx ** 2 + norm_dy ** 2

        # Radial displacement maps
        r_shift_x = x_coords - norm_dx * r_sq * (ca_strength * cx)
        r_shift_y = y_coords - norm_dy * r_sq * (ca_strength * cy)

        b_shift_x = x_coords + norm_dx * r_sq * (ca_strength * cx)
        b_shift_y = y_coords + norm_dy * r_sq * (ca_strength * cy)

        ch_r = cv2.remap(img_np[..., 0], r_shift_x, r_shift_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ch_g = img_np[..., 1]
        ch_b = cv2.remap(img_np[..., 2], b_shift_x, b_shift_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        img_np = np.stack([ch_r, ch_g, ch_b], axis=-1)

    # 4. Mild Highlight Bloom on Film Glare (Diffused specular barrier shrink-wrap reflections)
    if highlight_bloom > 0.01:
        # Extract brightest specular highlights (luminance > 200)
        gray = np.float32(0.299) * img_np[..., 0] + np.float32(0.587) * img_np[..., 1] + np.float32(0.114) * img_np[..., 2]
        bloom_thresh = 195.0
        specular_mask = np.clip((gray - bloom_thresh) / (255.0 - bloom_thresh + 1e-5), 0.0, 1.0).astype(np.float32)

        # Smooth wide Gaussian diffusion blur
        ksize = max(15, (int(w * 0.02) * 2) | 1)
        specular_diffused = cv2.GaussianBlur(specular_mask, (ksize, ksize), sigmaX=ksize * 0.5)

        bloom_gain = np.float32(min(1.0, highlight_bloom) * 0.28)
        bloom_glow = (specular_diffused[..., None] * bloom_gain) * img_np
        img_np += bloom_glow
        np.clip(img_np, 0.0, 255.0, out=img_np)

    # 5. Light Sensor Grain in Shadow Areas (Darker tones receive more thermal/shot noise)
    if grain > 0.01:
        gray_norm = (np.float32(0.299) * img_np[..., 0] + np.float32(0.587) * img_np[..., 1] + np.float32(0.114) * img_np[..., 2]) * np.float32(1.0 / 255.0)
        shadow_weight = (np.clip(np.float32(1.0) - gray_norm, 0.0, 1.0) ** 1.6).astype(np.float32)
        noise_sigma = min(1.0, grain) * 18.0
        noise = rng.normal(0.0, noise_sigma, img_np.shape).astype(np.float32)
        img_np += noise * shadow_weight[..., None]
        np.clip(img_np, 0.0, 255.0, out=img_np)

    # 6. Slight Vignette (Corner radial optical falloff)
    if vignette > 0.01:
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        radius_sq = xx ** 2 + yy ** 2
        vig_amount = np.float32(min(1.0, vignette) * 0.35)
        vignette_mask = np.float32(1.0) - np.clip(radius_sq * vig_amount, 0.0, vig_amount * 0.95).astype(np.float32)
        img_np *= vignette_mask[..., None]
        np.clip(img_np, 0.0, 255.0, out=img_np)

    # Convert back to PIL for softness and compression
    processed_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    # 7. Slight Lens Softness (Diffraction & unsharpened candid look)
    if softness > 0.01:
        blur_radius = min(1.0, softness) * 1.2
        soft_blurred = processed_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # Subtle blend factor
        processed_pil = Image.blend(processed_pil, soft_blurred, alpha=min(0.25, softness * 0.20))

    # 8. Realistic Smartphone JPEG Recompression (Quality 82-90)
    q = int(np.clip(jpeg_quality, 82, 90))
    buf = io.BytesIO()
    processed_pil.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    final_pil = Image.open(buf)

    # 9. Preserve Metadata and Add "AI-generated mockup" Provenance
    final_pil.info.update(orig_info)
    if hasattr(pil_img, "info") and pil_img.info:
        final_pil.info.update(pil_img.info)
    if realism_pass:
        final_pil.info["realism_pass"] = True
        setattr(final_pil, "_has_realism_pass", True)
    if getattr(pil_img, "_lighting_matched", False):
        setattr(final_pil, "_lighting_matched", True)
    attach_mockup_metadata(final_pil)

    return final_pil


# ==============================================================================
# FINISHER CLASS WRAPPER
# ==============================================================================

class Finisher:
    """Unified Finisher managing Stage A Repair and Stage B Phone Finishing."""

    def __init__(self, config: Optional[Union[Config, Dict[str, Any]]] = None):
        if isinstance(config, Config):
            self.config = config
            self.defaults = config.finishing_defaults.model_dump()
        elif isinstance(config, dict):
            self.config = load_config(require_api_key=False)
            self.defaults = config.get("finishing_defaults", {
                "grain": 0.15,
                "vignette": 0.20,
                "softness": 0.10,
                "chromatic_aberration": 0.15,
                "highlight_bloom": 0.20,
                "jpeg_quality": 85,
                "tilt_deg": 1.0,
                "off_center_crop": 0.15,
                "saturation": 0.90,
                "uneven_white_balance": 0.15,
                "motion_blur": 0.15,
                "realism_pass": True,
            })
        else:
            self.config = load_config(require_api_key=False)
            self.defaults = self.config.finishing_defaults.model_dump()

    def composite_tray(
        self,
        generated_img: Union[Image.Image, str, Path],
        tray_cutout: Union[Image.Image, str, Path],
        bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]],
        feather_radius: int = 5,
        shadow_strength: float = 0.45,
    ) -> Image.Image:
        """Stage A: Perspective-warps cutout onto region, feathers, matches color, adds shadow."""
        return composite_tray(
            generated_img=generated_img,
            tray_cutout=tray_cutout,
            bbox=bbox,
            feather_radius=feather_radius,
            shadow_strength=shadow_strength,
        )

    def fix_region(
        self,
        image: Union[Image.Image, str, Path],
        bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]],
        instruction: str,
        client: Optional[GenAIClient] = None,
        padding_ratio: float = 0.20,
        feather_radius: int = 15,
    ) -> Image.Image:
        """Stage A: Re-edits target region via client.image_edit and blends back."""
        return fix_region(
            image=image,
            bbox=bbox,
            instruction=instruction,
            client=client,
            padding_ratio=padding_ratio,
            feather_radius=feather_radius,
        )

    def replace_package(
        self,
        image: Union[Image.Image, str, Path],
        package_cutout: Optional[Union[Image.Image, str, Path]] = None,
        corners: Optional[List[Tuple[float, float]]] = None,
        client: Optional[GenAIClient] = None,
        placement_region: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
        feather_radius: int = 5,
        shadow_strength: float = 0.35,
    ) -> Image.Image:
        """Stage A.1: Always replaces model package with real cut-out perspective-warped with hand preservation."""
        return replace_packaged_tray(
            image=image,
            package_cutout=package_cutout,
            corners=corners,
            client=client,
            placement_region=placement_region,
            feather_radius=feather_radius,
            shadow_strength=shadow_strength,
            config=self.config,
        )

    def match_plate_lighting(
        self,
        image: Union[Image.Image, str, Path],
        plate: Union[Image.Image, str, Path],
        person_bbox: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
        relight_strength: Optional[float] = None,
        shadow_strength: Optional[float] = None,
        feather_radius: int = 25,
        client: Optional[GenAIClient] = None,
    ) -> Image.Image:
        """Stage A.2: Relights person region with plate ambient color grade and soft shadow."""
        r_str = relight_strength if relight_strength is not None else float(self.defaults.get("relight_strength", 0.35))
        s_str = shadow_strength if shadow_strength is not None else float(self.defaults.get("shadow_strength", 0.25))
        return match_plate_lighting(
            image=image,
            plate=plate,
            person_bbox=person_bbox,
            relight_strength=r_str,
            shadow_strength=s_str,
            feather_radius=feather_radius,
            client=client,
        )

    def apply_plate_matched(
        self,
        image: Union[Image.Image, str, Path],
        plate: Union[Image.Image, str, Path],
        edited_region: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
        face_region: Optional[Tuple[int, int, int, int]] = None,
        jpeg_quality: Optional[int] = None,
        feather_radius: int = 20,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Stage A.3: Applies plate-matched finishing pass to edited region (noise, sharpness, face softening, LAB match, shadow grain, global JPEG)."""
        jq = jpeg_quality if jpeg_quality is not None else int(self.defaults.get("jpeg_quality", 86))
        return apply_plate_matched_finishing(
            image=image,
            plate=plate,
            edited_region=edited_region,
            face_region=face_region,
            jpeg_quality=jq,
            feather_radius=feather_radius,
            seed=seed,
        )

    def apply(
        self,
        image: Union[Image.Image, str, Path],
        overrides: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Stage B: Applies smartphone optical finishing with defaults and overrides."""
        params = dict(self.defaults)
        if overrides:
            params.update(overrides)

        grain = float(params.get("grain", 0.15))
        vignette = float(params.get("vignette", 0.20))
        softness = float(params.get("softness", 0.10))
        chromatic_aberration = float(params.get("chromatic_aberration", 0.15))
        highlight_bloom = float(params.get("highlight_bloom", 0.20))
        jpeg_quality = int(params.get("jpeg_quality", 85))
        tilt_deg = float(params.get("tilt_deg", 1.0))
        off_center_crop = float(params.get("off_center_crop", 0.15))
        saturation = float(params.get("saturation", 0.90))
        uneven_white_balance = float(params.get("uneven_white_balance", 0.15))
        motion_blur = float(params.get("motion_blur", 0.15))
        realism_pass = bool(params.get("realism_pass", getattr(self.config, "realism_pass", True)))
        match_plate_lighting_flag = bool(params.get("match_plate_lighting", getattr(self.config, "match_plate_lighting", False)))
        plate = params.get("plate", None)
        person_bbox = params.get("person_bbox", None)
        relight_strength = float(params.get("relight_strength", 0.35))
        shadow_strength = float(params.get("shadow_strength", 0.25))

        logger.info(
            "Finisher applying optical profile: grain=%.2f, vig=%.2f, soft=%.2f, ca=%.2f, bloom=%.2f, "
            "tilt=%.1f deg, sat=%.2f, uneven_wb=%.2f, motion_blur=%.2f, realism_pass=%s, lighting_match=%s, quality=%d",
            grain, vignette, softness, chromatic_aberration, highlight_bloom,
            tilt_deg, saturation, uneven_white_balance, motion_blur, realism_pass, match_plate_lighting_flag, jpeg_quality
        )

        return apply_phone_finishing(
            image=image,
            grain=grain,
            vignette=vignette,
            softness=softness,
            chromatic_aberration=chromatic_aberration,
            highlight_bloom=highlight_bloom,
            jpeg_quality=jpeg_quality,
            tilt_deg=tilt_deg,
            off_center_crop=off_center_crop,
            saturation=saturation,
            uneven_white_balance=uneven_white_balance,
            motion_blur=motion_blur,
            realism_pass=realism_pass,
            plate=plate,
            person_bbox=person_bbox,
            match_plate_lighting=match_plate_lighting_flag,
            relight_strength=relight_strength,
            shadow_strength=shadow_strength,
            seed=seed,
        )


def apply_finishing(
    image: Union[Image.Image, str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """Convenience function to apply smartphone optical finishing."""
    finisher = Finisher()
    return finisher.apply(image=image, overrides=overrides, seed=seed)


def create_lighting_match_comparison(
    before_img: Union[Image.Image, str, Path],
    after_img: Union[Image.Image, str, Path],
    plate_img: Union[Image.Image, str, Path],
    output_path: Union[str, Path],
    title: str = "CirklaGen Lighting Match: Plate Ambient Harmonization",
) -> Path:
    """Generates a clean 3-panel comparison [Clean Plate | Before Relight | After Lighting Match]."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    b_im = Image.open(before_img) if isinstance(before_img, (str, Path)) else before_img.copy()
    a_im = Image.open(after_img) if isinstance(after_img, (str, Path)) else after_img.copy()
    p_im = Image.open(plate_img) if isinstance(plate_img, (str, Path)) else plate_img.copy()

    target_h = 600
    p_w = int(p_im.width * (target_h / float(p_im.height)))
    b_w = int(b_im.width * (target_h / float(b_im.height)))
    a_w = int(a_im.width * (target_h / float(a_im.height)))

    p_scaled = p_im.resize((p_w, target_h), Image.Resampling.LANCZOS).convert("RGB")
    b_scaled = b_im.resize((b_w, target_h), Image.Resampling.LANCZOS).convert("RGB")
    a_scaled = a_im.resize((a_w, target_h), Image.Resampling.LANCZOS).convert("RGB")

    banner_h = 75
    divider_w = 6
    total_w = p_w + b_w + a_w + (divider_w * 2)
    total_h = target_h + banner_h

    canvas = Image.new("RGB", (total_w, total_h), color=(22, 22, 26))
    draw = ImageDraw.Draw(canvas)

    # Paste panels
    x_offset = 0
    canvas.paste(p_scaled, (x_offset, banner_h))
    x_offset += p_w
    draw.rectangle([x_offset, banner_h, x_offset + divider_w, total_h], fill=(65, 65, 75))
    x_offset += divider_w

    canvas.paste(b_scaled, (x_offset, banner_h))
    x_offset += b_w
    draw.rectangle([x_offset, banner_h, x_offset + divider_w, total_h], fill=(65, 65, 75))
    x_offset += divider_w

    canvas.paste(a_scaled, (x_offset, banner_h))

    # Header and Subtitle
    draw.text((20, 16), title, fill=(245, 245, 250))
    draw.text(
        (20, 44),
        "Plate color temp & brightness sampled -> Person region relit with color grade & soft shadow (AI-generated mockup)",
        fill=(160, 160, 175),
    )

    # Panel Badges
    # 1. Plate Badge
    draw.rectangle([15, banner_h + 15, 280, banner_h + 45], fill=(30, 30, 38))
    draw.text((25, banner_h + 22), "CLEAN PLATE (Ambient Ref)", fill=(120, 190, 240))

    # 2. Before Badge
    b_badge_x = p_w + divider_w + 15
    draw.rectangle([b_badge_x, banner_h + 15, b_badge_x + 240, banner_h + 45], fill=(30, 30, 38))
    draw.text((b_badge_x + 10, banner_h + 22), "BEFORE (Unrelit Person)", fill=(230, 130, 130))

    # 3. After Badge
    a_badge_x = p_w + divider_w + b_w + divider_w + 15
    draw.rectangle([a_badge_x, banner_h + 15, a_badge_x + 260, banner_h + 45], fill=(30, 30, 38))
    draw.text((a_badge_x + 10, banner_h + 22), "AFTER (Plate Relit + Shadow)", fill=(110, 230, 150))

    png_info = PngInfo()
    png_info.add_text("Comment", "AI-generated mockup")
    png_info.add_text("Description", "AI-generated mockup")
    png_info.add_text("Software", "CirklaGen Candid Smartphone Synthesizer")

    canvas.save(out_p, format="PNG", pnginfo=png_info)
    logger.info("Saved lighting match comparison to %s", out_p)
    return out_p


# ==============================================================================
# DEMO & COMPARISON GENERATOR
# ==============================================================================

def create_plate_matched_comparison_sheet(
    variations: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "CirklaGen: Plate-Matched Finishing Pass Comparison (3 Variations)",
) -> Path:
    """Generates a clean 3-variation Before vs After comparison sheet.

    Each item in variations dict contains:
    - 'before': Image.Image or path
    - 'after': Image.Image or path
    - 'label': str title / description
    - 'metrics': optional dict of noise/sharpness/LAB stats
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_vars = len(variations)
    if n_vars == 0:
        raise ValueError("variations list must not be empty")

    card_h = 420
    header_h = 80
    row_gap = 25
    col_gap = 15

    # Determine dimensions based on first image aspect ratio
    first_b = Image.open(variations[0]["before"]) if isinstance(variations[0]["before"], (str, Path)) else variations[0]["before"]
    card_w = int(first_b.width * (card_h / float(first_b.height)))

    total_w = card_w * 2 + col_gap + 40
    total_h = header_h + n_vars * (card_h + 55 + row_gap) + 20

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 22, 26))
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text((25, 18), title, fill=(245, 245, 250))
    draw.text(
        (25, 46),
        "Plate-matched noise & sharpness, facial softening, LAB ambient color matching, mild shadow grain & global JPEG pass",
        fill=(160, 165, 180),
    )

    curr_y = header_h + 10

    for idx, var in enumerate(variations):
        b_im = Image.open(var["before"]) if isinstance(var["before"], (str, Path)) else var["before"]
        a_im = Image.open(var["after"]) if isinstance(var["after"], (str, Path)) else var["after"]
        lbl = var.get("label", f"Variation {idx + 1}")
        metrics_str = var.get("metrics_summary", "")

        # Row Title Banner
        draw.text((25, curr_y), f"Variation {idx + 1}: {lbl}", fill=(230, 230, 240))
        curr_y += 20
        if metrics_str:
            draw.text((25, curr_y), metrics_str, fill=(130, 190, 240))
            curr_y += 22
        else:
            curr_y += 8

        # Resize cards
        b_scaled = b_im.resize((card_w, card_h), Image.Resampling.LANCZOS).convert("RGB")
        a_scaled = a_im.resize((card_w, card_h), Image.Resampling.LANCZOS).convert("RGB")

        # Paste Before & After
        b_x = 20
        a_x = 20 + card_w + col_gap

        canvas.paste(b_scaled, (b_x, curr_y))
        canvas.paste(a_scaled, (a_x, curr_y))

        # Badges
        draw.rectangle([b_x + 10, curr_y + 10, b_x + 200, curr_y + 36], fill=(30, 32, 40))
        draw.text((b_x + 16, curr_y + 15), "BEFORE (Raw Candidate)", fill=(230, 120, 120))

        draw.rectangle([a_x + 10, curr_y + 10, a_x + 235, curr_y + 36], fill=(30, 32, 40))
        draw.text((a_x + 16, curr_y + 15), "AFTER (Plate-Matched Finish)", fill=(110, 230, 150))

        curr_y += card_h + row_gap

    png_info = PngInfo()
    png_info.add_text("Comment", "AI-generated mockup")
    png_info.add_text("Description", "CirklaGen Plate-Matched Finishing Pass 3-Variation Comparison")
    canvas.save(out_p, format="PNG", pnginfo=png_info)
    logger.info("Saved 3-variation plate-matched comparison sheet to %s", out_p)
    return out_p

def create_before_after_comparison(
    before_img: Image.Image,
    after_img: Image.Image,
    output_path: Union[str, Path],
    title: str = "CirklaGen Candid Smartphone Finishing",
) -> Path:
    """Generates a clean side-by-side comparison with title and labels."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Standardize image sizes
    target_h = 720
    ratio_before = target_h / float(before_img.height)
    w_before = int(before_img.width * ratio_before)
    before_scaled = before_img.resize((w_before, target_h), Image.Resampling.LANCZOS)

    ratio_after = target_h / float(after_img.height)
    w_after = int(after_img.width * ratio_after)
    after_scaled = after_img.resize((w_after, target_h), Image.Resampling.LANCZOS)

    banner_h = 70
    divider_w = 6
    total_w = w_before + w_after + divider_w
    total_h = target_h + banner_h

    canvas = Image.new("RGB", (total_w, total_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(canvas)

    # Paste before and after
    canvas.paste(before_scaled, (0, banner_h))
    # Divider line
    draw.rectangle([w_before, banner_h, w_before + divider_w, total_h], fill=(70, 70, 80))
    canvas.paste(after_scaled, (w_before + divider_w, banner_h))

    # Header and Badges
    draw.text((20, 16), title, fill=(240, 240, 245))
    draw.text((20, 42), "Natural framing, shadow grain, film bloom & JPEG texture (AI-generated mockup)", fill=(160, 160, 175))

    # Label badges
    # Before label
    draw.rectangle([15, banner_h + 15, 175, banner_h + 45], fill=(30, 30, 35))
    draw.text((25, banner_h + 22), "BEFORE (Raw Gen)", fill=(220, 120, 120))

    # After label
    after_x = w_before + divider_w + 15
    draw.rectangle([after_x, banner_h + 15, after_x + 195, banner_h + 45], fill=(30, 30, 35))
    draw.text((after_x + 10, banner_h + 22), "AFTER (Phone Finish)", fill=(100, 220, 140))

    # Save with metadata
    png_info = PngInfo()
    png_info.add_text("Comment", "AI-generated mockup")
    png_info.add_text("Description", "AI-generated mockup")
    png_info.add_text("Software", "CirklaGen Candid Smartphone Synthesizer")

    canvas.save(out_p, format="PNG", pnginfo=png_info)
    logger.info("Saved side-by-side demo comparison to %s", out_p)
    return out_p


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CirklaGen Finishing: Apply authentic candid smartphone optical artifacts and repairs."
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to input image. If omitted, uses latest run output.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate side-by-side before/after comparison image",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path for finished image output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible finishing (default 42)",
    )
    parser.add_argument(
        "--grain",
        type=float,
        default=None,
        help="Override sensor shadow grain strength (0-1)",
    )
    parser.add_argument(
        "--vignette",
        type=float,
        default=None,
        help="Override vignette falloff strength (0-1)",
    )
    parser.add_argument(
        "--softness",
        type=float,
        default=None,
        help="Override lens softness strength (0-1)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=None,
        help="Override JPEG recompression quality (82-90)",
    )
    parser.add_argument(
        "--plate",
        type=str,
        default=None,
        help="Path to clean background plate image for ambient lighting match",
    )
    parser.add_argument(
        "--lighting-match",
        action="store_true",
        help="Apply subtle relight (color grade + soft shadow) to person region based on plate",
    )

    args = parser.parse_args()

    # Discover target image if omitted
    target_path = args.image_path
    if not target_path:
        runs_dir = Path("outputs/runs")
        raw_images = [p for p in runs_dir.rglob("*.png") if "finished" not in p.name and "demo" not in p.name]
        if raw_images:
            target_path = str(max(raw_images, key=lambda p: p.stat().st_mtime))
            print(f"No image path specified. Using latest raw run image: {target_path}")
        else:
            target_path = "assets/plates/Trader Joe's Shelf.jpeg"
            print(f"No run image found. Using plate image: {target_path}")

    input_path = Path(target_path)
    if not input_path.is_file():
        print(f"Error: Target image file not found: {input_path}")
        return

    print(f"\n[CirklaGen Finisher] Processing: {input_path}")
    raw_img = Image.open(input_path)

    overrides: Dict[str, Any] = {}
    if args.grain is not None:
        overrides["grain"] = args.grain
    if args.vignette is not None:
        overrides["vignette"] = args.vignette
    if args.softness is not None:
        overrides["softness"] = args.softness
    if args.quality is not None:
        overrides["jpeg_quality"] = args.quality
    if args.lighting_match:
        overrides["match_plate_lighting"] = True
        plate_p = Path(args.plate) if args.plate else Path("assets/plates/Trader Joe's Shelf.jpeg")
        if plate_p.is_file():
            overrides["plate"] = plate_p
            print(f"Lighting match active: sampling clean plate from {plate_p}")

    finisher = Finisher()
    finished_img = finisher.apply(raw_img, overrides=overrides, seed=args.seed)

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = input_path.parent / f"{input_path.stem}_finished.png"

    png_info = PngInfo()
    png_info.add_text("Comment", "AI-generated mockup")
    png_info.add_text("Description", "AI-generated mockup")
    png_info.add_text("Software", "CirklaGen Candid Smartphone Synthesizer")

    finished_img.save(out_path, format="PNG", pnginfo=png_info)
    print(f"Finished image saved to: {out_path}")

    if args.demo:
        if args.lighting_match and "plate" in overrides:
            demo_path = input_path.parent / f"{input_path.stem}_lighting_match_comparison.png"
            comp_path = create_lighting_match_comparison(
                before_img=raw_img,
                after_img=finished_img,
                plate_img=overrides["plate"],
                output_path=demo_path,
            )
            print(f"Demo 3-panel lighting match comparison saved to: {comp_path}")
        else:
            demo_path = input_path.parent / f"{input_path.stem}_before_after.png"
            comp_path = create_before_after_comparison(
                before_img=raw_img,
                after_img=finished_img,
                output_path=demo_path,
            )
            print(f"Demo side-by-side comparison saved to: {comp_path}")

    print("[CirklaGen Finisher] Complete.\n")


if __name__ == "__main__":
    main()
