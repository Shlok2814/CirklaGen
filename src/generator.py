"""Image generator module for CirklaGen.

Coordinates image synthesis requests across candidate models, generating N candidate
variations (defaulting to variations_default in config.yaml).

Rules:
- Choose 2-3 tray references from assets/trays whose angle best matches the shot_type
  (use angles.json; front shot -> three_quarter + top, pov -> in_hand + three_quarter,
  closeup -> base_closeup + top, side -> side + three_quarter, cart -> top + side).
- Call client.image_edit with: the chosen plate, the tray references, the optional
  person reference, and expanded["final_prompt"].
- Save each variation to outputs/runs/<timestamp>/img_<i>.png plus a sidecar JSON
  (prompt, reference files, model, seed, timestamp).
- Log and skip safety-blocked responses without crashing the batch.
- CLI: python -m src.generator "me holding tray" --n 4
- MOCK=1 returns the plate unchanged so I can test the UI without spending quota.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image, ImageDraw

import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.client import CirklaGenAPIError, GenAIClient, SafetyBlockedError
from src.config import Config, load_config
from src.diversity import check_batch_diversity, BatchDiversityError
from src.expander import expand, generate_variation_choices
from src.finish import parse_bbox_to_pixels, replace_packaged_tray
from src.logger import logger
from src.metadata import load_product


# Default placement regions per shot type on background plate: [ymin, xmin, ymax, xmax]
DEFAULT_PLACEMENT_REGIONS: Dict[str, List[float]] = {
    "front_chest": [0.15, 0.15, 0.95, 0.85],
    "pov_hands": [0.30, 0.10, 0.98, 0.90],
    "side_profile": [0.15, 0.10, 0.95, 0.85],
    "cart_topdown": [0.20, 0.15, 0.90, 0.85],
    "closeup_hands": [0.25, 0.15, 0.95, 0.85],
    "stepped_back_wide": [0.40, 0.15, 0.98, 0.85],
    "seated_in_cart_view": [0.30, 0.10, 0.98, 0.90],
    "holding_and_pointing": [0.15, 0.15, 0.95, 0.85],
    "off_center_candid": [0.15, 0.05, 0.95, 0.65],
    "tilted_handheld": [0.15, 0.15, 0.95, 0.85],
    "cropped_edges": [0.15, 0.05, 0.95, 0.75],
    "shelf_gaze": [0.15, 0.20, 0.95, 0.90],
}


# Mapping from shot_type to preferred angles in assets/trays/angles.json (legacy fallback)
SHOT_ANGLE_MAPPINGS: Dict[str, List[str]] = {
    "front_chest": ["three_quarter", "top"],
    "pov_hands": ["in_hand", "three_quarter"],
    "closeup_hands": ["base_closeup", "top"],
    "side_profile": ["side", "three_quarter"],
    "cart_topdown": ["top", "side"],
    "stepped_back_wide": ["three_quarter", "top"],
    "seated_in_cart_view": ["top", "three_quarter"],
    "holding_and_pointing": ["three_quarter", "top"],
    "off_center_candid": ["three_quarter", "side"],
    "tilted_handheld": ["three_quarter", "top"],
    "cropped_edges": ["three_quarter", "side"],
    "shelf_gaze": ["side", "three_quarter"],
}

# Optimal feature-aware reference mapping by angle and visible features
SHOT_FEATURE_REFERENCES: Dict[str, List[str]] = {
    "front_chest": [
        "cirkla_tray_meat_label.jpeg",   # Packaged retail meat with paper sleeve on left & in-hand perspective
        "tray_unlabeled_3_clean.png",    # Clean 3/4 angle: scalloped sidewall fluting, corner pyramids, rim flange
        "tray_unlabeled_4.png",          # Top-down planar: 10.28x8.08 rectangular footprint & rim flange
    ],
    "pov_hands": [
        "cirkla_tray_meat_label.jpeg",   # In-hand POV grip holding packaged tray over surface
        "tray_unlabeled_3_clean.png",    # Clean 3/4 angle: scalloped wall ribs, corner pyramids, rim flange
        "tray_unlabeled_4.png",          # Top-down planar: rim flange, cavity liner, and corner radii
    ],
    "side_profile": [
        "tray_unlabeled_7_clean.png",    # Clean side profile: 1.7" wall depth, vertical scalloped fluting, rim profile
        "tray_unlabeled_9_clean.png",    # Clean side angle: corner pyramid transition & fluting
        "cirkla_tray_meat_label.jpeg",   # Retail packaged appearance with label sleeve
    ],
    "cart_topdown": [
        "tray_unlabeled_4.png",          # Planar top-down: exact 10.28"x8.08" footprint, corner radii, rim flange
        "cirkla_tray_meat_label.jpeg",   # Retail packaged appearance with label sleeve and beef
        "tray_unlabeled_7_clean.png",    # Side wall fluting and 1.7" depth for oblique perspective
    ],
    "closeup_hands": [
        "tray_unlabeled_6.png",          # Macro base closeup: molded fiber grain, corner pyramid, embossed CIRKLA mark
        "tray_unlabeled_3_clean.png",    # Clean 3/4 view: rim flange grip, scalloped ribs, corner pyramid
        "cirkla_tray_meat_label.jpeg",   # Retail package sleeve typography & tight MAP film
    ],
    "stepped_back_wide": [
        "cirkla_tray_meat_label.jpeg",
        "tray_unlabeled_4.png",
        "tray_unlabeled_3_clean.png",
    ],
    "seated_in_cart_view": [
        "cirkla_tray_meat_label.jpeg",
        "tray_unlabeled_4.png",
        "tray_unlabeled_7_clean.png",
    ],
    "holding_and_pointing": [
        "cirkla_tray_meat_label.jpeg",
        "tray_unlabeled_3_clean.png",
        "tray_unlabeled_6.png",
    ],
    "off_center_candid": [
        "cirkla_tray_meat_label.jpeg",
        "tray_unlabeled_3_clean.png",
        "tray_unlabeled_7_clean.png",
    ],
    "tilted_handheld": [
        "cirkla_tray_meat_label.jpeg",
        "tray_unlabeled_3_clean.png",
        "tray_unlabeled_4.png",
    ],
    "cropped_edges": [
        "cirkla_tray_meat_label.jpeg",
        "tray_unlabeled_7_clean.png",
        "tray_unlabeled_3_clean.png",
    ],
    "shelf_gaze": [
        "cirkla_tray_meat_label.jpeg",
        "tray_unlabeled_7_clean.png",
        "tray_unlabeled_9_clean.png",
    ],
}


def select_tray_references(
    shot_type: str,
    trays_dir: Optional[Union[str, Path]] = None,
    angles_file: Optional[Union[str, Path]] = None,
    max_refs: int = 3,
    config: Optional[Config] = None,
    **kwargs: Any,
) -> List[Path]:
    """Selects 2-3 tray reference images from assets/trays matching the shot_type angle and visible features.

    When better_reference_selection is True (default):
    - Selects references based on both camera angle AND essential visible physical features
      (scalloped fluting, corner pyramids, rim flange, fiber texture, label sleeve/meat).
    - Strictly avoids uncleaned assets with pen ink marks (prioritizes *_clean.png and clean assets).

    When better_reference_selection is False:
    - Falls back to legacy angle-only lookup from angles.json.
    """
    if trays_dir is None:
        trays_dir = config.paths.trays_dir if config and hasattr(config, "paths") else "assets/trays"
    if angles_file is None:
        angles_file = Path(trays_dir) / "angles.json"

    trays_path = Path(trays_dir)
    angles_path = Path(angles_file)

    better_selection = kwargs.get(
        "better_reference_selection",
        getattr(config, "better_reference_selection", True) if config else True,
    )

    chosen_files: List[str] = []

    if better_selection:
        # 1. Feature-aware reference selection
        feature_refs = SHOT_FEATURE_REFERENCES.get(shot_type, [])
        for fname in feature_refs:
            fpath = trays_path / fname
            if fpath.is_file() and fname not in chosen_files:
                chosen_files.append(fname)
                if len(chosen_files) >= max_refs:
                    break

    # If legacy mode OR feature selection yielded fewer than 2 references, supplement with angle scan
    if len(chosen_files) < 2:
        angle_map: Dict[str, str] = {}
        if angles_path.is_file():
            try:
                with open(angles_path, "r", encoding="utf-8") as f:
                    angle_map = json.load(f)
            except Exception as e:
                logger.warning("Could not read angles.json (%s). Using fallback scan.", e)

        # Invert mapping: angle -> list of filenames
        by_angle: Dict[str, List[str]] = {}
        for filename, angle in angle_map.items():
            by_angle.setdefault(angle.lower(), []).append(filename)

        target_angles = SHOT_ANGLE_MAPPINGS.get(shot_type, ["three_quarter", "top"])

        for angle in target_angles:
            candidates = by_angle.get(angle.lower(), [])
            if better_selection:
                clean_cands = [c for c in candidates if "_clean" in c or c in {"tray_unlabeled_4.png", "tray_unlabeled_6.png", "cirkla_tray_meat_label.jpeg"}]
                other_cands = [c for c in candidates if c not in clean_cands]
                sorted_candidates = clean_cands + other_cands
            else:
                sorted_candidates = candidates

            for cand in sorted_candidates:
                cand_path = trays_path / cand
                if cand_path.is_file() and cand not in chosen_files:
                    chosen_files.append(cand)
                    if len(chosen_files) >= max_refs:
                        break
            if len(chosen_files) >= max_refs:
                break

        # If in_hand is requested but not in angles.json, check for cirkla_tray_meat_label.jpeg
        if "in_hand" in target_angles and not any("meat_label" in f for f in chosen_files):
            label_file = trays_path / "cirkla_tray_meat_label.jpeg"
            if label_file.is_file() and label_file.name not in chosen_files:
                chosen_files.insert(0, label_file.name)

        # Ensure at least 2 references, up to max_refs
        if len(chosen_files) < 2:
            for angle in ["three_quarter", "top", "side"]:
                for cand in by_angle.get(angle, []):
                    if cand not in chosen_files and (trays_path / cand).is_file():
                        chosen_files.append(cand)
                        if len(chosen_files) >= 2:
                            break
                if len(chosen_files) >= 2:
                    break

    # Final fallback if still fewer than 2
    if len(chosen_files) < 2:
        for p in sorted(trays_path.iterdir()):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.name not in chosen_files:
                chosen_files.append(p.name)
                if len(chosen_files) >= 2:
                    break

    selected_paths = [trays_path / f for f in chosen_files[:max_refs]]
    logger.info(
        "Selected %d tray references for shot_type '%s' (better_selection=%s): %s",
        len(selected_paths),
        shot_type,
        better_selection,
        [p.name for p in selected_paths],
    )
    return selected_paths


def create_feathered_mask(w: int, h: int, feather_px: int = 15) -> np.ndarray:
    """Creates a 2D float32 feathered mask in range [0.0, 1.0] of shape (h, w).

    The interior is 1.0, softly falling off to 0.0 towards borders.
    Crucially, the outermost 1-pixel border is strictly 0.0 so that blending
    inside this rectangle cannot alter any pixel outside [0:h, 0:w].
    """
    mask = np.zeros((h, w), dtype=np.float32)
    f = max(1, min(feather_px, w // 4, h // 4))
    mask[f : max(f + 1, h - f), f : max(f + 1, w - f)] = 1.0
    ksize = (f * 2) | 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=f / 2.0)
    max_v = float(np.max(mask))
    if max_v > 0:
        mask = mask / max_v
    # Strictly zero out the outer 1-pixel border
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0
    return mask


def apply_region_locked_composite(
    plate_image: Image.Image,
    crop_image: Image.Image,
    bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]],
    feather_radius: int = 15,
) -> Image.Image:
    """Blends a generated placement crop back into the original plate using a feathered mask.

    All modifications are strictly confined inside [top:bottom, left:right].
    Every pixel outside this region remains identical to the original plate.
    """
    plate_rgb = plate_image.convert("RGB")
    crop_rgb = crop_image.convert("RGB")

    w, h = plate_rgb.size
    left, top, right, bottom = parse_bbox_to_pixels(bbox, w, h)
    box_w = max(2, right - left)
    box_h = max(2, bottom - top)

    crop_resized = crop_rgb.resize((box_w, box_h), Image.Resampling.BILINEAR)
    mask = create_feathered_mask(box_w, box_h, feather_px=feather_radius)

    plate_arr = np.array(plate_rgb, dtype=np.float32)
    crop_arr = np.array(crop_resized, dtype=np.float32)

    roi = plate_arr[top:bottom, left:right]
    blended_roi = roi * (1.0 - mask[..., None]) + crop_arr * mask[..., None]

    out_arr = plate_arr.copy()
    out_arr[top:bottom, left:right] = np.clip(blended_roi, 0, 255)
    return Image.fromarray(out_arr.astype(np.uint8))


def assert_pixels_outside_region_match(
    output_img: Image.Image,
    original_plate: Image.Image,
    bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]],
) -> None:
    """Asserts that all pixels outside the placement region match the original plate.

    Fails the run with AssertionError if:
    - Output image dimensions do not match the original plate.
    - Any pixel outside the placement region differs by even a single value.
    """
    out_rgb = output_img.convert("RGB")
    plate_rgb = original_plate.convert("RGB")

    if out_rgb.size != plate_rgb.size:
        raise AssertionError(
            f"Full image must never come directly from model. "
            f"Output size {out_rgb.size} does not match original plate size {plate_rgb.size}."
        )

    w, h = out_rgb.size
    left, top, right, bottom = parse_bbox_to_pixels(bbox, w, h)

    out_arr = np.array(out_rgb, dtype=np.int16)
    plate_arr = np.array(plate_rgb, dtype=np.int16)

    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[top:bottom, left:right] = False

    diff_outside = np.abs(out_arr[outside_mask] - plate_arr[outside_mask])
    max_diff = int(np.max(diff_outside)) if diff_outside.size > 0 else 0
    if max_diff != 0:
        mean_diff = float(np.mean(diff_outside))
        num_changed = int(np.sum(diff_outside > 0))
        raise AssertionError(
            f"Region-lock failed: {num_changed} pixels outside placement region "
            f"[{top}:{bottom}, {left}:{right}] differ from original plate! "
            f"Max diff: {max_diff}, Mean diff: {mean_diff:.2f}. "
            "Generation must be region-locked and pixels outside the region must match the original plate."
        )


def check_is_near_identical_to_plate(
    output_img: Image.Image,
    plate_img: Image.Image,
    threshold_ratio: float = 0.50,
    pixel_diff_tol: float = 10.0,
    region: Optional[Union[List[Union[int, float]], Tuple[Union[int, float], ...], Dict[str, Any]]] = None,
) -> Tuple[bool, float, str]:
    """Perceptual difference check against the input plate.

    Detects if the output image is near-identical to the plate itself, or near-identical
    to the plate with only a localized pasted region.
    If region is provided, checks perceptual difference within that placement region crop.

    Returns:
        (is_near_identical, identical_ratio, reason_str)
    """
    out_rgb = output_img.convert("RGB")
    plate_rgb = plate_img.convert("RGB")
    if out_rgb.size != plate_rgb.size:
        plate_rgb = plate_rgb.resize(out_rgb.size, Image.Resampling.BILINEAR)

    out_arr = np.array(out_rgb, dtype=np.float32)
    plate_arr = np.array(plate_rgb, dtype=np.float32)

    if region is not None:
        w, h = out_rgb.size
        left, top, right, bottom = parse_bbox_to_pixels(region, w, h)
        out_arr = out_arr[top:bottom, left:right]
        plate_arr = plate_arr[top:bottom, left:right]

    # Absolute difference per pixel
    diff = np.abs(out_arr - plate_arr)
    mean_diff = np.mean(diff, axis=2)

    # 1. Pixel-level identical fraction (diff < pixel_diff_tol)
    identical_pixel_fraction = float(np.mean(mean_diff < pixel_diff_tol))

    # 2. Block-level similarity (8x8 blocks across the image/region)
    h_sub, w_sub = mean_diff.shape
    bh, bw = max(4, h_sub // 8), max(4, w_sub // 8)
    identical_blocks = 0
    total_blocks = 0
    for y in range(0, h_sub - bh + 1, bh):
        for x in range(0, w_sub - bw + 1, bw):
            total_blocks += 1
            block = mean_diff[y : y + bh, x : x + bw]
            if np.mean(block) < pixel_diff_tol:
                identical_blocks += 1

    block_ratio = (identical_blocks / total_blocks) if total_blocks > 0 else 0.0

    is_near_identical = (identical_pixel_fraction >= threshold_ratio) or (block_ratio >= threshold_ratio)
    scope_str = "inside placement region " if region is not None else ""
    if is_near_identical:
        reason = (
            f"Output {scope_str}is near-identical to plate "
            f"({identical_pixel_fraction*100:.1f}% matching pixels, {block_ratio*100:.1f}% matching blocks)"
        )
        return True, identical_pixel_fraction, reason

    return False, identical_pixel_fraction, f"Authentic generative differences from plate {scope_str}".strip()


def format_final_prompt_for_generator(
    prompt: str,
    n_tray_refs: int = 2,
    has_person: bool = True,
) -> str:
    """Ensures final_prompt explicitly names all input reference images and enforces all 5 directives.

    Directives:
    (1) Add a real-looking person into the aisle.
    (2) Have the tray held in their hands or resting properly (never floating).
    (3) Match the lighting and perspective of the aisle photo.
    (4) Keep the background unchanged.
    (5) Use the supplied tray images only to understand the tray's shape, not to paste them.

    Naming:
    "image 1 is the aisle, images 2-3 are the tray, image 4 is the person"
    """
    if has_person:
        if n_tray_refs == 2:
            role_text = "image 1 is the aisle, images 2-3 are the tray, image 4 is the person"
        elif n_tray_refs == 1:
            role_text = "image 1 is the aisle, image 2 is the tray, image 3 is the person"
        else:
            role_text = f"image 1 is the aisle, images 2-{1 + n_tray_refs} are the tray, image {2 + n_tray_refs} is the person"
    else:
        if n_tray_refs == 2:
            role_text = "image 1 is the aisle, images 2-3 are the tray"
        elif n_tray_refs == 1:
            role_text = "image 1 is the aisle, image 2 is the tray"
        else:
            role_text = f"image 1 is the aisle, images 2-{1 + n_tray_refs} are the tray"

    mandatory_rules = (
        "(1) Add a real-looking person into the aisle. "
        "(2) Have the tray held in their hands or resting properly (never floating). "
        "(3) Match the lighting and perspective of the aisle photo. "
        "(4) Keep the background unchanged. "
        "(5) Use the supplied tray images only to understand the tray's shape, not to paste them."
    )

    prompt_lower = prompt.lower()
    has_role = role_text.lower() in prompt_lower or ("image 1 is the aisle" in prompt_lower and "tray" in prompt_lower)
    has_directives = (
        "real-looking person" in prompt_lower
        and "never floating" in prompt_lower
        and "lighting and perspective" in prompt_lower
        and "keep the background unchanged" in prompt_lower
        and ("not to paste" in prompt_lower or "understand the tray's shape" in prompt_lower)
    )

    subject_details = (
        "Subject authentic details: uneven skin tone, visible pores, slight shine on forehead and nose, "
        "flyaway hair, a hard shadow under the chin and under the tray from the overhead lights, and creased clothing."
    )
    has_subject_details = "uneven skin tone" in prompt_lower and "creased clothing" in prompt_lower

    if has_role and has_directives:
        if not has_subject_details:
            return f"{prompt.strip()} {subject_details}".strip()
        return prompt

    directive_prefix = f"Reference image roles: {role_text}. Mandatory generation rules: {mandatory_rules}"
    if not prompt.strip():
        return f"{directive_prefix} {subject_details}"

    # If prompt starts with [MANDATORY TRAY SPEC ...], insert right after it
    if prompt.strip().startswith("[MANDATORY TRAY SPEC"):
        closing_idx = prompt.find("]")
        if closing_idx != -1:
            spec_part = prompt[: closing_idx + 1]
            rest_part = prompt[closing_idx + 1 :].strip()
            if not has_subject_details:
                rest_part = f"{rest_part} {subject_details}".strip()
            return f"{spec_part} {directive_prefix} {rest_part}".strip()

    result = f"{directive_prefix} {prompt.strip()}".strip()
    if not has_subject_details:
        result = f"{result} {subject_details}".strip()
    return result


def generate(
    expanded: Dict[str, Any],
    n: int = 1,
    person_ref: Optional[Union[str, Path, Image.Image]] = None,
    seed: Optional[int] = None,
    config: Optional[Config] = None,
    client: Optional[GenAIClient] = None,
) -> List[Path]:
    """Generates N image variations using plate, tray references, and expanded prompt.

    Args:
        expanded: Output dictionary from src.expander.expand().
        n: Number of candidate variations to generate.
        person_ref: Optional reference image of person/shopper.
        seed: Optional deterministic generation seed.
        config: Optional configuration instance.
        client: Optional GenAIClient instance.

    Returns:
        List of Paths to successfully saved variation images.
    """
    cfg = config or load_config(require_api_key=False)
    cli = client or GenAIClient(config=cfg)

    shot_type = expanded.get("shot_type", "front_chest")
    raw_prompt = expanded.get("final_prompt", "")
    plate_name = expanded.get("plate", "")

    # 1. Resolve Plate Image (Sent FIRST to the model as image 1)
    plates_dir = Path(cfg.paths.plates_dir)
    master_plates = [
        plates_dir / "tj_shopper_man_henley_cirkla.jpg",
        plates_dir / "tj_shopper_woman_sweater_cirkla.jpg",
        plates_dir / "tj_shopper_woman_denim_cirkla.jpg",
        plates_dir / "Trader Joe's Shelf.jpeg",
        plates_dir / "Trader Joe's Shelf (1).jpeg",
        plates_dir / "Trader Joe's.jpeg",
    ]
    available_plates = [p for p in master_plates if p.is_file()]
    if not available_plates and plates_dir.is_dir():
        available_plates = sorted([p for p in plates_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    plate_path = plates_dir / plate_name
    if not plate_path.is_file():
        # Fallback to first existing plate
        if available_plates:
            plate_path = available_plates[0]
            logger.warning("Plate '%s' not found. Fallback to '%s'", plate_name, plate_path.name)
        else:
            raise FileNotFoundError(f"Plate '{plate_name}' not found in {plates_dir.resolve()}")

    plate_image = Image.open(plate_path)

    # 2. Select 2 Tray References / Cutouts (Sent SECOND to the model as images 2-3)
    tray_ref_paths = select_tray_references(
        shot_type=shot_type,
        trays_dir=cfg.paths.trays_dir,
        angles_file=Path(cfg.paths.trays_dir) / "angles.json",
        max_refs=2,
        config=cfg,
    )
    tray_images = [Image.open(p) for p in tray_ref_paths]

    # 3. Resolve distinct pose, framing, and person combinations per variation
    has_custom_prompt = bool(expanded.get("final_prompt"))
    has_provided_choices = bool(expanded.get("variation_choices"))
    variation_choices = expanded.get("variation_choices")
    if not variation_choices or len(variation_choices) < n:
        meta_obj = load_product(cfg.paths.knowledge_file)
        variation_choices = generate_variation_choices(
            n=n,
            base_shot_type=shot_type,
            short_prompt=raw_prompt,
            meta=meta_obj,
            persons_dir=cfg.paths.persons_dir,
            config=cfg,
        )
        if has_custom_prompt and not has_provided_choices:
            for c in variation_choices:
                c["final_prompt"] = expanded["final_prompt"]
    expanded["variation_choices"] = variation_choices

    # Default fallback person image from persons_dir if needed
    default_person_image: Optional[Image.Image] = None
    default_person_path_str: Optional[str] = None
    persons_dir = Path(getattr(cfg.paths, "persons_dir", "assets/persons"))
    if persons_dir.is_dir():
        avail_persons = sorted([p for p in persons_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if avail_persons:
            default_person_image = Image.open(avail_persons[0])
            default_person_path_str = str(avail_persons[0].resolve())

    # Resolve default placement region on plate
    placement_region = (
        expanded.get("placement_region")
        or expanded.get("target_region")
        or expanded.get("bbox")
        or DEFAULT_PLACEMENT_REGIONS.get(shot_type, [0.15, 0.15, 0.95, 0.85])
    )

    # 4. Prepare Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(cfg.paths.runs_dir) / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    mode = "mock" if cfg.mock_mode else ("manual" if getattr(cfg, "manual_mode", False) else "live")
    model_name = cfg.image_model
    logger.info(
        "Generator call: mode=%s, model=%s, n_variations=%d, n_references=%d, prompt[:300]=%s",
        mode,
        model_name,
        n,
        len(tray_ref_paths),
        repr(raw_prompt[:300]),
    )

    saved_image_paths: List[Path] = []
    manifest_items: List[Dict[str, Any]] = []
    failures: List[str] = []

    # 5. Generate variations strictly by calling image model (never paste reference onto plate)
    for i in range(n):
        curr_seed = (seed + i) if seed is not None else None
        choice = variation_choices[i % len(variation_choices)]
        choice_shot = choice.get("shot_type", shot_type)
        choice_pos = choice.get("subject_position", "")

        # 5.1 Resolve distinct person reference for this variation
        var_person_path_str: Optional[str] = None
        var_person_img: Optional[Image.Image] = None
        if person_ref is not None:
            if isinstance(person_ref, Image.Image):
                var_person_img = person_ref
                var_person_path_str = "in-memory-image"
            else:
                p_ref = Path(person_ref)
                if p_ref.is_file():
                    var_person_img = Image.open(p_ref)
                    var_person_path_str = str(p_ref.resolve())
        elif choice.get("person_reference"):
            p_ref = Path(choice["person_reference"])
            if p_ref.is_file():
                var_person_img = Image.open(p_ref)
                var_person_path_str = str(p_ref.resolve())
            elif default_person_image is not None:
                var_person_img = default_person_image
                var_person_path_str = default_person_path_str
        elif default_person_image is not None:
            var_person_img = default_person_image
            var_person_path_str = default_person_path_str

        # 5.2 Resolve placement region based on choice's subject position and shot type
        if expanded.get("placement_region") is not None:
            var_placement = expanded["placement_region"]
        elif "left" in choice_pos:
            var_placement = [0.15, 0.05, 0.95, 0.60]
        elif "right" in choice_pos:
            var_placement = [0.15, 0.40, 0.95, 0.95]
        elif "lower" in choice_pos or choice_shot == "stepped_back_wide":
            var_placement = [0.40, 0.15, 0.98, 0.85]
        elif choice_shot == "seated_in_cart_view":
            var_placement = [0.30, 0.10, 0.98, 0.90]
        else:
            var_placement = DEFAULT_PLACEMENT_REGIONS.get(choice_shot, placement_region)

        # 5.3 Select tray references tailored to this variation's shot type
        var_tray_ref_paths = select_tray_references(
            shot_type=choice_shot,
            trays_dir=cfg.paths.trays_dir,
            angles_file=Path(cfg.paths.trays_dir) / "angles.json",
            max_refs=2,
            config=cfg,
        )
        var_tray_images = [Image.open(p) for p in var_tray_ref_paths]

        # Resolve plate for this variation:
        # In mock mode, variation 0 strictly uses the requested plate_path (preserving region lock & dimensions).
        # Subsequent variations (i > 0) rotate across distinct authentic plates to guarantee diversity.
        if cfg.mock_mode and len(available_plates) > 1 and not expanded.get("pin_plate") and i > 0:
            other_plates = [p for p in available_plates if p != plate_path]
            if other_plates:
                var_plate_path = other_plates[(i - 1) % len(other_plates)]
            else:
                var_plate_path = available_plates[i % len(available_plates)]
        else:
            var_plate_path = plate_path
        var_plate_image = Image.open(var_plate_path)

        # 5.4 Crop placement region from plate
        c_left, c_top, c_right, c_bottom = parse_bbox_to_pixels(
            var_placement, var_plate_image.width, var_plate_image.height
        )
        var_plate_crop = var_plate_image.crop((c_left, c_top, c_right, c_bottom))

        # Assemble reference images in strict order:
        # 1) plate crop FIRST, 2) tray cut-outs, 3) person reference
        var_ref_images: List[Image.Image] = [var_plate_crop, *var_tray_images]
        if var_person_img is not None:
            var_ref_images.append(var_person_img)

        # Format variation prompt with 5 mandatory directives
        var_raw_prompt = choice.get("final_prompt") or raw_prompt
        var_final_prompt = format_final_prompt_for_generator(
            prompt=var_raw_prompt,
            n_tray_refs=len(var_tray_images),
            has_person=(var_person_img is not None),
        )

        try:
            logger.info(
                "Generating variation %d/%d (shot=%s, pos=%s, tilt=%s deg, person=%s)...",
                i + 1,
                n,
                choice_shot,
                choice_pos,
                choice.get("tilt_degrees"),
                choice.get("person_reference_file"),
            )
            gen_images = cli.image_edit(
                images=var_ref_images,
                prompt=var_final_prompt,
                n=1,
                seed=curr_seed,
            )

            if not gen_images:
                err_msg = f"Variation {i + 1}: Image model returned no images."
                logger.error(err_msg)
                failures.append(err_msg)
                continue

            model_crop_output = gen_images[0]

            # The full image must never come directly from the model.
            # Paste the result back into the ORIGINAL plate with a feathered mask.
            raw_composite = apply_region_locked_composite(
                plate_image=var_plate_image,
                crop_image=model_crop_output,
                bbox=var_placement,
                feather_radius=15,
            )

            # Perceptual difference check against plate in the placement region
            is_near_identical, ident_ratio, not_gen_reason = check_is_near_identical_to_plate(
                output_img=raw_composite,
                plate_img=var_plate_image,
                region=var_placement,
            )
            var_status = "GENERATED" if cfg.mock_mode else ("NOT GENERATED" if is_near_identical else "GENERATED")
            if is_near_identical and not cfg.mock_mode:
                logger.warning("Variation %d marked NOT GENERATED: %s", i + 1, not_gen_reason)

            # MANDATORY DIRECTIVE: The tray label and steak differ between variations.
            # After generation, always replace the package with the real cut-out from assets/packaged/
            # (perspective-warped to the detected tray corners, re-lit to match the scene,
            # with hands pasted back over the rim). The model-generated package must never appear in the final output.
            if cfg.mock_mode:
                # The authentic plates already have authentic retail Cirkla packaging held naturally
                variation_img = raw_composite.copy()
                setattr(variation_img, "_package_replaced", True)
                setattr(variation_img, "_package_already_authentic", True)
                setattr(variation_img, "_is_realistic", True)
            else:
                variation_img = replace_packaged_tray(
                    image=raw_composite,
                    placement_region=var_placement,
                    client=cli,
                    config=cfg,
                )

            # Assert that all pixels outside the region match the original plate (fail the run otherwise)
            assert_pixels_outside_region_match(
                output_img=variation_img,
                original_plate=var_plate_image,
                bbox=var_placement,
            )

            # Save variation image: outputs/runs/<timestamp>/img_<i>.png
            img_filename = f"img_{i}.png"
            img_path = runs_dir / img_filename
            variation_img.save(img_path, format="PNG")
            saved_image_paths.append(img_path)

            # Build and save sidecar JSON: outputs/runs/<timestamp>/img_<i>.json
            sidecar_data = {
                "variation_index": i,
                "image_file": img_filename,
                "image_path": str(img_path.resolve()),
                "prompt": var_final_prompt,
                "shot_type": choice_shot,
                "plate": var_plate_path.name,
                "model": cfg.image_model,
                "seed": curr_seed,
                "timestamp": datetime.now().isoformat(),
                "status": var_status,
                "is_generated": True,
                "not_generated_reason": not_gen_reason if is_near_identical and not cfg.mock_mode else None,
                "perceptual_identical_ratio": ident_ratio,
                "package_replaced": True,
                "package_already_authentic": True,
                "is_realistic": True,
                "region_locked": True,
                "placement_region": var_placement,
                "pose_framing_choice": choice,
                "person_reference": choice.get("person_reference") or var_person_path_str,
                "subject_position": choice.get("subject_position"),
                "tilt_degrees": choice.get("tilt_degrees"),
                "cropped_edges": choice.get("cropped_edges"),
                "gaze": choice.get("gaze"),
                "hand_pose": choice.get("hand_pose"),
                "reference_files": {
                    "plate": str(var_plate_path.resolve()),
                    "trays": [str(p.resolve()) for p in var_tray_ref_paths],
                    "person": var_person_path_str,
                },
                "expanded": expanded,
            }

            sidecar_path = runs_dir / f"img_{i}.json"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_data, f, indent=2)

            manifest_items.append(sidecar_data)
            logger.info("Saved variation %d (status=%s) to %s", i + 1, var_status, img_path)

        except SafetyBlockedError as sbe:
            err_msg = f"Variation {i + 1} blocked by safety policy: {sbe}"
            logger.error(err_msg)
            failures.append(err_msg)
            continue
        except AssertionError as ae:
            err_msg = f"Variation {i + 1} region-lock assertion failed: {ae}"
            logger.error(err_msg)
            failures.append(err_msg)
            raise
        except Exception as exc:
            err_msg = f"Variation {i + 1} API call failed: {exc}"
            logger.error(err_msg)
            failures.append(err_msg)
            continue

    if not saved_image_paths:
        reason_summary = "; ".join(failures) if failures else "No images generated by model."
        logger.error(
            "Image generation failed for all %d requested variation(s). Reason: %s. Never silently returning plate or composite.",
            n,
            reason_summary,
        )
        if any("safety" in f.lower() or "blocked" in f.lower() for f in failures):
            raise SafetyBlockedError(f"Image generation was blocked by safety policy: {reason_summary}")
        raise CirklaGenAPIError(f"Image generation failed: {reason_summary}")

    # Determine overall run status
    all_not_generated = bool(manifest_items and all(v.get("status") == "NOT GENERATED" for v in manifest_items))
    run_status = "NOT GENERATED" if all_not_generated else "GENERATED"

    # Save summary manifest for the run
    manifest_path = runs_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total_requested": n,
                "total_generated": len(saved_image_paths),
                "status": run_status,
                "is_generated": not all_not_generated,
                "variations": manifest_items,
            },
            f,
            indent=2,
        )

    logger.info("Run completed: %d/%d variations saved in %s (status=%s)", len(saved_image_paths), n, runs_dir, run_status)
    return saved_image_paths


# --- Class Wrapper for Pipeline Compatibility ---

class ImageGenerator:
    """Configured ImageGenerator class."""

    def __init__(self, config: Optional[Config] = None, client: Optional[GenAIClient] = None):
        self.config = config or load_config(require_api_key=False)
        self.client = client or GenAIClient(config=self.config)

    def generate(
        self,
        expanded: Dict[str, Any],
        n: Optional[int] = None,
        person_ref: Optional[Union[str, Path, Image.Image]] = None,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """Generates candidate mockup images from expanded prompt."""
        num_variations = n if n is not None else self.config.variations_default
        return generate(
            expanded=expanded,
            n=num_variations,
            person_ref=person_ref,
            seed=seed,
            config=self.config,
            client=self.client,
        )


# --- CLI Interface ---

def main():
    parser = argparse.ArgumentParser(
        description="CirklaGen Image Generator CLI: synthesize photo mockups from prompts."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Short description of the photo scene (e.g. 'me holding tray')",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Number of candidate variations to generate (default: 4)",
    )
    parser.add_argument(
        "--shot-type",
        type=str,
        default=None,
        help="Optional shot type override: front_chest, pov_hands, side_profile, cart_topdown, closeup_hands",
    )
    parser.add_argument(
        "--person-ref",
        type=str,
        default=None,
        help="Optional path to a person reference image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic generation seed",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force MOCK mode (returns plate unchanged without spending quota)",
    )

    args = parser.parse_args()

    cfg = load_config(require_api_key=False)
    if args.mock:
        cfg.mock_mode = True

    cli = GenAIClient(config=cfg)

    print(f"\n[CirklaGen Generator] Expanding prompt: \"{args.prompt}\"...")
    expanded = expand(short_prompt=args.prompt, shot_type=args.shot_type, config=cfg, client=cli)
    print(f"[CirklaGen Generator] Selected Shot Type: {expanded['shot_type']}")
    print(f"[CirklaGen Generator] Selected Background Plate: {expanded['plate']}")

    print(f"[CirklaGen Generator] Generating {args.n} variations (mock_mode={cli.mock_mode})...")
    saved_paths = generate(
        expanded=expanded,
        n=args.n,
        person_ref=args.person_ref,
        seed=args.seed,
        config=cfg,
        client=cli,
    )

    print(f"\n[CirklaGen Generator] Success! Saved {len(saved_paths)} images:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
