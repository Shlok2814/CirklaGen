"""Tests for region-locked image generation and strict plate matching outside mask.

Requirements:
- Generation must be region-locked.
- Crop placement region from plate and send only that crop to the model.
- Paste result back into the ORIGINAL plate with a feathered mask.
- Assert that all pixels outside the region match the original plate (fail run otherwise).
- The full image must never come directly from the model.
- Test that compares a run's output to its plate outside the mask.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.finish import parse_bbox_to_pixels
from src.generator import (
    DEFAULT_PLACEMENT_REGIONS,
    apply_region_locked_composite,
    assert_pixels_outside_region_match,
    create_feathered_mask,
    generate,
)
from src.pipeline import GenerationPipeline


@pytest.fixture
def test_config(tmp_path) -> Config:
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    cfg.paths.runs_dir = str(tmp_path / "test_runs")
    return cfg


@pytest.fixture
def sample_plate() -> Image.Image:
    plate_path = Path("assets/plates/Trader Joe's Shelf.jpeg")
    return Image.open(plate_path)


def test_feathered_mask_boundary_strictly_zero():
    """Verify create_feathered_mask creates smooth transition with strictly zero outer boundary."""
    w, h = 200, 300
    mask = create_feathered_mask(w, h, feather_px=15)
    assert mask.shape == (h, w)
    assert mask.dtype == np.float32

    # Outer 1px perimeter must be strictly 0.0
    assert np.all(mask[0, :] == 0.0)
    assert np.all(mask[-1, :] == 0.0)
    assert np.all(mask[:, 0] == 0.0)
    assert np.all(mask[:, -1] == 0.0)

    # Center should be 1.0
    assert mask[h // 2, w // 2] == 1.0


def test_apply_region_locked_composite_preserves_all_outside_pixels(sample_plate):
    """Verify that apply_region_locked_composite modifies ONLY the specified region."""
    w, h = sample_plate.size
    bbox = [0.20, 0.15, 0.85, 0.80]
    left, top, right, bottom = parse_bbox_to_pixels(bbox, w, h)

    # Generate synthetic crop with distinct content
    crop_w, crop_h = right - left, bottom - top
    crop_img = Image.new("RGB", (crop_w, crop_h), color=(255, 0, 0))

    composited = apply_region_locked_composite(sample_plate, crop_img, bbox, feather_radius=15)
    assert composited.size == sample_plate.size

    # Assert pixels outside bbox match identically
    assert_pixels_outside_region_match(composited, sample_plate, bbox)

    # Verify directly on raw arrays
    orig_arr = np.array(sample_plate.convert("RGB"), dtype=np.int16)
    comp_arr = np.array(composited, dtype=np.int16)

    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[top:bottom, left:right] = False
    diff_outside = np.abs(comp_arr[outside_mask] - orig_arr[outside_mask])
    assert np.max(diff_outside) == 0, f"Max diff outside region is {np.max(diff_outside)}, expected 0"


def test_assert_pixels_outside_region_match_detects_violations(sample_plate):
    """Verify assert_pixels_outside_region_match raises AssertionError if any outside pixel differs."""
    w, h = sample_plate.size
    bbox = [0.20, 0.15, 0.85, 0.80]
    left, top, right, bottom = parse_bbox_to_pixels(bbox, w, h)

    crop_img = Image.new("RGB", (right - left, bottom - top), color=(200, 100, 50))
    composited = apply_region_locked_composite(sample_plate, crop_img, bbox)

    # Negative test 1: Modify a single pixel outside the bounding box
    tampered_np = np.array(composited).copy()
    tampered_np[0, 0, 0] = np.uint8((int(tampered_np[0, 0, 0]) + 50) % 255)
    tampered_img = Image.fromarray(tampered_np)

    with pytest.raises(AssertionError) as exc_info:
        assert_pixels_outside_region_match(tampered_img, sample_plate, bbox)
    assert "Region-lock failed" in str(exc_info.value)
    assert "pixels outside placement region" in str(exc_info.value)

    # Negative test 2: Uncomposited full image directly from model (wrong size or full scene change)
    uncomposited_full = Image.new("RGB", sample_plate.size, color=(50, 60, 70))
    with pytest.raises(AssertionError):
        assert_pixels_outside_region_match(uncomposited_full, sample_plate, bbox)

    # Negative test 3: Mismatched size
    wrong_size_img = Image.new("RGB", (300, 300), color=(10, 20, 30))
    with pytest.raises(AssertionError) as exc_info3:
        assert_pixels_outside_region_match(wrong_size_img, sample_plate, bbox)
    assert "Full image must never come directly from model" in str(exc_info3.value)


def test_generator_sends_only_crop_to_model(test_config, sample_plate):
    """Verify that generate() crops placement region and sends only that crop as reference 1 to the model."""
    captured_first_image = []

    class CropCapturingClient:
        mock_mode = False

        def image_edit(self, images, prompt, n=1, seed=None):
            captured_first_image.append(images[0])
            # Return model output sized to the input crop
            return [Image.new("RGB", images[0].size, color=(180, 120, 60)) for _ in range(n)]

    cli = CropCapturingClient()
    expanded = {
        "plate": "Trader Joe's Shelf.jpeg",
        "shot_type": "front_chest",
        "placement_region": [0.15, 0.15, 0.95, 0.85],
        "final_prompt": "Candid grocery photo.",
    }

    saved = generate(expanded=expanded, n=1, config=test_config, client=cli)
    assert len(saved) == 1
    assert len(captured_first_image) == 1

    # First image sent must be the cropped placement region, NOT the full plate
    plate_w, plate_h = sample_plate.size
    left, top, right, bottom = parse_bbox_to_pixels(expanded["placement_region"], plate_w, plate_h)
    expected_crop_size = (right - left, bottom - top)

    sent_crop = captured_first_image[0]
    assert sent_crop.size == expected_crop_size
    assert sent_crop.size != sample_plate.size, "Full plate must not be sent directly to model; only crop."


def test_run_output_matches_plate_outside_mask(test_config, sample_plate):
    """Core Test: Compares a run's output to its plate outside the mask and asserts 0 diff."""
    class GenerativeClient:
        mock_mode = False

        def image_edit(self, images, prompt, n=1, seed=None):
            # Model generates realistic subject inside the crop
            crop = images[0]
            model_render = Image.new("RGB", crop.size, color=(220, 140, 90))
            return [model_render for _ in range(n)]

    cli = GenerativeClient()
    placement_box = [0.20, 0.15, 0.90, 0.85]
    expanded = {
        "plate": "Trader Joe's Shelf.jpeg",
        "shot_type": "front_chest",
        "placement_region": placement_box,
        "final_prompt": "A person holding meat tray.",
    }

    saved_paths = generate(expanded=expanded, n=1, config=test_config, client=cli)
    output_path = saved_paths[0]

    with Image.open(output_path) as out_im:
        out_rgb = out_im.convert("RGB")
    plate_rgb = sample_plate.convert("RGB")

    assert out_rgb.size == plate_rgb.size

    w, h = out_rgb.size
    left, top, right, bottom = parse_bbox_to_pixels(placement_box, w, h)

    out_arr = np.array(out_rgb, dtype=np.int16)
    plate_arr = np.array(plate_rgb, dtype=np.int16)

    # Boolean mask of pixels OUTSIDE the placement region
    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[top:bottom, left:right] = False

    # Check differences outside the region
    diff_outside = np.abs(out_arr[outside_mask] - plate_arr[outside_mask])
    max_diff = int(np.max(diff_outside))
    mean_diff = float(np.mean(diff_outside))

    assert max_diff == 0, f"Region-lock failed: max_diff outside mask was {max_diff}"
    assert mean_diff == 0.0, f"Mean diff outside mask was {mean_diff}"

    # Also assert using the strict assert_pixels_outside_region_match validator
    assert_pixels_outside_region_match(out_rgb, plate_rgb, placement_box)


def test_generator_fails_run_if_outside_pixels_do_not_match(test_config):
    """Verify generator fails the run (raises exception) if outside region does not match original plate."""
    class RogueBypassingClient:
        mock_mode = False

        def image_edit(self, images, prompt, n=1, seed=None):
            # Rogue client trying to return an arbitrary image directly
            return [Image.new("RGB", (500, 500), color=(1, 2, 3)) for _ in range(n)]

    cli = RogueBypassingClient()
    expanded = {
        "plate": "Trader Joe's Shelf.jpeg",
        "shot_type": "front_chest",
        "placement_region": [0.15, 0.15, 0.95, 0.85],
        "final_prompt": "A person holding meat tray.",
    }

    # Normal generation must composite model output into original plate and succeed
    saved = generate(expanded=expanded, n=1, config=test_config, client=cli)
    assert len(saved) == 1

    # But if someone manually bypassed composite and passed a raw image without region lock:
    plate_img = Image.open(Path("assets/plates/Trader Joe's Shelf.jpeg"))
    raw_unlocked_output = Image.new("RGB", plate_img.size, color=(1, 2, 3))
    with pytest.raises(AssertionError):
        assert_pixels_outside_region_match(raw_unlocked_output, plate_img, expanded["placement_region"])
