"""Tests for Finishing module src/finish.py covering Stage A Repair and Stage B Phone Finishing."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.finish import (
    Finisher,
    apply_finishing,
    apply_phone_finishing,
    composite_tray,
    create_before_after_comparison,
    fix_region,
    parse_bbox_to_pixels,
)


@pytest.fixture
def mock_config() -> Config:
    """Fixture providing Config in MOCK mode."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    """Fixture providing GenAIClient in MOCK mode."""
    return GenAIClient(config=mock_config)


@pytest.fixture
def sample_generated_image() -> Image.Image:
    """Creates a sample test RGB image simulating an AI-generated retail shot."""
    img = Image.new("RGB", (640, 480), color=(180, 160, 140))
    img.info["Camera"] = "Simulated Smartphone"
    return img


@pytest.fixture
def sample_tray_cutout() -> Image.Image:
    """Creates an RGBA tray cutout with a solid central core and transparent borders."""
    arr = np.zeros((200, 260, 4), dtype=np.uint8)
    # Core tray: beige color (210, 190, 160), alpha 255
    arr[20:180, 20:240, 0] = 210
    arr[20:180, 20:240, 1] = 190
    arr[20:180, 20:240, 2] = 160
    arr[20:180, 20:240, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def test_parse_bbox_to_pixels():
    """Verify parse_bbox_to_pixels handles normalized 0-1, normalized 0-1000, and pixel coordinates."""
    # 0.0 - 1.0 normalized
    l, t, r, b = parse_bbox_to_pixels([0.1, 0.2, 0.9, 0.8], img_width=1000, img_height=1000)
    assert (l, t, r, b) == (200, 100, 800, 900)

    # 0 - 1000 normalized
    l, t, r, b = parse_bbox_to_pixels([100, 200, 900, 800], img_width=2000, img_height=2000)
    assert (l, t, r, b) == (400, 200, 1600, 1800)

    # Absolute pixel coordinates
    l, t, r, b = parse_bbox_to_pixels([50, 50, 300, 300], img_width=500, img_height=500)
    assert (l, t, r, b) == (50, 50, 300, 300)


def test_stage_a_composite_tray(sample_generated_image, sample_tray_cutout):
    """Verify composite_tray perspective-warps cutout, feathers edges, and adds soft contact shadow."""
    bbox = [100, 120, 380, 520]
    composited = composite_tray(
        generated_img=sample_generated_image,
        tray_cutout=sample_tray_cutout,
        bbox=bbox,
        feather_radius=5,
        shadow_strength=0.5,
    )

    assert isinstance(composited, Image.Image)
    assert composited.size == sample_generated_image.size
    assert composited.mode == "RGB"
    assert "AI-generated mockup" in composited.info.get("Comment", "")

    # Pixels in composited area should differ from base image
    base_arr = np.array(sample_generated_image)
    comp_arr = np.array(composited)
    assert not np.array_equal(base_arr, comp_arr)


def test_stage_a_fix_region(mock_client, sample_generated_image):
    """Verify fix_region re-edits cropped region and seamlessly blends back."""
    bbox = [80, 80, 240, 240]
    fixed = fix_region(
        image=sample_generated_image,
        bbox=bbox,
        instruction="clean up finger anatomy and skin tone",
        client=mock_client,
        padding_ratio=0.15,
        feather_radius=10,
    )

    assert isinstance(fixed, Image.Image)
    assert fixed.size == sample_generated_image.size
    assert "AI-generated mockup" in fixed.info.get("Comment", "")


def test_stage_b_phone_finishing_seed_determinism(sample_generated_image):
    """Verify smartphone finishing is completely deterministic when seeded."""
    out_1 = apply_phone_finishing(sample_generated_image, seed=123)
    out_2 = apply_phone_finishing(sample_generated_image, seed=123)
    out_diff = apply_phone_finishing(sample_generated_image, seed=999)

    arr_1 = np.array(out_1)
    arr_2 = np.array(out_2)
    arr_diff = np.array(out_diff)

    # Identical seed yields exact same pixel values
    assert np.array_equal(arr_1, arr_2)
    # Different seed yields different optical realization
    assert not np.array_equal(arr_1, arr_diff)


def test_stage_b_provenance_and_metadata_preservation(sample_generated_image):
    """Verify existing metadata is preserved and AI-generated mockup field is added."""
    finished = apply_phone_finishing(sample_generated_image, seed=42)

    assert "Camera" in finished.info
    assert finished.info["Camera"] == "Simulated Smartphone"
    assert "AI-generated mockup" in finished.info["Comment"]
    assert "AI-generated mockup" in finished.info["Description"]


def test_finisher_class_wrapper(sample_generated_image, sample_tray_cutout):
    """Verify Finisher class wrapper methods and pipeline integration."""
    finisher = Finisher()

    # Stage A methods via class
    comp = finisher.composite_tray(sample_generated_image, sample_tray_cutout, bbox=[50, 50, 200, 200])
    assert isinstance(comp, Image.Image)

    # Stage B apply via class
    applied = finisher.apply(sample_generated_image, overrides={"grain": 0.10}, seed=42)
    assert isinstance(applied, Image.Image)
    assert applied.size == sample_generated_image.size


def test_create_before_after_comparison(tmp_path, sample_generated_image):
    """Verify side-by-side comparison image creation with banners and badges."""
    finished = apply_phone_finishing(sample_generated_image, seed=42)
    out_path = tmp_path / "comparison.png"

    saved_path = create_before_after_comparison(
        before_img=sample_generated_image,
        after_img=finished,
        output_path=out_path,
        title="Test Side by Side",
    )

    assert saved_path.is_file()
    comp_img = Image.open(saved_path)
    # Width should be approximately double of scaled images + divider
    assert comp_img.width > sample_generated_image.width
    assert comp_img.height > 720  # target_h + banner
