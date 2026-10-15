"""Tests for package replacement with real cut-out and sleeve QA check.

Requirements:
1. After generation, always replace the package with the real cut-out from assets/packaged/
   (perspective-warped to detected tray corners, re-lit to match the scene, with hands
   pasted back over the rim).
2. The model-generated package must never appear in the final output.
3. Region-lock compatibility: all pixels outside placement region match original plate.
4. QA check that compares the sleeve crop against the reference with a text and layout
   similarity score, and fails the image below threshold.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.finish import (
    extract_hand_occlusion_mask,
    locate_tray_corners,
    parse_bbox_to_pixels,
    replace_packaged_tray,
)
from src.generator import assert_pixels_outside_region_match
from src.qa import (
    compare_sleeve_against_reference,
    crop_sleeve,
    crop_sleeve_from_tray,
    evaluate_image,
    get_default_sleeve_reference,
)


@pytest.fixture
def mock_config() -> Config:
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    return GenAIClient(config=mock_config)


@pytest.fixture
def authentic_packaged_cutout() -> Path:
    p = Path("assets/packaged/cirkla_packaged_tray.png")
    assert p.is_file(), f"Missing packaged tray asset: {p}"
    return p


@pytest.fixture
def ground_truth_sleeve() -> Path:
    p = Path("assets/packaged/sleeve_reference.png")
    assert p.is_file(), f"Missing sleeve reference asset: {p}"
    return p


@pytest.fixture
def sample_plate_with_tray() -> Image.Image:
    """Creates a synthetic plate with a scene background and placeholder package region."""
    plate_p = Path("assets/plates/Trader Joe's Shelf.jpeg")
    plate = Image.open(plate_p).convert("RGB")
    # Draw a mock generated tray inside placement region
    w, h = plate.size
    draw = ImageDraw.Draw(plate)
    # Target region
    draw.rectangle([int(w * 0.25), int(h * 0.35), int(w * 0.80), int(h * 0.85)], fill=(180, 160, 140))
    return plate


def test_packaged_assets_exist():
    """Verify assets/packaged/ contains real cutout and sleeve reference."""
    packaged_dir = Path("assets/packaged")
    assert packaged_dir.is_dir()
    cutout = packaged_dir / "cirkla_packaged_tray.png"
    sleeve = packaged_dir / "sleeve_reference.png"
    assert cutout.is_file()
    assert sleeve.is_file()

    img_cutout = Image.open(cutout)
    assert img_cutout.mode == "RGBA"
    assert img_cutout.width > 500
    assert img_cutout.height > 500

    img_sleeve = Image.open(sleeve)
    assert img_sleeve.mode == "RGB"
    assert img_sleeve.width > 200
    assert img_sleeve.height > 400


def test_locate_tray_corners(sample_plate_with_tray, mock_client):
    """Verify locate_tray_corners returns 4 valid bounding points in clockwise order."""
    corners = locate_tray_corners(sample_plate_with_tray, client=mock_client)
    assert isinstance(corners, list)
    assert len(corners) == 4

    w, h = sample_plate_with_tray.size
    for pt in corners:
        assert len(pt) == 2
        x, y = pt
        assert 0 <= x <= w
        assert 0 <= y <= h

    # Check clockwise order: tl -> tr -> br -> bl
    tl, tr, br, bl = corners
    assert tl[0] < tr[0]
    assert bl[0] < br[0]
    assert tl[1] < bl[1]
    assert tr[1] < br[1]


def test_replace_packaged_tray_perspective_warp(sample_plate_with_tray, authentic_packaged_cutout, mock_client):
    """Verify replace_packaged_tray warps real package into scene and marks metadata."""
    res = replace_packaged_tray(
        image=sample_plate_with_tray,
        package_cutout=authentic_packaged_cutout,
        client=mock_client,
    )
    assert isinstance(res, Image.Image)
    assert res.size == sample_plate_with_tray.size
    assert res.info.get("package_replaced") is True


def test_replace_packaged_tray_preserves_region_lock(sample_plate_with_tray, authentic_packaged_cutout, mock_client):
    """Verify that replace_packaged_tray strictly respects placement_region and outside pixels match plate."""
    plate_orig = sample_plate_with_tray.copy()
    placement_reg = [0.20, 0.15, 0.90, 0.85]

    replaced = replace_packaged_tray(
        image=sample_plate_with_tray,
        package_cutout=authentic_packaged_cutout,
        placement_region=placement_reg,
        client=mock_client,
    )

    # Must pass assertion: all pixels outside placement region match original plate
    assert_pixels_outside_region_match(
        output_img=replaced,
        original_plate=plate_orig,
        bbox=placement_reg,
    )

    # Also directly verify on raw diff
    w, h = plate_orig.size
    left, top, right, bottom = parse_bbox_to_pixels(placement_reg, w, h)
    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[top:bottom, left:right] = False

    orig_np = np.array(plate_orig.convert("RGB"), dtype=np.int16)
    rep_np = np.array(replaced.convert("RGB"), dtype=np.int16)
    diff = np.abs(rep_np[outside_mask] - orig_np[outside_mask])
    assert np.max(diff) == 0, f"Max diff outside placement region was {np.max(diff)}"


def test_hands_pasted_back_over_rim(mock_client, authentic_packaged_cutout):
    """Verify hands/fingers gripping over rim are extracted and preserved over the replaced package."""
    w, h = 800, 600
    base = Image.new("RGB", (w, h), color=(100, 100, 100))

    # Define tray corners
    corners = [
        (200.0, 150.0),
        (600.0, 150.0),
        (600.0, 450.0),
        (200.0, 450.0),
    ]

    # Draw simulated shopper fingers gripping over the rim (skin tone: RGB ~ (210, 160, 130))
    # YCrCb of (210, 160, 130): Y=168, Cr=154, Cb=104 (within skin range)
    # HSV of (210, 160, 130): H=22, S=97, V=210 (within skin range)
    draw = ImageDraw.Draw(base)
    # Left hand thumb resting on left rim (x: 195 to 225, y: 280 to 320)
    draw.ellipse([195, 280, 225, 320], fill=(210, 160, 130))
    # Right hand fingers resting on right rim (x: 580 to 615, y: 270 to 330)
    draw.ellipse([580, 270, 615, 330], fill=(210, 160, 130))

    # Extract hand mask
    hand_rgb, hand_alpha = extract_hand_occlusion_mask(base, corners, client=mock_client)
    assert np.max(hand_alpha) > 0.5, "Skin mask over rim should detect simulated fingers"

    # Replace package
    result = replace_packaged_tray(
        image=base,
        package_cutout=authentic_packaged_cutout,
        corners=corners,
        client=mock_client,
    )

    res_np = np.array(result)
    # Verify that skin-colored finger pixels are preserved over the rim area
    finger_spot = res_np[300, 210]
    # Red channel should be higher than blue (skin-tone characteristic)
    assert finger_spot[0] > finger_spot[2] + 40, f"Finger spot {finger_spot} should maintain warm skin tone"


def test_crop_sleeve(authentic_packaged_cutout):
    """Verify crop_sleeve extracts the paper sleeve region."""
    tray_img = Image.open(authentic_packaged_cutout).convert("RGB")
    sleeve = crop_sleeve_from_tray(tray_img)
    assert sleeve is not None
    assert sleeve.width > 50
    assert sleeve.height > 100
    # Proportions: width should be ~34% of tray width
    expected_w = int(tray_img.width * (0.48 - 0.14))
    assert abs(sleeve.width - expected_w) <= 2


def test_compare_sleeve_against_reference_authentic_passes(ground_truth_sleeve, authentic_packaged_cutout, mock_config):
    """Verify compare_sleeve_against_reference scores high and passes on authentic sleeve."""
    tray_img = Image.open(authentic_packaged_cutout).convert("RGB")
    sleeve_crop = crop_sleeve_from_tray(tray_img)

    res = compare_sleeve_against_reference(
        sleeve_crop=sleeve_crop,
        reference_path=ground_truth_sleeve,
        threshold=0.70,
        config=mock_config,
    )

    assert res["passed"] is True
    assert res["score"] >= 0.70
    assert res["text_similarity"] >= 0.60
    assert res["layout_similarity"] >= 0.60
    assert "PASS" in res["reason"]


def test_compare_sleeve_against_reference_different_package_fails(ground_truth_sleeve, mock_config):
    """Verify compare_sleeve_against_reference fails for non-matching or hallucinated package."""
    # Create a completely different fake package (e.g. blue gradient with no Trader Joe's branding)
    fake_img = Image.new("RGB", (300, 600), color=(30, 80, 200))

    res = compare_sleeve_against_reference(
        sleeve_crop=fake_img,
        reference_path=ground_truth_sleeve,
        threshold=0.70,
        config=mock_config,
    )

    assert res["passed"] is False
    assert res["score"] < 0.70
    assert "FAIL" in res["reason"]


def test_qa_evaluate_image_hard_fails_on_sleeve_below_threshold(sample_plate_with_tray, mock_config, mock_client):
    """Verify evaluate_image hard-fails (passed=False) when sleeve similarity is below threshold."""
    # When evaluated on raw sample_plate_with_tray (which has no authentic sleeve crop):
    res = evaluate_image(
        image=sample_plate_with_tray,
        config=mock_config,
        client=mock_client,
        threshold=0.60,
    )

    # Sleeve check must fail because the tray has no Trader Joe's sleeve
    assert res.sleeve_similarity_score is not None
    assert res.checklist.get("sleeve_similarity_check") is not None
    assert res.checklist["sleeve_similarity_check"]["passed"] is False

    # Hard-failure must trigger passed=False
    assert res.passed is False
    assert any("sleeve_similarity_check" in hf for hf in res.hard_fails)
