"""Unit and integration tests for plate lighting match step in src/finish.py."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.finish import (
    Finisher,
    apply_phone_finishing,
    create_lighting_match_comparison,
    locate_person,
    match_plate_lighting,
    parse_bbox_to_pixels,
    sample_plate_lighting,
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
def sample_plate_cool() -> Image.Image:
    """Creates a sample cool/fluorescent plate image (higher blue/cyan)."""
    # Cold supermarket lighting: R=130, G=160, B=200 -> R/B = 0.65
    return Image.new("RGB", (640, 480), color=(130, 160, 200))


@pytest.fixture
def sample_plate_warm() -> Image.Image:
    """Creates a sample warm/tungsten plate image (higher red/yellow)."""
    # Warm lighting: R=210, G=170, B=120 -> R/B = 1.75
    return Image.new("RGB", (640, 480), color=(210, 170, 120))


@pytest.fixture
def sample_scene_with_person() -> Image.Image:
    """Creates a synthetic image with a person region in center and distinct background."""
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    # Person box: left=100, top=50, right=500, bottom=420
    # Neutral person color (180, 180, 180) -> R/B = 1.0
    arr = np.array(img)
    arr[50:420, 100:500] = [180, 180, 180]
    return Image.fromarray(arr)


def test_sample_plate_lighting_cool_vs_warm(sample_plate_cool, sample_plate_warm):
    """Verify sample_plate_lighting correctly calculates color temp and brightness."""
    person_box = [0.1, 0.15, 0.85, 0.8]

    cool_stats = sample_plate_lighting(sample_plate_cool, person_bbox=person_box)
    warm_stats = sample_plate_lighting(sample_plate_warm, person_bbox=person_box)

    assert "mean_rgb" in cool_stats
    assert "brightness" in cool_stats
    assert "color_temp_ratio" in cool_stats
    assert "lab_mean" in cool_stats
    assert "sample_box" in cool_stats

    # Cool plate should have a lower R/B ratio than warm plate
    assert cool_stats["color_temp_ratio"] < warm_stats["color_temp_ratio"]
    assert pytest.approx(cool_stats["color_temp_ratio"], 0.05) == 130.0 / 200.0
    assert pytest.approx(warm_stats["color_temp_ratio"], 0.05) == 210.0 / 120.0


def test_locate_person(mock_client, sample_scene_with_person):
    """Verify locate_person returns valid normalized bounding box."""
    box = locate_person(sample_scene_with_person, client=mock_client)
    assert box is not None
    assert len(box) == 4
    ymin, xmin, ymax, xmax = box
    assert 0.0 <= ymin < ymax <= 1.0
    assert 0.0 <= xmin < xmax <= 1.0


def test_match_plate_lighting_leaves_background_untouched(sample_scene_with_person, sample_plate_cool):
    """Verify background outside person bounding box is 100% untouched."""
    person_bbox = [0.10, 0.15, 0.85, 0.80]
    w, h = sample_scene_with_person.size
    left, top, right, bottom = parse_bbox_to_pixels(person_bbox, w, h)

    relit = match_plate_lighting(
        image=sample_scene_with_person,
        plate=sample_plate_cool,
        person_bbox=person_bbox,
        relight_strength=0.50,
        shadow_strength=0.30,
        feather_radius=10,
    )

    orig_np = np.array(sample_scene_with_person)
    relit_np = np.array(relit)

    # Outside the person box, all pixels must be exactly identical
    # Check top margin (0 to top)
    assert np.array_equal(orig_np[:top, :], relit_np[:top, :])
    # Check bottom margin (bottom to h)
    assert np.array_equal(orig_np[bottom:, :], relit_np[bottom:, :])
    # Check left margin (0 to left)
    assert np.array_equal(orig_np[:, :left], relit_np[:, :left])
    # Check right margin (right to w)
    assert np.array_equal(orig_np[:, right:], relit_np[:, right:])


def test_match_plate_lighting_color_shift_and_shadow(sample_scene_with_person, sample_plate_cool):
    """Verify person region shifts towards plate color temperature and applies soft shadow."""
    person_bbox = [0.10, 0.15, 0.85, 0.80]
    w, h = sample_scene_with_person.size
    left, top, right, bottom = parse_bbox_to_pixels(person_bbox, w, h)

    relit = match_plate_lighting(
        image=sample_scene_with_person,
        plate=sample_plate_cool,
        person_bbox=person_bbox,
        relight_strength=0.50,
        shadow_strength=0.30,
    )

    orig_np = np.array(sample_scene_with_person).astype(np.float32)
    relit_np = np.array(relit).astype(np.float32)

    # Inner core of person (avoiding feathered boundary)
    cx_l, cx_r = left + 50, right - 50
    cy_t, cy_b = top + 50, bottom - 50

    orig_roi = orig_np[cy_t:cy_b, cx_l:cx_r]
    relit_roi = relit_np[cy_t:cy_b, cx_l:cx_r]

    # Original person was neutral R/B = 1.0; cool plate has R/B = 0.65 (more blue)
    orig_ct = np.mean(orig_roi[..., 0]) / np.mean(orig_roi[..., 2])
    relit_ct = np.mean(relit_roi[..., 0]) / np.mean(relit_roi[..., 2])

    # Relit person should have shifted towards cooler tone (lower R/B)
    assert relit_ct < orig_ct

    # Verify downward soft shadow: upper person ROI should be brighter than lower person ROI
    mid_y = (cy_t + cy_b) // 2
    upper_lum = np.mean(relit_np[cy_t:mid_y, cx_l:cx_r])
    lower_lum = np.mean(relit_np[mid_y:cy_b, cx_l:cx_r])
    assert upper_lum > lower_lum


def test_lighting_match_is_optional(sample_scene_with_person, sample_plate_cool):
    """Verify lighting match is optional and does not modify image when disabled."""
    # When match_plate_lighting=False
    out_disabled = apply_phone_finishing(
        image=sample_scene_with_person,
        plate=sample_plate_cool,
        match_plate_lighting=False,
        grain=0.0,
        vignette=0.0,
        softness=0.0,
        chromatic_aberration=0.0,
        highlight_bloom=0.0,
        tilt_deg=0.0,
        off_center_crop=0.0,
        realism_pass=False,
    )

    # When match_plate_lighting=True
    out_enabled = apply_phone_finishing(
        image=sample_scene_with_person,
        plate=sample_plate_cool,
        match_plate_lighting=True,
        grain=0.0,
        vignette=0.0,
        softness=0.0,
        chromatic_aberration=0.0,
        highlight_bloom=0.0,
        tilt_deg=0.0,
        off_center_crop=0.0,
        realism_pass=False,
    )

    arr_dis = np.array(out_disabled)
    arr_en = np.array(out_enabled)

    # Enabled output should differ from disabled output in the person region
    assert not np.array_equal(arr_dis, arr_en)
    assert out_enabled.info.get("lighting_matched") is True


def test_finisher_class_integration(mock_config, sample_scene_with_person, sample_plate_warm):
    """Verify Finisher class properly supports match_plate_lighting method and apply overrides."""
    finisher = Finisher(config=mock_config)

    # 1. Direct method test
    relit_direct = finisher.match_plate_lighting(
        image=sample_scene_with_person,
        plate=sample_plate_warm,
        relight_strength=0.40,
        shadow_strength=0.20,
    )
    assert isinstance(relit_direct, Image.Image)
    assert relit_direct.info.get("lighting_matched") is True

    # 2. Finisher.apply override test
    finished = finisher.apply(
        image=sample_scene_with_person,
        overrides={
            "match_plate_lighting": True,
            "plate": sample_plate_warm,
            "relight_strength": 0.40,
        },
    )
    assert isinstance(finished, Image.Image)
    assert finished.info.get("lighting_matched") is True


def test_create_lighting_match_comparison(tmp_path, sample_scene_with_person, sample_plate_cool):
    """Verify create_lighting_match_comparison produces valid 3-panel image."""
    relit = match_plate_lighting(
        image=sample_scene_with_person,
        plate=sample_plate_cool,
    )

    comp_path = tmp_path / "test_lighting_comp.png"
    out_file = create_lighting_match_comparison(
        before_img=sample_scene_with_person,
        after_img=relit,
        plate_img=sample_plate_cool,
        output_path=comp_path,
    )

    assert out_file.is_file()
    comp_im = Image.open(out_file)
    # 3 panels side by side should be significantly wider than a single panel
    assert comp_im.width > sample_scene_with_person.width * 2
    assert comp_im.height >= 675
