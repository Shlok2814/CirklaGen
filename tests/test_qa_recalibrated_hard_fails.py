"""Tests for the 5 Recalibrated Hard-Fail QA checks in src/qa.py.

Checks tested:
1. background_matches_selected_plate (compare outside the edit region)
2. package_matches_reference (sleeve text and layout)
3. subject_is_centered_portrait (fail if subject center is within 10% of frame center and pose is symmetric)
4. skin_too_smooth (fail if skin lacks pores/texture/grain)
5. uniform_sharpness (fail if sharpness is artificially uniform across all blocks)
"""

from unittest.mock import MagicMock
from PIL import Image
import numpy as np
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.qa import (
    CHECKLIST_SCHEMA,
    HARD_FAIL_CHECKS,
    QAResult,
    check_background_matches_selected_plate,
    check_package_matches_reference,
    check_subject_is_centered_portrait,
    check_skin_too_smooth,
    check_uniform_sharpness,
    evaluate_image,
    format_qa_table,
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
def base_image() -> Image.Image:
    """Creates a realistic base test image with texture and variations."""
    np.random.seed(42)
    # 640x480 image with base color and grain
    arr = np.full((480, 640, 3), 180, dtype=np.uint8)
    noise = np.random.normal(0, 15, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_recalibrated_checks_exist_in_hard_fail_registry_and_schema():
    """Verify all 5 recalibrated checks are registered in HARD_FAIL_CHECKS and CHECKLIST_SCHEMA."""
    recalibrated = [
        "background_matches_selected_plate",
        "package_matches_reference",
        "subject_is_centered_portrait",
        "skin_too_smooth",
        "uniform_sharpness",
    ]

    for key in recalibrated:
        assert key in HARD_FAIL_CHECKS, f"{key} missing from HARD_FAIL_CHECKS"
        assert key in CHECKLIST_SCHEMA["properties"], f"{key} missing from CHECKLIST_SCHEMA properties"
        assert f"{key}_reason" in CHECKLIST_SCHEMA["properties"], f"{key}_reason missing from CHECKLIST_SCHEMA"
        assert key in CHECKLIST_SCHEMA["required"], f"{key} must be required in schema"

    # Verify expected values
    assert HARD_FAIL_CHECKS["background_matches_selected_plate"]["expected"] is True
    assert HARD_FAIL_CHECKS["package_matches_reference"]["expected"] is True
    assert HARD_FAIL_CHECKS["subject_is_centered_portrait"]["expected"] is False
    assert HARD_FAIL_CHECKS["skin_too_smooth"]["expected"] is False
    assert HARD_FAIL_CHECKS["uniform_sharpness"]["expected"] is False


def test_background_matches_selected_plate_failure_and_success():
    """Verify background_matches_selected_plate accurately detects outside differences."""
    plate = Image.new("RGB", (400, 400), color=(100, 120, 140))
    # Candidate identical outside edit region
    cand_match = Image.new("RGB", (400, 400), color=(100, 120, 140))
    # Edit region [50, 50, 350, 350]
    bbox = [50, 50, 350, 350]

    res_match = check_background_matches_selected_plate(
        image=cand_match,
        plate=plate,
        bbox=bbox,
        tolerance=5.0,
    )
    assert res_match["passed"] is True
    assert res_match["mean_diff"] <= 5.0

    # Candidate with altered background outside edit region
    cand_mismatch = np.array(cand_match)
    cand_mismatch[:45, :, :] = 255  # altered background band outside bbox
    res_mismatch = check_background_matches_selected_plate(
        image=Image.fromarray(cand_mismatch),
        plate=plate,
        bbox=bbox,
        tolerance=5.0,
    )
    assert res_mismatch["passed"] is False
    assert res_mismatch["mean_diff"] > 5.0


def test_package_matches_reference_tagged_failure(base_image):
    """Verify package_matches_reference fails if tagged as fake sleeve."""
    setattr(base_image, "_is_fake_sleeve", True)
    res = check_package_matches_reference(base_image)
    assert res["passed"] is False
    assert "FAIL" in res["reason"]


def test_subject_is_centered_portrait_evaluator():
    """Verify subject_is_centered_portrait fails on centered symmetric framing and passes on off-center."""
    img = Image.new("RGB", (640, 480), color=(180, 180, 180))

    # Centered placement box [0.15, 0.40, 0.85, 0.60] -> xc = 0.50, yc = 0.50
    centered_bbox = [0.15, 0.40, 0.85, 0.60]
    sidecar_centered = {
        "placement_region": centered_bbox,
        "pose_framing_choice": {"tilt_degrees": 0.0, "subject_position": "center"},
    }
    res_fail = check_subject_is_centered_portrait(img, sidecar=sidecar_centered)
    assert res_fail["passed"] is False
    assert res_fail["is_centered_portrait"] is True

    # Off-center placement box in left third -> xc = 0.25, yc = 0.50, tilt = 8 degrees
    off_center_bbox = [0.15, 0.05, 0.85, 0.45]
    sidecar_off_center = {
        "placement_region": off_center_bbox,
        "pose_framing_choice": {"tilt_degrees": 8.0, "subject_position": "left third"},
    }
    res_pass = check_subject_is_centered_portrait(img, sidecar=sidecar_off_center)
    assert res_pass["passed"] is True
    assert res_pass["is_centered_portrait"] is False


def test_skin_too_smooth_evaluator():
    """Verify skin_too_smooth fails when skin lacks pores and passes with realistic texture."""
    # Completely smooth image (zero texture)
    smooth_img = Image.new("RGB", (300, 300), color=(210, 160, 140))
    setattr(smooth_img, "_skin_too_smooth", True)
    res_smooth = check_skin_too_smooth(smooth_img)
    assert res_smooth["passed"] is False
    assert res_smooth["skin_too_smooth"] is True

    # Image with natural texture grain
    np.random.seed(123)
    textured_arr = np.full((300, 300, 3), 180, dtype=np.uint8)
    noise = np.random.normal(0, 12, (300, 300, 3)).astype(np.int16)
    textured_arr = np.clip(textured_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    textured_img = Image.fromarray(textured_arr, mode="RGB")
    res_textured = check_skin_too_smooth(textured_img, min_texture_sigma=2.0)
    assert res_textured["passed"] is True
    assert res_textured["skin_too_smooth"] is False


def test_uniform_sharpness_evaluator():
    """Verify uniform_sharpness fails on artificially uniform image and passes on realistic depth variance."""
    flat_img = Image.new("RGB", (300, 300), color=(150, 150, 150))
    setattr(flat_img, "_uniform_sharpness", True)
    res_flat = check_uniform_sharpness(flat_img)
    assert res_flat["passed"] is False
    assert res_flat["uniform_sharpness"] is True

    # Image with sharp foreground and blurred background (optical depth)
    np.random.seed(99)
    sharp_bg = np.zeros((300, 300), dtype=np.uint8)
    # One half high variance, other half low variance
    sharp_bg[:150, :] = np.random.randint(0, 255, (150, 300), dtype=np.uint8)
    sharp_bg[150:, :] = 128
    res_varying = check_uniform_sharpness(Image.fromarray(sharp_bg, mode="L"), min_sharpness_cv=0.15)
    assert res_varying["passed"] is True
    assert res_varying["uniform_sharpness"] is False


def test_any_of_five_recalibrated_checks_failing_marks_overall_as_fail(mock_config, mock_client, base_image):
    """Verify that ANY of the 5 checks failing causes evaluate_image to return passed=False (FAIL, not PASS)."""
    checks_to_test = [
        ("_is_bg_mismatch", "background_matches_selected_plate"),
        ("_is_fake_sleeve", "package_matches_reference"),
        ("_is_centered_portrait", "subject_is_centered_portrait"),
        ("_skin_too_smooth", "skin_too_smooth"),
        ("_uniform_sharpness", "uniform_sharpness"),
    ]

    for tag, check_name in checks_to_test:
        test_img = base_image.copy()
        setattr(test_img, tag, True)

        res = evaluate_image(
            image=test_img,
            config=mock_config,
            client=mock_client,
        )

        assert res.passed is False, f"Image with {tag}=True MUST be marked as FAIL, not PASS!"
        assert res.hard_checks is not None
        assert check_name in res.hard_checks
        assert res.hard_checks[check_name]["passed"] is False
        assert any(check_name in fail for fail in res.hard_fails)


def test_format_qa_table():
    """Verify format_qa_table outputs a valid markdown table with all 5 checks."""
    dummy_res = QAResult(
        similarity_max=0.88,
        similarity_mean=0.82,
        checklist={},
        passed=True,
        notes="All passed",
        overall_score=0.91,
        hard_checks={
            "background_matches_selected_plate": {"passed": True, "metrics": {"mean_diff": 2.1}},
            "package_matches_reference": {"passed": True, "metrics": {"score": 0.89}},
            "subject_is_centered_portrait": {"passed": True, "metrics": {"tilt": 7.0}},
            "skin_too_smooth": {"passed": True, "metrics": {"texture_sigma": 4.2}},
            "uniform_sharpness": {"passed": True, "metrics": {"sharpness_cv": 0.45}},
        },
    )

    table = format_qa_table([("Variation 1", dummy_res)])
    assert "| Image / Variation |" in table
    assert "| Background Match (Outside Edit) |" in table
    assert "| Package Match (Sleeve) |" in table
    assert "| Off-Center & Dynamic Pose |" in table
    assert "| Natural Skin Texture |" in table
    assert "| Optical Depth & Sharpness |" in table
    assert "| QA Decision |" in table
    assert "**Variation 1**" in table
    assert "**PASS**" in table
