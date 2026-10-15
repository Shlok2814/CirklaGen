"""Tests for HARD-FAIL QA checks in src/qa.py."""

from unittest.mock import MagicMock
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.qa import (
    CHECKLIST_SCHEMA,
    HARD_FAIL_CHECKS,
    QAResult,
    checklist,
    evaluate_image,
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
def normal_image() -> Image.Image:
    """Fixture creating a normal test image."""
    return Image.new("RGB", (640, 480), color=(220, 200, 180))


@pytest.fixture
def floating_pasted_tray_image() -> Image.Image:
    """Fixture creating a test image tagged as a floating pasted tray."""
    img = Image.new("RGB", (640, 480), color=(200, 180, 160))
    setattr(img, "_is_floating_pasted", True)
    setattr(img, "_hard_fail", True)
    return img


def test_hard_fail_checks_in_schema():
    """Verify that all 6 required HARD-FAIL checks exist in CHECKLIST_SCHEMA."""
    props = CHECKLIST_SCHEMA["properties"]
    required = CHECKLIST_SCHEMA["required"]

    expected_hard_checks = [
        "person_present_and_holding_tray",
        "tray_physically_supported",
        "looks_pasted",
        "scene_matches_request",
        "lighting_consistent",
        "visible_rectangular_border_or_background_leak",
    ]

    for check in expected_hard_checks:
        assert check in props, f"Missing {check} in schema properties"
        assert f"{check}_reason" in props, f"Missing {check}_reason in schema properties"
        assert check in required, f"{check} must be required in schema"


def test_floating_pasted_tray_triggers_hard_fails(mock_config, mock_client, floating_pasted_tray_image):
    """Verify floating pasted tray fails all 6 hard checks and results in passed=False regardless of score."""
    res = evaluate_image(
        image=floating_pasted_tray_image,
        config=mock_config,
        client=mock_client,
        threshold=0.5,
    )

    assert isinstance(res, QAResult)
    assert res.passed is False, "Floating pasted tray MUST have passed=False regardless of score!"
    assert res.hard_fails is not None
    assert len(res.hard_fails) > 0

    # Verify hard_checks dictionary has all 6 checks
    assert res.hard_checks is not None
    for k in HARD_FAIL_CHECKS.keys():
        assert k in res.hard_checks
        check_res = res.hard_checks[k]
        assert "passed" in check_res
        assert "value" in check_res
        assert "reason" in check_res
        assert len(check_res["reason"]) > 0

    # Specifically check failing items on floating pasted tray
    assert res.hard_checks["person_present_and_holding_tray"]["passed"] is False
    assert res.hard_checks["person_present_and_holding_tray"]["value"] is False

    assert res.hard_checks["tray_physically_supported"]["passed"] is False
    assert res.hard_checks["tray_physically_supported"]["value"] is False

    assert res.hard_checks["looks_pasted"]["passed"] is False
    assert res.hard_checks["looks_pasted"]["value"] is True

    assert res.hard_checks["scene_matches_request"]["passed"] is False
    assert res.hard_checks["scene_matches_request"]["value"] is False

    assert res.hard_checks["lighting_consistent"]["passed"] is False
    assert res.hard_checks["lighting_consistent"]["value"] is False

    assert res.hard_checks["visible_rectangular_border_or_background_leak"]["passed"] is False
    assert res.hard_checks["visible_rectangular_border_or_background_leak"]["value"] is True

    # Notes must explicitly mention HARD-FAIL
    assert "HARD-FAIL" in res.notes


def test_single_hard_fail_overrides_perfect_score(mock_config, normal_image):
    """Verify that even a single hard-fail forces passed=False even if all other criteria pass 10/10 with 1.0 similarity."""
    mock_cli = MagicMock()
    # locate_tray returns valid crop
    mock_cli.vision.side_effect = [
        # locate_tray
        {"found": True, "box_2d": [100, 100, 800, 800]},
        # checklist: all pass EXCEPT looks_pasted=True
        {
            "person_present_and_holding_tray": True,
            "person_present_and_holding_tray_reason": "Customer is holding tray",
            "tray_physically_supported": True,
            "tray_physically_supported_reason": "Held securely",
            "looks_pasted": True,  # FAILS!
            "looks_pasted_reason": "Edges look artificially cut out and pasted onto the background",
            "scene_matches_request": True,
            "scene_matches_request_reason": "Front chest matches",
            "lighting_consistent": True,
            "lighting_consistent_reason": "Lighting matches",
            "visible_rectangular_border_or_background_leak": False,
            "visible_rectangular_border_or_background_leak_reason": "Clean edges",
            "exterior_beige": True,
            "exterior_beige_reason": "Natural beige",
            "scalloped_ribs_visible": True,
            "scalloped_ribs_reason": "Ribs distinct",
            "corner_pyramids_visible": True,
            "corner_pyramids_reason": "Pyramids visible",
            "rim_intact": True,
            "rim_intact_reason": "Rim intact",
            "no_pen_marks": True,
            "no_pen_marks_reason": "No pen",
            "label_sleeve_present": True,
            "label_sleeve_reason": "Sleeve present",
            "scale_plausible": True,
            "scale_plausible_reason": "Scale OK",
            "hands_anatomy_ok": True,
            "hands_anatomy_reason": "Hands OK",
            "looks_like_ai": False,
            "looks_like_ai_reason": "No AI artifacts",
            "looks_like_real_phone_photo": True,
            "looks_like_real_phone_photo_reason": "Authentic photo",
            "background_consistent": True,
            "background_consistent_reason": "Background consistent",
        },
    ]

    res = evaluate_image(
        image=normal_image,
        config=mock_config,
        client=mock_cli,
        threshold=0.5,
    )

    # Even though standard checklist is 11/11 (100%), looks_pasted=True must make passed=False
    assert res.passed is False, "Hard check failure must force passed=False regardless of overall score!"
    assert any("looks_pasted" in fail for fail in res.hard_fails)
    assert "HARD-FAIL" in res.notes


def test_similarity_is_supporting_only_and_cannot_override_hard_fail(mock_config, normal_image):
    """Verify high DINOv2 similarity cannot override a hard check failure."""
    mock_cli = MagicMock()
    mock_cli.vision.side_effect = [
        {"found": True, "box_2d": [100, 100, 800, 800]},
        {
            "person_present_and_holding_tray": False,  # No person holding tray!
            "person_present_and_holding_tray_reason": "No person holding tray",
            "tray_physically_supported": False,  # Floating!
            "tray_physically_supported_reason": "Floating in mid-air",
            "looks_pasted": True,  # Pasted!
            "looks_pasted_reason": "Tray is pasted cutout",
            "scene_matches_request": False,
            "scene_matches_request_reason": "No customer",
            "lighting_consistent": False,
            "lighting_consistent_reason": "Harsh cutout lighting",
            "visible_rectangular_border_or_background_leak": True,
            "visible_rectangular_border_or_background_leak_reason": "Border visible",
            # Standard tray checks pass
            "exterior_beige": True,
            "exterior_beige_reason": "Beige",
            "scalloped_ribs_visible": True,
            "scalloped_ribs_reason": "Ribs",
            "corner_pyramids_visible": True,
            "corner_pyramids_reason": "Pyramids",
            "rim_intact": True,
            "rim_intact_reason": "Rim",
            "no_pen_marks": True,
            "no_pen_marks_reason": "No pen",
            "label_sleeve_present": True,
            "label_sleeve_reason": "Sleeve",
            "scale_plausible": True,
            "scale_plausible_reason": "Scale",
            "hands_anatomy_ok": True,
            "hands_anatomy_reason": "N/A",
            "looks_like_ai": False,
            "looks_like_ai_reason": "No glitches",
            "looks_like_real_phone_photo": True,
            "looks_like_real_phone_photo_reason": "Real photo",
            "background_consistent": True,
            "background_consistent_reason": "Aisle",
        },
    ]

    # Even with similarity forced to 1.0
    res = evaluate_image(
        image=normal_image,
        config=mock_config,
        client=mock_cli,
        threshold=0.1,  # very low threshold
    )

    assert res.passed is False
    assert len(res.hard_fails) >= 5
    assert "HARD-FAIL" in res.notes
