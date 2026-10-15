"""Tests for Quality Assurance (QA) module src/qa.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.qa import (
    CHECKLIST_SCHEMA,
    LOCATE_TRAY_SCHEMA,
    QAResult,
    QualityAssurance,
    checklist,
    evaluate_image,
    locate_tray,
    qa,
    similarity,
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
def sample_test_image() -> Image.Image:
    """Creates a sample test RGB image."""
    return Image.new("RGB", (640, 480), color=(220, 200, 180))


@pytest.fixture
def sample_tray_references() -> list[Image.Image]:
    """Creates dummy tray reference images."""
    return [
        Image.new("RGB", (256, 256), color=(225, 205, 185)),
        Image.new("RGB", (256, 256), color=(215, 195, 175)),
    ]


def test_locate_tray_found(mock_client, sample_test_image):
    """Verify locate_tray successfully returns a cropped image when tray is found."""
    crop = locate_tray(sample_test_image, client=mock_client)
    assert crop is not None
    assert isinstance(crop, Image.Image)
    assert crop.size[0] > 0
    assert crop.size[1] > 0


def test_locate_tray_not_found(sample_test_image):
    """Verify locate_tray returns None when tray is not detected."""
    mock_cli = MagicMock()
    mock_cli.vision.return_value = {"found": False, "box_2d": [0, 0, 0, 0]}

    crop = locate_tray(sample_test_image, client=mock_cli)
    assert crop is None


def test_locate_tray_invalid_box(sample_test_image):
    """Verify locate_tray returns None when box coordinates are degenerate."""
    mock_cli = MagicMock()
    mock_cli.vision.return_value = {"found": True, "box_2d": [500, 500, 500, 500]}

    crop = locate_tray(sample_test_image, client=mock_cli)
    assert crop is None


def test_similarity_empty_inputs():
    """Verify similarity returns (0.0, 0.0) when crop or refs are missing."""
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    assert similarity(None, [img]) == (0.0, 0.0)
    assert similarity(img, []) == (0.0, 0.0)


def test_similarity_computation(sample_test_image, sample_tray_references):
    """Verify DINOv2 cosine similarity returns valid float range [-1.0, 1.0]."""
    sim_max, sim_mean = similarity(sample_test_image, sample_tray_references)
    assert isinstance(sim_max, float)
    assert isinstance(sim_mean, float)
    assert -1.0 <= sim_max <= 1.0
    assert -1.0 <= sim_mean <= 1.0
    assert sim_max >= sim_mean


def test_checklist_criteria(mock_client, sample_test_image):
    """Verify checklist returns all 10 required audit fields with boolean and reason."""
    checks = checklist(sample_test_image, client=mock_client)
    assert isinstance(checks, dict)

    expected_keys = [
        "exterior_beige",
        "scalloped_ribs_visible",
        "corner_pyramids_visible",
        "rim_intact",
        "no_pen_marks",
        "label_sleeve_present",
        "scale_plausible",
        "hands_anatomy_ok",
        "looks_like_ai",
        "looks_like_real_phone_photo",
        "background_consistent",
    ]

    for key in expected_keys:
        assert key in checks, f"Missing checklist key: {key}"
        assert "passed" in checks[key]
        assert "reason" in checks[key]
        assert isinstance(checks[key]["passed"], bool)
        assert isinstance(checks[key]["reason"], str)


def test_evaluate_image_returns_qa_result(mock_config, mock_client, sample_test_image):
    """Verify evaluate_image returns complete QAResult dataclass."""
    res = evaluate_image(
        image=sample_test_image,
        config=mock_config,
        client=mock_client,
        threshold=0.6,
    )

    assert isinstance(res, QAResult)
    assert hasattr(res, "similarity_max")
    assert hasattr(res, "similarity_mean")
    assert hasattr(res, "checklist")
    assert hasattr(res, "passed")
    assert hasattr(res, "notes")
    assert hasattr(res, "overall_score")
    assert hasattr(res, "tray_detected")

    assert isinstance(res.similarity_max, float)
    assert isinstance(res.similarity_mean, float)
    assert isinstance(res.passed, bool)
    assert isinstance(res.notes, str)
    assert len(res.notes) > 0
    assert res.overall_score >= 0.0


def test_evaluate_image_no_tray_fails(mock_config, sample_test_image):
    """Verify evaluate_image fails with score 0 when no tray is located."""
    mock_cli = MagicMock()
    mock_cli.vision.return_value = {"found": False, "box_2d": [0, 0, 0, 0]}

    res = evaluate_image(
        image=sample_test_image,
        config=mock_config,
        client=mock_cli,
    )

    assert res.passed is False
    assert res.tray_detected is False
    assert res.overall_score == 0.0
    assert "No meat tray" in res.notes


def test_qa_function_alias(mock_config, mock_client, sample_test_image):
    """Verify qa alias is identical to evaluate_image."""
    res = qa(sample_test_image, config=mock_config, client=mock_client)
    assert isinstance(res, QAResult)


def test_quality_assurance_class_wrapper(mock_config, mock_client, sample_test_image):
    """Verify QualityAssurance wrapper returns dictionary formatted for pipeline."""
    qa_inst = QualityAssurance(config=mock_config, client=mock_client)
    res_dict = qa_inst.evaluate(sample_test_image)

    assert isinstance(res_dict, dict)
    assert "qa_score" in res_dict
    assert "overall_score" in res_dict
    assert "passed" in res_dict
    assert "feedback" in res_dict
    assert "checklist" in res_dict
    assert "similarity_max" in res_dict
    assert "similarity_mean" in res_dict
