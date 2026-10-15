"""Tests for QA automatic hand anatomy repair detection and fix_region workflow."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.finish import fix_region
from src.qa import (
    CHECKLIST_SCHEMA,
    FIX_HANDS_INSTRUCTION,
    LOCATE_HANDS_SCHEMA,
    QAResult,
    QualityAssurance,
    checklist,
    evaluate_image,
    locate_hands,
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


def test_fix_hands_instruction_exact_text():
    """Verify FIX_HANDS_INSTRUCTION exactly matches user specifications."""
    expected = (
        "fix the hands so there are exactly five fingers on each hand with a natural grip on the tray; change nothing else"
    )
    assert FIX_HANDS_INSTRUCTION == expected


def test_locate_hands_detected(mock_client, sample_test_image):
    """Verify locate_hands detects bounding box normalized to 0.0-1.0."""
    bbox = locate_hands(sample_test_image, client=mock_client)
    assert bbox is not None
    assert len(bbox) == 4
    ymin, xmin, ymax, xmax = bbox
    assert 0.0 <= ymin < ymax <= 1.0
    assert 0.0 <= xmin < xmax <= 1.0
    # In mock client, returns [450, 150, 950, 850] normalized to [0.45, 0.15, 0.95, 0.85]
    assert pytest.approx(ymin, 0.01) == 0.45
    assert pytest.approx(xmin, 0.01) == 0.15
    assert pytest.approx(ymax, 0.01) == 0.95
    assert pytest.approx(xmax, 0.01) == 0.85


def test_locate_hands_not_found(sample_test_image):
    """Verify locate_hands returns None when no hands are detected."""
    mock_cli = MagicMock()
    mock_cli.vision.return_value = {"found": False, "box_2d": [0, 0, 0, 0]}

    bbox = locate_hands(sample_test_image, client=mock_cli)
    assert bbox is None


def test_locate_hands_invalid_box(sample_test_image):
    """Verify locate_hands returns None when bounding box is degenerate."""
    mock_cli = MagicMock()
    mock_cli.vision.return_value = {"found": True, "box_2d": [500, 500, 500, 500]}

    bbox = locate_hands(sample_test_image, client=mock_cli)
    assert bbox is None


def test_qa_auto_fix_hands_triggers_and_keeps_better_version(mock_config, mock_client, sample_test_image):
    """Verify QA detects hands_anatomy_ok failure, calls fix_region, and keeps the repaired version."""
    # Tag image with bad hand anatomy
    bad_hands_img = sample_test_image.copy()
    setattr(bad_hands_img, "_hands_anatomy_bad", True)
    bad_hands_img.info["hands_bad"] = True

    with patch("src.finish.fix_region", wraps=fix_region) as spy_fix_region:
        res = evaluate_image(
            image=bad_hands_img,
            config=mock_config,
            client=mock_client,
            auto_fix_hands=True,
        )

        # 1. Verify fix_region was invoked
        assert spy_fix_region.called
        call_args = spy_fix_region.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["instruction"] == FIX_HANDS_INSTRUCTION
        # Bounding box should match detected hand region
        bbox = kwargs["bbox"]
        assert pytest.approx(bbox[0], 0.01) == 0.45
        assert pytest.approx(bbox[1], 0.01) == 0.15
        assert pytest.approx(bbox[2], 0.01) == 0.95
        assert pytest.approx(bbox[3], 0.01) == 0.85

        # 2. Verify result kept the better (repaired) version
        assert res.repaired is True
        assert res.repair_type == "fix_region_hands"
        assert res.image is not None
        assert res.checklist["hands_anatomy_ok"]["passed"] is True
        assert res.overall_score >= 0.70

        # 3. Verify dictionary serialization includes repair metadata
        res_dict = res.to_dict()
        assert res_dict["repaired"] is True
        assert res_dict["repair_type"] == "fix_region_hands"


def test_qa_auto_fix_hands_keeps_original_if_repair_worse(mock_config, mock_client, sample_test_image):
    """Verify QA keeps the original version if the repaired version scores worse."""
    bad_hands_img = sample_test_image.copy()
    setattr(bad_hands_img, "_hands_anatomy_bad", True)
    bad_hands_img.info["hands_bad"] = True

    corrupted = Image.new("RGB", (640, 480), color=(0, 0, 0))
    from src.qa import locate_tray as real_locate_tray

    def mock_locate_tray_fn(img, *args, **kwargs):
        if img is corrupted:
            return None
        return real_locate_tray(img, *args, **kwargs)

    with patch("src.finish.fix_region", return_value=corrupted), \
         patch("src.qa.locate_tray", side_effect=mock_locate_tray_fn):
        res = evaluate_image(
            image=bad_hands_img,
            config=mock_config,
            client=mock_client,
            auto_fix_hands=True,
        )

        # Repaired version scored 0.0 (no tray detected), so original (score ~0.78) must be kept
        assert res.repaired is False
        assert res.repair_type is None
        assert res.checklist["hands_anatomy_ok"]["passed"] is False


def test_qa_auto_fix_hands_disabled_flag(mock_config, mock_client, sample_test_image):
    """Verify auto_fix_hands=False suppresses automatic repair."""
    bad_hands_img = sample_test_image.copy()
    setattr(bad_hands_img, "_hands_anatomy_bad", True)
    bad_hands_img.info["hands_bad"] = True

    with patch("src.finish.fix_region") as mock_fix_region:
        res = evaluate_image(
            image=bad_hands_img,
            config=mock_config,
            client=mock_client,
            auto_fix_hands=False,
        )

        assert not mock_fix_region.called
        assert res.repaired is False
        assert res.checklist["hands_anatomy_ok"]["passed"] is False


def test_quality_assurance_class_wrapper_auto_fix(mock_config, mock_client, sample_test_image):
    """Verify QualityAssurance wrapper class propagates auto_fix_hands."""
    bad_hands_img = sample_test_image.copy()
    setattr(bad_hands_img, "_hands_anatomy_bad", True)
    bad_hands_img.info["hands_bad"] = True

    qa = QualityAssurance(config=mock_config, client=mock_client)
    res_dict = qa.evaluate(bad_hands_img, auto_fix_hands=True)

    assert isinstance(res_dict, dict)
    assert res_dict["repaired"] is True
    assert res_dict["repair_type"] == "fix_region_hands"
    assert res_dict["checklist"]["hands_anatomy_ok"]["passed"] is True


def test_pipeline_hands_repair_flow(mock_config, mock_client, sample_test_image):
    """Verify pipeline integrates QA hand repair, saving img_0_repaired.png."""
    from src.pipeline import GenerationPipeline

    bad_hands_img = sample_test_image.copy()
    setattr(bad_hands_img, "_hands_anatomy_bad", True)
    bad_hands_img.info["hands_bad"] = True

    pipe = GenerationPipeline(config=mock_config, client=mock_client)
    res = pipe.run(
        user_prompt="POV hands holding tray in aisle",
        manual_mode=True,
        manual_image=bad_hands_img,
    )

    assert res["status"] == "success"
    run_dir = Path(res["run_dir"])
    repaired_path = run_dir / "img_0_repaired.png"
    assert repaired_path.is_file()

    manifest_path = Path(res["manifest_path"])
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)

    var0 = manifest["variations"][0]
    assert var0["repaired"] is True
    assert var0["repair_type"] == "fix_region_hands"
    assert var0["qa_evaluation"]["checklist"]["hands_anatomy_ok"]["passed"] is True

