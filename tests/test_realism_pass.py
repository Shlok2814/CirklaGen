"""Tests for the Realism Pass features across finish, expander, QA, client, and pipeline."""

import cv2
import numpy as np
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.expander import enforce_final_prompt_constraints, expand, REALISM_CLUTTER_INSTRUCTION
from src.finish import apply_phone_finishing, Finisher
from src.metadata import load_product
from src.pipeline import GenerationPipeline
from src.qa import checklist, evaluate_image, CHECKLIST_SCHEMA


@pytest.fixture
def test_config() -> Config:
    """Fixture providing Config in MOCK mode."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def test_client(test_config: Config) -> GenAIClient:
    """Fixture providing GenAIClient in MOCK mode."""
    return GenAIClient(config=test_config)


@pytest.fixture
def vibrant_test_image() -> Image.Image:
    """Creates a colorful high-saturation image to test realism pass optical modifications."""
    img_np = np.zeros((200, 200, 3), dtype=np.uint8)
    # Bright colorful stripes
    img_np[:, :100] = [240, 60, 60]    # Vibrant red
    img_np[:, 100:] = [60, 200, 240]   # Vibrant cyan
    return Image.fromarray(img_np)


def test_finish_realism_pass_saturation_reduction(vibrant_test_image):
    """Verify realism pass reduces mean color saturation."""
    before_np = np.array(vibrant_test_image)
    before_hsv = cv2.cvtColor(before_np, cv2.COLOR_RGB2HSV)
    mean_sat_before = float(np.mean(before_hsv[..., 1]))

    # Finished with realism_pass=True
    after_real = apply_phone_finishing(
        vibrant_test_image,
        saturation=0.85,
        uneven_white_balance=0.0,
        motion_blur=0.0,
        realism_pass=True,
        seed=42,
    )
    after_np = np.array(after_real)
    after_hsv = cv2.cvtColor(after_np, cv2.COLOR_RGB2HSV)
    mean_sat_after = float(np.mean(after_hsv[..., 1]))

    # Saturation after must be strictly lower
    assert mean_sat_after < mean_sat_before
    assert after_real.info.get("realism_pass") is True


def test_finish_realism_pass_uneven_white_balance(vibrant_test_image):
    """Verify realism pass introduces vertical spatial color temperature gradient."""
    # White neutral canvas
    neutral_img = Image.new("RGB", (200, 200), color=(180, 180, 180))

    finished = apply_phone_finishing(
        neutral_img,
        saturation=1.0,
        uneven_white_balance=0.20,
        motion_blur=0.0,
        grain=0.0,
        vignette=0.0,
        softness=0.0,
        chromatic_aberration=0.0,
        highlight_bloom=0.0,
        tilt_deg=0.0,
        off_center_crop=0.0,
        realism_pass=True,
        seed=42,
    )
    res_np = np.array(finished).astype(np.float32)

    # Top vs bottom red/blue balance should differ due to spatial ramp
    top_red = np.mean(res_np[:40, :, 0])
    bottom_red = np.mean(res_np[-40:, :, 0])
    top_blue = np.mean(res_np[:40, :, 2])
    bottom_blue = np.mean(res_np[-40:, :, 2])

    # Bottom has positive y_ramp (warmer red), top has negative y_ramp (cooler)
    assert bottom_red > top_red
    assert top_blue > bottom_blue


def test_finish_realism_pass_motion_blur(vibrant_test_image):
    """Verify realism pass softens sharp edge gradients via directional micro-motion blur."""
    sharp_img = Image.new("RGB", (200, 200), color=(0, 0, 0))
    # Draw high contrast sharp white bar
    draw_np = np.array(sharp_img)
    draw_np[80:120, :] = 255
    sharp_img = Image.fromarray(draw_np)

    no_mb = apply_phone_finishing(
        sharp_img,
        motion_blur=0.0,
        grain=0.0,
        softness=0.0,
        vignette=0.0,
        chromatic_aberration=0.0,
        highlight_bloom=0.0,
        tilt_deg=0.0,
        off_center_crop=0.0,
        realism_pass=False,
    )

    with_mb = apply_phone_finishing(
        sharp_img,
        motion_blur=0.25,
        grain=0.0,
        softness=0.0,
        vignette=0.0,
        chromatic_aberration=0.0,
        highlight_bloom=0.0,
        tilt_deg=0.0,
        off_center_crop=0.0,
        realism_pass=True,
    )

    # Check Laplacian variance (sharpness metric)
    lap_no_mb = cv2.Laplacian(cv2.cvtColor(np.array(no_mb), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    lap_with_mb = cv2.Laplacian(cv2.cvtColor(np.array(with_mb), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

    assert lap_with_mb < lap_no_mb


def test_expander_clutter_instruction_injection(test_config):
    """Verify expander injects background retail clutter instructions when realism_pass is True."""
    meta = load_product(test_config.paths.knowledge_file)

    base_prompt = "A photo of customer holding the meat tray."

    # When realism_pass is True
    prompt_with_clutter = enforce_final_prompt_constraints(
        base_prompt,
        meta,
        strict_must_keep=True,
        realism_pass=True,
    )
    assert "clutter" in prompt_with_clutter.lower()
    assert "shelf-talker" in prompt_with_clutter.lower() or "price tags" in prompt_with_clutter.lower()

    # When realism_pass is False
    prompt_no_clutter = enforce_final_prompt_constraints(
        base_prompt,
        meta,
        strict_must_keep=True,
        realism_pass=False,
    )
    assert REALISM_CLUTTER_INSTRUCTION not in prompt_no_clutter


def test_qa_looks_like_real_phone_photo_scoring(test_config, test_client, vibrant_test_image):
    """Verify QA evaluates looks_like_real_phone_photo and returns boolean + reason."""
    # Test on image finished with realism pass
    finished_real = apply_phone_finishing(vibrant_test_image, realism_pass=True)
    checks_real = checklist(finished_real, client=test_client)

    assert "looks_like_real_phone_photo" in checks_real
    item_real = checks_real["looks_like_real_phone_photo"]
    assert item_real["passed"] is True
    assert isinstance(item_real["reason"], str)
    assert len(item_real["reason"]) > 0

    # Test on image finished without realism pass
    raw_img = vibrant_test_image.copy()
    if hasattr(raw_img, "_has_realism_pass"):
        delattr(raw_img, "_has_realism_pass")
    test_config.realism_pass = False
    client_no_real = GenAIClient(config=test_config)

    checks_unreal = checklist(raw_img, client=client_no_real)
    assert "looks_like_real_phone_photo" in checks_unreal
    item_unreal = checks_unreal["looks_like_real_phone_photo"]
    assert item_unreal["passed"] is False
    assert "synthetic" in item_unreal["reason"].lower() or "oversaturated" in item_unreal["reason"].lower()


def test_pipeline_realism_pass_toggle(test_config, test_client):
    """Verify GenerationPipeline.run respects realism_pass flag."""
    pipeline = GenerationPipeline(config=test_config, client=test_client)

    # Run with realism_pass=True
    res_true = pipeline.run(
        short_prompt="front chest view holding tray",
        shot_type="front_chest",
        n=1,
        realism_pass=True,
    )
    assert res_true["status"] == "success"
    var_true = res_true["variations"][0]
    assert var_true.get("realism_pass") is True
    finished_im_true = Image.open(var_true["finished_image"])
    assert finished_im_true.info.get("realism_pass") in (True, "True")

    # Run with realism_pass=False
    res_false = pipeline.run(
        short_prompt="front chest view holding tray",
        shot_type="front_chest",
        n=1,
        realism_pass=False,
    )
    assert res_false["status"] == "success"
    var_false = res_false["variations"][0]
    assert var_false.get("realism_pass") is False
    finished_im_false = Image.open(var_false["finished_image"])
    assert finished_im_false.info.get("realism_pass") not in (True, "True")
