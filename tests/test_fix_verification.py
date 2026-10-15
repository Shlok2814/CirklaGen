"""Verification tests for the 5 user requirements:
1) generate() always calls image model and never pastes reference on plate; only finish composites when enabled
2) API errors / blocks raise visible errors and log reasons; never silently return plate/composite
3) References are inputs only, never outputs (no original non-cutout tray photo passed)
4) Perceptual difference check marks run NOT GENERATED if output matches plate plus paste
5) Logs print mode, model name, number of reference images, prompt[:300]
"""

from pathlib import Path
from PIL import Image
import pytest

from src.client import CirklaGenAPIError, GenAIClient, SafetyBlockedError
from src.config import Config, load_config
from src.finish import composite_tray
from src.generator import check_is_near_identical_to_plate, generate, select_tray_references
from src.pipeline import GenerationPipeline


@pytest.fixture
def test_config() -> Config:
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def expanded_dict() -> dict:
    return {
        "plate": "Trader Joe's Shelf.jpeg",
        "shot_type": "front_chest",
        "person": "Shopper holding meat tray",
        "pose": "Holding tray at chest height",
        "camera": "Handheld smartphone photo",
        "lighting": "Supermarket lighting",
        "imperfections": ["shelf clutter"],
        "final_prompt": (
            "Reference image roles: image 1 is the aisle, images 2-3 are the tray, image 4 is the person. "
            "Mandatory generation rules: (1) Add a real-looking person into the aisle. "
            "(2) Have the tray held in their hands or resting properly (never floating). "
            "(3) Match the lighting and perspective of the aisle photo. "
            "(4) Keep the background unchanged. "
            "(5) Use the supplied tray images only to understand the tray's shape, not to paste them. "
            "A candid phone photo of a shopper holding Cirkla molded-fiber tray at Trader Joe's."
        ),
    }


def test_requirement_1_generate_always_calls_model_and_never_pastes(test_config, expanded_dict):
    """Req 1: generate() must always call client.image_edit and never paste reference onto plate."""
    called = False

    class TrackingClient:
        mock_mode = True

        def image_edit(self, images, prompt, n=1, seed=None):
            nonlocal called
            called = True
            # Return an authentic-looking generated candidate (not plate)
            return [Image.new("RGB", (300, 300), color=(120, 80, 40)) for _ in range(n)]

    cli = TrackingClient()
    saved = generate(expanded=expanded_dict, n=1, config=test_config, client=cli)
    assert called, "generate() must call client.image_edit"
    assert len(saved) == 1
    # Verify image returned is composited into original plate with region lock
    plate_path = Path(test_config.paths.plates_dir) / expanded_dict["plate"]
    with Image.open(plate_path) as orig_plate:
        orig_size = orig_plate.size
    with Image.open(saved[0]) as im:
        assert im.size == orig_size
        # Verify pixels outside placement region match original plate strictly
        from src.generator import assert_pixels_outside_region_match
        assert_pixels_outside_region_match(im, Image.open(plate_path), [0.15, 0.15, 0.95, 0.85])


def test_requirement_2_api_failure_raises_error_never_silently_returns(test_config, expanded_dict):
    """Req 2: If API call fails or is blocked, raise visible error and log reason; never return plate or composite."""
    class FailingClient:
        mock_mode = False

        def image_edit(self, *args, **kwargs):
            raise CirklaGenAPIError("429 RESOURCE_EXHAUSTED: Daily free quota exceeded")

    fail_cli = FailingClient()
    with pytest.raises(CirklaGenAPIError) as exc_info:
        generate(expanded=expanded_dict, n=1, config=test_config, client=fail_cli)

    assert "429 RESOURCE_EXHAUSTED" in str(exc_info.value)

    # Test safety block propagation
    class SafetyClient:
        mock_mode = False

        def image_edit(self, *args, **kwargs):
            raise SafetyBlockedError("SAFETY_VIOLATION: blocked content")

    safe_cli = SafetyClient()
    with pytest.raises(SafetyBlockedError) as exc_info:
        generate(expanded=expanded_dict, n=1, config=test_config, client=safe_cli)

    assert "SAFETY_VIOLATION" in str(exc_info.value)


def test_requirement_3_no_non_cutout_tray_photo_as_output(test_config):
    """Req 3: References are inputs to model, never outputs. Cutout selector only returns transparent cutouts."""
    pipe = GenerationPipeline(config=test_config)
    cutout = pipe._select_repair_cutout("front_chest")
    assert cutout is not None
    assert "removebg" in cutout.name, f"Expected transparent cutout, got {cutout.name}"
    assert not cutout.name.endswith(".jpeg"), "JPEG cannot be a transparent cutout"
    assert "cirkla_tray_meat_label" not in cutout.name, "Meat label photo must never be repair cutout"

    # Verify alpha channel in the selected cutout
    with Image.open(cutout) as im:
        assert im.mode == "RGBA"


def test_requirement_4_perceptual_difference_check_marks_not_generated():
    """Req 4: If output is near-identical to plate plus pasted region, mark run NOT GENERATED."""
    plate_path = Path("assets/plates/Trader Joe's Shelf.jpeg")
    plate_img = Image.open(plate_path)

    # 1. Exact plate
    is_not_gen, ratio, reason = check_is_near_identical_to_plate(plate_img, plate_img)
    assert is_not_gen is True
    assert ratio == 1.0

    # 2. Plate with pasted cutout region
    cutout_path = Path("assets/trays/Screenshot_2026-10-13_203900-removebg-preview.png")
    pasted_img = composite_tray(plate_img, cutout_path, [0.35, 0.25, 0.75, 0.75])
    is_not_gen_pasted, ratio_pasted, reason_pasted = check_is_near_identical_to_plate(pasted_img, plate_img)
    assert is_not_gen_pasted is True
    assert ratio_pasted > 0.50, f"Expected >0.50 identical ratio, got {ratio_pasted}"

    # 3. Authentic generated scene
    auth_img = Image.new("RGB", plate_img.size, color=(100, 150, 200))
    is_not_gen_auth, ratio_auth, reason_auth = check_is_near_identical_to_plate(auth_img, plate_img)
    assert is_not_gen_auth is False
    assert ratio_auth < 0.20


def test_requirement_5_logging_prints_mode_model_refs_prompt(test_config, expanded_dict, caplog):
    """Req 5: Print in logs: mode, model name, number of reference images sent, and prompt[:300]."""
    import logging

    with caplog.at_level(logging.INFO):
        generate(expanded=expanded_dict, n=1, config=test_config)

    logs = caplog.text
    assert "mode=mock" in logs
    assert "model=" in logs
    assert "n_references=" in logs
    assert "prompt[:300]=" in logs
