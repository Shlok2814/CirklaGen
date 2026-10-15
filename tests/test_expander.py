"""Unit tests for src/expander.py in both MOCK and optional LIVE mode."""

import os
from pathlib import Path
from dotenv import load_dotenv
import pytest

# Load environment variables from .env
load_dotenv()

from src.client import GenAIClient
from src.config import Config, load_config
from src.expander import (
    ExpanderError,
    PromptExpander,
    expand,
    get_available_plates,
    infer_shot_type_from_text,
    normalize_shot_type,
)


@pytest.fixture
def mock_config() -> Config:
    """Fixture providing Config with mock_mode=True."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    """Fixture providing GenAIClient with mock_mode=True."""
    return GenAIClient(config=mock_config)


REQUIRED_KEYS = [
    "plate",
    "shot_type",
    "person",
    "pose",
    "camera",
    "lighting",
    "imperfections",
    "final_prompt",
]

FIVE_PROMPTS_AND_SHOTS = [
    ("me holding tray", "front_chest"),
    ("POV hands with tray in the aisle", "pov_hands"),
    ("side profile looking at the tray", "side_profile"),
    ("tray in a cart from above", "cart_topdown"),
    ("close up of hands holding the tray", "closeup_hands"),
]


@pytest.mark.parametrize("short_prompt, expected_shot_type", FIVE_PROMPTS_AND_SHOTS)
def test_expand_five_shot_types_mock_mode(short_prompt, expected_shot_type, mock_config, mock_client):
    """Verify expand() correctly expands all 5 shot types in MOCK mode.

    Asserts:
    - All 8 required schema keys are present and non-empty.
    - The returned plate exists on disk in assets/plates.
    - The shot_type matches the expected shot type.
    - final_prompt contains 'keep the background unchanged' and reference to supplied tray images.
    """
    res = expand(short_prompt, shot_type=expected_shot_type, config=mock_config, client=mock_client)

    # 1. Assert structure
    assert isinstance(res, dict)
    for key in REQUIRED_KEYS:
        assert key in res, f"Missing required key '{key}' in expander output"
        assert res[key], f"Key '{key}' is empty in expander output"

    # 2. Assert plate exists
    plates_dir = Path("assets/plates")
    plate_path = plates_dir / res["plate"]
    assert plate_path.is_file(), f"Selected plate '{res['plate']}' does not exist in {plates_dir}"

    # 3. Assert shot type
    assert res["shot_type"] == expected_shot_type

    # 4. Assert final_prompt mandates
    final_lower = res["final_prompt"].lower()
    assert "keep the background unchanged" in final_lower
    assert "supplied tray" in final_lower or "exact cirkla" in final_lower
    assert len(res["final_prompt"]) >= 150


def test_prompt_expander_class_wrapper(mock_config, mock_client):
    """Verify PromptExpander class methods delegate properly."""
    expander_inst = PromptExpander(config=mock_config, client=mock_client)
    res = expander_inst.expand("me holding tray", shot_type="front_chest")
    assert res["shot_type"] == "front_chest"
    assert (Path("assets/plates") / res["plate"]).is_file()


def test_normalize_and_infer_shot_types():
    """Verify shot type normalization and inference from casual keywords."""
    assert normalize_shot_type("pov") == "pov_hands"
    assert normalize_shot_type("side") == "side_profile"
    assert normalize_shot_type("cart") == "cart_topdown"
    assert normalize_shot_type("closeup") == "closeup_hands"
    assert normalize_shot_type("front") == "front_chest"

    assert infer_shot_type_from_text("POV hands in meat aisle") == "pov_hands"
    assert infer_shot_type_from_text("tray in a red cart from above") == "cart_topdown"
    assert infer_shot_type_from_text("close up shot of tray ribbing") == "closeup_hands"
    assert infer_shot_type_from_text("side profile inspecting") == "side_profile"
    assert infer_shot_type_from_text("holding tray in front") == "front_chest"


def test_expander_retry_and_readable_error_on_invalid_data(monkeypatch, mock_config):
    """Verify expander retries once on invalid JSON, then raises readable ExpanderError."""
    attempt_count = 0

    class BrokenClient:
        mock_mode = False

        def text(self, *args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            # Return malformed JSON that fails schema validation
            return "This is completely invalid and not JSON"

    broken_cli = BrokenClient()

    with pytest.raises(ExpanderError, match="Prompt expansion failed after 2 attempts"):
        expand("me holding tray", config=mock_config, client=broken_cli)

    assert attempt_count == 2, f"Expected exactly 2 attempts before raising error, got {attempt_count}"


# --- Optional Live Test with Gemini API ---

has_api_key = bool(os.getenv("GEMINI_API_KEY"))


@pytest.mark.skipif(not has_api_key, reason="GEMINI_API_KEY not configured for live test")
def test_live_expander_three_prompts():
    """Live test calling Gemini API with real prompt expansion.

    Prints and verifies three distinct expanded prompts:
    1. me holding tray (front_chest)
    2. POV hands with tray in the aisle (pov_hands)
    3. tray in a cart from above (cart_topdown)
    """
    live_config = load_config(require_api_key=True)
    live_config.mock_mode = False
    live_client = GenAIClient(config=live_config)

    test_cases = [
        ("me holding tray", "front_chest"),
        ("POV hands with tray in the aisle", "pov_hands"),
        ("tray in a cart from above", "cart_topdown"),
    ]

    print("\n" + "=" * 80)
    print("LIVE EXPANDER TEST RESULTS")
    print("=" * 80)

    for idx, (short_p, shot) in enumerate(test_cases, start=1):
        res = expand(short_p, shot_type=shot, config=live_config, client=live_client)

        print(f"\n--- [Live Test {idx}/3] Prompt: '{short_p}' | Shot Type: '{res['shot_type']}' ---")
        print(f"Plate Selected: {res['plate']}")
        print(f"Person: {res['person']}")
        print(f"Pose: {res['pose']}")
        print(f"Camera: {res['camera']}")
        print(f"Lighting: {res['lighting']}")
        print(f"Imperfections: {res['imperfections']}")
        print(f"\nFinal Prompt ({len(res['final_prompt'])} chars):\n{res['final_prompt']}\n")

        # Assertions
        assert res["shot_type"] == shot
        assert (Path("assets/plates") / res["plate"]).is_file(), f"Plate '{res['plate']}' must exist on disk"
        final_lower = res["final_prompt"].lower()
        assert "keep the background unchanged" in final_lower, "Must say 'keep the background unchanged'"
        assert len(res["final_prompt"]) > 300, "Live expanded prompt must be long and detailed"

    print("=" * 80)
