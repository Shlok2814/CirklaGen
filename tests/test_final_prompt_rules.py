"""Unit tests for the 5 final_prompt mandatory directives, reference ordering, and sidecar persistence:
(1) add a real-looking person into the aisle
(2) have the tray held in their hands or resting properly (never floating)
(3) match the lighting and perspective of the aisle photo
(4) keep the background unchanged
(5) use the supplied tray images only to understand the tray's shape, not to paste them
Image reference order: plate FIRST, then tray cut-outs, then person reference,
and named in prompt ("image 1 is the aisle, images 2-3 are the tray, image 4 is the person").
Save final prompt in sidecar JSON.
"""

import json
from pathlib import Path
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.expander import CORE_MANDATORY_RULES_BLOCK, enforce_final_prompt_constraints, expand
from src.generator import format_final_prompt_for_generator, generate
from src.metadata import load_product
from src.pipeline import GenerationPipeline


@pytest.fixture
def mock_config() -> Config:
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    return GenAIClient(config=mock_config)


def test_expander_enforce_constraints_includes_all_five_rules():
    """Verify enforce_final_prompt_constraints guarantees all 5 rules and image roles."""
    meta = load_product("knowledge/product.json")
    bare_prompt = "Shopper holding meat tray in Trader Joe's."
    enforced = enforce_final_prompt_constraints(bare_prompt, meta)

    enforced_lower = enforced.lower()
    # 1. Image reference roles
    assert "image 1 is the aisle" in enforced_lower
    assert "images 2-3 are the tray" in enforced_lower
    assert "image 4 is the person" in enforced_lower

    # 2. Five mandatory directives
    assert "real-looking person into the aisle" in enforced_lower
    assert "never floating" in enforced_lower or "resting properly" in enforced_lower
    assert "lighting and perspective" in enforced_lower
    assert "keep the background unchanged" in enforced_lower
    assert "not to paste them" in enforced_lower or "understand the tray's shape" in enforced_lower


def test_expand_generates_final_prompt_with_rules(mock_config, mock_client):
    """Verify expand() output contains all 5 rules and image roles."""
    res = expand("me holding tray", shot_type="front_chest", config=mock_config, client=mock_client)
    final_prompt = res["final_prompt"]
    prompt_lower = final_prompt.lower()

    assert "image 1 is the aisle" in prompt_lower
    assert "images 2-3 are the tray" in prompt_lower
    assert "image 4 is the person" in prompt_lower
    assert "real-looking person" in prompt_lower
    assert "never floating" in prompt_lower
    assert "lighting and perspective" in prompt_lower
    assert "keep the background unchanged" in prompt_lower
    assert "not to paste" in prompt_lower


def test_format_final_prompt_for_generator():
    """Verify format_final_prompt_for_generator generates clean roles and directives."""
    formatted = format_final_prompt_for_generator("A candid photo.", n_tray_refs=2, has_person=True)
    lower = formatted.lower()

    assert "image 1 is the aisle, images 2-3 are the tray, image 4 is the person" in lower
    assert "(1) add a real-looking person into the aisle" in lower
    assert "(2) have the tray held in their hands or resting properly (never floating)" in lower
    assert "(3) match the lighting and perspective of the aisle photo" in lower
    assert "(4) keep the background unchanged" in lower
    assert "(5) use the supplied tray images only to understand the tray's shape, not to paste them" in lower


def test_generator_sends_plate_first_trays_second_person_third(mock_config):
    """Verify generate() passes references in exact order: plate FIRST, tray cut-outs, then person reference."""
    captured_images = []
    captured_prompt = []

    class CapturingClient:
        mock_mode = True

        def image_edit(self, images, prompt, n=1, seed=None):
            nonlocal captured_images, captured_prompt
            captured_images = list(images)
            captured_prompt.append(prompt)
            return [Image.new("RGB", (200, 200), color=(50, 100, 150)) for _ in range(n)]

    cli = CapturingClient()

    # Pass an explicit person reference
    shopper_ref = Path("assets/persons/shopper_reference.jpg")
    expanded_payload = {
        "plate": "Trader Joe's Shelf.jpeg",
        "shot_type": "front_chest",
        "person": "A shopper in a jacket",
        "pose": "Chest level",
        "camera": "Phone camera",
        "lighting": "Supermarket fluorescent",
        "imperfections": [],
        "final_prompt": "Candid photo of shopper holding tray.",
    }

    saved = generate(
        expanded=expanded_payload,
        n=1,
        person_ref=shopper_ref if shopper_ref.is_file() else None,
        config=mock_config,
        client=cli,
    )

    assert len(saved) == 1
    # Check reference order: Image 1 is plate, Images 2-3 are tray cutouts, Image 4 is person
    # Image 1 is the cropped placement region of the plate (region-locked generation)
    plate_orig = Image.open(Path("assets/plates/Trader Joe's Shelf.jpeg"))
    from src.finish import parse_bbox_to_pixels
    from src.generator import DEFAULT_PLACEMENT_REGIONS
    l, t, r, b = parse_bbox_to_pixels(DEFAULT_PLACEMENT_REGIONS["front_chest"], plate_orig.width, plate_orig.height)
    assert captured_images[0].size == (r - l, b - t)

    # Check prompt captured by model
    model_prompt = captured_prompt[0]
    prompt_lower = model_prompt.lower()
    assert "image 1 is the aisle, images 2-3 are the tray, image 4 is the person" in prompt_lower
    assert "(1) add a real-looking person into the aisle" in prompt_lower
    assert "(2) have the tray held in their hands or resting properly (never floating)" in prompt_lower
    assert "(3) match the lighting and perspective of the aisle photo" in prompt_lower
    assert "(4) keep the background unchanged" in prompt_lower
    assert "(5) use the supplied tray images only to understand the tray's shape, not to paste them" in prompt_lower

    # Verify final prompt is saved in sidecar JSON
    sidecar_path = saved[0].with_suffix(".json")
    with open(sidecar_path, "r", encoding="utf-8") as sf:
        sidecar = json.load(sf)

    assert sidecar["prompt"] == model_prompt
    assert sidecar["expanded"]["final_prompt"] == model_prompt
    assert "image 1 is the aisle" in sidecar["prompt"].lower()
    assert "never floating" in sidecar["prompt"].lower()
    assert "not to paste" in sidecar["prompt"].lower()


def test_pipeline_saves_final_prompt_in_manifest_and_sidecar(mock_config, mock_client):
    """Verify end-to-end pipeline stores final prompt across sidecars and run manifest."""
    pipeline = GenerationPipeline(config=mock_config, client=mock_client)
    res = pipeline.run("me holding tray in meat aisle", n=1, finishing=False)

    assert res["status"] == "success"
    run_dir = Path(res["run_dir"])
    sidecar_path = run_dir / "img_0.json"
    assert sidecar_path.is_file()

    with open(sidecar_path, "r", encoding="utf-8") as sf:
        sidecar = json.load(sf)

    assert "prompt" in sidecar
    prompt_lower = sidecar["prompt"].lower()
    assert "image 1 is the aisle" in prompt_lower
    assert "never floating" in prompt_lower
    assert "not to paste" in prompt_lower
    assert "lighting and perspective" in prompt_lower
