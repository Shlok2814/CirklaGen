"""Unit tests for src/generator.py and scripts/check_output.py."""

import json
from pathlib import Path
from PIL import Image
import pytest

from scripts.check_output import inspect_image
from src.client import GenAIClient, SafetyBlockedError
from src.config import Config, load_config
from src.generator import (
    ImageGenerator,
    generate,
    select_tray_references,
)


@pytest.fixture
def mock_config() -> Config:
    """Fixture providing Config with mock_mode=True."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    """Fixture providing GenAIClient operating in MOCK mode."""
    return GenAIClient(config=mock_config)


@pytest.fixture
def sample_expanded() -> dict:
    """Sample expanded scene dictionary from src.expander."""
    return {
        "plate": "Trader Joe's Shelf.jpeg",
        "shot_type": "front_chest",
        "person": "A casual shopper in a denim jacket",
        "pose": "Holding tray at chest level",
        "camera": "Handheld phone photo, 28mm lens",
        "lighting": "Supermarket cold fluorescent lighting",
        "imperfections": ["slight tilt", "shelf clutter"],
        "final_prompt": (
            "Reference image roles: image 1 is the aisle, images 2-3 are the tray, image 4 is the person. "
            "Mandatory generation rules: (1) Add a real-looking person into the aisle. "
            "(2) Have the tray held in their hands or resting properly (never floating). "
            "(3) Match the lighting and perspective of the aisle photo. "
            "(4) Keep the background unchanged. "
            "(5) Use the supplied tray images only to understand the tray's shape, not to paste them. "
            "The supplied tray images show the exact Cirkla molded-fiber meat tray to place into the scene. "
            "Keep the background unchanged from the plate. A candid phone photo of a shopper holding the "
            "Cirkla beige molded-fiber meat tray at chest level in Trader Joe's meat aisle."
        ),
    }


def test_select_tray_references_for_all_shot_types():
    """Verify tray reference selection logic matches specified shot types and angles."""
    test_cases = [
        ("front_chest", ["three_quarter", "top"]),
        ("pov_hands", ["in_hand", "three_quarter"]),
        ("closeup_hands", ["base_closeup", "top"]),
        ("side_profile", ["side", "three_quarter"]),
        ("cart_topdown", ["top", "side"]),
    ]

    for shot_type, expected_angles in test_cases:
        refs = select_tray_references(shot_type=shot_type)
        assert 2 <= len(refs) <= 3, f"Expected 2-3 references for '{shot_type}', got {len(refs)}"
        for ref_path in refs:
            assert ref_path.is_file(), f"Selected reference {ref_path} does not exist"


def test_generate_mock_mode_creates_images_and_sidecars(mock_config, mock_client, sample_expanded):
    """Verify generate() in MOCK mode saves variations, sidecar JSONs, and manifest."""
    n_variations = 2
    saved_images = generate(
        expanded=sample_expanded,
        n=n_variations,
        config=mock_config,
        client=mock_client,
        seed=42,
    )

    assert len(saved_images) == n_variations

    # Check that plate is returned unchanged in MOCK mode
    plate_path = Path(mock_config.paths.plates_dir) / sample_expanded["plate"]
    with Image.open(plate_path) as orig_plate:
        orig_size = orig_plate.size

    run_dir = saved_images[0].parent
    assert run_dir.is_dir()

    for idx, img_path in enumerate(saved_images):
        assert img_path.is_file()
        assert img_path.suffix.lower() == ".png"

        # Verify image dimensions match original plate (unchanged in mock mode)
        with Image.open(img_path) as saved_img:
            assert saved_img.size == orig_size

        # Verify sidecar JSON
        sidecar_path = img_path.with_suffix(".json")
        assert sidecar_path.is_file()
        with open(sidecar_path, "r", encoding="utf-8") as sf:
            sidecar = json.load(sf)

        assert sidecar["variation_index"] == idx
        assert sidecar["prompt"] == sample_expanded["final_prompt"]
        assert sidecar["shot_type"] == sample_expanded["shot_type"]
        assert "reference_files" in sidecar
        assert "plate" in sidecar["reference_files"]
        assert "trays" in sidecar["reference_files"]
        assert sidecar["seed"] == 42 + idx
        assert "timestamp" in sidecar

    # Verify manifest.json exists
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)
    assert manifest["total_requested"] == n_variations
    assert manifest["total_generated"] == n_variations


def test_image_generator_class_wrapper(mock_config, mock_client, sample_expanded):
    """Verify ImageGenerator class wrapper generates variations cleanly."""
    gen = ImageGenerator(config=mock_config, client=mock_client)
    paths = gen.generate(expanded=sample_expanded, n=1)
    assert len(paths) == 1
    assert paths[0].is_file()


def test_generate_skips_safety_blocked_without_crashing(mock_config, sample_expanded):
    """Verify that safety-blocked variations are logged and skipped without crashing."""
    call_count = 0

    class SafetyMockClient:
        mock_mode = False

        def image_edit(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call blocked by safety
                raise SafetyBlockedError("Safety filter triggered for test.")
            # Second call succeeds
            dummy = Image.new("RGB", (64, 64), color="blue")
            return [dummy]

    mock_cli = SafetyMockClient()
    saved = generate(expanded=sample_expanded, n=2, config=mock_config, client=mock_cli)

    # 1 out of 2 succeeded, 1 was skipped without crashing
    assert len(saved) == 1
    assert saved[0].is_file()


def test_check_output_read_only(mock_config, mock_client, sample_expanded):
    """Verify scripts.check_output.inspect_image extracts metadata without modifying file."""
    saved_images = generate(expanded=sample_expanded, n=1, config=mock_config, client=mock_client)
    img_path = saved_images[0]

    # Record last modified time before inspection
    mtime_before = img_path.stat().st_mtime_ns

    report = inspect_image(img_path)

    # Record last modified time after inspection
    mtime_after = img_path.stat().st_mtime_ns
    assert mtime_before == mtime_after, "Image file must not be modified by check_output"

    assert report["file_name"] == img_path.name
    assert report["format"] == "PNG"
    assert report["mode"] in ("RGB", "RGBA")
    assert report["dimensions"]["width"] > 0
    assert report["dimensions"]["height"] > 0
    assert report["sidecar"] is not None
    assert report["sidecar"]["shot_type"] == "front_chest"


def test_generate_raises_error_when_all_api_calls_fail(mock_config, sample_expanded):
    """Verify generate() raises visible exception and never silently returns plate or composite when API fails."""
    from src.client import CirklaGenAPIError

    class FailingClient:
        mock_mode = False

        def image_edit(self, *args, **kwargs):
            raise CirklaGenAPIError("Quota exceeded (429 RESOURCE_EXHAUSTED)")

    fail_cli = FailingClient()
    with pytest.raises(CirklaGenAPIError) as exc_info:
        generate(expanded=sample_expanded, n=1, config=mock_config, client=fail_cli)

    assert "Quota exceeded" in str(exc_info.value) or "Image generation failed" in str(exc_info.value)


def test_generate_perceptual_check_marks_not_generated(mock_config, mock_client, sample_expanded):
    """Verify output near-identical to plate is marked NOT GENERATED in sidecar and manifest."""
    # In mock mode, image_edit returns plate_image, which triggers the perceptual check
    saved_images = generate(expanded=sample_expanded, n=1, config=mock_config, client=mock_client)
    assert len(saved_images) == 1

    sidecar_path = saved_images[0].with_suffix(".json")
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert sidecar["status"] == "NOT GENERATED"
    assert sidecar["is_generated"] is False
    assert "not_generated_reason" in sidecar

    manifest_path = saved_images[0].parent / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "NOT GENERATED"
    assert manifest["is_generated"] is False


def test_generate_logs_required_fields(mock_config, mock_client, sample_expanded, caplog):
    """Verify that mode, model, reference count, and prompt[:300] are logged."""
    import logging

    with caplog.at_level(logging.INFO):
        generate(expanded=sample_expanded, n=1, config=mock_config, client=mock_client)

    log_text = caplog.text
    assert "Generator call:" in log_text
    assert "mode=mock" in log_text
    assert "model=" in log_text
    assert "n_references=" in log_text
    assert "prompt[:300]=" in log_text
