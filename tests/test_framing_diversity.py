"""Unit and integration tests for pose and framing variety and perceptual-hash batch rejection.

Verifies:
1. knowledge/product.json contains all required shot types and framing/pose variety options.
2. Expander chooses a distinct combination per variation (different person reference, pose,
   subject position, 5-15 deg tilt, cropped edges, gaze).
3. Generator applies per-variation placement, preserves region lock, and records choices in sidecar JSONs.
4. Pipeline records choices in manifest.json and records batch diversity metrics.
5. Batch is rejected when two images have a perceptual-hash distance below the threshold.
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.diversity import (
    BatchDiversityError,
    BatchDiversityResult,
    check_batch_diversity,
    compute_phash,
    phash_distance,
    phash_to_hex,
)
from src.expander import (
    VALID_SHOT_TYPES,
    expand,
    expand_variations,
    generate_variation_choices,
    get_available_person_references,
    infer_shot_type_from_text,
    normalize_shot_type,
)
from src.generator import generate
from src.metadata import load_product
from src.pipeline import GenerationPipeline


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    """Fixture providing Config in MOCK mode."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    cfg.paths.runs_dir = str(tmp_path / "test_runs")
    return cfg


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    """Fixture providing GenAIClient in MOCK mode."""
    return GenAIClient(config=mock_config)


def test_product_json_expanded_shot_types():
    """Verify knowledge/product.json contains all requested shot types and framing varieties."""
    meta = load_product("knowledge/product.json")
    shot_types_dict = meta.shot_types.model_dump()

    required_shot_types = [
        "stepped_back_wide",
        "seated_in_cart_view",
        "holding_and_pointing",
        "off_center_candid",
        "tilted_handheld",
        "cropped_edges",
        "shelf_gaze",
    ]

    for st in required_shot_types:
        assert st in shot_types_dict, f"Expected shot_type '{st}' in product.json"
        detail = shot_types_dict[st]
        assert detail is not None
        assert "pose" in detail
        assert "camera_distance" in detail
        assert "tray_orientation" in detail
        assert "what_the_camera_sees" in detail

    # Verify framing and pose variety dictionary
    assert meta.framing_and_pose_variety is not None
    variety = meta.framing_and_pose_variety
    assert "subject_positions" in variety
    assert "handheld_tilts" in variety
    assert "cropped_edges" in variety
    assert "gaze_directions" in variety
    assert "hand_poses" in variety


def test_expander_shot_type_inference_and_aliases():
    """Verify inference keywords and normalization aliases map to the new shot types."""
    assert infer_shot_type_from_text("Shopper with a stepped back wide shot") == "stepped_back_wide"
    assert infer_shot_type_from_text("Shopper seated in cart view inspecting meat") == "seated_in_cart_view"
    assert infer_shot_type_from_text("one hand holding and one hand pointing to the label") == "holding_and_pointing"
    assert infer_shot_type_from_text("candid photo off center in right third") == "off_center_candid"
    assert infer_shot_type_from_text("handheld photo with 5 to 15 degree tilt") == "tilted_handheld"
    assert infer_shot_type_from_text("cropped edges shoulder out of frame") == "cropped_edges"
    assert infer_shot_type_from_text("shopper looking at the shelf rather than the camera") == "shelf_gaze"

    assert normalize_shot_type("stepped_back_wide") == "stepped_back_wide"
    assert normalize_shot_type("wide") == "stepped_back_wide"
    assert normalize_shot_type("seated_in_cart") == "seated_in_cart_view"
    assert normalize_shot_type("pointing") == "holding_and_pointing"
    assert normalize_shot_type("off_center") == "off_center_candid"
    assert normalize_shot_type("dutch_angle") == "tilted_handheld"


def test_person_references_exist():
    """Verify distinct authentic person reference assets exist in assets/persons/."""
    persons = get_available_person_references("assets/persons")
    assert len(persons) >= 4, f"Expected at least 4 person references, found {len(persons)}"
    filenames = {p.name for p in persons}
    assert "shopper_woman_cardigan.jpg" in filenames
    assert "shopper_man_henley.jpg" in filenames
    assert "shopper_woman_fleece.jpg" in filenames
    assert "shopper_reference.jpg" in filenames


def test_generate_variation_choices_diversity(mock_config):
    """Verify expander picks a different person reference and pose each time."""
    n = 4
    choices = generate_variation_choices(
        n=n,
        meta=load_product(mock_config.paths.knowledge_file),
        persons_dir=mock_config.paths.persons_dir,
        config=mock_config,
    )

    assert len(choices) == n

    person_refs = [c["person_reference"] for c in choices]
    # Each variation must have a different person reference
    assert len(set(person_refs)) == n, f"Person references must be unique per variation: {person_refs}"

    # Poses and hand poses must be distinct
    poses = [c["pose"] for c in choices]
    assert len(set(poses)) == n, f"Poses must be unique per variation: {poses}"

    # Each choice must satisfy framing constraints
    for c in choices:
        assert 5 <= c["tilt_degrees"] <= 15, f"Tilt {c['tilt_degrees']} must be between 5 and 15 degrees"
        assert c["subject_position"] in ["left third", "right third", "lower third"]
        assert any(term in c["cropped_edges"] for term in ["partly out of frame", "cropped"])
        assert "rather than the camera" in c["gaze"]
        assert "final_prompt" in c
        assert "[MANDATORY TRAY SPEC" in c["final_prompt"]
        assert "image 1 is the aisle, images 2-3 are the tray, image 4 is the person" in c["final_prompt"]


def test_expand_attaches_variation_choices(mock_config, mock_client):
    """Verify expand() returns variation_choices list matching requested variation count."""
    res = expand(
        short_prompt="Candid phone photo in meat aisle",
        n=4,
        config=mock_config,
        client=mock_client,
    )

    assert "variation_choices" in res
    assert len(res["variation_choices"]) == 4

    # Verify expand_variations returns N distinct expanded scene dicts
    expanded_scenes = expand_variations(
        short_prompt="Candid phone photo in meat aisle",
        n=4,
        config=mock_config,
        client=mock_client,
    )
    assert len(expanded_scenes) == 4
    for idx, sc in enumerate(expanded_scenes):
        assert sc["variation_index"] == idx
        assert "pose_framing_choice" in sc


def test_generator_records_choices_in_sidecars(mock_config, mock_client):
    """Verify generator executes per-variation choices and records choices in sidecar JSONs."""
    expanded = expand(
        short_prompt="Shopper holding beef tray in meat aisle",
        n=2,
        config=mock_config,
        client=mock_client,
    )

    saved_images = generate(
        expanded=expanded,
        n=2,
        config=mock_config,
        client=mock_client,
    )

    assert len(saved_images) == 2
    sidecar0 = json.loads(saved_images[0].with_suffix(".json").read_text(encoding="utf-8"))
    sidecar1 = json.loads(saved_images[1].with_suffix(".json").read_text(encoding="utf-8"))

    assert "pose_framing_choice" in sidecar0
    assert "pose_framing_choice" in sidecar1

    # Verify distinct person references and subject positions
    assert sidecar0["person_reference"] != sidecar1["person_reference"]
    assert sidecar0["pose_framing_choice"]["subject_position"] != sidecar1["pose_framing_choice"]["subject_position"]
    assert sidecar0["region_locked"] is True
    assert sidecar1["region_locked"] is True


def test_phash_distance_computation():
    """Verify 64-bit pHash computation and Hamming distance metric."""
    im1 = Image.open("assets/plates/Trader Joe's Shelf.jpeg")
    im1_copy = im1.copy()
    im2 = Image.open("assets/persons/shopper_woman_cardigan.jpg")

    h1 = compute_phash(im1)
    h1_copy = compute_phash(im1_copy)
    h2 = compute_phash(im2)

    assert phash_distance(h1, h1_copy) == 0
    assert phash_distance(h1, h2) >= 8
    assert len(phash_to_hex(h1)) == 16


def test_check_batch_diversity_passes_and_fails():
    """Verify check_batch_diversity accepts diverse images and rejects duplicate/near-identical ones."""
    plate = Image.open("assets/plates/Trader Joe's Shelf.jpeg")
    plate_rot = plate.rotate(15)

    # Diverse batch passes
    res_pass = check_batch_diversity([plate, plate_rot], threshold=8)
    assert res_pass.passed is True
    assert res_pass.min_distance >= 8
    assert res_pass.violating_pair is None

    # Duplicate batch fails
    res_fail = check_batch_diversity([plate, plate.copy()], threshold=8)
    assert res_fail.passed is False
    assert res_fail.min_distance == 0
    assert res_fail.violating_pair == (0, 1)
    assert "below diversity threshold 8" in res_fail.reason

    # Verify raise_on_failure raises BatchDiversityError
    with pytest.raises(BatchDiversityError):
        check_batch_diversity([plate, plate.copy()], threshold=8, raise_on_failure=True)


def test_pipeline_records_choices_and_diversity_in_manifest(mock_config, mock_client):
    """Verify full pipeline records pose_framing_choices, per-variation choices, and batch diversity in manifest."""
    pipeline = GenerationPipeline(config=mock_config, client=mock_client)
    res = pipeline.run(
        user_prompt="Shopper holding meat tray at chest level",
        variations_count=2,
    )

    assert res["status"] == "success"
    manifest_path = Path(res["manifest_path"])
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Top-level manifest records
    assert "pose_framing_choices" in manifest
    assert len(manifest["pose_framing_choices"]) == 2
    assert "batch_diversity" in manifest
    assert manifest["batch_diversity"]["passed"] is True
    assert manifest["batch_diversity"]["min_distance"] >= mock_config.phash_diversity_threshold

    # Per-variation records
    for var in manifest["variations"]:
        assert "pose_framing_choice" in var
        assert var["pose_framing_choice"] is not None
        assert "person_reference" in var["pose_framing_choice"]
        assert "subject_position" in var["pose_framing_choice"]
        assert 5 <= var["pose_framing_choice"]["tilt_degrees"] <= 15


def test_pipeline_rejects_batch_below_phash_threshold(mock_config, mock_client):
    """Verify pipeline rejects a batch if two images have a perceptual-hash distance below threshold."""
    pipeline = GenerationPipeline(config=mock_config, client=mock_client)

    # Set an impossibly high diversity threshold to trigger batch rejection on candidate variations
    mock_config.phash_diversity_threshold = 64

    res = pipeline.run(
        user_prompt="Shopper holding meat tray in aisle",
        variations_count=2,
    )

    assert res["status"] == "rejected"
    assert res["run_status"] == "REJECTED_LOW_DIVERSITY"
    assert "Batch rejected" in res["error_message"]
    assert "below diversity threshold 64" in res["rejection_reason"]
    assert res["batch_diversity"]["passed"] is False

    # Check that manifest records REJECTED_LOW_DIVERSITY
    manifest_path = Path(res["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "REJECTED_LOW_DIVERSITY"
    assert manifest["run_status"] == "REJECTED_LOW_DIVERSITY"
