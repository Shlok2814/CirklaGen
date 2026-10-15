"""Unit and integration tests for plate-matched finishing pass and prompt realism descriptors."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.config import Config, load_config
from src.expander import (
    build_variation_final_prompt,
    enforce_final_prompt_constraints,
    expand,
    generate_variation_choices,
    load_product,
)
from src.finish import (
    Finisher,
    apply_mild_shadow_grain,
    apply_plate_matched_finishing,
    create_plate_matched_comparison_sheet,
    match_lab_color_to_surrounding,
    match_noise_to_plate,
    match_sharpness_to_plate,
    measure_plate_noise,
    measure_sharpness,
    sample_surrounding_plate_pixels,
    soften_face_region,
)
from src.generator import format_final_prompt_for_generator


@pytest.fixture
def mock_config() -> Config:
    """Provides test configuration in mock mode."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def sample_plate() -> Image.Image:
    """Loads actual Trader Joe's shelf plate."""
    return Image.open("assets/plates/Trader Joe's Shelf.jpeg")


@pytest.fixture
def sample_composite(sample_plate) -> Image.Image:
    """Creates a sample composite with a person crop in the center."""
    comp = sample_plate.copy()
    w, h = comp.size
    # Paste person reference into center [0.15, 0.20, 0.90, 0.80]
    person = Image.open("assets/persons/shopper_woman_cardigan.jpg").convert("RGB")
    left, top, right, bottom = int(w * 0.20), int(h * 0.15), int(w * 0.80), int(h * 0.90)
    person_resized = person.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    comp.paste(person_resized, (left, top))
    comp.info["placement_region"] = [0.15, 0.20, 0.90, 0.80]
    return comp


def test_generation_prompt_includes_all_realism_descriptors(mock_config):
    """Verify that expanded prompts include all 6 required natural realism descriptors."""
    required_descriptors = [
        "uneven skin tone",
        "visible pores",
        "slight shine on forehead and nose",
        "flyaway hair",
        "a hard shadow under the chin and under the tray from the overhead lights",
        "creased clothing",
    ]

    meta = load_product(mock_config.paths.knowledge_file)
    choices = generate_variation_choices(
        n=4,
        base_shot_type="front_chest",
        short_prompt="Shopper holding meat tray at chest level",
        meta=meta,
        persons_dir=mock_config.paths.persons_dir,
        config=mock_config,
    )

    for choice in choices:
        prompt = choice["final_prompt"].lower()
        for desc in required_descriptors:
            assert desc.lower() in prompt, f"Descriptor '{desc}' missing from choice prompt: {prompt}"

    # Also test enforce_final_prompt_constraints on arbitrary prompt
    custom_prompt = "A candid phone photo of a person holding tray"
    enforced = enforce_final_prompt_constraints(custom_prompt, meta=meta).lower()
    for desc in required_descriptors:
        assert desc.lower() in enforced, f"Descriptor '{desc}' missing from enforced prompt: {enforced}"

    # Also test format_final_prompt_for_generator
    gen_prompt = format_final_prompt_for_generator(
        "A photo in the aisle", n_tray_refs=2, has_person=True
    ).lower()
    for desc in required_descriptors:
        assert desc.lower() in gen_prompt, f"Descriptor '{desc}' missing from generator prompt: {gen_prompt}"


def test_measure_plate_noise_and_sharpness(sample_plate):
    """Verify noise and sharpness estimation from plate image."""
    noise_sigma = measure_plate_noise(sample_plate, edited_bbox=(200, 200, 800, 800))
    assert 1.0 <= noise_sigma <= 25.0, f"Expected realistic sensor noise sigma, got {noise_sigma}"

    sharpness = measure_sharpness(sample_plate)
    assert sharpness > 10.0, f"Expected non-zero sharpness on textured grocery plate, got {sharpness}"


def test_match_lab_color_to_surrounding():
    """Verify Reinhard LAB color matching against surrounding pixels."""
    # Synthetic warm ROI (R=220, G=160, B=100) and cool surrounding (R=120, G=150, B=210)
    warm_roi = np.full((100, 100, 3), [220, 160, 100], dtype=np.float32)
    cool_surround = np.full((50, 50, 3), [120, 150, 210], dtype=np.float32)

    matched_roi, stats = match_lab_color_to_surrounding(warm_roi, cool_surround, strength=0.70)

    assert "surround_lab_mean" in stats
    assert "roi_lab_mean_before" in stats
    assert "roi_lab_mean_after" in stats

    # Blue channel should increase, Red channel should decrease
    assert np.mean(matched_roi[..., 2]) > np.mean(warm_roi[..., 2])
    assert np.mean(matched_roi[..., 0]) < np.mean(warm_roi[..., 0])


def test_soften_face_region():
    """Verify facial softening using bilateral filtering inside face region."""
    # Create synthetic face crop with high-frequency noise
    rng = np.random.RandomState(42)
    face_im = np.full((200, 200, 3), 180.0, dtype=np.float32)
    # Add sharp texture
    face_im += rng.normal(0, 15.0, face_im.shape)
    np.clip(face_im, 0, 255, out=face_im)

    sharpness_before = measure_sharpness(face_im)
    softened_roi, face_box = soften_face_region(
        face_im, face_bbox=(20, 20, 180, 180), strength=0.60
    )
    sharpness_after = measure_sharpness(softened_roi)

    assert face_box is not None
    assert sharpness_after < sharpness_before, "Facial softening must reduce hyper-sharp skin noise"


def test_apply_mild_shadow_grain():
    """Verify mild shadow grain adds noise to dark regions while leaving bright highlights clean."""
    rng = np.random.RandomState(42)
    # Gradient image from 20 (shadow) to 240 (bright)
    grad = np.linspace(20, 240, 100, dtype=np.float32)
    img = np.tile(grad[:, None, None], (1, 100, 3))

    grained = apply_mild_shadow_grain(img, noise_sigma=12.0, rng=rng)

    # Difference in shadow top row (luminance ~ 20)
    shadow_diff = np.mean(np.abs(grained[:15, :] - img[:15, :]))
    # Difference in bright bottom row (luminance ~ 240)
    bright_diff = np.mean(np.abs(grained[85:, :] - img[85:, :]))

    assert shadow_diff > bright_diff * 3.0, "Shadow grain must be concentrated in low-luminance areas"


def test_apply_plate_matched_finishing_full(sample_composite, sample_plate):
    """Verify full plate-matched finishing pass executes all steps and embeds provenance."""
    finished = apply_plate_matched_finishing(
        image=sample_composite,
        plate=sample_plate,
        edited_region=[0.15, 0.20, 0.90, 0.80],
        jpeg_quality=86,
        feather_radius=15,
        seed=42,
    )

    assert finished.size == sample_composite.size
    assert finished.mode == "RGB"

    # Check info metadata
    meta = finished.info.get("plate_matched_finishing")
    assert meta is not None
    assert "plate_noise_sigma" in meta
    assert "plate_sharpness" in meta
    assert "roi_noise_before" in meta
    assert "roi_noise_after" in meta
    assert "roi_sharpness_before" in meta
    assert "roi_sharpness_after" in meta
    assert meta["global_jpeg_quality"] == 86


def test_finisher_class_apply_plate_matched(sample_composite, sample_plate):
    """Verify Finisher class wrapper applies plate-matched pass cleanly."""
    finisher = Finisher()
    finished = finisher.apply_plate_matched(
        image=sample_composite,
        plate=sample_plate,
        edited_region=[0.15, 0.20, 0.90, 0.80],
    )
    assert finished.size == sample_composite.size
    assert getattr(finished, "_plate_matched", False) is True


def test_create_plate_matched_comparison_sheet(sample_composite, sample_plate, tmp_path):
    """Verify 3-variation comparison sheet generation."""
    # Create 3 synthetic variations
    vars_list = []
    for i in range(3):
        comp = sample_composite.copy()
        fin = apply_plate_matched_finishing(
            image=comp,
            plate=sample_plate,
            edited_region=[0.15, 0.20, 0.90, 0.80],
            seed=42 + i,
        )
        vars_list.append({
            "before": comp,
            "after": fin,
            "label": f"Shopper Candidate {i + 1}",
            "metrics_summary": f"Noise: 2.4 -> 3.8 | Sharpness: 420 -> 310 | JPEG: Q86",
        })

    out_sheet = tmp_path / "test_comparison_sheet.png"
    result_path = create_plate_matched_comparison_sheet(
        variations=vars_list,
        output_path=out_sheet,
    )

    assert result_path.is_file()
    with Image.open(result_path) as sheet_img:
        assert sheet_img.width > 500
        assert sheet_img.height > 1000
