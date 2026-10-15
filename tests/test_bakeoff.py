"""Unit tests for src/adapters and scripts/bakeoff.py."""

import csv
from pathlib import Path
from PIL import Image
import pytest

from src.adapters import (
    NotConfigured,
    NotConfiguredError,
    fal,
    gemini,
    generate as unified_generate,
    get_adapter_for_model,
    openai,
)
from src.config import load_config
from scripts.bakeoff import get_fixed_scene, run_bakeoff


def test_adapters_not_configured_errors(monkeypatch):
    """Verify that unconfigured openai and fal adapters raise clear NotConfigured error."""
    # Ensure keys are absent
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_API_KEY", raising=False)

    dummy_plate = Image.new("RGB", (100, 100), (200, 200, 200))
    dummy_refs = [Image.new("RGB", (50, 50), (100, 100, 100))]
    dummy_prompt = "Test prompt for bakeoff"

    # Test OpenAI
    with pytest.raises(NotConfigured) as exc_openai:
        openai.generate(plate=dummy_plate, refs=dummy_refs, prompt=dummy_prompt, n=1)
    assert "OPENAI_API_KEY" in str(exc_openai.value)

    # Test Fal
    with pytest.raises(NotConfigured) as exc_fal:
        fal.generate(plate=dummy_plate, refs=dummy_refs, prompt=dummy_prompt, n=1)
    assert "FAL_KEY" in str(exc_fal.value)


def test_get_adapter_for_model():
    """Verify get_adapter_for_model routes candidate model names to correct adapters."""
    assert get_adapter_for_model("imagen-3.0-generate-002") is gemini
    assert get_adapter_for_model("gemini-2.5-flash") is gemini
    assert get_adapter_for_model("fal-ai/flux/dev") is fal
    assert get_adapter_for_model("dall-e-3") is openai
    assert get_adapter_for_model("openai/gpt-4o") is openai


def test_gemini_adapter_mock_generation():
    """Verify gemini adapter generates n images matching (plate, refs, prompt, n) -> list[Image]."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True

    dummy_plate = Image.new("RGB", (256, 256), (120, 140, 160))
    dummy_refs = [Image.new("RGB", (128, 128), (200, 180, 150))]
    prompt = "Candid photo of shopper with Cirkla tray"

    images = gemini.generate(
        plate=dummy_plate,
        refs=dummy_refs,
        prompt=prompt,
        n=2,
        config=cfg,
    )

    assert isinstance(images, list)
    assert len(images) == 2
    for img in images:
        assert isinstance(img, Image.Image)


def test_get_fixed_scene():
    """Verify fixed benchmark scene returns valid plate, refs, and prompt."""
    cfg = load_config(require_api_key=False)
    plate_path, tray_refs, prompt = get_fixed_scene(cfg)

    assert plate_path.is_file()
    assert len(tray_refs) >= 2
    for r in tray_refs:
        assert r.is_file()
    assert len(prompt) > 20
    assert "Cirkla" in prompt


def test_run_bakeoff_execution(tmp_path, monkeypatch):
    """Verify run_bakeoff generates CSV, skips unconfigured models without failing."""
    # Ensure FAL_KEY is not set so fal model is skipped
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_API_KEY", raising=False)

    out_dir = tmp_path / "bakeoff_test"
    csv_path, rows = run_bakeoff(
        n=2,
        mock=True,
        output_dir=out_dir,
        models=["imagen-3.0-generate-002", "fal-ai/flux/dev"],
    )

    assert csv_path.is_file()

    # Read CSV and verify schema and values
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_fields = ["model", "pass_rate", "mean_similarity", "avg_latency_seconds", "notes"]
        assert reader.fieldnames == expected_fields
        read_rows = list(reader)

    assert len(read_rows) == 2

    # Check imagen row (configured and succeeded)
    imagen_row = read_rows[0]
    assert imagen_row["model"] == "imagen-3.0-generate-002"
    assert float(imagen_row["pass_rate"]) > 0.0
    assert float(imagen_row["mean_similarity"]) >= 0.0
    assert float(imagen_row["avg_latency_seconds"]) >= 0.0
    assert "passed QA" in imagen_row["notes"]

    # Check fal row (skipped without failing)
    fal_row = read_rows[1]
    assert fal_row["model"] == "fal-ai/flux/dev"
    assert float(fal_row["pass_rate"]) == 0.0
    assert "Skipped: FAL_KEY is not configured" in fal_row["notes"]
