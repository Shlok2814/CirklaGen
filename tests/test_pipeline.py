"""Unit and integration tests for src/pipeline.py including Manual Generation Mode and Repairs."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock
import zipfile
from PIL import Image
import pytest

from src.client import GenAIClient
from src.config import Config, load_config
from src.pipeline import GenerationPipeline, create_reference_bundle_zip, run
from src.qa import QAResult


@pytest.fixture
def mock_config() -> Config:
    """Fixture providing Config with mock_mode=True."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    """Fixture providing GenAIClient in MOCK mode."""
    return GenAIClient(config=mock_config)


@pytest.fixture
def dummy_manual_image() -> Image.Image:
    """Creates a sample test image representing a manually generated photo."""
    return Image.new("RGB", (512, 512), color=(210, 190, 160))


def test_create_reference_bundle_zip():
    """Verify in-memory reference bundle ZIP generation."""
    plate_file = Path("assets/plates/Trader Joe's Shelf.jpeg")
    tray_files = [
        Path("assets/trays/tray_unlabeled_3_clean.png"),
        Path("assets/trays/tray_unlabeled_4.png"),
    ]
    prompt = "A candid photo of customer holding Cirkla tray"

    zip_bytes = create_reference_bundle_zip(
        plate_path=plate_file,
        tray_paths=tray_files,
        prompt_text=prompt,
        metadata={"test": True},
    )

    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert any("plate" in n for n in namelist)
        assert any("trays" in n for n in namelist)
        assert "prompt_for_gemini.txt" in namelist
        assert "expansion_metadata.json" in namelist

        extracted_prompt = zf.read("prompt_for_gemini.txt").decode("utf-8")
        assert extracted_prompt == prompt


def test_pipeline_manual_mode_waiting_step(mock_config, mock_client):
    """Verify pipeline returns copy-ready prompt and zip bundle when manual image is pending."""
    pipe = GenerationPipeline(config=mock_config, client=mock_client)
    res = pipe.run(
        user_prompt="me holding tray",
        manual_mode=True,
        manual_image=None,
    )

    assert res["status"] == "waiting_for_manual_upload"
    assert res["manual_mode"] is True
    assert res["mode"] == "manual"
    assert len(res["copy_ready_prompt"]) > 50
    assert len(res["zip_bytes"]) > 0
    assert "Trader Joe's Shelf.jpeg" in res["files_to_upload"] or len(res["files_to_upload"]) >= 2


def test_pipeline_manual_mode_complete_flow(mock_config, mock_client, dummy_manual_image):
    """Verify end-to-end execution of manual mode with uploaded image."""
    pipe = GenerationPipeline(config=mock_config, client=mock_client)
    res = pipe.run(
        user_prompt="POV hands with tray in the aisle",
        shot_type="pov_hands",
        manual_mode=True,
        manual_image=dummy_manual_image,
    )

    assert res["status"] == "success"
    assert res["manual"] is True
    assert res["mode"] == "manual"

    run_dir = Path(res["run_dir"])
    assert run_dir.is_dir()

    raw_img = run_dir / "img_0.png"
    finished_img = run_dir / "img_0_finished.png"
    assert raw_img.is_file()
    assert finished_img.is_file()

    sidecar_path = run_dir / "img_0.json"
    assert sidecar_path.is_file()
    with open(sidecar_path, "r", encoding="utf-8") as sf:
        sidecar = json.load(sf)
    assert sidecar["manual"] is True
    assert sidecar["mode"] == "manual"
    assert "qa_evaluation" in sidecar

    manifest_path = Path(res["manifest_path"])
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)

    assert manifest["manual"] is True
    assert manifest["mode"] == "manual"
    assert manifest["total_variations"] == 1
    assert manifest["variations"][0]["manual"] is True
    assert manifest["variations"][0]["passed_qa"] is True


def test_pipeline_automated_mode_flow(mock_config, mock_client):
    """Verify pipeline automated flow with generator and QA."""
    pipe = GenerationPipeline(config=mock_config, client=mock_client)
    res = pipe.run(
        user_prompt="me holding tray",
        num_variations=1,
        manual_mode=False,
    )

    assert res["status"] == "success"
    assert res["manual"] is False
    assert res["mode"] == "automated"

    run_dir = Path(res["run_dir"])
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)

    assert manifest["manual"] is False
    assert manifest["mode"] == "automated"


def test_pipeline_run_end_to_end_with_progress_and_best_image(mock_config, mock_client):
    """Verify end-to-end execution with progress callback and best_image manifest ranking."""
    pipe = GenerationPipeline(config=mock_config, client=mock_client)

    progress_log = []

    def on_progress(pct: float, msg: str):
        progress_log.append((pct, msg))

    res = pipe.run(
        short_prompt="me holding tray in market",
        shot_type="front_chest",
        n=2,
        use_composite=False,
        finishing=True,
        progress_callback=on_progress,
    )

    assert res["status"] == "success"
    assert len(progress_log) >= 5
    assert progress_log[0][0] <= 0.15
    assert progress_log[-1][0] == 1.0

    run_dir = Path(res["run_dir"])
    assert run_dir.is_dir()

    manifest_path = Path(res["manifest_path"])
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)

    assert manifest["short_prompt"] == "me holding tray in market"
    assert len(manifest["expanded_prompt"]) > 20
    assert "best_image" in manifest
    assert manifest["best_image"] is not None
    assert "checklist_passes" in manifest["best_image"]
    assert "similarity_max" in manifest["best_image"]
    assert "file" in manifest["best_image"]
    assert len(manifest["variations"]) == 2


def test_pipeline_top_level_run_convenience(mock_config, mock_client):
    """Verify top-level module run function works cleanly."""
    res = run(
        short_prompt="POV customer inspecting steak",
        shot_type="pov_hands",
        n=1,
        finishing=False,
        config=mock_config,
        client=mock_client,
    )

    assert res["status"] == "success"
    assert res["short_prompt"] == "POV customer inspecting steak"
    assert Path(res["manifest_path"]).is_file()


def test_pipeline_repair_flow_when_qa_fails(mock_config, mock_client, monkeypatch):
    """Verify repair is triggered when an image fails QA."""
    pipe = GenerationPipeline(config=mock_config, client=mock_client)

    call_count = 0

    def mock_evaluate_image(img, config=None, client=None, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: fail QA with bad hands
            return QAResult(
                similarity_max=0.45,
                similarity_mean=0.30,
                checklist={
                    "exterior_beige": {"passed": True, "reason": "ok"},
                    "hands_anatomy_ok": {"passed": False, "reason": "extra finger detected"},
                    "looks_like_ai": {"passed": False, "reason": "none"},
                },
                passed=False,
                notes="FAIL: hands anatomy invalid",
                overall_score=0.40,
                tray_detected=True,
                crop=None,
            )
        else:
            # Re-run QA after repair: passes
            return QAResult(
                similarity_max=0.75,
                similarity_mean=0.60,
                checklist={
                    "exterior_beige": {"passed": True, "reason": "ok"},
                    "hands_anatomy_ok": {"passed": True, "reason": "repaired"},
                    "looks_like_ai": {"passed": False, "reason": "none"},
                },
                passed=True,
                notes="PASS: repaired successfully",
                overall_score=0.85,
                tray_detected=True,
                crop=None,
            )

    import src.pipeline
    monkeypatch.setattr(src.pipeline, "evaluate_image", mock_evaluate_image)

    res = pipe.run(
        short_prompt="holding tray",
        n=1,
        use_composite=False,
        finishing=True,
    )

    assert res["status"] == "success"
    var = res["variations"][0]
    assert var["repaired"] is True
    assert var["repair_type"] == "fix_region_hands"
    assert var["passed_qa"] is True
    assert var["repaired_image"] is not None
    assert Path(var["repaired_image"]).is_file()


def test_pipeline_enterprise_mode_graceful_fallback(mock_config, monkeypatch):
    """Verify that when image generation raises Enterprise mode/billing requirement, pipeline gracefully falls back to realistic candidates."""
    from src.client import CirklaGenAPIError

    def failing_generate(*args, **kwargs):
        raise CirklaGenAPIError(
            "Image generation failed: Variation 1 API call failed: Image edit model 'imagen-3.0-generate-002' "
            "requires Gemini Enterprise mode or billing. Set MOCK=1 to test offline."
        )

    import src.pipeline
    monkeypatch.setattr(src.pipeline, "generate", failing_generate)

    pipe = GenerationPipeline(config=mock_config)
    res = pipe.run(
        short_prompt="front chest holding tray",
        shot_type="front_chest",
        n=2,
        finishing=True,
    )

    assert res["status"] == "success"
    assert res["fallback"] is True
    assert len(res["variations"]) == 2
    for var in res["variations"]:
        assert var["fallback"] is True
        assert Path(var["raw_image"]).is_file()
        assert Path(var["finished_image"]).is_file()
        assert var["status"] == "GENERATED"

