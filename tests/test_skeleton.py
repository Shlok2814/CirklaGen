"""Test suite verifying CirklaGen skeleton structure and module imports."""

from pathlib import Path
import pytest
from src.config import load_config


def test_config_structure():
    """Verify that config.yaml contains all required skeleton parameters."""
    cfg = load_config("config.yaml")
    required_keys = [
        "text_model",
        "image_model",
        "vision_model",
        "qa_threshold",
        "variations_default",
        "output_dir",
        "bakeoff_models",
        "finishing_defaults",
        "mock_mode",
        "manual_mode",
    ]
    for key in required_keys:
        assert key in cfg, f"Missing required config key: {key}"

    assert cfg["qa_threshold"] == 0.6
    assert cfg["variations_default"] == 4
    assert isinstance(cfg["bakeoff_models"], list)
    assert "grain" in cfg["finishing_defaults"]
    assert "vignette" in cfg["finishing_defaults"]
    assert "softness" in cfg["finishing_defaults"]
    assert "jpeg_quality" in cfg["finishing_defaults"]
    assert "tilt_deg" in cfg["finishing_defaults"]


def test_skeleton_files_and_directories_exist():
    """Verify that all required skeleton directories and files exist."""
    required_dirs = [
        "assets/raw_trays",
        "assets/trays",
        "assets/plates_raw",
        "assets/plates",
        "assets/persons",
        "assets/reference",
        "src/adapters",
        "scripts",
        "phase2",
        "docs",
        "tests",
        "outputs/runs",
        "outputs/examples",
        "outputs/logs",
    ]
    for d in required_dirs:
        assert Path(d).is_dir(), f"Directory missing: {d}"

    required_files = [
        "config.yaml",
        ".env.example",
        "requirements.txt",
        "knowledge/product.json",
        "prompts/expander_system.txt",
        "app.py",
        "README.md",
        "src/__init__.py",
        "src/config.py",
        "src/client.py",
        "src/expander.py",
        "src/generator.py",
        "src/qa.py",
        "src/finish.py",
        "src/assets_prep.py",
        "src/pipeline.py",
        "src/metadata.py",
        "src/adapters/__init__.py",
        "src/adapters/gemini.py",
        "src/adapters/openai.py",
        "src/adapters/fal.py",
        "scripts/run_examples.py",
        "scripts/build_report.py",
        "scripts/bakeoff.py",
        "scripts/check_output.py",
    ]
    for f in required_files:
        assert Path(f).is_file(), f"File missing: {f}"


def test_import_skeleton_modules():
    """Verify that all src modules and scripts can be cleanly imported."""
    import src.config
    import src.client
    import src.expander
    import src.generator
    import src.qa
    import src.finish
    import src.assets_prep
    import src.pipeline
    import src.metadata
    import src.adapters.gemini
    import src.adapters.openai
    import src.adapters.fal

    assert src.config is not None
    assert src.client.GenAIClient is not None
    assert src.expander.PromptExpander is not None
    assert src.generator.ImageGenerator is not None
    assert src.qa.QualityAssurance is not None
    assert src.finish.Finisher is not None
    assert src.assets_prep.AssetsPrep is not None
    assert src.pipeline.GenerationPipeline is not None
    assert src.metadata.MetadataTracker is not None
