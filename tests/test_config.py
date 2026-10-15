"""Unit tests for configuration module."""

import os
from pathlib import Path
import pytest
from src.config import Config, load_config, get_api_key


def test_load_default_config():
    """Verify that default config.yaml loads with typed Config object."""
    config = load_config("config.yaml")
    assert isinstance(config, Config)
    # Test typed attribute access
    assert config.text_model == "gemini-3.5-flash-lite"
    assert config.image_model == "imagen-3.0-generate-002"
    assert config.vision_model == "gemini-3.5-flash-lite"
    assert config.qa_threshold == 0.6
    assert config.variations_default == 4
    assert config.output_dir == "outputs/runs"
    assert isinstance(config.bakeoff_models, list)
    assert config.finishing_defaults.jpeg_quality == 85

    # Test dict-like backward-compatibility access
    assert config["text_model"] == "gemini-3.5-flash-lite"
    assert config["finishing_defaults"]["grain"] == 0.15


def test_missing_config_file_raises():
    """Verify that a missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")


def test_missing_gemini_api_key_raises_clear_error(monkeypatch):
    """Verify that missing GEMINI_API_KEY raises a clear, informative ValueError."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing from environment"):
        load_config("config.yaml", env_file=None)


def test_missing_key_allowed_in_mock_mode(monkeypatch):
    """Verify that mock_mode=True allows empty GEMINI_API_KEY without failing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("MOCK_MODE", "true")
    config = load_config("config.yaml", env_file=None)
    assert config.mock_mode is True
    assert config.gemini_api_key == ""


def test_missing_key_allowed_with_mock_env(monkeypatch):
    """Verify that MOCK=1 sets mock_mode=True."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("MOCK", "1")
    config = load_config("config.yaml", env_file=None)
    assert config.mock_mode is True



def test_get_api_key_from_env(monkeypatch):
    """Verify API key resolution from environment variable."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_123")
    assert get_api_key() == "test_gemini_key_123"


def test_anti_drift_config_flags():
    """Verify default anti-drift flags in Config object."""
    config = load_config("config.yaml")
    assert config.better_reference_selection is True
    assert config.strict_must_keep is True
    assert config.auto_composite_fallback is True
