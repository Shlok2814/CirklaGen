"""Unit tests for src/client.py running strictly in MOCK mode."""

import json
import pytest
from PIL import Image

from src.client import (
    CirklaGenAPIError,
    GenAIClient,
    RateLimitExceededError,
    SafetyBlockedError,
    image_edit,
    text,
    vision,
)
from src.config import Config


@pytest.fixture
def mock_config() -> Config:
    """Fixture providing a Config object with mock_mode=True."""
    return Config(
        mock_mode=True,
        text_model="gemini-2.5-flash",
        image_model="imagen-3.0-generate-002",
        vision_model="gemini-2.5-flash",
    )


@pytest.fixture
def mock_client(mock_config: Config) -> GenAIClient:
    """Fixture providing a GenAIClient instance operating in MOCK mode."""
    return GenAIClient(config=mock_config)


@pytest.fixture
def dummy_image() -> Image.Image:
    """Fixture creating a distinct 64x64 PIL Image for testing."""
    return Image.new("RGB", (64, 64), color=(255, 100, 50))


def test_text_returns_canned_json(mock_client: GenAIClient):
    """Verify that text() in MOCK mode returns canned JSON string."""
    prompt = "A person in Trader Joe's holding a meat tray"
    result = mock_client.text(prompt)

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed.get("mock") is True
    assert parsed.get("status") == "success"
    assert "expanded_prompt" in parsed


def test_text_with_json_schema_returns_dict(mock_client: GenAIClient):
    """Verify that text() with json_schema in MOCK mode returns a parsed dictionary."""
    prompt = "A person in Trader Joe's holding a meat tray"
    schema = {"type": "object", "properties": {"expanded_prompt": {"type": "string"}}}
    result = mock_client.text(prompt, json_schema=schema)

    assert isinstance(result, dict)
    assert result.get("mock") is True
    assert "expanded_prompt" in result


def test_vision_returns_canned_json(mock_client: GenAIClient, dummy_image: Image.Image):
    """Verify that vision() in MOCK mode returns canned evaluation JSON."""
    result = mock_client.vision(dummy_image, prompt="Evaluate tray authenticity")

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed.get("mock") is True
    assert "qa_score" in parsed
    assert parsed.get("passed") is True


def test_vision_with_json_schema_returns_dict(mock_client: GenAIClient, dummy_image: Image.Image):
    """Verify that vision() with json_schema in MOCK mode returns a dictionary."""
    schema = {"type": "object"}
    result = mock_client.vision(dummy_image, prompt="Evaluate tray authenticity", json_schema=schema)

    assert isinstance(result, dict)
    assert result.get("mock") is True
    assert "qa_score" in result


def test_image_edit_returns_first_input_image(mock_client: GenAIClient, dummy_image: Image.Image):
    """Verify that image_edit() in MOCK mode returns the first input image duplicated n times."""
    second_image = Image.new("RGB", (32, 32), color=(0, 0, 255))
    images = [dummy_image, second_image]

    # Test with n=3
    results = mock_client.image_edit(images, prompt="Change background to meat aisle", n=3)

    assert isinstance(results, list)
    assert len(results) == 3
    for img in results:
        assert isinstance(img, Image.Image)
        # Must match the first image's dimensions and color
        assert img.size == dummy_image.size
        assert img.getpixel((0, 0)) == dummy_image.getpixel((0, 0))


def test_image_edit_single_image_input(mock_client: GenAIClient, dummy_image: Image.Image):
    """Verify image_edit() works when passed a single PIL.Image instead of a list."""
    results = mock_client.image_edit(dummy_image, prompt="Adjust lighting", n=1)

    assert len(results) == 1
    assert results[0].size == dummy_image.size
    assert results[0].getpixel((10, 10)) == dummy_image.getpixel((10, 10))


def test_image_edit_empty_images_raises(mock_client: GenAIClient):
    """Verify that passing an empty image list raises ValueError."""
    with pytest.raises(ValueError, match="requires at least one input image"):
        mock_client.image_edit([], prompt="Test prompt")


def test_module_level_functions_in_mock_mode(mock_config: Config, dummy_image: Image.Image):
    """Verify that top-level module functions text, vision, and image_edit work with mock_config."""
    # Top-level text()
    text_res = text("Test prompt", config=mock_config)
    assert isinstance(text_res, str)
    assert json.loads(text_res)["mock"] is True

    # Top-level vision()
    vision_res = vision(dummy_image, "Test prompt", config=mock_config)
    assert isinstance(vision_res, str)
    assert json.loads(vision_res)["mock"] is True

    # Top-level image_edit()
    edit_res = image_edit(dummy_image, "Test prompt", n=2, config=mock_config)
    assert len(edit_res) == 2
    assert edit_res[0].size == dummy_image.size


def test_custom_exception_hierarchy():
    """Verify that specific exceptions inherit from CirklaGenAPIError."""
    assert issubclass(RateLimitExceededError, CirklaGenAPIError)
    assert issubclass(SafetyBlockedError, CirklaGenAPIError)

    rate_err = RateLimitExceededError("Rate limit (HTTP 429) hit after 5 retries.")
    assert "Rate limit" in str(rate_err)

    safety_err = SafetyBlockedError("Prompt blocked due to safety policy violation.")
    assert "safety policy" in str(safety_err)
