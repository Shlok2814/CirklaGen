"""Configuration loader and schema definition for CirklaGen.

Loads config.yaml and .env, exposes a typed Config object, and fails with a clear
error message if GEMINI_API_KEY is missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class FinishingDefaults(BaseModel):
    """Candid smartphone camera optical finishing settings."""
    grain: float = 0.15
    vignette: float = 0.20
    softness: float = 0.10
    jpeg_quality: int = 85
    tilt_deg: float = 1.5
    chromatic_aberration: float = 0.15
    highlight_bloom: float = 0.20
    off_center_crop: float = 0.15
    saturation: float = 0.90
    uneven_white_balance: float = 0.15
    motion_blur: float = 0.15
    realism_pass: bool = True
    match_plate_lighting: bool = False
    relight_strength: float = 0.35
    shadow_strength: float = 0.25

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


class PathsConfig(BaseModel):
    """Filesystem locations for knowledge, assets, scripts, and runs."""
    knowledge_file: str = "knowledge/product.json"
    expander_prompt: str = "prompts/expander_system.txt"
    trays_dir: str = "assets/trays"
    trays_work_dir: str = "assets/trays_work"
    raw_trays_dir: str = "assets/raw_trays"
    plates_dir: str = "assets/plates"
    plates_raw_dir: str = "assets/plates_raw"
    persons_dir: str = "assets/persons"
    packaged_dir: str = "assets/packaged"
    reference_dir: str = "assets/reference"
    runs_dir: str = "outputs/runs"
    examples_dir: str = "outputs/examples"
    logs_dir: str = "outputs/logs"
    log_file: str = "outputs/logs/cirklagen.log"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class ResilienceConfig(BaseModel):
    """Retry and backoff parameters for API reliability."""
    max_retries: int = 5
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 32.0
    backoff_multiplier: float = 2.0
    timeout_seconds: float = 90.0

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class Config(BaseModel):
    """Typed configuration object for CirklaGen."""
    gemini_api_key: str = Field(default="", description="Google GenAI API Key")
    text_model: str = Field(default="gemini-3.5-flash-lite", description="Model for prompt expansion")
    image_model: str = Field(default="imagen-3.0-generate-002", description="Model for image editing/generation")
    vision_model: str = Field(default="gemini-3.5-flash-lite", description="Model for QA vision evaluation")
    rembg_model: str = Field(default="u2net", description="Model for background removal (rembg)")
    qa_threshold: float = Field(default=0.6, description="Minimum QA score threshold")
    sleeve_similarity_threshold: float = Field(
        default=0.70, description="Minimum sleeve text and layout similarity threshold"
    )
    phash_diversity_threshold: int = Field(
        default=8, description="Minimum pairwise perceptual-hash distance threshold to accept a batch"
    )
    variations_default: int = Field(default=4, description="Default number of image candidates")
    output_dir: str = Field(default="outputs/runs", description="Default directory for run outputs")
    bakeoff_models: List[str] = Field(
        default_factory=lambda: ["imagen-3.0-generate-002", "gemini-2.5-flash", "fal-ai/flux/dev"]
    )
    finishing_defaults: FinishingDefaults = Field(default_factory=FinishingDefaults)
    mock_mode: bool = Field(default=False, description="When true, client returns mock data")
    manual_mode: bool = Field(default=False, description="Manual human-in-the-loop review mode")
    better_reference_selection: bool = Field(
        default=True,
        description="Select tray references by both angle and specific visible features",
    )
    strict_must_keep: bool = Field(
        default=True,
        description="Place a strict, dense must_keep block at the start of final_prompt",
    )
    auto_composite_fallback: bool = Field(
        default=True,
        description="Automatically composite clean tray cutout when QA fails twice",
    )
    realism_pass: bool = Field(
        default=True,
        description="Enable realism pass: slightly lower saturation, uneven white balance, small motion blur, and clutter",
    )
    match_plate_lighting: bool = Field(
        default=False,
        description="Enable optional lighting match to plate for person region",
    )
    paths: PathsConfig = Field(default_factory=PathsConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


def load_config(
    config_path: str | Path = "config.yaml",
    require_api_key: bool = True,
    env_file: Optional[str | Path] = ".env",
) -> Config:
    """Loads configuration from config.yaml and environment variables (.env).

    Exposes a typed Config object.
    Fails with a clear message if GEMINI_API_KEY is missing.

    Args:
        config_path: Path to config.yaml.
        require_api_key: If True (default), fails when GEMINI_API_KEY is not configured
                         (unless mock_mode is enabled).
        env_file: Path to .env file to load.

    Returns:
        Config: Typed validated configuration object.

    Raises:
        FileNotFoundError: If config.yaml is not found.
        ValueError: If GEMINI_API_KEY is missing from environment/.env.
    """
    if env_file:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Check for mock_mode override from environment variable if present
    env_mock = os.getenv("MOCK_MODE") or os.getenv("MOCK")
    if env_mock is not None:
        data["mock_mode"] = env_mock.lower() in ("1", "true", "yes")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass

    # Fail with a clear message if GEMINI_API_KEY is missing (unless mock_mode is active)
    is_mock = data.get("mock_mode", False)
    if require_api_key and not is_mock and not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from environment or .env file. "
            "Please copy .env.example to .env and assign your Google GenAI API key: "
            "GEMINI_API_KEY=your_key_here"
        )

    data["gemini_api_key"] = api_key
    return Config(**data)


def get_api_key() -> Optional[str]:
    """Retrieves GEMINI_API_KEY from environment variables or Streamlit secrets."""
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass
    return key


if __name__ == "__main__":
    print("--- Running src.config demo ---")
    try:
        cfg = load_config()
        print("Config successfully loaded:")
        print(f"  Text Model: {cfg.text_model}")
        print(f"  Image Model: {cfg.image_model}")
        print(f"  Vision Model: {cfg.vision_model}")
        print(f"  QA Threshold: {cfg.qa_threshold}")
        print(f"  Variations Default: {cfg.variations_default}")
        print(f"  Mock Mode: {cfg.mock_mode}")
        print(f"  Better Reference Selection: {cfg.better_reference_selection}")
        print(f"  Strict Must-Keep: {cfg.strict_must_keep}")
        print(f"  Auto Composite Fallback: {cfg.auto_composite_fallback}")
        print(f"  API Key Present: {bool(cfg.gemini_api_key)}")
    except ValueError as e:
        print(f"Configuration validation notice: {e}")
