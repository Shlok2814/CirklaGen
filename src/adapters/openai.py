"""OpenAI adapter stub for CirklaGen bakeoff comparisons.

Allows benchmarking prompt expansion and image generation against OpenAI models
(e.g. GPT-4o, DALL-E) as listed in config.yaml bakeoff_models.
Reads OPENAI_API_KEY from .env and raises NotConfigured if missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.adapters.base import NotConfigured, NotConfiguredError


class OpenAIAdapter:
    """Stub adapter for OpenAI models."""

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise NotConfigured("OPENAI_API_KEY is not configured in .env")

    def generate(
        self,
        plate: Union[str, Path, Image.Image],
        refs: List[Union[str, Path, Image.Image]],
        prompt: str,
        n: int = 1,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Unified bakeoff generation interface: generate(plate, refs, prompt, n) -> list[Image]."""
        if not self.api_key:
            raise NotConfigured("OPENAI_API_KEY is not configured in .env")
        raise NotImplementedError("OpenAI image generation adapter is a stub.")

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Invokes OpenAI text model stub."""
        if not self.api_key:
            raise NotConfigured("OPENAI_API_KEY is not configured in .env")
        raise NotImplementedError("OpenAI text generation adapter is a stub.")

    def generate_image(self, prompt: str, **kwargs: Any) -> List[Image.Image]:
        """Invokes OpenAI image model stub."""
        if not self.api_key:
            raise NotConfigured("OPENAI_API_KEY is not configured in .env")
        raise NotImplementedError("OpenAI image generation adapter is a stub.")


def generate(
    plate: Union[str, Path, Image.Image],
    refs: List[Union[str, Path, Image.Image]],
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> List[Image.Image]:
    """Module-level function matching generate(plate, refs, prompt, n) -> list[Image]."""
    adapter = OpenAIAdapter(api_key=api_key)
    return adapter.generate(plate=plate, refs=refs, prompt=prompt, n=n, model=model, **kwargs)


if __name__ == "__main__":
    print("--- Demo: src.adapters.openai ---")
    try:
        adapter = OpenAIAdapter()
        print("OpenAIAdapter initialized.")
    except NotConfigured as e:
        print(f"Expected NotConfigured caught: {e}")
