"""Fal.ai adapter stub for CirklaGen bakeoff comparisons.

Allows benchmarking image generation against open-source diffusion models
(e.g. Flux Dev / Schnell) hosted on Fal.ai as listed in config.yaml bakeoff_models.
Reads FAL_KEY from .env and raises NotConfigured if missing.
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


class FalAdapter:
    """Stub adapter for Fal.ai models."""

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        if not self.api_key:
            raise NotConfigured("FAL_KEY is not configured in .env")

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
            raise NotConfigured("FAL_KEY is not configured in .env")
        raise NotImplementedError("Fal.ai image generation adapter is a stub.")

    def generate_image(
        self,
        prompt: str,
        model: str = "fal-ai/flux/dev",
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Invokes Fal model for image generation stub."""
        if not self.api_key:
            raise NotConfigured("FAL_KEY is not configured in .env")
        raise NotImplementedError("Fal.ai image generation adapter is a stub.")


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
    adapter = FalAdapter(api_key=api_key)
    return adapter.generate(plate=plate, refs=refs, prompt=prompt, n=n, model=model, **kwargs)


if __name__ == "__main__":
    print("--- Demo: src.adapters.fal ---")
    try:
        adapter = FalAdapter()
        print("FalAdapter initialized.")
    except NotConfigured as e:
        print(f"Expected NotConfigured caught: {e}")
