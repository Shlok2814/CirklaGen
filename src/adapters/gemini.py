"""Google GenAI adapter for CirklaGen.

Implements text synthesis, photorealistic image generation (Imagen 3 / Gemini multimodal),
and vision QA inspection using the official google-genai SDK.
Secrets live exclusively in .env.
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
from src.client import GenAIClient
from src.config import Config, load_config
from src.logger import logger


def _load_pil_image(img_input: Union[str, Path, Image.Image]) -> Image.Image:
    """Loads a PIL Image from path or returns if already an Image."""
    if isinstance(img_input, Image.Image):
        return img_input
    path = Path(img_input)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {path}")
    return Image.open(path).convert("RGB")


class GeminiAdapter:
    """Adapter for Google GenAI models (Imagen 3, Gemini 2.5/3.5)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        load_dotenv()
        self.config = config or load_config(require_api_key=False)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or getattr(self.config, "gemini_api_key", None)
        
        # Check if configured
        if not self.api_key and not getattr(self.config, "mock_mode", False):
            raise NotConfigured("GEMINI_API_KEY is not configured in .env")

        self.model = model or getattr(self.config, "image_model", "imagen-3.0-generate-002")
        self.client = GenAIClient(config=self.config, api_key=self.api_key)

    def generate(
        self,
        plate: Union[str, Path, Image.Image],
        refs: List[Union[str, Path, Image.Image]],
        prompt: str,
        n: int = 1,
        model: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Unified bakeoff generation interface: generate(plate, refs, prompt, n) -> list[Image]."""
        if config is not None:
            self.config = config
            self.client.config = config
            self.client.mock_mode = config.mock_mode

        target_model = model or self.model
        plate_img = _load_pil_image(plate)
        ref_imgs = [_load_pil_image(r) for r in refs]
        input_images = [plate_img] + ref_imgs

        orig_model = self.config.image_model
        try:
            self.config.image_model = target_model
            logger.info("GeminiAdapter generating %d image(s) with model '%s'...", n, target_model)
            images = self.client.image_edit(
                images=input_images,
                prompt=prompt,
                n=n,
                **kwargs,
            )
            return images
        finally:
            self.config.image_model = orig_model

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Invokes Gemini text generation with exponential backoff retries."""
        result = self.client.text(prompt=prompt, system=system_instruction)
        return str(result)

    def generate_image(self, prompt: str, aspect_ratio: str = "3:4", **kwargs: Any) -> List[Image.Image]:
        """Invokes Imagen 3 image generation with provenance watermark preservation."""
        dummy_base = Image.new("RGB", (768, 1024), color=(250, 250, 250))
        return self.generate(plate=dummy_base, refs=[], prompt=prompt, n=1, **kwargs)

    def evaluate_image(self, image: Image.Image, criteria: str) -> Dict[str, Any]:
        """Invokes Gemini vision for QA evaluation."""
        result = self.client.vision(images=[image], prompt=criteria)
        if isinstance(result, dict):
            return result
        return {"evaluation": str(result)}


def generate(
    plate: Union[str, Path, Image.Image],
    refs: List[Union[str, Path, Image.Image]],
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    config: Optional[Config] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> List[Image.Image]:
    """Module-level function matching generate(plate, refs, prompt, n) -> list[Image]."""
    adapter = GeminiAdapter(api_key=api_key, model=model, config=config)
    return adapter.generate(plate=plate, refs=refs, prompt=prompt, n=n, model=model, config=config, **kwargs)


if __name__ == "__main__":
    print("--- Demo: src.adapters.gemini ---")
    adapter = GeminiAdapter()
    print("GeminiAdapter initialized successfully.")
