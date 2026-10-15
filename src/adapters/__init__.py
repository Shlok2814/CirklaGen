"""Model backend adapters for CirklaGen.

Provides a unified interface to multiple generative AI providers:
- Gemini / Imagen (google-genai SDK)
- OpenAI (GPT-4o, DALL-E)
- Fal.ai (Flux models for multi-model bakeoff)

All adapters expose:
    generate(plate, refs, prompt, n) -> list[Image]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union
from PIL import Image

from src.adapters.base import NotConfigured, NotConfiguredError
from src.adapters import fal, gemini, openai
from src.adapters.fal import FalAdapter
from src.adapters.gemini import GeminiAdapter
from src.adapters.openai import OpenAIAdapter


def get_adapter_for_model(model_name: str):
    """Returns the adapter module matching the model name string."""
    m = model_name.lower()
    if m.startswith("fal") or "flux" in m:
        return fal
    elif m.startswith("openai") or m.startswith("dall-e") or m.startswith("gpt"):
        return openai
    elif m.startswith("gemini") or m.startswith("imagen"):
        return gemini
    else:
        return gemini


def generate(
    plate: Union[str, Path, Image.Image],
    refs: List[Union[str, Path, Image.Image]],
    prompt: str,
    n: int = 1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> List[Image.Image]:
    """Unified entrypoint: routes generation to the correct adapter.

    Signature: generate(plate, refs, prompt, n) -> list[Image]
    """
    target_model = model or "imagen-3.0-generate-002"
    adapter_mod = get_adapter_for_model(target_model)
    return adapter_mod.generate(
        plate=plate,
        refs=refs,
        prompt=prompt,
        n=n,
        model=target_model,
        **kwargs,
    )


__all__ = [
    "gemini",
    "openai",
    "fal",
    "GeminiAdapter",
    "OpenAIAdapter",
    "FalAdapter",
    "NotConfigured",
    "NotConfiguredError",
    "get_adapter_for_model",
    "generate",
]
