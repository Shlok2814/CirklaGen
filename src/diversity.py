"""Perceptual hashing and batch diversity verification for CirklaGen.

Uses 64-bit 2D Discrete Cosine Transform (DCT) perceptual hashing (pHash)
to measure compositional and framing distance between generated candidate images.
Rejects a candidate batch if any two images have a perceptual-hash Hamming distance
below a specified threshold (e.g. duplicate or near-identical framing/pose).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.logger import logger


class BatchDiversityError(ValueError):
    """Raised when a batch of images fails the perceptual hash diversity check."""
    pass


@dataclass
class BatchDiversityResult:
    """Detailed results of a batch perceptual diversity audit."""
    passed: bool
    min_distance: int
    threshold: int
    violating_pair: Optional[Tuple[int, int]]
    pairwise_distances: List[Dict[str, Any]]
    hashes: List[str]
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result to a clean dictionary."""
        return {
            "passed": self.passed,
            "min_distance": self.min_distance,
            "threshold": self.threshold,
            "violating_pair": list(self.violating_pair) if self.violating_pair else None,
            "pairwise_distances": self.pairwise_distances,
            "hashes": self.hashes,
            "reason": self.reason,
        }


def compute_phash(image: Union[Image.Image, Path, str], hash_size: int = 8) -> int:
    """Computes a 64-bit 2D DCT perceptual hash (pHash) for an image.

    Steps:
    1. Standardize image to grayscale.
    2. Resize to (hash_size * 4, hash_size * 4) = 32x32 using area interpolation.
    3. Compute 2D Discrete Cosine Transform (DCT).
    4. Extract low-frequency 8x8 corner (excluding DC component).
    5. Compare against the median coefficient to form a 64-bit integer hash.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    else:
        pil_img = image

    gray_np = np.array(pil_img.convert("L"), dtype=np.uint8)
    sample_dim = hash_size * 4
    resized = cv2.resize(gray_np, (sample_dim, sample_dim), interpolation=cv2.INTER_AREA)

    # 2D DCT on float32 representation
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]

    # Exclude DC component (0, 0) for median calculation to focus on spatial structure
    med = float(np.median(dct_low.flatten()[1:]))
    diff = dct_low > med

    h = 0
    for b in diff.flatten():
        h = (h << 1) | int(b)
    return int(h)


def phash_to_hex(h: int, hash_size: int = 8) -> str:
    """Formats an integer hash into a zero-padded hex string."""
    n_hex = (hash_size * hash_size + 3) // 4
    return f"{h:0{n_hex}x}"


def phash_distance(h1: Union[int, str], h2: Union[int, str]) -> int:
    """Computes the Hamming distance (differing bit count) between two pHashes."""
    val1 = int(h1, 16) if isinstance(h1, str) else int(h1)
    val2 = int(h2, 16) if isinstance(h2, str) else int(h2)
    return bin(val1 ^ val2).count("1")


def check_batch_diversity(
    images: List[Union[Image.Image, Path, str]],
    threshold: int = 8,
    raise_on_failure: bool = False,
) -> BatchDiversityResult:
    """Evaluates pairwise perceptual-hash distances across all images in a batch.

    Rejects the batch if any pair of images has a distance below the threshold.

    Args:
        images: List of PIL Images or image filepaths in the batch.
        threshold: Minimum required Hamming distance (default 8). Distance < threshold fails.
        raise_on_failure: If True, raises BatchDiversityError on failure.

    Returns:
        BatchDiversityResult with pass/fail status, min distance, and pair details.
    """
    n = len(images)
    if n <= 1:
        return BatchDiversityResult(
            passed=True,
            min_distance=64,
            threshold=threshold,
            violating_pair=None,
            pairwise_distances=[],
            hashes=[phash_to_hex(compute_phash(images[0]))] if n == 1 else [],
            reason=None,
        )

    # Compute pHash for each image in the batch
    int_hashes: List[int] = []
    hex_hashes: List[str] = []
    for img in images:
        h_int = compute_phash(img)
        int_hashes.append(h_int)
        hex_hashes.append(phash_to_hex(h_int))

    pairwise: List[Dict[str, Any]] = []
    min_dist = 64
    violating_pair: Optional[Tuple[int, int]] = None

    for i in range(n):
        for j in range(i + 1, n):
            dist = phash_distance(int_hashes[i], int_hashes[j])
            is_below = dist < threshold
            pairwise.append({
                "pair": [i, j],
                "distance": dist,
                "below_threshold": is_below,
            })
            if dist < min_dist:
                min_dist = dist
            if is_below and violating_pair is None:
                violating_pair = (i, j)

    passed = (violating_pair is None)
    reason = None
    if not passed and violating_pair is not None:
        reason = (
            f"Perceptual-hash distance between variation {violating_pair[0]} and variation "
            f"{violating_pair[1]} is {min_dist}, which is below diversity threshold {threshold}."
        )
        logger.warning("Batch diversity check FAILED: %s", reason)
        if raise_on_failure:
            raise BatchDiversityError(reason)
    else:
        logger.info(
            "Batch diversity check PASSED: min_distance=%d >= threshold=%d across %d pairs.",
            min_dist,
            threshold,
            len(pairwise),
        )

    return BatchDiversityResult(
        passed=passed,
        min_distance=min_dist,
        threshold=threshold,
        violating_pair=violating_pair,
        pairwise_distances=pairwise,
        hashes=hex_hashes,
        reason=reason,
    )
