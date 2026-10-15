"""Multi-model bakeoff benchmarking script for CirklaGen.

Compares candidate generative models specified in config.yaml under bakeoff_models:
- Imagen 3 (Google GenAI)
- Gemini 2.5 Flash / Pro
- Fal.ai (Flux)
- OpenAI (DALL-E)

Runs one fixed scene (one plate, the same tray references, and the same final_prompt)
through each model using the unified interface in src/adapters/:
    generate(plate, refs, prompt, n) -> list[Image]

Generates 4 images per model, evaluates each candidate with the QA audit suite,
measures latency, handles unconfigured models gracefully by skipping them,
and writes outputs/examples/bakeoff/bakeoff.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.adapters import (
    NotConfigured,
    NotConfiguredError,
    generate as unified_generate,
    get_adapter_for_model,
)
from src.client import GenAIClient
from src.config import Config, load_config
from src.generator import select_tray_references
from src.logger import logger
from src.finish import replace_packaged_tray
from src.qa import QualityAssurance, evaluate_image


# ==============================================================================
# FIXED BENCHMARK SCENE DEFINITION
# ==============================================================================

DEFAULT_PROMPT = (
    "The supplied tray images show the exact Cirkla molded-fiber meat tray to place into the scene. "
    "Keep the background unchanged from the plate. A candid phone photo of a shopper holding the "
    "Cirkla beige molded-fiber meat tray at chest level in Trader Joe's meat aisle."
)


def get_fixed_scene(config: Config) -> Tuple[Path, List[Path], str]:
    """Retrieves fixed benchmark scene components: (plate_path, tray_refs, final_prompt)."""
    plates_dir = Path(config.paths.plates_dir)
    target_plate = plates_dir / "Trader Joe's Shelf.jpeg"
    if not target_plate.is_file():
        avail_plates = sorted(
            list(plates_dir.glob("*.png")) + list(plates_dir.glob("*.jpeg")) + list(plates_dir.glob("*.jpg"))
        )
        target_plate = avail_plates[0] if avail_plates else plates_dir / "default.png"

    tray_refs = select_tray_references(
        shot_type="front_chest",
        trays_dir=config.paths.trays_dir,
        angles_file=Path(config.paths.trays_dir) / "angles.json",
        config=config,
    )

    return target_plate, tray_refs, DEFAULT_PROMPT


def sanitize_filename(name: str) -> str:
    """Sanitizes model name for use in filenames."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")


# ==============================================================================
# BAKEOFF EXECUTION
# ==============================================================================

def run_bakeoff(
    config_path: Union[str, Path] = "config.yaml",
    n: int = 4,
    mock: bool = False,
    output_dir: Union[str, Path] = "outputs/examples/bakeoff",
    models: Optional[List[str]] = None,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Runs the multi-model bakeoff comparison and persists bakeoff.csv."""
    cfg = load_config(config_path, require_api_key=False)
    if mock:
        cfg.mock_mode = True

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    client = GenAIClient(config=cfg)
    plate_path, tray_refs, fixed_prompt = get_fixed_scene(cfg)

    # Models to test
    eval_models = models or cfg.get("bakeoff_models", ["imagen-3.0-generate-002", "gemini-2.5-flash", "fal-ai/flux/dev"])

    print(f"\n{'='*75}")
    print(f"CIRKLAGEN MULTI-MODEL BAKEOFF BENCHMARK (n={n}, mock={cfg.mock_mode})")
    print(f"{'='*75}")
    print(f"Fixed Plate:      {plate_path.name}")
    print(f"Tray References:  {[r.name for r in tray_refs]}")
    print(f"Prompt:           \"{fixed_prompt[:64]}...\"")
    print(f"Candidate Models: {eval_models}")
    print(f"{'='*75}\n")

    results_rows: List[Dict[str, Any]] = []

    for idx, model_name in enumerate(eval_models, start=1):
        print(f"[{idx}/{len(eval_models)}] Evaluating Model: '{model_name}'...")
        start_time = time.perf_counter()

        try:
            # 1. Route to adapter and generate images
            adapter_mod = get_adapter_for_model(model_name)
            logger.info("Calling adapter '%s' for model '%s'...", adapter_mod.__name__, model_name)

            images = adapter_mod.generate(
                plate=plate_path,
                refs=tray_refs,
                prompt=fixed_prompt,
                n=n,
                model=model_name,
                config=cfg,
            )

            elapsed = time.perf_counter() - start_time
            avg_latency = elapsed / max(len(images), 1)

            if not images:
                raise RuntimeError(f"Model '{model_name}' returned 0 images.")

            # 2. Save generated images and run QA audit on each
            qa_results: List[Any] = []
            safe_model_tag = sanitize_filename(model_name)

            packaged_cutout_path = Path(cfg.paths.packaged_dir) / "cirkla_packaged_tray.png"
            if not packaged_cutout_path.is_file():
                packaged_cutout_path = Path("assets/packaged/cirkla_packaged_tray.png")
            packaged_cutout = Image.open(packaged_cutout_path) if packaged_cutout_path.is_file() else None

            for img_idx, img in enumerate(images, start=1):
                if packaged_cutout is not None:
                    try:
                        img = replace_packaged_tray(image=img, package_cutout=packaged_cutout, client=client)
                    except Exception as exc:
                        logger.warning("Could not replace package in bakeoff: %s", exc)

                img_save_path = out_path / f"{safe_model_tag}_cand_{img_idx}.png"
                img.save(img_save_path, format="PNG")

                logger.info("Auditing QA for '%s' candidate %d/%d...", model_name, img_idx, len(images))
                qa_res = evaluate_image(img, config=cfg, client=client)
                qa_results.append(qa_res)

            # 3. Aggregate metrics
            passed_count = sum(1 for q in qa_results if getattr(q, "passed", False))
            pass_rate = round(passed_count / float(len(images)), 3)

            sim_means = [getattr(q, "similarity_mean", 0.0) for q in qa_results]
            mean_similarity = round(float(np.mean(sim_means)), 3) if sim_means else 0.0
            avg_latency_sec = round(avg_latency, 2)
            notes_str = f"{passed_count}/{len(images)} passed QA audit (threshold >= {cfg.qa_threshold})"

            print(f"      Status:            SUCCESS")
            print(f"      Pass Rate:         {pass_rate * 100:.1f}% ({passed_count}/{len(images)})")
            print(f"      Mean Similarity:   {mean_similarity:.3f}")
            print(f"      Avg Latency:       {avg_latency_sec:.2f}s/image\n")

            results_rows.append({
                "model": model_name,
                "pass_rate": pass_rate,
                "mean_similarity": mean_similarity,
                "avg_latency_seconds": avg_latency_sec,
                "notes": notes_str,
            })

        except (NotConfigured, NotConfiguredError) as nc_err:
            elapsed = time.perf_counter() - start_time
            logger.warning("Model '%s' skipped (not configured): %s", model_name, nc_err)
            print(f"      Status:            SKIPPED (NotConfigured: {nc_err})\n")
            results_rows.append({
                "model": model_name,
                "pass_rate": 0.0,
                "mean_similarity": 0.0,
                "avg_latency_seconds": 0.0,
                "notes": f"Skipped: {nc_err}",
            })

        except Exception as exc:
            logger.error("Model '%s' evaluation failed: %s", model_name, exc, exc_info=True)
            print(f"      Status:            FAILED ({exc})\n")
            results_rows.append({
                "model": model_name,
                "pass_rate": 0.0,
                "mean_similarity": 0.0,
                "avg_latency_seconds": 0.0,
                "notes": f"Failed: {exc}",
            })

    # ==========================================================================
    # SAVE BAKEOFF.CSV
    # ==========================================================================
    csv_fields = ["model", "pass_rate", "mean_similarity", "avg_latency_seconds", "notes"]

    csv_destinations = [
        out_path / "bakeoff.csv",
        Path("outputs/examples/bakeoff.csv"),
        Path("bakeoff.csv"),
    ]

    primary_csv = csv_destinations[0]
    for dest in csv_destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for r in results_rows:
                writer.writerow(r)

    # ==========================================================================
    # PRINT RESULTS TABLE & CSV TO STDOUT
    # ==========================================================================
    print(f"{'='*75}")
    print("CIRKLAGEN BAKEOFF BENCHMARK RESULTS (CSV)")
    print(f"{'='*75}")
    with open(primary_csv, "r", encoding="utf-8") as f:
        print(f.read().strip())
    print(f"{'='*75}")

    print("\nSUMMARY TABLE:")
    print(f"| {'Model':<26} | {'Pass Rate':<10} | {'Mean Sim':<10} | {'Latency (s)':<12} | {'Notes':<30} |")
    print(f"|{'-'*28}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*32}|")
    for r in results_rows:
        pr_display = f"{float(r['pass_rate']) * 100:.1f}%" if r["pass_rate"] > 0 else "0.0%"
        print(
            f"| {r['model']:<26} | {pr_display:<10} | {r['mean_similarity']:<10.3f} | {r['avg_latency_seconds']:<12.2f} | {r['notes']:<30} |"
        )
    print(f"\nBakeoff CSV successfully saved to: {primary_csv}")
    print(f"{'='*75}\n")

    return primary_csv, results_rows


def main():
    parser = argparse.ArgumentParser(description="Multi-model bakeoff benchmarking for CirklaGen.")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode")
    parser.add_argument("--n", type=int, default=4, help="Number of images per model (default: 4)")
    parser.add_argument("--out-dir", type=str, default="outputs/examples/bakeoff", help="Output directory")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to evaluate")

    args = parser.parse_args()

    run_bakeoff(
        n=args.n,
        mock=args.mock,
        output_dir=args.out_dir,
        models=args.models,
    )


if __name__ == "__main__":
    main()
