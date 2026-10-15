"""Asset preparation module and CLI for CirklaGen.

Handles reference tray conversion, background cutouts, plate normalization,
mark cleaning, and asset inventory:
1. convert: HEIC to PNG conversion using pillow-heif with EXIF rotation and 2048px scaling.
2. cutout: Background removal using rembg, transparent PNG saving, and angles.json initialization.
3. plates: Store plate normalization to max 2048px dimensions.
4. clean_marks: Generative inpainting/removal of pen marks via client.image_edit without overwriting.
5. summary: Tabular asset inventory across repository asset directories.

CLI Usage:
    python -m src.assets_prep convert
    python -m src.assets_prep cutout
    python -m src.assets_prep plates
    python -m src.assets_prep clean_marks <path>
    python -m src.assets_prep summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from PIL import Image, ImageOps
import pillow_heif
import rembg

from src.config import Config, load_config
from src import client

# Configure module-level logger
logger = logging.getLogger("cirklagen.assets_prep")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Register pillow-heif opener for HEIC/HEIF compatibility
pillow_heif.register_heif_opener()

# Allowed tray angle categories
VALID_ANGLES: List[str] = [
    "top",
    "three_quarter",
    "side",
    "underside",
    "base_closeup",
    "in_hand",
]
DEFAULT_ANGLE = "unlabeled"


def resize_longest_side(img: Image.Image, target_longest: int = 2048) -> Image.Image:
    """Resizes an image so its longest dimension is exactly target_longest px."""
    w, h = img.size
    longest = max(w, h)
    if longest <= 0:
        return img
    if longest != target_longest:
        scale = target_longest / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def resize_max_dimension(img: Image.Image, max_dim: int = 2048) -> Image.Image:
    """Downscales an image so its longest dimension does not exceed max_dim px."""
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def convert(
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    target_longest: int = 2048,
    config: Optional[Config] = None,
) -> List[Path]:
    """Converts every HEIC image in raw_trays to PNG with EXIF rotation and 2048px resize.

    Saves results to assets/trays_work/.
    """
    cfg = config or load_config(require_api_key=False)
    in_path = Path(input_dir or cfg.paths.raw_trays_dir)
    out_path = Path(output_dir or cfg.paths.trays_work_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        logger.warning("Input raw trays directory does not exist: %s", in_path)
        return []

    # Find all HEIC/HEIF files
    heic_files = sorted(
        [
            p
            for p in in_path.iterdir()
            if p.is_file() and p.suffix.lower() in (".heic", ".heif")
        ],
        key=lambda p: p.name.lower(),
    )

    if not heic_files:
        logger.info("No HEIC files found in %s", in_path)
        return []

    logger.info("Converting %d HEIC file(s) from %s -> %s", len(heic_files), in_path, out_path)
    converted_paths: List[Path] = []

    for src_file in heic_files:
        try:
            with Image.open(src_file) as raw_img:
                # Correct orientation according to EXIF tags
                img_fixed = ImageOps.exif_transpose(raw_img)
                if img_fixed is None:
                    img_fixed = raw_img

                # Resize so longest side is target_longest
                img_resized = resize_longest_side(img_fixed, target_longest=target_longest)

                # Ensure RGB or RGBA mode for PNG encoding
                if img_resized.mode not in ("RGB", "RGBA"):
                    img_resized = img_resized.convert("RGB")

                dest_file = out_path / f"{src_file.stem}.png"
                img_resized.save(dest_file, format="PNG")
                converted_paths.append(dest_file)
                logger.info("Converted: %s -> %s (%dx%d)", src_file.name, dest_file.name, *img_resized.size)
        except Exception as e:
            logger.error("Failed to convert %s: %s", src_file, e)
            raise

    logger.info("Successfully converted %d files to %s", len(converted_paths), out_path)
    return converted_paths


def cutout(
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    config: Optional[Config] = None,
) -> List[Path]:
    """Removes background from tray images using rembg and saves transparent PNGs.

    Outputs to assets/trays/ as tray_<angle>_<n>.png.
    Generates/updates assets/trays/angles.json with 'unlabeled' values and prints instructions.
    """
    cfg = config or load_config(require_api_key=False)
    in_path = Path(input_dir or cfg.paths.trays_work_dir)
    out_path = Path(output_dir or cfg.paths.trays_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # If trays_work_dir does not exist or has no images, fallback to raw_trays_dir if available
    input_candidates = []
    if in_path.exists():
        input_candidates = sorted(
            [
                p
                for p in in_path.iterdir()
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".heif")
            ],
            key=lambda p: p.name.lower(),
        )

    if not input_candidates:
        fallback_path = Path(cfg.paths.raw_trays_dir)
        if fallback_path.exists() and fallback_path != in_path:
            logger.warning(
                "No images in %s. Please run 'convert' first, or falling back to %s",
                in_path,
                fallback_path,
            )
            input_candidates = sorted(
                [
                    p
                    for p in fallback_path.iterdir()
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".heif")
                ],
                key=lambda p: p.name.lower(),
            )

    if not input_candidates:
        logger.warning("No input images found for cutout in %s", in_path)
        return []

    # Initialize rembg session
    active_model = model_name or cfg.rembg_model or "u2net"
    logger.info("Initializing rembg session with model: %s", active_model)
    session = rembg.new_session(active_model)

    # Read existing angles.json if present to preserve user's manual edits
    angles_file = out_path / "angles.json"
    angles_map: Dict[str, str] = {}
    if angles_file.exists():
        try:
            with open(angles_file, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
                if isinstance(loaded, dict):
                    angles_map = loaded
        except Exception as e:
            logger.warning("Could not parse existing %s: %s", angles_file, e)

    cutout_paths: List[Path] = []
    for idx, src_img_path in enumerate(input_candidates, start=1):
        target_filename = f"tray_{DEFAULT_ANGLE}_{idx}.png"
        dest_file = out_path / target_filename

        try:
            with Image.open(src_img_path) as src_img:
                src_fixed = ImageOps.exif_transpose(src_img)
                if src_fixed is None:
                    src_fixed = src_img

                # Remove background with rembg
                cutout_img = rembg.remove(src_fixed, session=session)
                if cutout_img.mode != "RGBA":
                    cutout_img = cutout_img.convert("RGBA")

                cutout_img.save(dest_file, format="PNG")
                cutout_paths.append(dest_file)

                # Initialize in angles mapping if not already labeled
                if target_filename not in angles_map:
                    angles_map[target_filename] = DEFAULT_ANGLE

                logger.info(
                    "Cutout saved: %s -> %s (%dx%d)",
                    src_img_path.name,
                    dest_file.name,
                    *cutout_img.size,
                )
        except Exception as e:
            logger.error("Failed background removal on %s: %s", src_img_path, e)
            raise

    # Write angles.json
    try:
        with open(angles_file, "w", encoding="utf-8") as fp:
            json.dump(angles_map, fp, indent=2)
        logger.info("Generated/Updated %s with %d entries", angles_file, len(angles_map))
    except Exception as e:
        logger.error("Failed to write %s: %s", angles_file, e)
        raise

    # Print clear human instructions for manual labeling
    _print_angle_labeling_instructions(angles_file, list(angles_map.keys()))

    return cutout_paths


def _print_angle_labeling_instructions(angles_file: Path, filenames: Sequence[str]) -> None:
    """Prints terminal instructions for manual angle classification."""
    border = "=" * 80
    print("\n" + border)
    print("ACTION REQUIRED: Tray Angle Visual Review & Labeling")
    print(border)
    print(f"File created: {angles_file.resolve()}")
    print("\nPlease inspect each transparent cutout in assets/trays/ by eye,")
    print("open angles.json, and replace 'unlabeled' with one of the allowed angle values:\n")
    for angle in VALID_ANGLES:
        print(f"    - {angle}")
    print("\nExample configuration snippet:")
    print("    {")
    sample_files = list(filenames[:3]) if filenames else ["tray_unlabeled_1.png", "tray_unlabeled_2.png"]
    for i, fname in enumerate(sample_files):
        sample_angle = VALID_ANGLES[i % len(VALID_ANGLES)]
        comma = "," if i < len(sample_files) - 1 else ""
        print(f'      "{fname}": "{sample_angle}"{comma}')
    print("    }")
    print(border + "\n")


def plates(
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    max_dim: int = 2048,
    config: Optional[Config] = None,
) -> List[Path]:
    """Copies images from plates_raw to plates, resized to max 2048px, with no other changes."""
    cfg = config or load_config(require_api_key=False)
    in_path = Path(input_dir or cfg.paths.plates_raw_dir)
    out_path = Path(output_dir or cfg.paths.plates_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        logger.warning("Input raw plates directory does not exist: %s", in_path)
        return []

    # Find all images
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    plate_files = sorted(
        [p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions],
        key=lambda p: p.name.lower(),
    )

    if not plate_files:
        logger.info("No plate images found in %s", in_path)
        return []

    logger.info("Processing %d plate(s) from %s -> %s", len(plate_files), in_path, out_path)
    processed_paths: List[Path] = []

    for src_file in plate_files:
        try:
            with Image.open(src_file) as img:
                # Correct orientation
                img_fixed = ImageOps.exif_transpose(img)
                if img_fixed is None:
                    img_fixed = img

                # Resize to max 2048px if larger; if <= 2048px, make no dimension changes
                w, h = img_fixed.size
                if max(w, h) > max_dim:
                    img_final = resize_max_dimension(img_fixed, max_dim=max_dim)
                else:
                    img_final = img_fixed

                # Determine target filename and format
                ext = src_file.suffix.lower()
                if ext in (".heic", ".heif"):
                    dest_file = out_path / f"{src_file.stem}.png"
                    save_format = "PNG"
                    save_kwargs = {}
                else:
                    dest_file = out_path / src_file.name
                    save_format = img.format or ("JPEG" if ext in (".jpg", ".jpeg") else "PNG")
                    save_kwargs = {"quality": 95} if save_format == "JPEG" else {}

                img_final.save(dest_file, format=save_format, **save_kwargs)
                processed_paths.append(dest_file)
                logger.info(
                    "Plate prepared: %s -> %s (%dx%d)",
                    src_file.name,
                    dest_file.name,
                    *img_final.size,
                )
        except Exception as e:
            logger.error("Failed to process plate %s: %s", src_file, e)
            raise

    logger.info("Successfully processed %d plates in %s", len(processed_paths), out_path)
    return processed_paths


def clean_marks(
    path: Union[str, Path],
    config: Optional[Config] = None,
) -> Path:
    """Uses client.image_edit to remove pen marks on the tray, changing nothing else.

    Saves as <name>_clean.png in the same directory.
    Never overwrites an existing file.
    """
    in_file = Path(path).resolve()
    if not in_file.is_file():
        raise FileNotFoundError(f"Source image not found: {in_file}")

    out_file = in_file.parent / f"{in_file.stem}_clean.png"
    if out_file.exists():
        raise FileExistsError(
            f"Target clean image already exists and will not be overwritten: {out_file}"
        )

    cfg = config or load_config(require_api_key=False)
    instruction = "remove the pen marks on the tray, change nothing else"

    logger.info("Loading image for mark cleaning: %s", in_file)
    with Image.open(in_file) as raw_img:
        img_fixed = ImageOps.exif_transpose(raw_img)
        if img_fixed is None:
            img_fixed = raw_img.copy()
        else:
            img_fixed = img_fixed.copy()

    logger.info("Calling client.image_edit with instruction: '%s'", instruction)
    edited_images = client.image_edit(
        images=img_fixed,
        prompt=instruction,
        n=1,
        config=cfg,
    )

    if not edited_images:
        raise RuntimeError("client.image_edit returned an empty list of images.")

    clean_img = edited_images[0]
    clean_img.save(out_file, format="PNG")
    logger.info("Successfully saved cleaned image: %s (%dx%d)", out_file, *clean_img.size)
    return out_file


def summary(
    assets_dir: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None,
) -> List[Dict[str, Any]]:
    """Gathers and prints a table of every asset, dimensions, mode, and file size."""
    cfg = config or load_config(require_api_key=False)
    root_path = Path(assets_dir or "assets").resolve()

    if not root_path.exists():
        print(f"Assets directory not found: {root_path}")
        return []

    valid_extensions = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp", ".bmp", ".tiff"}
    records: List[Dict[str, Any]] = []

    for dirpath, _, filenames in os.walk(root_path):
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() not in valid_extensions:
                continue

            try:
                stat = fpath.stat()
                file_size_bytes = stat.st_size
                if file_size_bytes >= 1024 * 1024:
                    size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
                else:
                    size_str = f"{file_size_bytes / 1024:.1f} KB"

                with Image.open(fpath) as img:
                    w, h = img.size
                    mode = img.mode
                    fmt = img.format or fpath.suffix.upper().lstrip(".")

                try:
                    rel_path = str(fpath.relative_to(Path.cwd()))
                except ValueError:
                    rel_path = str(fpath)

                records.append(
                    {
                        "Asset": rel_path,
                        "Dimensions": f"{w}x{h}",
                        "Mode": mode,
                        "Format": fmt,
                        "File Size": size_str,
                        "_bytes": file_size_bytes,
                    }
                )
            except Exception as e:
                logger.debug("Could not inspect asset %s: %s", fpath, e)

    # Format table for terminal output
    if records:
        df = pd.DataFrame(records)[["Asset", "Dimensions", "Mode", "Format", "File Size"]]
        print("\n" + "=" * 90)
        print("CIRKLAGEN ASSET INVENTORY SUMMARY")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)
        print(f"Total Assets: {len(records)}\n")
    else:
        print(f"\nNo image assets found in {root_path}\n")

    return records


class AssetsPrep:
    """Object-oriented interface for asset preparation workflows."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config(require_api_key=False)

    def convert(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> List[Path]:
        return convert(input_dir=input_dir, output_dir=output_dir, config=self.config)

    def cutout(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> List[Path]:
        return cutout(input_dir=input_dir, output_dir=output_dir, config=self.config)

    def plates(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> List[Path]:
        return plates(input_dir=input_dir, output_dir=output_dir, config=self.config)

    def clean_marks(self, path: Union[str, Path]) -> Path:
        return clean_marks(path=path, config=self.config)

    def summary(self, assets_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        return summary(assets_dir=assets_dir, config=self.config)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for assets_prep."""
    parser = argparse.ArgumentParser(
        prog="python -m src.assets_prep",
        description="CirklaGen Asset Preparation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # convert
    convert_p = subparsers.add_parser("convert", help="Convert raw HEIC trays to 2048px PNG in trays_work")
    convert_p.add_argument("--input-dir", type=str, default=None, help="Input raw trays directory")
    convert_p.add_argument("--output-dir", type=str, default=None, help="Output trays_work directory")

    # cutout
    cutout_p = subparsers.add_parser("cutout", help="Remove background with rembg and generate angles.json")
    cutout_p.add_argument("--input-dir", type=str, default=None, help="Input work trays directory")
    cutout_p.add_argument("--output-dir", type=str, default=None, help="Output transparent trays directory")
    cutout_p.add_argument("--model", type=str, default=None, help="Rembg model name (default: from config)")

    # plates
    plates_p = subparsers.add_parser("plates", help="Copy and resize raw plates to max 2048px")
    plates_p.add_argument("--input-dir", type=str, default=None, help="Input raw plates directory")
    plates_p.add_argument("--output-dir", type=str, default=None, help="Output plates directory")

    # clean_marks
    clean_p = subparsers.add_parser(
        "clean_marks",
        aliases=["clean-marks"],
        help="Remove pen marks on tray using client.image_edit",
    )
    clean_p.add_argument("path", type=str, help="Path to tray image file with pen marks")

    # summary
    summary_p = subparsers.add_parser("summary", help="Print table of all assets, dimensions, and modes")
    summary_p.add_argument("--assets-dir", type=str, default=None, help="Assets root directory to scan")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point for python -m src.assets_prep <command>."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = args.command
    if cmd in ("clean-marks", "clean_marks"):
        clean_marks(args.path)
    elif cmd == "convert":
        convert(input_dir=args.input_dir, output_dir=args.output_dir)
    elif cmd == "cutout":
        cutout(input_dir=args.input_dir, output_dir=args.output_dir, model_name=getattr(args, "model", None))
    elif cmd == "plates":
        plates(input_dir=args.input_dir, output_dir=args.output_dir)
    elif cmd == "summary":
        summary(assets_dir=getattr(args, "assets_dir", None))
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
