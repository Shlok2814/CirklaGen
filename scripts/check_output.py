"""Output validation and image metadata inspection script for CirklaGen.

Given an image path, prints its dimensions, format, mode, and all metadata/EXIF fields
so you can inspect what the API actually returned.

Rules:
- Strictly read-only: never modifies or overwrites any image or metadata.
- Preserves all provenance, watermarks, and EXIF tags.
- CLI: python scripts/check_output.py <image_path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import ExifTags, Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def inspect_image(image_path: str | Path) -> Dict[str, Any]:
    """Inspects an image file and extracts all metadata in read-only mode.

    Args:
        image_path: Path to the image file to inspect.

    Returns:
        Dictionary containing size, format, mode, info, EXIF tags, and sidecar data.
    """
    file_path = Path(image_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path.resolve()}")

    file_bytes = file_path.stat().st_size

    # Strictly read-only inspection using context manager
    with Image.open(file_path) as img:
        img_format = img.format
        img_mode = img.mode
        img_width, img_height = img.size
        raw_info = dict(img.info) if hasattr(img, "info") and img.info else {}

        # Extract EXIF tags
        exif_data: Dict[str, Any] = {}
        try:
            exif_obj = img.getexif()
            if exif_obj:
                for tag_id, value in exif_obj.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Convert bytes to string for readability if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="replace")
                        except Exception:
                            value = f"<{len(value)} raw bytes>"
                    exif_data[tag_name] = value
        except Exception as e:
            exif_data["_exif_error"] = str(e)

        # Extract PNG text chunks
        png_text: Dict[str, str] = {}
        if hasattr(img, "text") and isinstance(img.text, dict):
            png_text = dict(img.text)

    # Check for corresponding sidecar JSON (img_<i>.json)
    sidecar_path = file_path.with_suffix(".json")
    sidecar_data: Optional[Dict[str, Any]] = None
    if sidecar_path.is_file():
        try:
            with open(sidecar_path, "r", encoding="utf-8") as sf:
                sidecar_data = json.load(sf)
        except Exception as e:
            sidecar_data = {"_error": f"Failed to parse sidecar JSON: {e}"}

    report = {
        "file_path": str(file_path.resolve()),
        "file_name": file_path.name,
        "file_size_bytes": file_bytes,
        "file_size_kb": round(file_bytes / 1024, 2),
        "dimensions": {
            "width": img_width,
            "height": img_height,
            "aspect_ratio": f"{round(img_width / img_height, 2)}:1" if img_height > 0 else "unknown",
        },
        "format": img_format,
        "mode": img_mode,
        "info": {k: str(v) for k, v in raw_info.items() if k != "exif"},
        "exif": exif_data,
        "png_text": png_text,
        "sidecar": sidecar_data,
    }
    return report


def print_inspection_report(report: Dict[str, Any]) -> None:
    """Prints a structured, human-readable terminal report of image metadata."""
    print("\n" + "=" * 70)
    print(f"CIRKLAGEN OUTPUT INSPECTION: {report['file_name']}")
    print("=" * 70)
    print(f"Path:         {report['file_path']}")
    print(f"File Size:    {report['file_size_kb']} KB ({report['file_size_bytes']} bytes)")
    dims = report["dimensions"]
    print(f"Dimensions:   {dims['width']} x {dims['height']} px (Aspect Ratio: {dims['aspect_ratio']})")
    print(f"Format:       {report['format']}")
    print(f"Color Mode:   {report['mode']}")

    # Print Image Info Dictionary
    print("\n[Image Info Properties]")
    if report["info"]:
        for k, v in report["info"].items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            print(f"  - {k}: {val_str}")
    else:
        print("  (None)")

    # Print EXIF Metadata
    print("\n[EXIF Metadata Tags]")
    if report["exif"]:
        for k, v in report["exif"].items():
            print(f"  - {k}: {v}")
    else:
        print("  (No EXIF metadata found)")

    # Print PNG Text Chunks
    if report["png_text"]:
        print("\n[PNG Text Chunks]")
        for k, v in report["png_text"].items():
            print(f"  - {k}: {v}")

    # Print Sidecar JSON Info
    if report["sidecar"]:
        sc = report["sidecar"]
        print("\n[Generation Sidecar JSON]")
        print(f"  - Model:     {sc.get('model', 'N/A')}")
        print(f"  - Shot Type: {sc.get('shot_type', 'N/A')}")
        print(f"  - Seed:      {sc.get('seed', 'N/A')}")
        print(f"  - Timestamp: {sc.get('timestamp', 'N/A')}")
        refs = sc.get("reference_files", {})
        if refs:
            print(f"  - Plate:     {Path(refs.get('plate', '')).name}")
            trays = [Path(t).name for t in refs.get("trays", [])]
            print(f"  - Tray Refs: {', '.join(trays)}")
        prompt_snippet = sc.get("prompt", "")
        if len(prompt_snippet) > 120:
            prompt_snippet = prompt_snippet[:117] + "..."
        print(f"  - Prompt:    {prompt_snippet}")

    print("=" * 70 + "\n")


def find_latest_run_image(runs_dir: str | Path = "outputs/runs") -> Optional[Path]:
    """Finds the most recently created image in outputs/runs/."""
    r_dir = Path(runs_dir)
    if not r_dir.is_dir():
        return None
    images = list(r_dir.rglob("*.png")) + list(r_dir.rglob("*.jpg")) + list(r_dir.rglob("*.jpeg"))
    if not images:
        return None
    return max(images, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect image file metadata, dimensions, mode, and EXIF data (strictly read-only)."
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the image to inspect. If omitted, inspects the latest image in outputs/runs.",
    )

    args = parser.parse_args()

    target_path_str = args.image_path
    if not target_path_str:
        latest = find_latest_run_image()
        if latest:
            print(f"No image path specified. Inspecting latest run image: {latest}")
            target_path_str = str(latest)
        else:
            print("Error: No image path specified and no generated runs found in outputs/runs/.")
            sys.exit(1)

    try:
        report = inspect_image(target_path_str)
        print_inspection_report(report)
    except Exception as exc:
        print(f"Error inspecting image '{target_path_str}': {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
