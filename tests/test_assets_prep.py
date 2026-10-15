"""Tests for src/assets_prep.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from src.assets_prep import (
    VALID_ANGLES,
    clean_marks,
    convert,
    cutout,
    main,
    plates,
    resize_longest_side,
    resize_max_dimension,
    summary,
)
from src.config import Config


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Fixture providing a test configuration with temporary paths."""
    cfg = Config()
    cfg.mock_mode = True
    cfg.paths.raw_trays_dir = str(tmp_path / "raw_trays")
    cfg.paths.trays_work_dir = str(tmp_path / "trays_work")
    cfg.paths.trays_dir = str(tmp_path / "trays")
    cfg.paths.plates_raw_dir = str(tmp_path / "plates_raw")
    cfg.paths.plates_dir = str(tmp_path / "plates")
    return cfg


def test_resize_longest_side():
    """Verify that images are correctly resized to have longest side equal to target."""
    # Portrait
    img_portrait = Image.new("RGB", (3000, 4000), color="red")
    resized_p = resize_longest_side(img_portrait, target_longest=2048)
    assert max(resized_p.size) == 2048
    assert resized_p.size == (1536, 2048)

    # Landscape
    img_landscape = Image.new("RGB", (4000, 3000), color="blue")
    resized_l = resize_longest_side(img_landscape, target_longest=2048)
    assert max(resized_l.size) == 2048
    assert resized_l.size == (2048, 1536)

    # Already target size
    img_exact = Image.new("RGB", (2048, 1024), color="green")
    resized_e = resize_longest_side(img_exact, target_longest=2048)
    assert resized_e.size == (2048, 1024)


def test_resize_max_dimension():
    """Verify downscaling only happens if dimension exceeds max_dim."""
    # Oversized image
    img_big = Image.new("RGB", (3000, 1500), color="yellow")
    resized_big = resize_max_dimension(img_big, max_dim=2048)
    assert max(resized_big.size) == 2048
    assert resized_big.size == (2048, 1024)

    # Undersized image remains unchanged
    img_small = Image.new("RGB", (800, 600), color="white")
    resized_small = resize_max_dimension(img_small, max_dim=2048)
    assert resized_small.size == (800, 600)


def test_plates_preparation(tmp_path: Path, test_config: Config):
    """Test standardizing plate images: downscaling large plates and preserving small ones."""
    plates_raw = tmp_path / "plates_raw"
    plates_dir = tmp_path / "plates"
    plates_raw.mkdir(parents=True, exist_ok=True)

    # 1. Large image (3000x2000)
    img_large = Image.new("RGB", (3000, 2000), color="red")
    img_large.save(plates_raw / "store_large.jpeg", format="JPEG")

    # 2. Normal image (1200x800)
    img_normal = Image.new("RGB", (1200, 800), color="green")
    img_normal.save(plates_raw / "store_normal.jpeg", format="JPEG")

    processed = plates(
        input_dir=plates_raw,
        output_dir=plates_dir,
        max_dim=2048,
        config=test_config,
    )

    assert len(processed) == 2

    # Check processed dimensions
    with Image.open(plates_dir / "store_large.jpeg") as im:
        assert max(im.size) == 2048
        assert im.size == (2048, 1365)

    with Image.open(plates_dir / "store_normal.jpeg") as im:
        assert im.size == (1200, 800)


def test_clean_marks_flow_and_no_overwrite(tmp_path: Path, test_config: Config):
    """Verify clean_marks calls client in mock mode and never overwrites."""
    test_img_path = tmp_path / "tray_sample.png"
    sample_img = Image.new("RGB", (500, 500), color="gray")
    sample_img.save(test_img_path, format="PNG")

    # First cleaning call: should create tray_sample_clean.png
    out_clean = clean_marks(test_img_path, config=test_config)
    assert out_clean.exists()
    assert out_clean.name == "tray_sample_clean.png"

    # Second cleaning call: must raise FileExistsError and never overwrite
    with pytest.raises(FileExistsError, match="Target clean image already exists"):
        clean_marks(test_img_path, config=test_config)


def test_clean_marks_missing_file(tmp_path: Path, test_config: Config):
    """Verify FileNotFoundError if input file does not exist."""
    missing_path = tmp_path / "non_existent.png"
    with pytest.raises(FileNotFoundError):
        clean_marks(missing_path, config=test_config)


def test_cutout_generation_and_angles_json(tmp_path: Path, test_config: Config):
    """Test cutout process with mocked rembg and verify angles.json structure."""
    work_dir = tmp_path / "trays_work"
    trays_dir = tmp_path / "trays"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create 2 work images
    for i in (1, 2):
        im = Image.new("RGB", (400, 400), color="white")
        im.save(work_dir / f"work_tray_{i}.png", format="PNG")

    # Mock rembg.remove to return RGBA image
    def mock_remove(img, **kwargs):
        return Image.new("RGBA", img.size, (200, 180, 150, 255))

    with patch("rembg.remove", side_effect=mock_remove), patch("rembg.new_session", return_value="dummy_session"):
        cutouts = cutout(input_dir=work_dir, output_dir=trays_dir, config=test_config)

    assert len(cutouts) == 2
    assert (trays_dir / "tray_unlabeled_1.png").exists()
    assert (trays_dir / "tray_unlabeled_2.png").exists()

    # Verify angles.json
    angles_json_file = trays_dir / "angles.json"
    assert angles_json_file.exists()

    with open(angles_json_file, "r", encoding="utf-8") as fp:
        angles_map = json.load(fp)

    assert angles_map.get("tray_unlabeled_1.png") == "unlabeled"
    assert angles_map.get("tray_unlabeled_2.png") == "unlabeled"

    # Verify allowed angles are defined
    assert "top" in VALID_ANGLES
    assert "three_quarter" in VALID_ANGLES
    assert "in_hand" in VALID_ANGLES


def test_summary_inventory(tmp_path: Path, test_config: Config, capsys):
    """Test asset summary inventory report."""
    test_assets = tmp_path / "assets"
    sub = test_assets / "trays"
    sub.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (1024, 768), color=(255, 0, 0, 128))
    img.save(sub / "test_tray.png", format="PNG")

    records = summary(assets_dir=test_assets, config=test_config)
    assert len(records) == 1
    assert records[0]["Dimensions"] == "1024x768"
    assert records[0]["Mode"] == "RGBA"

    captured = capsys.readouterr()
    assert "CIRKLAGEN ASSET INVENTORY SUMMARY" in captured.out
    assert "1024x768" in captured.out


def test_cli_parsing(tmp_path: Path):
    """Test CLI parser accepts each command."""
    sample_file = tmp_path / "sample.png"
    sample_file.write_text("test")

    with patch("src.assets_prep.convert") as mock_conv:
        main(["convert"])
        mock_conv.assert_called_once()

    with patch("src.assets_prep.cutout") as mock_cut:
        main(["cutout"])
        mock_cut.assert_called_once()

    with patch("src.assets_prep.plates") as mock_plates:
        main(["plates"])
        mock_plates.assert_called_once()

    with patch("src.assets_prep.clean_marks") as mock_clean:
        main(["clean_marks", str(sample_file)])
        mock_clean.assert_called_once_with(str(sample_file))

    with patch("src.assets_prep.summary") as mock_sum:
        main(["summary"])
        mock_sum.assert_called_once()
