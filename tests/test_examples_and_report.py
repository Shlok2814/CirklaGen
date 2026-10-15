"""Unit tests for scripts/run_examples.py and scripts/build_report.py."""

import csv
from pathlib import Path
from PIL import Image
import pytest

from scripts.build_report import build_report, generate_markdown_report
from scripts.run_examples import (
    EXAMPLE_SCENARIOS,
    create_example_side_by_side_panel,
    run_all_examples,
)
from src.config import load_config


def test_example_scenarios_definition():
    """Verify that EXAMPLE_SCENARIOS contains exactly 5 shot types."""
    assert len(EXAMPLE_SCENARIOS) == 5

    expected_shot_types = {
        "front_chest",
        "pov_hands",
        "side_profile",
        "cart_topdown",
        "closeup_hands",
    }
    scen_types = {s["shot_type"] for s in EXAMPLE_SCENARIOS}
    assert scen_types == expected_shot_types

    for scen in EXAMPLE_SCENARIOS:
        assert "prompt" in scen and len(scen["prompt"].strip()) > 0
        assert "shot_type" in scen
        assert "filename" in scen and scen["filename"].endswith(".png")


def test_create_example_side_by_side_panel(tmp_path):
    """Verify side-by-side panel creation produces valid 5-panel PNG."""
    # Create dummy plate image
    plate_path = tmp_path / "dummy_plate.png"
    Image.new("RGB", (400, 400), color=(180, 180, 180)).save(plate_path)

    # Create dummy tray ref
    tray_ref = tmp_path / "dummy_tray.png"
    Image.new("RGBA", (200, 200), color=(210, 190, 160, 255)).save(tray_ref)

    # Create dummy best candidate
    best_img = tmp_path / "dummy_best.png"
    Image.new("RGB", (400, 400), color=(100, 150, 200)).save(best_img)

    out_panel = tmp_path / "panel_test.png"

    qa_dummy = {
        "passed": True,
        "overall_score": 0.88,
        "similarity_max": 0.35,
        "similarity_mean": 0.20,
        "checklist": {
            "exterior_beige": True,
            "scalloped_ribs_visible": True,
            "corner_pyramids_visible": True,
            "rim_intact": True,
            "no_pen_marks": True,
            "label_sleeve_present": True,
            "scale_plausible": True,
            "hands_anatomy_ok": True,
            "looks_like_ai": False,
            "background_consistent": True,
        },
        "notes": "Test audit notes",
    }

    result_path = create_example_side_by_side_panel(
        plate_path=plate_path,
        tray_ref_paths=[tray_ref],
        short_prompt="test prompt",
        shot_type="front_chest",
        expanded_prompt="Expanded prompt text for testing layout and word wrap in panel 3.",
        best_image_path=best_img,
        qa_result=qa_dummy,
        output_path=out_panel,
    )

    assert result_path.is_file()
    with Image.open(result_path) as im:
        assert im.format == "PNG"
        assert im.width > 2000  # 5 panels side by side
        assert im.height > 600


def test_build_report_with_csv(tmp_path):
    """Verify generate_markdown_report parses results.csv and formats summary table."""
    csv_file = tmp_path / "test_results.csv"
    fieldnames = ["prompt", "shot_type", "n_generated", "n_passed", "best_score", "checklist_failures"]

    rows = [
        {
            "prompt": "me holding tray in meat aisle",
            "shot_type": "front_chest",
            "n_generated": 4,
            "n_passed": 4,
            "best_score": 0.888,
            "checklist_failures": "none",
        },
        {
            "prompt": "POV hands holding ribeye tray over refrigerated shelf",
            "shot_type": "pov_hands",
            "n_generated": 4,
            "n_passed": 3,
            "best_score": 0.750,
            "checklist_failures": "hands_anatomy_ok",
        },
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md_report = generate_markdown_report(csv_file)
    assert "# 🛒 CirklaGen Benchmark Evaluation Report" in md_report
    assert "Tray Pass Rate per Shot Type" in md_report
    assert "`front_chest`" in md_report
    assert "`pov_hands`" in md_report
    assert "100.0%" in md_report
    assert "75.0%" in md_report
    assert "hands_anatomy_ok" in md_report

    # Test build_report CLI convenience wrapper
    report_output = build_report(csv_path=csv_file, save_markdown=True)
    assert "front_chest" in report_output
    saved_md = tmp_path / "benchmark_report.md"
    assert saved_md.is_file()


def test_build_report_missing_file_raises():
    """Verify build_report raises FileNotFoundError when csv is missing."""
    with pytest.raises(FileNotFoundError):
        generate_markdown_report(Path("non_existent_results_path_12345.csv"))


def test_existing_examples_output():
    """Verify that existing outputs/examples/ contains valid results.csv and panels."""
    results_csv = Path("outputs/examples/results.csv")
    assert results_csv.is_file(), "outputs/examples/results.csv must exist"

    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_cols = ["prompt", "shot_type", "n_generated", "n_passed", "best_score", "checklist_failures"]
        assert reader.fieldnames == expected_cols

        rows = list(reader)
        assert len(rows) == 5
        shot_types = [r["shot_type"] for r in rows]
        for st in ["front_chest", "pov_hands", "side_profile", "cart_topdown", "closeup_hands"]:
            assert st in shot_types

    # Verify that panels exist and are valid PNG images
    for scen in EXAMPLE_SCENARIOS:
        panel_file = Path("outputs/examples") / scen["filename"]
        assert panel_file.is_file(), f"Expected panel file {panel_file} missing"
        with Image.open(panel_file) as im:
            assert im.format == "PNG"
            assert im.width > 2000
