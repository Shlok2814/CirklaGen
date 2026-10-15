"""Batch runner script for executing CirklaGen across 5 predefined example scenarios.

Runs 5 predefined short prompts (one per shot type) with n=4 through the pipeline.
For each prompt, saves to outputs/examples/ one side-by-side PNG with panels:
[clean plate | tray references used | short prompt & expanded prompt | best output | QA score & checklist ticks]

Also writes outputs/examples/results.csv with columns:
prompt, shot_type, n_generated, n_passed, best_score, checklist_failures
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.client import GenAIClient
from src.config import Config, load_config
from src.logger import logger
from src.pipeline import GenerationPipeline


# ==============================================================================
# 5 CURATED PREDEFINED EXAMPLE SCENARIOS (ONE PER SHOT TYPE)
# ==============================================================================

EXAMPLE_SCENARIOS = [
    {
        "shot_type": "front_chest",
        "prompt": "me holding tray in meat aisle",
        "description": "Shopper holding Cirkla molded-fiber beef package at chest level facing camera",
        "filename": "example_1_front_chest.png",
    },
    {
        "shot_type": "pov_hands",
        "prompt": "POV hands holding ribeye tray over refrigerated shelf",
        "description": "First-person perspective of hands holding packaged beef over cooler",
        "filename": "example_2_pov_hands.png",
    },
    {
        "shot_type": "side_profile",
        "prompt": "customer side profile inspecting Cirkla tray fluting",
        "description": "Shopper viewed from 45-degree angle inspecting sidewall scalloped ribs",
        "filename": "example_3_side_profile.png",
    },
    {
        "shot_type": "cart_topdown",
        "prompt": "looking down into red shopping cart with beef package inside",
        "description": "High-angle top-down view into red wire cart holding meat package",
        "filename": "example_4_cart_topdown.png",
    },
    {
        "shot_type": "closeup_hands",
        "prompt": "close up of hands holding molded fiber tray rim",
        "description": "Macro close-up of fingers grasping wide rim flange and corner pyramids",
        "filename": "example_5_closeup_hands.png",
    },
]


# ==============================================================================
# FONT HELPER
# ==============================================================================

def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Safely loads a clean TrueType font or falls back to default."""
    font_names = ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf", "DejaVuSans.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ==============================================================================
# PANEL COMPOSITOR (5 PANELS SIDE-BY-SIDE)
# ==============================================================================

def create_example_side_by_side_panel(
    plate_path: Path,
    tray_ref_paths: List[Path],
    short_prompt: str,
    shot_type: str,
    expanded_prompt: str,
    best_image_path: Path,
    qa_result: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Builds a cohesive 5-panel side-by-side benchmark image for evaluators.

    Panels:
    1. Clean plate
    2. Tray references used
    3. Short prompt & expanded prompt text
    4. Best output candidate
    5. QA score and checklist ticks
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel_w = 540
    panel_h = 760
    header_h = 90
    margin = 14
    divider_w = 4
    num_panels = 5

    total_w = panel_w * num_panels + divider_w * (num_panels - 1) + margin * 2
    total_h = panel_h + header_h + margin * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(248, 250, 252))  # Clean light background
    draw = ImageDraw.Draw(canvas)

    font_title = get_font(26, bold=True)
    font_subtitle = get_font(15, bold=False)
    font_panel_hdr = get_font(18, bold=True)
    font_body = get_font(14, bold=False)
    font_body_bold = get_font(14, bold=True)
    font_small = get_font(12, bold=False)

    # 1. Header Banner
    draw.rectangle([0, 0, total_w, header_h], fill=(30, 41, 59))  # Dark slate banner
    draw.text((margin + 10, 16), "CIRKLAGEN RETAIL MOCKUP BENCHMARK", font=font_title, fill=(255, 255, 255))
    banner_sub = f"Shot Type: {shot_type.upper()}  |  Prompt: \"{short_prompt}\"  |  Single Source of Truth: knowledge/product.json"
    draw.text((margin + 10, 52), banner_sub, font=font_subtitle, fill=(203, 213, 225))
    draw.text((total_w - 320, 52), "AI-generated mockup for evaluation", font=font_subtitle, fill=(148, 163, 184))

    card_y = header_h + margin
    card_h = panel_h

    # Helper to calculate x offset for each panel
    def get_panel_x(idx: int) -> int:
        return margin + idx * (panel_w + divider_w)

    # Draw vertical divider bars between panels
    for i in range(1, num_panels):
        dx = margin + i * panel_w + (i - 1) * divider_w
        draw.rectangle([dx, card_y, dx + divider_w, card_y + card_h], fill=(226, 232, 240))

    # --------------------------------------------------------------------------
    # PANEL 1: CLEAN PLATE
    # --------------------------------------------------------------------------
    p1_x = get_panel_x(0)
    draw.rectangle([p1_x, card_y, p1_x + panel_w, card_y + card_h], fill=(255, 255, 255), outline=(226, 232, 240))
    # Panel Title Tab
    draw.rectangle([p1_x, card_y, p1_x + panel_w, card_y + 44], fill=(241, 245, 249))
    draw.text((p1_x + 14, card_y + 12), "PANEL 1: Clean Background Plate", font=font_panel_hdr, fill=(15, 23, 42))

    if plate_path and plate_path.is_file():
        plate_img = Image.open(plate_path).convert("RGB")
        # Resize to fit inner panel
        target_img_w = panel_w - 20
        target_img_h = card_h - 100
        p_ratio = min(target_img_w / plate_img.width, target_img_h / plate_img.height)
        nw, nh = int(plate_img.width * p_ratio), int(plate_img.height * p_ratio)
        scaled_plate = plate_img.resize((nw, nh), Image.Resampling.LANCZOS)
        paste_x = p1_x + 10 + (target_img_w - nw) // 2
        paste_y = card_y + 54 + (target_img_h - nh) // 2
        canvas.paste(scaled_plate, (paste_x, paste_y))
        draw.text((p1_x + 14, card_y + card_h - 36), f"Asset: {plate_path.name} ({plate_img.width}x{plate_img.height})", font=font_small, fill=(100, 116, 139))
    else:
        draw.text((p1_x + 20, card_y + 100), "Plate image not found", font=font_body, fill=(239, 68, 68))

    # --------------------------------------------------------------------------
    # PANEL 2: TRAY REFERENCES USED
    # --------------------------------------------------------------------------
    p2_x = get_panel_x(1)
    draw.rectangle([p2_x, card_y, p2_x + panel_w, card_y + card_h], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.rectangle([p2_x, card_y, p2_x + panel_w, card_y + 44], fill=(241, 245, 249))
    draw.text((p2_x + 14, card_y + 12), f"PANEL 2: Tray References ({len(tray_ref_paths)})", font=font_panel_hdr, fill=(15, 23, 42))

    # Tile tray references vertically
    if tray_ref_paths:
        avail_h = (card_h - 60) // len(tray_ref_paths)
        for t_idx, tp in enumerate(tray_ref_paths):
            sub_y = card_y + 50 + t_idx * avail_h
            # Draw sub-card outline
            draw.rectangle([p2_x + 8, sub_y, p2_x + panel_w - 8, sub_y + avail_h - 6], fill=(248, 250, 252), outline=(241, 245, 249))
            if tp.is_file():
                t_img = Image.open(tp)
                if t_img.mode != "RGBA":
                    t_img = t_img.convert("RGBA")
                # Scale
                max_tw = panel_w - 30
                max_th = avail_h - 38
                tr_scale = min(max_tw / t_img.width, max_th / t_img.height)
                tnw, tnh = int(t_img.width * tr_scale), int(t_img.height * tr_scale)
                t_scaled = t_img.resize((tnw, tnh), Image.Resampling.LANCZOS)
                tx = p2_x + 15 + (max_tw - tnw) // 2
                ty = sub_y + 6 + (max_th - tnh) // 2
                canvas.paste(t_scaled, (tx, ty), mask=t_scaled)
                draw.text((p2_x + 14, sub_y + avail_h - 26), f"Ref #{t_idx+1}: {tp.name}", font=font_small, fill=(71, 85, 105))

    # --------------------------------------------------------------------------
    # PANEL 3: SHORT PROMPT & EXPANDED PROMPT TEXT
    # --------------------------------------------------------------------------
    p3_x = get_panel_x(2)
    draw.rectangle([p3_x, card_y, p3_x + panel_w, card_y + card_h], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.rectangle([p3_x, card_y, p3_x + panel_w, card_y + 44], fill=(241, 245, 249))
    draw.text((p3_x + 14, card_y + 12), "PANEL 3: Prompt Grounding", font=font_panel_hdr, fill=(15, 23, 42))

    cur_y = card_y + 56
    # Short Prompt Box
    draw.rectangle([p3_x + 12, cur_y, p3_x + panel_w - 12, cur_y + 70], fill=(241, 245, 249), outline=(226, 232, 240))
    draw.text((p3_x + 20, cur_y + 8), "USER SHORT PROMPT:", font=font_body_bold, fill=(71, 85, 105))
    draw.text((p3_x + 20, cur_y + 32), f"\"{short_prompt}\"", font=font_body, fill=(15, 23, 42))

    cur_y += 84
    # Shot Type & Grounding Constraints
    draw.text((p3_x + 16, cur_y), f"Target Shot Type: {shot_type}", font=font_body_bold, fill=(30, 41, 59))
    cur_y += 26
    draw.text((p3_x + 16, cur_y), "Must Keep: natural beige fiber, fluted ribs, corner pyramids", font=font_small, fill=(100, 116, 139))
    cur_y += 20
    draw.text((p3_x + 16, cur_y), "Must Not: pen marks, white/gray color, floating grip, bad scale", font=font_small, fill=(100, 116, 139))

    cur_y += 32
    # Expanded Prompt Header
    draw.text((p3_x + 16, cur_y), "EXPANDED IMAGE-EDIT PROMPT:", font=font_body_bold, fill=(15, 23, 42))
    cur_y += 24

    # Text wrapping for expanded prompt
    wrapped_lines = textwrap.wrap(expanded_prompt, width=48)
    for line in wrapped_lines[:26]:  # Limit lines to fit panel
        draw.text((p3_x + 18, cur_y), line, font=font_small, fill=(51, 65, 85))
        cur_y += 18

    # --------------------------------------------------------------------------
    # PANEL 4: BEST OUTPUT
    # --------------------------------------------------------------------------
    p4_x = get_panel_x(3)
    # Highlighted card border for best output
    draw.rectangle([p4_x, card_y, p4_x + panel_w, card_y + card_h], fill=(255, 255, 255), outline=(234, 179, 8), width=2)
    draw.rectangle([p4_x, card_y, p4_x + panel_w, card_y + 44], fill=(254, 240, 138))  # Golden tab
    draw.text((p4_x + 14, card_y + 12), "PANEL 4: Best Output (Finished)", font=font_panel_hdr, fill=(113, 63, 18))

    if best_image_path and best_image_path.is_file():
        out_img = Image.open(best_image_path).convert("RGB")
        target_img_w = panel_w - 20
        target_img_h = card_h - 100
        o_ratio = min(target_img_w / out_img.width, target_img_h / out_img.height)
        onw, onh = int(out_img.width * o_ratio), int(out_img.height * o_ratio)
        scaled_out = out_img.resize((onw, onh), Image.Resampling.LANCZOS)
        paste_x = p4_x + 10 + (target_img_w - onw) // 2
        paste_y = card_y + 54 + (target_img_h - onh) // 2
        canvas.paste(scaled_out, (paste_x, paste_y))
        draw.text((p4_x + 14, card_y + card_h - 36), f"Result: {best_image_path.name} | Smartphone Finished", font=font_small, fill=(100, 116, 139))
    else:
        draw.text((p4_x + 20, card_y + 100), "Output image not found", font=font_body, fill=(239, 68, 68))

    # --------------------------------------------------------------------------
    # PANEL 5: QA SCORE AND CHECKLIST TICKS
    # --------------------------------------------------------------------------
    p5_x = get_panel_x(4)
    draw.rectangle([p5_x, card_y, p5_x + panel_w, card_y + card_h], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.rectangle([p5_x, card_y, p5_x + panel_w, card_y + 44], fill=(241, 245, 249))
    draw.text((p5_x + 14, card_y + 12), "PANEL 5: QA Audit & Checklist", font=font_panel_hdr, fill=(15, 23, 42))

    qa_y = card_y + 56
    is_passed = qa_result.get("passed", False)
    qa_score = float(qa_result.get("overall_score", qa_result.get("qa_score", 0.0)))
    sim_max = float(qa_result.get("similarity_max", 0.0))
    sim_mean = float(qa_result.get("similarity_mean", 0.0))

    # Score Badge Box
    badge_bg = (220, 252, 231) if is_passed else (254, 226, 226)
    badge_border = (34, 197, 94) if is_passed else (239, 68, 68)
    badge_fg = (21, 128, 61) if is_passed else (185, 28, 28)
    status_label = "PASSED" if is_passed else "FAILED"

    draw.rectangle([p5_x + 12, qa_y, p5_x + panel_w - 12, qa_y + 64], fill=badge_bg, outline=badge_border)
    draw.text((p5_x + 22, qa_y + 10), f"AUDIT DECISION: {status_label}", font=font_body_bold, fill=badge_fg)
    draw.text((p5_x + 22, qa_y + 36), f"Overall Score: {qa_score:.3f} (Threshold >= 0.60)", font=font_body, fill=(15, 23, 42))

    qa_y += 76
    # DINOv2 Similarity Metric
    draw.text((p5_x + 16, qa_y), f"DINOv2 Cosine Sim: max={sim_max:.3f}, mean={sim_mean:.3f}", font=font_body_bold, fill=(71, 85, 105))
    qa_y += 24
    draw.text((p5_x + 16, qa_y), "Checklist is primary decision; similarity is supporting.", font=font_small, fill=(148, 163, 184))
    qa_y += 30

    # 10 Checklist Items
    draw.text((p5_x + 16, qa_y), "10-POINT VISUAL CHECKLIST:", font=font_body_bold, fill=(15, 23, 42))
    qa_y += 24

    checklist = qa_result.get("checklist", {})
    standard_checks = [
        "exterior_beige",
        "scalloped_ribs_visible",
        "corner_pyramids_visible",
        "rim_intact",
        "no_pen_marks",
        "label_sleeve_present",
        "scale_plausible",
        "hands_anatomy_ok",
        "looks_like_ai",
        "background_consistent",
    ]

    for check_name in standard_checks:
        c_item = checklist.get(check_name, {})
        if isinstance(c_item, dict):
            c_val = c_item.get("passed", False)
            c_reason = c_item.get("reason", "")
        else:
            c_val = bool(c_item)
            c_reason = ""

        # For looks_like_ai: False is good
        if check_name == "looks_like_ai":
            item_pass = not c_val
            tick = "[✔] PASS" if item_pass else "[✖] FAIL"
            color = (21, 128, 61) if item_pass else (185, 28, 28)
            label = "looks_like_ai (clean)" if item_pass else "looks_like_ai (glitches)"
        else:
            item_pass = c_val
            tick = "[✔] PASS" if item_pass else "[✖] FAIL"
            color = (21, 128, 61) if item_pass else (185, 28, 28)
            label = check_name

        draw.text((p5_x + 18, qa_y), tick, font=font_body_bold, fill=color)
        draw.text((p5_x + 105, qa_y), label, font=font_body, fill=(30, 41, 59))
        qa_y += 24

    # Bottom notes
    notes = qa_result.get("notes") or qa_result.get("qa_feedback", "")
    if notes:
        qa_y += 10
        wrapped_notes = textwrap.wrap(f"Audit Notes: {notes}", width=50)
        for nl in wrapped_notes[:3]:
            draw.text((p5_x + 16, qa_y), nl, font=font_small, fill=(100, 116, 139))
            qa_y += 18

    # Save output image with provenance metadata
    png_info = PngInfo()
    png_info.add_text("Comment", "AI-generated mockup")
    png_info.add_text("Description", "CirklaGen 5-Panel Benchmark")
    png_info.add_text("Software", "CirklaGen Benchmark Runner")

    canvas.save(output_path, format="PNG", pnginfo=png_info)
    logger.info("Saved side-by-side benchmark panel to %s", output_path)
    return output_path


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_all_examples(
    out_dir: Union[str, Path] = "outputs/examples",
    mock: bool = False,
    n_variations: int = 4,
    use_composite: bool = False,
    finishing: bool = True,
    better_reference_selection: Optional[bool] = None,
    strict_must_keep: Optional[bool] = None,
    auto_composite_fallback: Optional[bool] = None,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Executes 5 predefined prompts through the pipeline and creates benchmark artifacts."""
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    cfg = load_config(require_api_key=False)
    if mock:
        cfg.mock_mode = True
    if better_reference_selection is not None:
        cfg.better_reference_selection = better_reference_selection
    if strict_must_keep is not None:
        cfg.strict_must_keep = strict_must_keep
    if auto_composite_fallback is not None:
        cfg.auto_composite_fallback = auto_composite_fallback

    cli = GenAIClient(config=cfg)
    pipeline = GenerationPipeline(config=cfg, client=cli)

    csv_path = output_directory / "results.csv"
    results_rows: List[Dict[str, Any]] = []

    print(f"\n{'='*75}")
    print(
        f"CIRKLAGEN 5-SCENARIO BENCHMARK EXECUTION (n={n_variations}, mock={cfg.mock_mode}, "
        f"better_refs={cfg.better_reference_selection}, strict_keep={cfg.strict_must_keep}, "
        f"auto_comp={cfg.auto_composite_fallback})"
    )
    print(f"{'='*75}\n")

    for idx, scenario in enumerate(EXAMPLE_SCENARIOS, start=1):
        s_type = scenario["shot_type"]
        s_prompt = scenario["prompt"]
        panel_filename = scenario["filename"]
        panel_path = output_directory / panel_filename

        print(f"[{idx}/5] Running Shot Type: '{s_type}'...")
        print(f"      Prompt: \"{s_prompt}\"")

        res = pipeline.run(
            short_prompt=s_prompt,
            shot_type=s_type,
            n=n_variations,
            use_composite=use_composite,
            finishing=finishing,
            manual_mode=False,
            better_reference_selection=better_reference_selection,
            strict_must_keep=strict_must_keep,
            auto_composite_fallback=auto_composite_fallback,
        )

        variations = res.get("variations", [])
        best_candidate = res.get("best_image", {})
        expanded = res.get("expanded", {})

        n_generated = len(variations)
        n_passed = sum(1 for v in variations if v.get("passed_qa", False))
        best_score = float(best_candidate.get("qa_score", 0.0))

        # Collect checklist failures
        failures: List[str] = []
        for v in variations:
            qa_e = v.get("qa_evaluation", {})
            chk = qa_e.get("checklist", {})
            for k, val in chk.items():
                if isinstance(val, dict):
                    p_val = val.get("passed", False)
                else:
                    p_val = bool(val)
                if k == "looks_like_ai":
                    if p_val:
                        failures.append("looks_like_ai")
                else:
                    if not p_val:
                        failures.append(k)

        unique_failures = sorted(list(set(failures)))
        failure_str = "; ".join(unique_failures) if unique_failures else "none"

        # Record CSV row
        row = {
            "prompt": s_prompt,
            "shot_type": s_type,
            "n_generated": n_generated,
            "n_passed": n_passed,
            "best_score": round(best_score, 3),
            "checklist_failures": failure_str,
        }
        results_rows.append(row)

        # Build 5-panel side-by-side PNG
        plates_dir = Path(cfg.paths.plates_dir)
        plate_name = expanded.get("plate", "")
        plate_file = plates_dir / plate_name
        if not plate_file.is_file():
            avail_plates = sorted(list(plates_dir.glob("*.png")) + list(plates_dir.glob("*.jpeg")) + list(plates_dir.glob("*.jpg")))
            plate_file = avail_plates[0] if avail_plates else Path("assets/plates/default.png")

        # Discover exact tray references used for this shot type
        from src.generator import select_tray_references
        tray_refs = select_tray_references(
            s_type,
            trays_dir=cfg.paths.trays_dir,
            angles_file=Path(cfg.paths.trays_dir) / "angles.json",
            config=cfg,
        )

        best_img_p = Path(best_candidate.get("file", best_candidate.get("finished_image", "")))

        qa_eval_dict = best_candidate.get("qa_evaluation") or {}
        if not qa_eval_dict:
            # Fallback to variation qa_evaluation
            best_idx = best_candidate.get("variation_index")
            best_var = next((v for v in variations if v.get("variation_index") == best_idx), variations[0] if variations else {})
            qa_eval_dict = best_var.get("qa_evaluation", {})

        create_example_side_by_side_panel(
            plate_path=plate_file,
            tray_ref_paths=tray_refs,
            short_prompt=s_prompt,
            shot_type=s_type,
            expanded_prompt=expanded.get("final_prompt", ""),
            best_image_path=best_img_p,
            qa_result=qa_eval_dict,
            output_path=panel_path,
        )

        # Also save unnumbered alias (example_{shot_type}.png) for compatibility
        alias_path = output_directory / f"example_{s_type}.png"
        if alias_path.resolve() != panel_path.resolve():
            import shutil
            shutil.copyfile(panel_path, alias_path)

        print(f"      Passed: {n_passed}/{n_generated} | Best Score: {best_score:.3f} | Failures: {failure_str}")
        print(f"      Saved Panel: {panel_path.name}\n")

    # Write results.csv
    csv_fields = ["prompt", "shot_type", "n_generated", "n_passed", "best_score", "checklist_failures"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results_rows:
            writer.writerow(r)

    print(f"Benchmark results CSV saved to: {csv_path}")
    print(f"{'='*75}\n")
    return csv_path, results_rows


def run_ablation_comparison(
    out_dir: Union[str, Path] = "outputs/examples",
    mock: bool = False,
    n_variations: int = 4,
    finishing: bool = True,
) -> Dict[str, Any]:
    """Runs the 5 benchmark prompts across flag ablation combinations and prints a pass rate comparison table."""
    base_out = Path(out_dir)
    comp_dir = base_out / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        {
            "name": "All Anti-Drift Flags ON",
            "tag": "all_on",
            "better_refs": True,
            "strict_keep": True,
            "auto_comp": True,
            "description": "Full fidelity: feature+angle refs, strict front-loaded spec, auto-composite fallback",
        },
        {
            "name": "(1) Better Ref Selection OFF",
            "tag": "no_better_refs",
            "better_refs": False,
            "strict_keep": True,
            "auto_comp": True,
            "description": "Legacy angle lookup only (may pick uncleaned assets / lacks feature pairing)",
        },
        {
            "name": "(2) Strict Must-Keep OFF",
            "tag": "no_strict_must_keep",
            "better_refs": True,
            "strict_keep": False,
            "auto_comp": True,
            "description": "No front-loaded mandatory spec block at prompt start",
        },
        {
            "name": "(3) Auto Composite Fallback OFF",
            "tag": "no_auto_composite",
            "better_refs": True,
            "strict_keep": True,
            "auto_comp": False,
            "description": "No automatic photographic composite recovery when QA fails twice",
        },
        {
            "name": "All Anti-Drift Flags OFF",
            "tag": "all_off",
            "better_refs": False,
            "strict_keep": False,
            "auto_comp": False,
            "description": "Baseline unconstrained generation: high risk of material, rib, and pen drift",
        },
    ]

    summary_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("STARTING CIRKLAGEN ANTI-DRIFT ABLATION COMPARISON")
    print("Evaluating 5 conditions across 5 retail shot types (n=4 variations each, 20 images/condition)")
    print("=" * 80 + "\n")

    for cond in conditions:
        print(f"\n>>> Running Condition: {cond['name']} ({cond['tag']})")
        cond_dir = comp_dir / cond["tag"]
        csv_p, rows = run_all_examples(
            out_dir=cond_dir,
            mock=mock,
            n_variations=n_variations,
            use_composite=False,
            finishing=finishing,
            better_reference_selection=cond["better_refs"],
            strict_must_keep=cond["strict_keep"],
            auto_composite_fallback=cond["auto_comp"],
        )

        total_gen = sum(r["n_generated"] for r in rows)
        total_passed = sum(r["n_passed"] for r in rows)
        pass_rate_pct = (total_passed / total_gen * 100.0) if total_gen > 0 else 0.0
        avg_best_score = float(np.mean([r["best_score"] for r in rows])) if rows else 0.0

        # Unique failures across condition
        cond_failures = set()
        for r in rows:
            f_str = r.get("checklist_failures", "none")
            if f_str and f_str != "none":
                for item in f_str.split("; "):
                    cond_failures.add(item.strip())
        failures_display = ", ".join(sorted(cond_failures)) if cond_failures else "None (100% compliant)"

        summary_results.append({
            "name": cond["name"],
            "tag": cond["tag"],
            "better_refs": cond["better_refs"],
            "strict_keep": cond["strict_keep"],
            "auto_comp": cond["auto_comp"],
            "total_gen": total_gen,
            "total_passed": total_passed,
            "pass_rate_pct": pass_rate_pct,
            "avg_best_score": round(avg_best_score, 3),
            "failures": failures_display,
            "rows": rows,
        })

    # Also update the primary outputs/examples/results.csv with all_on results
    all_on_rows = summary_results[0]["rows"]
    main_csv = base_out / "results.csv"
    csv_fields = ["prompt", "shot_type", "n_generated", "n_passed", "best_score", "checklist_failures"]
    with open(main_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_on_rows:
            writer.writerow(r)

    # Print markdown comparison table
    print("\n" + "=" * 80)
    print("ANTI-DRIFT ABLATION BENCHMARK RESULTS")
    print("=" * 80 + "\n")

    markdown_table = (
        "| Condition | Better Refs (1) | Strict Must-Keep (2) | Auto Composite (3) | Generated | Passed | Pass Rate | Mean Score | Primary Failures |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
    )

    for s in summary_results:
        b_ref = "ON" if s["better_refs"] else "OFF"
        s_keep = "ON" if s["strict_keep"] else "OFF"
        a_comp = "ON" if s["auto_comp"] else "OFF"
        markdown_table += (
            f"| **{s['name']}** | {b_ref} | {s_keep} | {a_comp} | "
            f"{s['total_gen']} | {s['total_passed']} | **{s['pass_rate_pct']:.1f}%** | "
            f"{s['avg_best_score']:.3f} | {s['failures']} |\n"
        )

    print(markdown_table)
    return {
        "summary": summary_results,
        "markdown_table": markdown_table,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CirklaGen Benchmark: Run 5 example prompts and produce side-by-side evaluator panels."
    )
    parser.add_argument("--mock", action="store_true", help="Force MOCK mode for offline benchmarking")
    parser.add_argument("--n", type=int, default=4, help="Variations per prompt (default: 4)")
    parser.add_argument("--composite", action="store_true", help="Enable composite repair on failure")
    parser.add_argument("--no-finishing", action="store_true", help="Disable phone optical finishing")
    parser.add_argument("--out-dir", type=str, default="outputs/examples", help="Output directory")

    # Anti-drift flag toggles
    parser.add_argument("--better-refs", dest="better_refs", action="store_true", default=None, help="Enable better reference selection")
    parser.add_argument("--no-better-refs", dest="better_refs", action="store_false", help="Disable better reference selection")
    parser.add_argument("--strict-must-keep", dest="strict_must_keep", action="store_true", default=None, help="Enable strict must-keep block at start of prompt")
    parser.add_argument("--no-strict-must-keep", dest="strict_must_keep", action="store_false", help="Disable strict must-keep block")
    parser.add_argument("--auto-composite-fallback", dest="auto_composite_fallback", action="store_true", default=None, help="Enable auto-composite fallback on double QA failure")
    parser.add_argument("--no-auto-composite-fallback", dest="auto_composite_fallback", action="store_false", help="Disable auto-composite fallback")
    parser.add_argument("--compare-flags", action="store_true", help="Run comparative ablation across all flag combinations and print summary table")

    args = parser.parse_args()

    if args.compare_flags:
        run_ablation_comparison(
            out_dir=args.out_dir,
            mock=args.mock,
            n_variations=args.n,
            finishing=not args.no_finishing,
        )
    else:
        run_all_examples(
            out_dir=args.out_dir,
            mock=args.mock,
            n_variations=args.n,
            use_composite=args.composite,
            finishing=not args.no_finishing,
            better_reference_selection=args.better_refs,
            strict_must_keep=args.strict_must_keep,
            auto_composite_fallback=args.auto_composite_fallback,
        )


if __name__ == "__main__":
    main()
