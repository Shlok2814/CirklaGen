"""Unit tests for anti-drift improvements:
1. Better reference selection by angle and visible features (config.better_reference_selection)
2. Stricter, shorter must_keep block placed at the start of final_prompt (config.strict_must_keep)
3. Automatic composite fallback when QA fails twice (config.auto_composite_fallback)
"""

from pathlib import Path
from PIL import Image
import pytest

from src.config import Config, load_config
from src.expander import (
    STRICT_MUST_KEEP_BLOCK,
    enforce_final_prompt_constraints,
    expand,
)
from src.generator import SHOT_FEATURE_REFERENCES, select_tray_references
from src.metadata import load_product
from src.pipeline import GenerationPipeline
from src.qa import QAResult


@pytest.fixture
def test_config() -> Config:
    """Fixture providing mock Config."""
    cfg = load_config(require_api_key=False)
    cfg.mock_mode = True
    return cfg


# ==============================================================================
# FEATURE 1: BETTER REFERENCE SELECTION BY ANGLE AND VISIBLE FEATURES
# ==============================================================================

def test_better_reference_selection_enabled(test_config):
    """Verify better_reference_selection selects feature-rich clean references."""
    for shot_type, expected_features in SHOT_FEATURE_REFERENCES.items():
        refs = select_tray_references(
            shot_type=shot_type,
            trays_dir=test_config.paths.trays_dir,
            better_reference_selection=True,
            config=test_config,
        )
        assert len(refs) >= 2
        ref_names = [r.name for r in refs]

        # Ensure no pen-marked uncleaned references are selected
        assert "tray_unlabeled_1.png" not in ref_names
        assert "tray_unlabeled_3.png" not in ref_names  # Only tray_unlabeled_3_clean.png is allowed

        # Verify key visible feature file was included
        first_expected = expected_features[0]
        if (Path(test_config.paths.trays_dir) / first_expected).is_file():
            assert first_expected in ref_names


def test_better_reference_selection_disabled(test_config):
    """Verify disabling better_reference_selection uses legacy angle lookup."""
    refs_legacy = select_tray_references(
        shot_type="front_chest",
        trays_dir=test_config.paths.trays_dir,
        better_reference_selection=False,
        config=test_config,
    )
    assert len(refs_legacy) >= 2


# ==============================================================================
# FEATURE 2: STRICT MUST_KEEP BLOCK AT START OF FINAL_PROMPT
# ==============================================================================

def test_strict_must_keep_block_at_start_of_prompt(test_config):
    """Verify strict must_keep block is placed at character 0 of final_prompt."""
    meta = load_product(test_config.paths.knowledge_file)
    sample_prompt = "A customer in the Trader Joe's meat aisle holding organic ribeye."

    # When strict_must_keep is True
    strict_prompt = enforce_final_prompt_constraints(
        final_prompt=sample_prompt,
        meta=meta,
        strict_must_keep=True,
    )
    assert strict_prompt.startswith(STRICT_MUST_KEEP_BLOCK)
    assert "MANDATORY TRAY SPEC" in strict_prompt
    assert "vertical scalloped fluted sidewalls" in strict_prompt

    # When strict_must_keep is False
    loose_prompt = enforce_final_prompt_constraints(
        final_prompt=sample_prompt,
        meta=meta,
        strict_must_keep=False,
    )
    assert not loose_prompt.startswith(STRICT_MUST_KEEP_BLOCK)
    assert "Candid phone photo:" in loose_prompt


def test_expander_honors_strict_must_keep_flag(test_config):
    """Verify expand() respects strict_must_keep flag."""
    res_strict = expand(
        short_prompt="holding tray in aisle",
        shot_type="front_chest",
        config=test_config,
        strict_must_keep=True,
    )
    assert res_strict["final_prompt"].startswith(STRICT_MUST_KEEP_BLOCK)

    res_loose = expand(
        short_prompt="holding tray in aisle",
        shot_type="front_chest",
        config=test_config,
        strict_must_keep=False,
    )
    assert not res_loose["final_prompt"].startswith(STRICT_MUST_KEEP_BLOCK)


# ==============================================================================
# FEATURE 3: AUTOMATIC COMPOSITE FALLBACK WHEN QA FAILS TWICE
# ==============================================================================

def test_auto_composite_fallback_triggers_on_double_failure(test_config, monkeypatch):
    """Verify auto composite fallback triggers when QA fails twice and recovers the candidate."""
    test_config.auto_composite_fallback = True
    pipe = GenerationPipeline(config=test_config)

    qa_call_count = 0

    def mock_evaluate_image(img, config=None, client=None):
        nonlocal qa_call_count
        qa_call_count += 1
        # Call 1 fails (initial candidate)
        # Call 2 is after auto_composite_fallback -> passes
        if qa_call_count == 1:
            return QAResult(
                similarity_max=0.30,
                similarity_mean=0.15,
                checklist={
                    "exterior_beige": {"passed": False, "reason": "drift to plastic"},
                    "scalloped_ribs_visible": {"passed": False, "reason": "smooth wall"},
                    "looks_like_ai": {"passed": False, "reason": "none"},
                },
                passed=False,
                notes="FAIL: tray geometry drifted",
                overall_score=0.35,
                tray_detected=True,
                crop=None,
            )
        else:
            return QAResult(
                similarity_max=0.90,
                similarity_mean=0.60,
                checklist={
                    "exterior_beige": {"passed": True, "reason": "authentic fiber"},
                    "scalloped_ribs_visible": {"passed": True, "reason": "fluting verified"},
                    "looks_like_ai": {"passed": False, "reason": "none"},
                },
                passed=True,
                notes="PASS: recovered via composite",
                overall_score=0.92,
                tray_detected=True,
                crop=None,
            )

    import src.pipeline
    monkeypatch.setattr(src.pipeline, "evaluate_image", mock_evaluate_image)

    res = pipe.run(
        short_prompt="me holding tray",
        n=1,
        use_composite=False,
        finishing=False,
        auto_composite_fallback=True,
    )

    var = res["variations"][0]
    assert var["repaired"] is True
    assert var["repair_type"] == "auto_composite_fallback"
    assert var["passed_qa"] is True
    assert qa_call_count == 2


def test_auto_composite_fallback_disabled_leaves_candidate_failed(test_config, monkeypatch):
    """Verify disabling auto composite fallback leaves a twice-failing candidate as failed."""
    test_config.auto_composite_fallback = False
    pipe = GenerationPipeline(config=test_config)

    def mock_evaluate_image_always_fails(img, config=None, client=None):
        return QAResult(
            similarity_max=0.30,
            similarity_mean=0.15,
            checklist={
                "exterior_beige": {"passed": False, "reason": "drift"},
                "scalloped_ribs_visible": {"passed": False, "reason": "drift"},
                "looks_like_ai": {"passed": False, "reason": "none"},
            },
            passed=False,
            notes="FAIL: tray drift",
            overall_score=0.35,
            tray_detected=True,
            crop=None,
        )

    import src.pipeline
    monkeypatch.setattr(src.pipeline, "evaluate_image", mock_evaluate_image_always_fails)

    res = pipe.run(
        short_prompt="me holding tray",
        n=1,
        use_composite=False,
        finishing=False,
        auto_composite_fallback=False,
    )

    var = res["variations"][0]
    assert var["passed_qa"] is False
    assert var["repair_type"] != "auto_composite_fallback"
