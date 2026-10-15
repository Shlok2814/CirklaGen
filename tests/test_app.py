"""Automated tests for Streamlit app.py using Streamlit AppTest framework."""

import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_test():
    """Initializes AppTest runner for app.py."""
    app_path = (Path(__file__).parent.parent / "app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    return at


def test_app_initial_render_and_sidebar_controls(app_test):
    """Verify app.py renders cleanly with all required sidebar and main page controls."""
    app_test.run(timeout=30)
    assert not app_test.exception, f"AppTest raised exception: {app_test.exception}"

    # Verify Sidebar Controls
    # Model names box
    assert any("Text Expander" in m.value for m in app_test.sidebar.markdown)

    # Variations slider (1-6)
    var_sliders = [s for s in app_test.sidebar.slider if "Variations" in s.label]
    assert len(var_sliders) == 1
    assert var_sliders[0].min == 1
    assert var_sliders[0].max == 6

    # Shot type selectbox (Auto + 5 types)
    shot_selects = [s for s in app_test.sidebar.selectbox if "Shot Type" in s.label]
    assert len(shot_selects) == 1
    options = shot_selects[0].options
    assert "Auto" in options
    assert "front_chest" in options
    assert "pov_hands" in options
    assert "side_profile" in options
    assert "cart_topdown" in options
    assert "closeup_hands" in options

    # Person reference uploader
    uploaders = [u for u in app_test.sidebar.file_uploader if "Person Reference" in u.label]
    assert len(uploaders) == 1

    # Toggles: composite real tray, phone finishing, manual mode, mock mode
    toggle_labels = [t.label for t in app_test.sidebar.toggle]
    assert any("Composite Real Tray" in l for l in toggle_labels)
    assert any("Phone Optical Finishing" in l for l in toggle_labels)
    assert any("Manual Generation Mode" in l for l in toggle_labels)
    assert any("MOCK Mode" in l for l in toggle_labels)

    # Verify Tabs: Studio, Product Memory, Run History
    tabs = [t.label for t in app_test.tabs]
    assert any("Studio" in t for t in tabs)
    assert any("Product Memory" in t for t in tabs)
    assert any("Run History" in t for t in tabs)

    # Verify 5 Example Buttons
    button_labels = [b.label for b in app_test.button]
    assert "Front Chest" in button_labels
    assert "POV Hands" in button_labels
    assert "Side Profile" in button_labels
    assert "Cart Topdown" in button_labels
    assert "Closeup Hands" in button_labels

    # Verify Generate Button
    assert any("Generate Mockups" in b for b in button_labels)

    # Verify Footer
    assert any("All images are AI-generated mockups" in m.value for m in app_test.markdown)


def test_app_click_example_prompt_updates_input(app_test):
    """Verify clicking an example prompt button updates the prompt input text box."""
    app_test.run(timeout=30)
    assert not app_test.exception

    # Find and click "POV Hands" example button
    pov_btn = [b for b in app_test.button if b.label == "POV Hands"][0]
    pov_btn.click().run(timeout=30)
    assert not app_test.exception

    # Check that text input was updated
    text_inputs = [ti for ti in app_test.text_input if "Prompt" in ti.label]
    assert len(text_inputs) == 1
    assert "POV hands" in text_inputs[0].value


def test_app_product_memory_tab(app_test):
    """Verify Product Memory tab renders product.json, metrics, and save button."""
    app_test.run(timeout=30)
    assert not app_test.exception

    # Verify metrics
    metrics = [m.label for m in app_test.metric]
    assert any("Brand & Product" in m for m in metrics)
    assert any("Size" in m for m in metrics)

    # Verify JSON text area
    text_areas = [ta for ta in app_test.text_area if "JSON" in ta.label or "product.json" in ta.label]
    assert len(text_areas) >= 1
    content = text_areas[0].value
    parsed = json.loads(content)
    assert "product" in parsed
    assert "geometry" in parsed

    # Verify Save button
    save_btns = [b for b in app_test.button if "Save Product Memory" in b.label]
    assert len(save_btns) == 1


def test_app_mock_generation_flow(app_test):
    """Verify full generation in MOCK mode produces gallery with badges, scores, and expanders."""
    runs_dir = Path("outputs/runs")
    existing_folders = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()

    try:
        app_test.run(timeout=30)
        assert not app_test.exception

        # Turn ON MOCK mode toggle
        mock_toggle = [t for t in app_test.sidebar.toggle if "MOCK Mode" in t.label][0]
        mock_toggle.set_value(True)

        # Set variations to 2
        var_slider = [s for s in app_test.sidebar.slider if "Variations" in s.label][0]
        var_slider.set_value(2)

        # Click Generate Mockups button
        gen_btn = [b for b in app_test.button if "Generate Mockups" in b.label][0]
        gen_btn.click().run(timeout=120)
        assert not app_test.exception

        # Verify gallery rendered: check markdown badges
        markdown_values = [m.value for m in app_test.markdown]
        assert any("PASS" in m or "FAIL" in m for m in markdown_values)
        assert any("BEST MATCH" in m for m in markdown_values)

        # Verify expanded prompt text area appeared
        exp_text_areas = [ta for ta in app_test.text_area if "Expanded" in ta.label or "regenerating" in ta.label]
        assert len(exp_text_areas) >= 1
        assert len(exp_text_areas[0].value) > 50

        # Verify Regenerate button appeared
        regen_btns = [b for b in app_test.button if "Regenerate" in b.label]
        assert len(regen_btns) >= 1
    finally:
        # Clean up any test runs created during test execution
        import shutil
        if runs_dir.is_dir():
            for folder in set(runs_dir.iterdir()) - existing_folders:
                if folder.is_dir():
                    shutil.rmtree(folder, ignore_errors=True)
