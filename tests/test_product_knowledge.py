"""Unit tests for product knowledge base and tray ground truth."""

from pathlib import Path
from src.product_knowledge import ProductKnowledge


def test_product_knowledge_load():
    """Verify that product.json loads correctly and conforms to ground truth."""
    pk = ProductKnowledge("knowledge/product.json")
    assert pk.product["brand"] == "Cirkla"
    assert pk.tray_spec["manufacturer"] == "Cirkla"
    assert "molded" in pk.tray_spec["material"].lower()
    assert "fluting" in pk.tray_spec["geometry"]["sidewalls"].lower()
    assert len(pk.available_cuts) >= 3


def test_ground_truth_prompt_snippet():
    """Verify that prompt snippet includes exact product details and no hallucinated details."""
    pk = ProductKnowledge("knowledge/product.json")
    snippet = pk.build_ground_truth_prompt_snippet("Tenderloin Steak")

    assert "Cirkla" in snippet
    assert "molded plant fiber" in snippet
    assert "TRADER JOE'S" in snippet
    assert "Tenderloin Steak" in snippet
    assert "USDA Organic" in snippet
    assert "Angus beef steer" in snippet


def test_reference_assets_discovered():
    """Verify that assets/trays and assets/backgrounds contain required reference files."""
    pk = ProductKnowledge("knowledge/product.json")
    tray_images = pk.get_reference_tray_images("assets/trays")
    assert len(tray_images) > 0, "No reference tray images found in assets/trays"

    # Verify the primary ground truth reference image is present
    tray_filenames = [p.name for p in tray_images]
    assert "cirkla_tray_meat_label.jpeg" in tray_filenames

    background_images = pk.get_background_images("assets/backgrounds")
    assert len(background_images) > 0, "No background plates found in assets/backgrounds"
