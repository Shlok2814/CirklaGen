"""Unit tests for metadata loader, validator, and serialization in src/metadata.py."""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.metadata import (
    AngleViewsInfo,
    AsSoldInfo,
    GeometryInfo,
    MarkingsInfo,
    ProductInfo,
    ProductMetadata,
    ScaleInfo,
    SceneDefaultsInfo,
    ShotTypeDetail,
    ShotTypesInfo,
    load_product,
    save_product,
)


def test_load_product_success():
    """Verify load_product correctly parses and validates knowledge/product.json."""
    meta = load_product("knowledge/product.json")
    assert isinstance(meta, ProductMetadata)

    # 1. Product section
    assert meta.product.name == "Cirkla Molded Fiber Meat Tray"
    assert meta.product.brand == "Cirkla"
    assert meta.product.material == "beige molded-fiber"
    assert "10.28" in meta.product.size_in and "8.08" in meta.product.size_in
    assert meta.product.size_label == "#5"

    # 2. Geometry section
    assert "beige" in meta.geometry.exterior_color.lower()
    assert "rim" in meta.geometry.rim.lower() or "flange" in meta.geometry.rim.lower()
    assert "recessed" in meta.geometry.base_panel.lower()
    assert "scalloped" in meta.geometry.wall_ribs.lower() or "ribs" in meta.geometry.wall_ribs.lower()
    assert "pyramid" in meta.geometry.corner_features.lower()
    assert "glossy white" in meta.geometry.liner.lower()

    # 3. Markings section
    assert "CIRKLA" in meta.markings.embossed_logo
    assert "base" in meta.markings.location.lower()

    # 4. As sold section
    assert "modified atmosphere packaging" in meta.as_sold.film.lower() or "map" in meta.as_sold.film.lower()
    assert "steak" in meta.as_sold.contents.lower()
    assert "label" in meta.as_sold.label_sleeve.lower() or "sleeve" in meta.as_sold.label_sleeve.lower()
    assert len(meta.as_sold.glare_notes) > 0

    # 5. Must keep list
    assert isinstance(meta.must_keep, list)
    assert len(meta.must_keep) >= 5
    keep_str = " ".join(meta.must_keep).lower()
    assert "beige" in keep_str
    assert "ribs" in keep_str or "fluting" in keep_str
    assert "pyramid" in keep_str

    # 6. Must not list
    assert isinstance(meta.must_not, list)
    not_str = " ".join(meta.must_not).lower()
    assert "pen marks" in not_str
    assert "white or gray exterior" in not_str
    assert "extra compartments" in not_str
    assert "changed rim thickness" in not_str
    assert "garbled embossing" in not_str
    assert "floating tray" in not_str
    assert "wrong scale" in not_str

    # 7. Scale section
    assert "10.28" in meta.scale.hand_to_tray_notes or "#5" in meta.scale.hand_to_tray_notes

    # 8. Angle views section
    for view in ["front", "side", "top", "pov", "closeup"]:
        assert getattr(meta.angle_views, view) is not None
        assert len(getattr(meta.angle_views, view)) > 0

    # 9. Scene defaults section
    defaults_str = (
        f"{meta.scene_defaults.camera} {meta.scene_defaults.lighting} "
        f"{meta.scene_defaults.imperfections} {meta.scene_defaults.skin} {meta.scene_defaults.framing}"
    ).lower()
    assert "phone" in defaults_str
    assert "fluorescent" in defaults_str
    assert "tilt" in defaults_str
    assert "unretouched" in defaults_str
    assert "framing" in defaults_str or "clutter" in defaults_str

    # 10. Shot types section
    shots = meta.shot_types
    for shot_name in ["front_chest", "pov_hands", "side_profile", "cart_topdown", "closeup_hands"]:
        shot = getattr(shots, shot_name)
        assert isinstance(shot, ShotTypeDetail)
        assert len(shot.pose) > 0
        assert len(shot.camera_distance) > 0
        assert len(shot.tray_orientation) > 0
        assert len(shot.what_the_camera_sees) > 0


def test_load_product_missing_file_raises():
    """Verify load_product raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        load_product("non_existent_directory/non_existent_file.json")


def test_load_product_invalid_json_schema(tmp_path):
    """Verify load_product raises ValidationError on invalid schema data."""
    bad_data = {
        "product": {
            "name": "Invalid Tray",
            # missing brand, material, size_in, size_label
        }
    }
    bad_file = tmp_path / "bad_product.json"
    bad_file.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_product(bad_file)


def test_save_product_validates_and_saves(tmp_path):
    """Verify save_product validates data before writing to disk."""
    meta = load_product("knowledge/product.json")

    out_file = tmp_path / "saved_product.json"
    result_path = save_product(meta, out_file)
    assert result_path.is_file()

    reloaded = load_product(result_path)
    assert reloaded.product.name == meta.product.name
    assert reloaded.geometry.wall_ribs == meta.geometry.wall_ribs
    assert reloaded.must_not == meta.must_not


def test_save_product_from_dict_and_roundtrip(tmp_path):
    """Verify save_product accepts a dict, validates it, and writes valid JSON."""
    meta = load_product("knowledge/product.json")
    meta_dict = meta.model_dump()

    out_file = tmp_path / "dict_saved_product.json"
    save_product(meta_dict, out_file)

    reloaded = load_product(out_file)
    assert reloaded.product.brand == "Cirkla"
    assert reloaded.geometry.corner_features == meta.geometry.corner_features


def test_save_product_invalid_data_rejects_without_writing(tmp_path):
    """Verify save_product rejects invalid data and never writes unvalidated files."""
    invalid_dict = {
        "product": {
            "name": "Malformed Tray"
            # Missing all required fields
        }
    }
    target_file = tmp_path / "should_not_exist.json"

    with pytest.raises(ValidationError):
        save_product(invalid_dict, target_file)

    assert not target_file.exists(), "Target file should not have been created on validation failure"
