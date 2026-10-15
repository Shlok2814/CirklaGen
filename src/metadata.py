"""Product metadata loader, validator, and serialization module for CirklaGen.

Enforces schema validation on knowledge/product.json using Pydantic:
- product: {name, brand, material, size_in, size_label}
- geometry: {exterior_color, rim, base_panel, wall_ribs, corner_features, liner}
- markings: {embossed_logo, location}
- as_sold: {film, contents, label_sleeve, glare_notes}
- must_keep: list
- must_not: list
- scale: {hand_to_tray_notes}
- angle_views: what is visible from front, side, top, pov, closeup
- scene_defaults: {camera, lighting, imperfections, skin, framing}
- shot_types: {front_chest, pov_hands, side_profile, cart_topdown, closeup_hands}
  each with: pose, camera_distance, tray_orientation, what_the_camera_sees

Rule:
knowledge/product.json is the single source of truth about the tray. Never invent product details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ProductInfo(BaseModel):
    """Core product identity, manufacturer, and physical dimensions."""
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Product name")
    brand: str = Field(..., description="Brand or manufacturer, e.g. Cirkla")
    material: str = Field(..., description="Base physical material, e.g. beige molded-fiber")
    size_in: str = Field(..., description="Dimensions in inches (e.g. 10.28 x 8.08 in)")
    size_label: str = Field(..., description="Standard packaging size label (e.g. #5)")
    retailer: Optional[str] = Field(default=None, description="Retailer or grocery brand")
    cuts: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available cuts of meat")


class GeometryInfo(BaseModel):
    """Detailed structural geometry of the molded fiber tray."""
    model_config = ConfigDict(extra="allow")

    exterior_color: str = Field(..., description="Exterior pulp color")
    rim: str = Field(..., description="Perimeter sealing flange")
    base_panel: str = Field(..., description="Recessed central base floor")
    wall_ribs: str = Field(..., description="Vertical wall fluting / ribs")
    corner_features: str = Field(..., description="Corner pyramid reinforcement features")
    liner: str = Field(..., description="Internal moisture-barrier liner")


class MarkingsInfo(BaseModel):
    """Embossed brandings and patent numbers on tray body."""
    model_config = ConfigDict(extra="allow")

    embossed_logo: str = Field(..., description="Embossed CIRKLA logo and text")
    location: str = Field(..., description="Location of the markings")
    additional_text: Optional[str] = Field(default=None, description="Additional debossed text")


class AsSoldInfo(BaseModel):
    """Retail packaged appearance including film, meat cut, sleeve, and glares."""
    model_config = ConfigDict(extra="allow")

    film: str = Field(..., description="Protective barrier film sealing")
    contents: str = Field(..., description="Fresh raw meat contents")
    label_sleeve: str = Field(..., description="Paper label sleeve details")
    glare_notes: str = Field(..., description="Lighting glares and reflections on plastic film")


class ScaleInfo(BaseModel):
    """Human hand scale and proportion guidelines."""
    model_config = ConfigDict(extra="allow")

    hand_to_tray_notes: str = Field(..., description="Hand size and grasp relative to tray dimensions")


class AngleViewsInfo(BaseModel):
    """Visual elements visible from standard viewing perspectives."""
    model_config = ConfigDict(extra="allow")

    front: str = Field(..., description="What is visible from the front view")
    side: str = Field(..., description="What is visible from the side view")
    top: str = Field(..., description="What is visible from the top view")
    pov: str = Field(..., description="What is visible from the first-person POV")
    closeup: str = Field(..., description="What is visible from a closeup / macro view")


class SceneDefaultsInfo(BaseModel):
    """Environmental settings for candid supermarket photography."""
    model_config = ConfigDict(extra="allow")

    camera: str = Field(..., description="Handheld smartphone camera style")
    lighting: str = Field(..., description="Supermarket cold fluorescent illumination")
    imperfections: str = Field(..., description="Handheld camera angle tilt, blur, and grain")
    skin: str = Field(..., description="Authentic unretouched skin texture and hand tone")
    framing: str = Field(..., description="Off-center candid framing and shelf clutter")


class ShotTypeDetail(BaseModel):
    """Configuration for a specific shot composition."""
    model_config = ConfigDict(extra="allow")

    pose: str = Field(..., description="Subject pose and hand position")
    camera_distance: str = Field(..., description="Distance from camera to subject/tray")
    tray_orientation: str = Field(..., description="Tray 3D spatial orientation")
    what_the_camera_sees: str = Field(..., description="Framed visible content")


class ShotTypesInfo(BaseModel):
    """Prescribed photographic shot compositions."""
    model_config = ConfigDict(extra="allow")

    front_chest: ShotTypeDetail = Field(..., description="Frontal shot at chest level")
    pov_hands: ShotTypeDetail = Field(..., description="First-person view of hands holding tray")
    side_profile: ShotTypeDetail = Field(..., description="Side view showing depth and ribs")
    cart_topdown: ShotTypeDetail = Field(..., description="Top-down shot in shopping cart")
    closeup_hands: ShotTypeDetail = Field(..., description="Close-up macro of hands and tray")
    stepped_back_wide: Optional[ShotTypeDetail] = Field(None, description="Stepped-back wide view with subject in lower third")
    seated_in_cart_view: Optional[ShotTypeDetail] = Field(None, description="View from inside shopping cart basket looking across/up")
    holding_and_pointing: Optional[ShotTypeDetail] = Field(None, description="One hand holding tray and one hand pointing to or touching label")
    off_center_candid: Optional[ShotTypeDetail] = Field(None, description="Off-center subject position in left or right third")
    tilted_handheld: Optional[ShotTypeDetail] = Field(None, description="Candid 5 to 15 degree handheld camera tilt")
    cropped_edges: Optional[ShotTypeDetail] = Field(None, description="Cropped edges with hand or shoulder partly out of frame")
    shelf_gaze: Optional[ShotTypeDetail] = Field(None, description="Subject looking at the tray or shelf rather than camera")


class ProductMetadata(BaseModel):
    """Complete validated ground-truth metadata schema for CirklaGen."""
    model_config = ConfigDict(extra="allow")

    product: ProductInfo
    geometry: GeometryInfo
    markings: MarkingsInfo
    as_sold: AsSoldInfo
    must_keep: List[str]
    must_not: List[str]
    scale: ScaleInfo
    angle_views: AngleViewsInfo
    scene_defaults: SceneDefaultsInfo
    shot_types: ShotTypesInfo
    framing_and_pose_variety: Optional[Dict[str, Any]] = None


def load_product(path: Union[str, Path] = "knowledge/product.json") -> ProductMetadata:
    """Loads and validates knowledge/product.json against ProductMetadata schema.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If JSON data fails Pydantic schema validation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Product metadata file not found at: {file_path.resolve()}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return ProductMetadata.model_validate(raw_data)


def save_product(
    data: Union[ProductMetadata, Dict[str, Any]],
    path: Union[str, Path] = "knowledge/product.json",
    indent: int = 2,
) -> Path:
    """Validates and writes product metadata to JSON.

    Validates data against ProductMetadata schema before writing.
    Never writes invalid or unvalidated product metadata.
    """
    if isinstance(data, dict):
        validated = ProductMetadata.model_validate(data)
    elif isinstance(data, ProductMetadata):
        validated = data
    else:
        raise TypeError(f"Expected ProductMetadata or dict, got {type(data)}")

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    json_str = validated.model_dump_json(indent=indent)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    return target_path


# Backward compatibility alias for skeleton test imports
MetadataTracker = ProductMetadata


if __name__ == "__main__":
    print("--- Validating knowledge/product.json ---")
    meta = load_product("knowledge/product.json")
    print(f"Loaded product: {meta.product.name} ({meta.product.brand})")
    print(f"Material: {meta.product.material}")
    print(f"Size: {meta.product.size_in} [{meta.product.size_label}]")
    print(f"Must keep rules: {len(meta.must_keep)}")
    print(f"Must not rules: {len(meta.must_not)}")
    print(f"Shot types: {list(meta.shot_types.model_dump().keys())}")
    print("Schema validation successful!")
