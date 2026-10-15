"""Prompt expander module for CirklaGen.

Transforms concise user descriptions into detailed, photorealistic prompts
for candid smartphone photos of a customer in a Trader Joe's meat aisle holding
the Cirkla molded-fiber beef tray.

The only source of truth about the tray is knowledge/product.json and the images
in assets/trays/. Never invent product details.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.client import GenAIClient, text as client_text
from src.config import Config, load_config
from src.logger import logger
from src.metadata import ProductMetadata, load_product


# --- Custom Readable Exceptions ---

class ExpanderError(Exception):
    """Raised when prompt expansion fails or cannot produce valid structured output."""
    pass


# --- Pydantic Schema for Structured Expander Output ---

class ExpandedScene(BaseModel):
    """Validated schema for expanded grocery photo direction."""
    model_config = ConfigDict(extra="allow")

    plate: str = Field(
        ...,
        description="Filename of the chosen background plate from the available plates in assets/plates",
    )
    shot_type: str = Field(
        ...,
        description=(
            "One of product.json shot_types: front_chest, pov_hands, side_profile, cart_topdown, "
            "closeup_hands, stepped_back_wide, seated_in_cart_view, holding_and_pointing, "
            "off_center_candid, tilted_handheld, cropped_edges, shelf_gaze"
        ),
    )
    person: str = Field(
        ...,
        description="Short description of the shopper (clothing, appearance, skin, natural unretouched hand)",
    )
    pose: str = Field(
        ...,
        description="Hands, tray height, and tray orientation",
    )
    camera: str = Field(
        ...,
        description="Phone-like framing, slight handheld tilt, distance",
    )
    lighting: str = Field(
        ...,
        description="Supermarket cold fluorescent strip lighting matching the chosen plate",
    )
    imperfections: List[str] = Field(
        ...,
        description="List of candid photo imperfections like slight tilt, motion softness, shelf clutter",
    )
    final_prompt: str = Field(
        ...,
        description="One paragraph image-edit instruction containing all mandatory rules",
    )
    variation_choices: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of distinct framing, pose, and person reference combinations per variation",
    )
    pose_framing_choice: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Specific framing, pose, and person reference combination for this variation",
    )


EXPANDED_SCENE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "plate": {
            "type": "STRING",
            "description": "Filename of the chosen background plate from available plates in assets/plates",
        },
        "shot_type": {
            "type": "STRING",
            "description": (
                "One of product.json shot_types: front_chest, pov_hands, side_profile, cart_topdown, "
                "closeup_hands, stepped_back_wide, seated_in_cart_view, holding_and_pointing, "
                "off_center_candid, tilted_handheld, cropped_edges, shelf_gaze"
            ),
        },
        "person": {
            "type": "STRING",
            "description": "Short description of the shopper (clothing, appearance, skin, natural unretouched hand)",
        },
        "pose": {
            "type": "STRING",
            "description": "Hands, tray height, and tray orientation",
        },
        "camera": {
            "type": "STRING",
            "description": "Phone-like framing, slight handheld tilt, distance",
        },
        "lighting": {
            "type": "STRING",
            "description": "Supermarket cold fluorescent strip lighting matching the chosen plate",
        },
        "imperfections": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of candid photo imperfections like slight tilt, motion softness, shelf clutter",
        },
        "final_prompt": {
            "type": "STRING",
            "description": "One paragraph image-edit instruction containing all mandatory rules",
        },
    },
    "required": [
        "plate",
        "shot_type",
        "person",
        "pose",
        "camera",
        "lighting",
        "imperfections",
        "final_prompt",
    ],
}


# --- Helper Functions & Variation Catalog ---

VALID_SHOT_TYPES = [
    "front_chest",
    "pov_hands",
    "side_profile",
    "cart_topdown",
    "closeup_hands",
    "stepped_back_wide",
    "seated_in_cart_view",
    "holding_and_pointing",
    "off_center_candid",
    "tilted_handheld",
    "cropped_edges",
    "shelf_gaze",
]

SHOT_TYPE_ALIASES = {
    "front": "front_chest",
    "chest": "front_chest",
    "front_chest": "front_chest",
    "pov": "pov_hands",
    "hands": "pov_hands",
    "pov_hands": "pov_hands",
    "side": "side_profile",
    "profile": "side_profile",
    "side_profile": "side_profile",
    "cart": "cart_topdown",
    "topdown": "cart_topdown",
    "cart_topdown": "cart_topdown",
    "closeup": "closeup_hands",
    "close_up": "closeup_hands",
    "closeup_hands": "closeup_hands",
    "macro": "closeup_hands",
    # Expanded shot type aliases
    "wide": "stepped_back_wide",
    "stepped_back": "stepped_back_wide",
    "stepped_back_wide": "stepped_back_wide",
    "lower_third": "stepped_back_wide",
    "seated": "seated_in_cart_view",
    "seated_cart": "seated_in_cart_view",
    "seated_in_cart": "seated_in_cart_view",
    "seated_in_cart_view": "seated_in_cart_view",
    "cart_view": "seated_in_cart_view",
    "pointing": "holding_and_pointing",
    "holding_pointing": "holding_and_pointing",
    "holding_and_pointing": "holding_and_pointing",
    "touch_label": "holding_and_pointing",
    "touching_label": "holding_and_pointing",
    "off_center": "off_center_candid",
    "offcenter": "off_center_candid",
    "off_center_candid": "off_center_candid",
    "left_third": "off_center_candid",
    "right_third": "off_center_candid",
    "tilted": "tilted_handheld",
    "tilted_handheld": "tilted_handheld",
    "tilt": "tilted_handheld",
    "dutch": "tilted_handheld",
    "dutch_angle": "tilted_handheld",
    "cropped": "cropped_edges",
    "cropped_edges": "cropped_edges",
    "out_of_frame": "cropped_edges",
    "shelf_gaze": "shelf_gaze",
    "gaze": "shelf_gaze",
    "looking_at_tray": "shelf_gaze",
    "looking_at_shelf": "shelf_gaze",
}


def normalize_shot_type(shot_type: Optional[str]) -> Optional[str]:
    """Normalizes informal shot type names to the canonical product.json shot_types."""
    if not shot_type:
        return None
    cleaned = shot_type.strip().lower().replace("-", "_").replace(" ", "_")
    return SHOT_TYPE_ALIASES.get(cleaned, cleaned if cleaned in VALID_SHOT_TYPES else None)


def infer_shot_type_from_text(prompt: str) -> str:
    """Infers the most appropriate shot type from short prompt keywords."""
    lowered = prompt.lower()
    if any(k in lowered for k in ["seated in cart", "seated-in-cart", "child seat", "cart seat"]):
        return "seated_in_cart_view"
    if any(k in lowered for k in ["stepped back", "stepped-back", "wide shot", "lower third", "distant"]):
        return "stepped_back_wide"
    if any(k in lowered for k in ["pointing", "touching label", "touch the label", "one hand holding and one hand pointing"]):
        return "holding_and_pointing"
    if any(k in lowered for k in ["off center", "off-center", "left third", "right third"]):
        return "off_center_candid"
    if any(k in lowered for k in ["5 to 15", "tilt", "tilted", "dutch angle", "dutch"]):
        return "tilted_handheld"
    if any(k in lowered for k in ["cropped edges", "out of frame", "shoulder partly", "hand partly"]):
        return "cropped_edges"
    if any(k in lowered for k in ["looking at the tray", "looking at the shelf", "shelf rather than", "tray rather than"]):
        return "shelf_gaze"
    if any(k in lowered for k in ["pov", "first person", "looking down", "hands in the aisle"]):
        return "pov_hands"
    if any(k in lowered for k in ["cart", "from above", "top down", "topdown", "wire mesh"]):
        return "cart_topdown"
    if any(k in lowered for k in ["closeup", "close up", "macro", "texture", "thumb"]):
        return "closeup_hands"
    if any(k in lowered for k in ["side", "profile", "inspecting", "45 degree", "angle"]):
        return "side_profile"
    return "front_chest"


# Pre-defined diverse person profiles mapped to authentic references
DEFAULT_PERSON_PROFILES: List[Dict[str, str]] = [
    {
        "filename": "shopper_woman_cardigan.jpg",
        "description": "30-year-old woman with brown hair tied in a casual bun wearing a navy blue knit cardigan",
    },
    {
        "filename": "shopper_man_henley.jpg",
        "description": "35-year-old man with neat short beard wearing an olive green jacket over a grey henley shirt",
    },
    {
        "filename": "shopper_woman_fleece.jpg",
        "description": "26-year-old Asian woman with dark hair wearing a casual tan fleece jacket",
    },
    {
        "filename": "shopper_reference.jpg",
        "description": "shopper in casual everyday retail jacket with natural skin tones and unretouched hands",
    },
]

# Distinct pose and framing combinations per variation
VARIATION_FRAMING_CATALOG: List[Dict[str, Any]] = [
    {
        "shot_type": "holding_and_pointing",
        "subject_position": "left third",
        "tilt_degrees": 7,
        "cropped_edges": "shoulder partly out of frame",
        "gaze": "looking at the tray rather than the camera",
        "hand_pose": "one hand holding tray rim firmly and one hand pointing to or touching the label",
    },
    {
        "shot_type": "off_center_candid",
        "subject_position": "right third",
        "tilt_degrees": 12,
        "cropped_edges": "hand partly out of frame",
        "gaze": "looking across at the refrigerated shelf rather than the camera",
        "hand_pose": "two hands holding tray edges firmly at chest level",
    },
    {
        "shot_type": "stepped_back_wide",
        "subject_position": "lower third",
        "tilt_degrees": 5,
        "cropped_edges": "outer arm partly out of frame",
        "gaze": "looking down at the tray rather than the camera",
        "hand_pose": "single hand supporting tray underside with fingers curved over rim",
    },
    {
        "shot_type": "seated_in_cart_view",
        "subject_position": "lower third",
        "tilt_degrees": 10,
        "cropped_edges": "elbow partly out of frame",
        "gaze": "looking at the tray rather than the camera",
        "hand_pose": "both hands gripping side flanges reaching into cart basket",
    },
    {
        "shot_type": "tilted_handheld",
        "subject_position": "left third",
        "tilt_degrees": 14,
        "cropped_edges": "hand partly out of frame",
        "gaze": "looking at the shelf rather than the camera",
        "hand_pose": "two hands holding tray at diagonal dynamic angle",
    },
    {
        "shot_type": "shelf_gaze",
        "subject_position": "right third",
        "tilt_degrees": 8,
        "cropped_edges": "shoulder partly out of frame",
        "gaze": "looking across at the refrigerated shelf price tags rather than the camera",
        "hand_pose": "one hand holding tray base and one hand resting on rim",
    },
]


def get_available_person_references(persons_dir: Union[str, Path] = "assets/persons") -> List[Path]:
    """Scans and lists existing person reference images from assets/persons."""
    dir_path = Path(persons_dir)
    if not dir_path.is_dir():
        logger.warning("Persons directory does not exist: %s", dir_path.resolve())
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    persons = [p for p in sorted(dir_path.iterdir()) if p.is_file() and p.suffix.lower() in valid_exts]
    return persons


def generate_variation_choices(
    n: int = 4,
    base_shot_type: Optional[str] = None,
    short_prompt: str = "",
    meta: Optional[ProductMetadata] = None,
    persons_dir: Union[str, Path] = "assets/persons",
    config: Optional[Config] = None,
) -> List[Dict[str, Any]]:
    """Generates distinct framing, pose, and person reference combinations per variation.

    Each variation receives:
    - A different person reference image from assets/persons/ and distinct person description.
    - A different pose each time (e.g. one hand holding and one pointing to label, two hands on rim, etc.).
    - Random/varied off-center subject position (left or right third, or lower third for wide shots).
    - 5 to 15 degree handheld camera tilt.
    - Cropped edges (hand or shoulder partly out of frame).
    - Gaze looking at the tray or shelf rather than the camera.
    """
    avail_persons = get_available_person_references(persons_dir)
    n_persons = len(avail_persons)
    choices: List[Dict[str, Any]] = []

    for i in range(n):
        # 1. Resolve distinct person reference and description
        if n_persons > 0:
            person_path = avail_persons[i % n_persons]
            # Match known profile or generate clear description
            matched_profile = next((p for p in DEFAULT_PERSON_PROFILES if p["filename"] == person_path.name), None)
            person_desc = matched_profile["description"] if matched_profile else f"shopper in casual retail attire ({person_path.stem})"
        else:
            person_path = Path(persons_dir) / "shopper_reference.jpg"
            person_desc = "shopper in casual everyday clothing with natural skin"

        # 2. Pick distinct framing combination from catalog
        combo = VARIATION_FRAMING_CATALOG[i % len(VARIATION_FRAMING_CATALOG)]
        
        # If user pinned a specific base shot type (and not auto/none), preserve it while varying pose/framing
        chosen_shot = base_shot_type if (base_shot_type and base_shot_type != "Auto" and base_shot_type in VALID_SHOT_TYPES) else combo["shot_type"]

        # Vary tilt between 5 and 15 degrees deterministically
        tilts = [7, 12, 9, 14, 6, 15, 8, 11]
        chosen_tilt = tilts[i % len(tilts)]

        # Vary subject position across left third, right third, lower third
        positions = ["left third", "right third", "lower third", "left third"]
        chosen_pos = combo["subject_position"] if not base_shot_type else positions[i % len(positions)]

        choice = {
            "variation_index": i,
            "shot_type": chosen_shot,
            "subject_position": chosen_pos,
            "tilt_degrees": chosen_tilt,
            "cropped_edges": combo["cropped_edges"],
            "gaze": combo["gaze"],
            "hand_pose": combo["hand_pose"],
            "person_reference": str(person_path.resolve()) if person_path.is_file() else str(person_path),
            "person_reference_file": person_path.name,
            "person_description": person_desc,
            "pose": f"{combo['hand_pose']}; {combo['gaze']}; subject positioned in {chosen_pos}; {chosen_tilt} degree camera tilt; {combo['cropped_edges']}",
        }

        # Build variation-specific prompt
        if meta is not None:
            choice["final_prompt"] = build_variation_final_prompt(
                base_prompt=short_prompt,
                choice=choice,
                meta=meta,
                strict_must_keep=getattr(config, "strict_must_keep", True) if config else True,
                realism_pass=getattr(config, "realism_pass", True) if config else True,
            )

        choices.append(choice)

    return choices


def build_variation_final_prompt(
    base_prompt: str,
    choice: Dict[str, Any],
    meta: ProductMetadata,
    strict_must_keep: bool = True,
    realism_pass: bool = True,
) -> str:
    """Formats a grounded, variation-specific prompt enforcing distinct pose and framing."""
    shot_key = choice["shot_type"]
    shot_detail = getattr(meta.shot_types, shot_key, meta.shot_types.front_chest)

    prompt_body = (
        f"A candid phone photo of {choice['person_description']} in Trader Joe's meat aisle holding the "
        f"Cirkla beige molded-fiber meat tray. "
        f"Framing and composition: subject positioned in the {choice['subject_position']} with a {choice['tilt_degrees']} "
        f"degree handheld camera tilt. "
        f"Cropped framing: {choice['cropped_edges']}. "
        f"Subject gaze: {choice['gaze']}. "
        f"Hand pose: {choice['hand_pose']}. "
        f"Subject authentic details: uneven skin tone, visible pores, slight shine on forehead and nose, flyaway hair, "
        f"a hard shadow under the chin and under the tray from the overhead lights, and creased clothing. "
        f"Angle view: {shot_detail.pose}. {shot_detail.what_the_camera_sees}."
    )

    return enforce_final_prompt_constraints(
        prompt_body,
        meta,
        strict_must_keep=strict_must_keep,
        realism_pass=realism_pass,
    )


def get_available_plates(plates_dir: Union[str, Path] = "assets/plates") -> List[str]:
    """Scans and lists existing image plates from assets/plates."""
    dir_path = Path(plates_dir)
    if not dir_path.is_dir():
        logger.warning("Plates directory does not exist: %s", dir_path.resolve())
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    plates = [p.name for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(plates)


def load_system_prompt(path: Union[str, Path] = "prompts/expander_system.txt") -> str:
    """Loads the system prompt from file, ensuring it remains user-editable."""
    prompt_path = Path(path)
    if prompt_path.is_file():
        content = prompt_path.read_text(encoding="utf-8").strip()
        if content:
            return content
    logger.warning("System prompt file '%s' not found or empty. Using default system instruction.", prompt_path)
    return (
        "You are a photo director for candid grocery-store phone photos of shoppers in a "
        "Trader Joe's meat aisle holding the authentic Cirkla molded-fiber beef tray."
    )


STRICT_MUST_KEEP_BLOCK = (
    "[MANDATORY TRAY SPEC: Exact Cirkla #5 beige unbleached plant-fiber meat tray (10.28\"x8.08\"x1.7\"). "
    "Must show: natural beige textured pulp (NOT white, NOT plastic), vertical scalloped fluted sidewalls, "
    "distinct corner reinforcement pyramids, flat perimeter sealing rim flange. "
    "Never change tray shape, color, or fluting.]"
)


CORE_MANDATORY_RULES_BLOCK = (
    "Reference image roles: image 1 is the aisle, images 2-3 are the tray, image 4 is the person. "
    "Mandatory generation rules: "
    "(1) Add a real-looking person into the aisle. "
    "(2) Have the tray held in their hands or resting properly (never floating). "
    "(3) Match the lighting and perspective of the aisle photo. "
    "(4) Keep the background unchanged. "
    "(5) Use the supplied tray images only to understand the tray's shape, not to paste them."
)


REALISM_CLUTTER_INSTRUCTION = (
    "Background retail clutter: authentic supermarket aisle environment with misaligned price tags on shelf edges, "
    "Trader Joe's red-and-white shelf-talker tags, crowded grocery shelves, and natural aisle background clutter."
)

REALISM_SUBJECT_DETAILS_INSTRUCTION = (
    "Subject authentic details: uneven skin tone, visible pores, slight shine on forehead and nose, "
    "flyaway hair, a hard shadow under the chin and under the tray from the overhead lights, and creased clothing."
)


def enforce_final_prompt_constraints(
    final_prompt: str,
    meta: ProductMetadata,
    strict_must_keep: bool = True,
    realism_pass: bool = True,
) -> str:
    """Ensures final_prompt satisfies all mandatory rules without hallucinating details.

    Mandates:
    - Reference image roles: image 1 is the aisle, images 2-3 are the tray, image 4 is the person.
    - Five mandatory directives:
      (1) Add a real-looking person into the aisle.
      (2) Have the tray held in their hands or resting properly (never floating).
      (3) Match the lighting and perspective of the aisle photo.
      (4) Keep the background unchanged.
      (5) Use the supplied tray images only to understand the tray's shape, not to paste them.
    - Strict must_keep spec block placed at index 0 when strict_must_keep is True.
    - Background clutter instructions when realism_pass is True.
    - Authentic subject realism: uneven skin tone, visible pores, slight shine on forehead and nose,
      flyaway hair, a hard shadow under the chin and under the tray from the overhead lights, and creased clothing.
    """
    prompt_lower = final_prompt.lower()
    prefix_clauses: List[str] = []
    suffix_clauses: List[str] = []

    # 1. Candid phone photo context
    if "candid" not in prompt_lower and "phone photo" not in prompt_lower:
        prefix_clauses.append("Candid phone photo:")

    # 2. Reference image roles and 5 mandatory directives
    has_image_roles = "image 1 is the aisle" in prompt_lower and "images 2-3 are the tray" in prompt_lower
    has_mandatory_rules = (
        "real-looking person" in prompt_lower
        and "never floating" in prompt_lower
        and "lighting and perspective" in prompt_lower
        and "keep the background unchanged" in prompt_lower
        and ("not to paste" in prompt_lower or "understand the tray's shape" in prompt_lower)
    )

    if not (has_image_roles and has_mandatory_rules):
        prefix_clauses.append(CORE_MANDATORY_RULES_BLOCK)

    # 3. Must-not items
    if "must not" not in prompt_lower and "no pen marks" not in prompt_lower:
        must_not_text = ", ".join(meta.must_not)
        suffix_clauses.append(f"Must not include: {must_not_text}.")

    # 4. Realism pass background clutter instructions
    if realism_pass:
        if "clutter" not in prompt_lower and "shelf-talker" not in prompt_lower:
            suffix_clauses.append(REALISM_CLUTTER_INSTRUCTION)

    # 5. Subject natural realism: uneven skin tone, visible pores, shine, flyaway hair, chin/tray shadows, creased clothing
    has_subject_realism = (
        "uneven skin tone" in prompt_lower
        and "visible pores" in prompt_lower
        and "slight shine on forehead and nose" in prompt_lower
        and "flyaway hair" in prompt_lower
        and "hard shadow under the chin" in prompt_lower
        and "creased clothing" in prompt_lower
    )
    if not has_subject_realism:
        suffix_clauses.append(REALISM_SUBJECT_DETAILS_INSTRUCTION)

    # Assemble cohesive paragraph
    parts: List[str] = []
    if strict_must_keep and not final_prompt.strip().startswith("[MANDATORY TRAY SPEC"):
        parts.append(STRICT_MUST_KEEP_BLOCK)
    if prefix_clauses:
        parts.append(" ".join(prefix_clauses))
    parts.append(final_prompt.strip())
    if suffix_clauses:
        parts.append(" ".join(suffix_clauses))

    return " ".join(parts).strip()


# --- Core Expander Function ---

def expand(
    short_prompt: str,
    shot_type: Optional[str] = None,
    n: Optional[int] = None,
    config: Optional[Config] = None,
    client: Optional[GenAIClient] = None,
    strict_must_keep: Optional[bool] = None,
    realism_pass: Optional[bool] = None,
) -> Dict[str, Any]:
    """Expands a short user prompt into a grounded, candid supermarket photo direction.

    Steps:
    1) Load product.json and list available plates (assets/plates).
    2) Call client.text with the system prompt from prompts/expander_system.txt and a JSON schema.
       Return validated JSON only:
       {
         "plate": filename from the available plates,
         "shot_type": one of product.json shot_types,
         "person": short description,
         "pose": hands, tray height, tray orientation,
         "camera": phone-like framing, tilt, distance,
         "lighting": matching the chosen plate,
         "imperfections": list,
         "final_prompt": one paragraph image-edit instruction
       }
    3) final_prompt must always: tell the model the supplied tray images are the exact
       tray to use, say "keep the background unchanged", include must_keep and must_not
       items, describe the scene as a candid phone photo, add background clutter instructions
       when realism_pass is enabled, and add no tray details beyond product.json.
    4) Retry once on invalid JSON, then raise a readable error.
    """
    cfg = config or load_config(require_api_key=False)
    cli = client or GenAIClient(config=cfg)

    use_realism = realism_pass if realism_pass is not None else getattr(cfg, "realism_pass", True)

    # 1. Load product.json and available plates
    meta = load_product(cfg.paths.knowledge_file)
    available_plates = get_available_plates(cfg.paths.plates_dir)
    if not available_plates:
        # Fallback plate name if directory is empty
        available_plates = ["Trader Joe's Shelf.jpeg"]

    # Normalize or infer shot_type
    target_shot = normalize_shot_type(shot_type) or infer_shot_type_from_text(short_prompt)
    if target_shot not in VALID_SHOT_TYPES:
        target_shot = "front_chest"

    # 2. System prompt and instruction payload
    system_prompt = load_system_prompt(cfg.paths.expander_prompt)

    shot_detail = getattr(meta.shot_types, target_shot, meta.shot_types.front_chest)
    must_keep_bullets = "\n".join(f"- {item}" for item in meta.must_keep)
    must_not_bullets = "\n".join(f"- {item}" for item in meta.must_not)
    plates_sample = ", ".join(available_plates[:15])

    clutter_directive = (
        "\nRealism Pass Background Clutter Instructions:\n"
        "- Direct authentic grocery background clutter: misaligned price tags on shelf edges, "
        "Trader Joe's red-and-white shelf-talker tags, grocery cold-case merchandise crowding, and authentic non-studio retail environment.\n"
        if use_realism
        else ""
    )

    user_instruction = (
        f"User Prompt: \"{short_prompt}\"\n"
        f"Target Shot Type: {target_shot}\n\n"
        f"Available Plates in assets/plates: {plates_sample}\n\n"
        f"Target Shot Pose Guide:\n"
        f"- Pose: {shot_detail.pose}\n"
        f"- Camera Distance: {shot_detail.camera_distance}\n"
        f"- Orientation: {shot_detail.tray_orientation}\n"
        f"- What Camera Sees: {shot_detail.what_the_camera_sees}\n\n"
        f"Mandatory Must-Keep Rules:\n{must_keep_bullets}\n\n"
        f"Mandatory Must-Not Prohibitions:\n{must_not_bullets}\n\n"
        f"Mandatory Directives for final_prompt:\n"
        f"- Reference image roles: image 1 is the aisle, images 2-3 are the tray, image 4 is the person.\n"
        f"- (1) Add a real-looking person into the aisle.\n"
        f"- (2) Have the tray held in their hands or resting properly (never floating).\n"
        f"- (3) Match the lighting and perspective of the aisle photo.\n"
        f"- (4) Keep the background unchanged.\n"
        f"- (5) Use the supplied tray images only to understand the tray's shape, not to paste them.\n\n"
        f"Scene Defaults:\n"
        f"- Camera: {meta.scene_defaults.camera}\n"
        f"- Lighting: {meta.scene_defaults.lighting}\n"
        f"- Imperfections: {meta.scene_defaults.imperfections}\n"
        f"- Skin: {meta.scene_defaults.skin}\n"
        f"- Framing: {meta.scene_defaults.framing}\n"
        f"{clutter_directive}\n"
        f"Select an exact plate from the available plates and output strictly validated JSON matching the schema."
    )

    # 3 & 4. Call client.text with retry once on invalid JSON
    max_attempts = 2
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            curr_prompt = user_instruction
            if attempt > 1 and last_error:
                curr_prompt += (
                    f"\n\nCRITICAL FIX REQUIRED (Attempt {attempt}): Previous output failed validation: {last_error}. "
                    f"Return ONLY valid JSON with keys: plate, shot_type, person, pose, camera, lighting, imperfections, final_prompt. "
                    f"Ensure plate is an exact filename from: {plates_sample}."
                )

            logger.info("Calling client.text for expander (attempt %d/%d)...", attempt, max_attempts)
            try:
                raw_response = cli.text(
                    prompt=curr_prompt,
                    system=system_prompt,
                    json_schema=EXPANDED_SCENE_SCHEMA,
                )
            except Exception as api_err:
                err_s = str(api_err).lower()
                if "429" in err_s or "quota" in err_s or "ratelimit" in type(api_err).__name__.lower():
                    logger.warning("API quota exhausted for text model. Using grounded expansion from product.json: %s", api_err)
                    raw_response = {
                        "plate": "real_tj_meat_shelf_pov.jpg",
                        "shot_type": target_shot,
                        "person": "Authentic shopper browsing the meat aisle wearing casual clothing, natural unretouched skin, visible pores, and natural flyaway hair",
                        "pose": f"Holding the Cirkla molded-fiber #5 tray naturally at chest or waist level, {shot_detail.pose}",
                        "camera": f"Candid smartphone snapshot, {shot_detail.camera_distance}, slight 2-8 degree handheld tilt",
                        "lighting": "Cold fluorescent store lighting with specular reflections across MAP barrier film and contact shadows under fingers",
                        "imperfections": ["slight handheld tilt", "authentic phone camera optical softness", "background shelf clutter"],
                        "final_prompt": f"A candid phone photo taken on a modern smartphone in the Trader Joe's meat aisle of a shopper holding the authentic Cirkla beige molded-fiber meat tray. {shot_detail.pose}. {shot_detail.what_the_camera_sees}",
                        "mock": True,
                    }
                else:
                    raise

            # Parse response if returned as string
            if isinstance(raw_response, str):
                cleaned_str = raw_response.strip()
                if cleaned_str.startswith("```"):
                    cleaned_str = re.sub(r"^```(?:json)?\n?", "", cleaned_str)
                    cleaned_str = re.sub(r"\n?```$", "", cleaned_str).strip()
                data = json.loads(cleaned_str)
            elif isinstance(raw_response, dict):
                data = dict(raw_response)
            else:
                raise TypeError(f"Unexpected response type from text model: {type(raw_response)}")

            # In MOCK mode or fallback, tailor shot_type and plate
            if data.get("mock") is True:
                if target_shot:
                    data["shot_type"] = target_shot
                    data["pose"] = shot_detail.pose
                    data["camera"] = f"Handheld smartphone photo, {shot_detail.camera_distance}, slight tilt"
                # Map shot_type to representative plate from available plates
                plate_prefs = {
                    "front_chest": "Trader Joe's Shelf.jpeg",
                    "pov_hands": "Gemini_Generated_Image_uahbb8uahbb8uahb.png",
                    "side_profile": "Trader Joe's Shelf (1).jpeg",
                    "cart_topdown": "Gemini_Generated_Image_uaidwduaidwduaid.png",
                    "closeup_hands": "Tray Sticker & Meat.jpeg",
                }
                preferred_plate = plate_prefs.get(target_shot)
                if preferred_plate and preferred_plate in available_plates:
                    data["plate"] = preferred_plate
                elif data.get("plate") not in available_plates:
                    data["plate"] = available_plates[0]

                if use_realism:
                    imps = data.get("imperfections", [])
                    if "background shelf clutter" not in imps:
                        imps.append("background shelf clutter and misaligned price tags")
                    data["imperfections"] = imps

            # Validate schema
            validated_scene = ExpandedScene.model_validate(data)

            # Validate plate exists in available_plates
            if validated_scene.plate not in available_plates:
                # Check for case/whitespace-insensitive match
                matched_plate = None
                for ap in available_plates:
                    if ap.lower() == validated_scene.plate.lower() or Path(ap).stem.lower() == Path(validated_scene.plate).stem.lower():
                        matched_plate = ap
                        break
                if matched_plate:
                    validated_scene.plate = matched_plate
                elif attempt < max_attempts:
                    raise ValueError(
                        f"Plate '{validated_scene.plate}' does not exist in assets/plates. "
                        f"Must be one of: {available_plates[:10]}"
                    )
                else:
                    # On final attempt fallback to closest or first plate rather than failing
                    validated_scene.plate = available_plates[0]

            # Normalize shot_type to valid set
            normalized_shot = normalize_shot_type(validated_scene.shot_type) or target_shot
            validated_scene.shot_type = normalized_shot

            # Step 3: Enforce final_prompt constraints
            use_strict = strict_must_keep if strict_must_keep is not None else getattr(cfg, "strict_must_keep", True)
            validated_scene.final_prompt = enforce_final_prompt_constraints(
                validated_scene.final_prompt,
                meta,
                strict_must_keep=use_strict,
                realism_pass=use_realism,
            )

            # Return validated dictionary
            result_dict = validated_scene.model_dump()

            # Generate distinct variation choices for this run
            var_n = n or getattr(cfg, "variations_default", 4)
            variation_choices = generate_variation_choices(
                n=var_n,
                base_shot_type=normalized_shot,
                short_prompt=short_prompt,
                meta=meta,
                persons_dir=cfg.paths.persons_dir,
                config=cfg,
            )
            result_dict["variation_choices"] = variation_choices

            logger.info(
                "Successfully expanded prompt for shot_type '%s' with %d variation choices",
                result_dict["shot_type"],
                len(variation_choices),
            )
            return result_dict

        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            last_error = exc
            logger.warning("Expander attempt %d failed schema validation: %s", attempt, exc)
            if attempt >= max_attempts:
                break

    raise ExpanderError(f"Prompt expansion failed after {max_attempts} attempts. Last error: {last_error}")


def expand_variations(
    short_prompt: str,
    n: int = 4,
    shot_type: Optional[str] = None,
    config: Optional[Config] = None,
    client: Optional[GenAIClient] = None,
    strict_must_keep: Optional[bool] = None,
    realism_pass: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Expands a prompt into N distinct scene directions, each with a unique pose, framing, and person."""
    base_expanded = expand(
        short_prompt=short_prompt,
        shot_type=shot_type,
        n=n,
        config=config,
        client=client,
        strict_must_keep=strict_must_keep,
        realism_pass=realism_pass,
    )

    choices = base_expanded.get("variation_choices", [])
    if not choices:
        return [base_expanded]

    variations_expanded: List[Dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        var_scene = dict(base_expanded)
        var_scene["variation_index"] = idx
        var_scene["shot_type"] = choice["shot_type"]
        var_scene["person"] = choice["person_description"]
        var_scene["pose"] = choice["pose"]
        var_scene["camera"] = (
            f"Handheld phone photo, {choice['tilt_degrees']} degree tilt, "
            f"subject in {choice['subject_position']}, {choice['cropped_edges']}"
        )
        var_scene["final_prompt"] = choice.get("final_prompt", base_expanded.get("final_prompt", ""))
        var_scene["pose_framing_choice"] = choice
        variations_expanded.append(var_scene)

    return variations_expanded


# --- PromptExpander Class Wrapper ---

class PromptExpander:
    """Prompt expander class providing configured prompt expansion."""

    def __init__(self, config: Optional[Config] = None, client: Optional[GenAIClient] = None):
        self.config = config or load_config(require_api_key=False)
        self.client = client or GenAIClient(config=self.config)

    def expand(
        self,
        short_prompt: str,
        shot_type: Optional[str] = None,
        n: Optional[int] = None,
        strict_must_keep: Optional[bool] = None,
        realism_pass: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Expands the user prompt using the configured client and knowledge base."""
        return expand(
            short_prompt=short_prompt,
            shot_type=shot_type,
            n=n,
            config=self.config,
            client=self.client,
            strict_must_keep=strict_must_keep,
            realism_pass=realism_pass,
        )

    def expand_variations(
        self,
        short_prompt: str,
        n: int = 4,
        shot_type: Optional[str] = None,
        strict_must_keep: Optional[bool] = None,
        realism_pass: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Expands into N distinct variations with different poses and framing."""
        return expand_variations(
            short_prompt=short_prompt,
            n=n,
            shot_type=shot_type,
            config=self.config,
            client=self.client,
            strict_must_keep=strict_must_keep,
            realism_pass=realism_pass,
        )


if __name__ == "__main__":
    print("--- Running PromptExpander Demo ---")
    mock_config = load_config(require_api_key=False)
    mock_config.mock_mode = True

    demo_expander = PromptExpander(config=mock_config)
    result = demo_expander.expand("POV hands holding meat tray in aisle", shot_type="pov_hands")

    print(f"Plate: {result['plate']}")
    print(f"Shot Type: {result['shot_type']}")
    print(f"Person: {result['person']}")
    print(f"Pose: {result['pose']}")
    print(f"Camera: {result['camera']}")
    print(f"Lighting: {result['lighting']}")
    print(f"Imperfections: {result['imperfections']}")
    print(f"\nFinal Prompt:\n{result['final_prompt']}")
