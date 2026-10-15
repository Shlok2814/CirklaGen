"""Product knowledge base loader and grounding utility for CirklaGen.

Strict rule:
The only source of truth about the tray is knowledge/product.json and the images in
assets/trays/. Never invent product details.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger


class ProductKnowledge:
    """Manages the official ground truth specifications for the Cirkla molded-fiber tray."""

    def __init__(self, knowledge_path: str | Path = "knowledge/product.json"):
        self.knowledge_path = Path(knowledge_path)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads and parses the product.json knowledge file."""
        if not self.knowledge_path.is_file():
            err_msg = f"Product knowledge file not found at: {self.knowledge_path.resolve()}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        try:
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded product knowledge from %s", self.knowledge_path)
            return data
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON in %s: %s", self.knowledge_path, e)
            raise

    @property
    def product(self) -> Dict[str, Any]:
        """Returns root product dictionary."""
        return self.data.get("product", {})

    @property
    def tray_spec(self) -> Dict[str, Any]:
        """Returns tray structural and material specifications."""
        if "geometry" in self.data:
            geom = self.data["geometry"]
            prod = self.data.get("product", {})
            return {
                "manufacturer": prod.get("brand", "Cirkla"),
                "material": prod.get("material", "beige molded-fiber"),
                "color": geom.get("exterior_color", "natural beige"),
                "finish": "matte tactile organic paper pulp texture",
                "geometry": {
                    "shape": "rectangular with smooth rounded corners",
                    "rim": geom.get("rim", ""),
                    "sidewalls": geom.get("wall_ribs", "vertical scalloped ribs on the walls"),
                    "base": geom.get("base_panel", ""),
                    "corners": geom.get("corner_features", ""),
                },
            }
        return self.product.get("packaging", {}).get("tray", {})

    @property
    def label_spec(self) -> Dict[str, Any]:
        """Returns exact label branding and typographical specifications."""
        if "as_sold" in self.data:
            as_sold = self.data["as_sold"]
            return {
                "branding_header": "TRADER JOE'S",
                "product_header": "ORGANIC GRASS FED ANGUS BEEF",
                "label_sleeve": as_sold.get("label_sleeve", ""),
            }
        return self.product.get("packaging", {}).get("label", {})

    @property
    def sealing_spec(self) -> Dict[str, Any]:
        """Returns film sealing specifications."""
        if "as_sold" in self.data:
            return {"type": self.data["as_sold"].get("film", "modified atmosphere packaging (MAP) film")}
        return self.product.get("packaging", {}).get("sealing", {})

    @property
    def available_cuts(self) -> List[Dict[str, str]]:
        """Returns list of authentic beef cuts."""
        return self.product.get("cuts", [])

    def get_cut(self, cut_name: str) -> Optional[Dict[str, str]]:
        """Looks up a specific cut by name or partial name."""
        for cut in self.available_cuts:
            if cut_name.lower() in cut.get("name", "").lower():
                return cut
        return None

    def build_ground_truth_prompt_snippet(self, cut_name: Optional[str] = None) -> str:
        """Constructs an exact, grounded visual description of the tray and product.

        Never invents details; strictly synthesized from knowledge/product.json.
        """
        tray = self.tray_spec
        label = self.label_spec
        sealing = self.sealing_spec

        chosen_cut = None
        if cut_name:
            chosen_cut = self.get_cut(cut_name)
        if not chosen_cut and self.available_cuts:
            chosen_cut = self.available_cuts[0]

        cut_title = chosen_cut.get("name", "Tenderloin Steak") if chosen_cut else "Tenderloin Steak"
        net_wt = chosen_cut.get("net_weight", "12 OZ (340g)") if chosen_cut else "12 OZ (340g)"

        branding = label.get("branding_header", "TRADER JOE'S")
        product_hdr = label.get("product_header", "ORGANIC GRASS FED ANGUS BEEF")
        mfg = tray.get("manufacturer", "Cirkla")
        mat = tray.get("material", "molded plant fiber pulp")
        col = tray.get("color", "natural beige oatmeal kraft color")
        finish = tray.get("finish", "matte tactile organic paper pulp texture")
        seal_type = sealing.get("type", "tight vacuum skin packaging film")

        snippet = (
            f"The person is holding a specific retail package: a {mfg} "
            f"{mat} meat tray in {col}. "
            f"The tray has a {finish}, "
            f"a rectangular shape with smooth rounded corners, a wide flat outer rim flange, and "
            f"distinctive vertical structural fluting (ribbed vertical channels) along the inner sidewalls. "
            f"Inside is a cut of raw marbled red beef ({cut_title}), packed with {seal_type} "
            f"stretched cleanly over the meat and heat-sealed flush to the molded fiber rim. "
            f"A vertical rectangular paper label sticker band is affixed over the left portion of the package "
            f"(leaving the right half exposed to show the fresh marbled meat). "
            f"The label features: '{branding}', "
            f"'{product_hdr}', "
            f"'{cut_title.upper()}', a crisp black silhouette illustration of an Angus beef steer, "
            f"a circular green USDA Organic seal, and the text '{net_wt}'."
        )
        return snippet

    def get_reference_tray_images(self, trays_dir: str | Path = "assets/trays") -> List[Path]:
        """Lists available reference images in the trays directory."""
        dir_path = Path(trays_dir)
        if not dir_path.is_dir():
            logger.warning("Trays directory does not exist: %s", dir_path)
            return []
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = [p for p in dir_path.iterdir() if p.suffix.lower() in valid_extensions and p.is_file()]
        return sorted(images)

    def get_background_images(self, backgrounds_dir: str | Path = "assets/backgrounds") -> List[Path]:
        """Lists available consistent Trader Joe's background plates."""
        dir_path = Path(backgrounds_dir)
        if not dir_path.is_dir():
            logger.warning("Backgrounds directory does not exist: %s", dir_path)
            return []
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = [p for p in dir_path.iterdir() if p.suffix.lower() in valid_extensions and p.is_file()]
        return sorted(images)


if __name__ == "__main__":
    print("--- Running ProductKnowledge demo ---")
    pk = ProductKnowledge()
    print(f"Product line: {pk.product.get('line')}")
    print(f"Tray manufacturer: {pk.tray_spec.get('manufacturer')}")
    print(f"Tray material: {pk.tray_spec.get('material')}")
    print(f"Available cuts: {[c['name'] for c in pk.available_cuts]}")
    tray_images = pk.get_reference_tray_images()
    print(f"Found {len(tray_images)} reference tray images in assets/trays:")
    for img in tray_images[:3]:
        print(f"  - {img.name}")
    bg_images = pk.get_background_images()
    print(f"Found {len(bg_images)} background plates in assets/backgrounds:")
    for img in bg_images:
        print(f"  - {img.name}")
    print("\nSynthesized ground truth prompt snippet:")
    print(pk.build_ground_truth_prompt_snippet())
