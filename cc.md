# CIRKLAGEN: CLAUDE CODE & PIPELINE ARCHITECTURE SPECIFICATION (cc.md)
> **Codebase Context & Technical Pipeline Blueprint for Claude**  
> *Project: CirklaGen – Authentic Retail Packaging Visual Mockup Pipeline*  
> *Target Environment: Python 3.11, Google GenAI SDK (`google-genai`), PyTorch (DINOv2), OpenCV, Pillow, Streamlit*

---

## 1. PROJECT EXECUTIVE SUMMARY & MISSION

**CirklaGen** is an automated, AI-driven retail mockup generation and quality assurance system. It converts short human prompts (e.g., *"me holding ribeye tray in meat aisle"*) into candid, photorealistic smartphone-style photos of shoppers holding an authentic **Cirkla molded plant-fiber meat tray (#5 size)** inside a **Trader Joe's refrigerated meat aisle**.

### Key Value Proposition
Instead of organizing expensive physical photoshoots with packaging prototypes in commercial grocery stores, CirklaGen:
1. Reuses authentic store background plates so the retail aisle context (bunkers, wire racks, Trader Joe's signage, price tags) remains 100% consistent across all shots.
2. Inpaints the customer and the packaging exclusively within a defined placement region using multi-reference multimodal generative models.
3. Enforces strict ground-truth adherence to Cirkla's physical geometry and material properties using DINOv2 visual embeddings and a 10-point checklist.
4. Harmonizes generated subjects with authentic smartphone optical defects (plate-matched sensor grain, color temperature matching, skin texture recovery).

---

## 2. REPOSITORY MAP & DIRECTORY STRUCTURE

```text
cirklagen/
├── .env                       # API keys and secrets (GEMINI_API_KEY)
├── .env.example               # Template for environment secrets
├── config.yaml                # Master configuration (models, thresholds, paths, finishing)
├── knowledge/
│   └── product.json           # Ground-truth specifications for Cirkla #5 meat tray
├── prompts/
│   └── expander_system.txt    # System prompt for Gemini Flash grounded expansion
├── assets/
│   ├── trays/                 # Cutout reference photos of Cirkla trays (with angles.json)
│   ├── raw_trays/             # Original unprocessed photoshoot images of trays
│   ├── plates/                # Trader Joe's refrigerated meat aisle background plates
│   ├── plates_raw/            # High-res raw retail aisle photographs
│   ├── persons/               # Diverse shopper reference photographs (with gender/ethnicity)
│   ├── packaged/              # As-sold reference cuts with paper sleeves & MAP wrap
│   └── reference/             # Benchmark visual reference gallery
├── src/
│   ├── __init__.py            # Package root & public exports
│   ├── config.py              # YAML loader with strong typing and path normalization
│   ├── client.py              # Resilient Google GenAI SDK wrapper (backoff, retry, mock mode)
│   ├── expander.py            # LLM prompt grounding against product.json & shot presets
│   ├── generator.py           # Multi-reference inpainting generator with region-lock enforcement
│   ├── qa.py                  # DINOv2 feature similarity + vision LLM 10-point checklist
│   ├── finish.py              # Plate-matched noise, LAB white balance, shadow grain, optical finishing
│   ├── pipeline.py            # Master orchestration pipeline (stages 1-7, repair, manifest)
│   ├── metadata.py            # Strict schema validation & persistence for product.json
│   ├── assets_prep.py         # Inpainting pen removal and automated tray cutout utilities
│   ├── diversity.py           # Perceptual hash (pHash) batch diversity validation
│   ├── logger.py              # Centralized logging to console and outputs/logs/cirklagen.log
│   └── adapters/              # Multi-model bakeoff adapters
│       ├── __init__.py
│       ├── gemini.py          # Google Imagen 3 adapter
│       ├── openai.py          # OpenAI DALL-E adapter
│       └── fal.py             # Fal.ai Flux Dev adapter
├── app.py                     # High-contrast dark-mode Streamlit UI (ElevenLabs aesthetic)
├── scripts/
│   ├── run_examples.py        # Runs predefined benchmark retail scenarios
│   ├── build_report.py        # Generates HTML & markdown bakeoff evaluation reports
│   ├── bakeoff.py             # Multi-model competitive evaluation harness
│   └── check_output.py        # Headless output integrity and region lock verification
├── tests/                     # 25+ automated pytest suites (AppTest, QA, region lock, diversity)
└── outputs/
    ├── runs/                  # Generated timestamped run folders (images, sidecars, manifest)
    ├── examples/              # Benchmark scenario outputs
    └── logs/                  # System operational log files
```

---

## 3. GROUND TRUTH SPECIFICATIONS (`knowledge/product.json`)

All pipeline components must strictly honor `knowledge/product.json`. Product details must **never be hallucinated**.

```json
{
  "product": {
    "name": "Cirkla Molded Fiber Meat Tray",
    "brand": "Cirkla",
    "material": "beige molded-fiber",
    "size_in": "10.28 x 8.08 in",
    "size_label": "#5",
    "retailer": "Trader Joe's",
    "cuts": [
      { "name": "Tenderloin Steak", "net_weight": "12 OZ (340g)", "label_title": "ORGANIC GRASS FED ANGUS BEEF TENDERLOIN STEAK" },
      { "name": "Top Sirloin Steak", "net_weight": "8 OZ (227g)", "label_title": "ORGANIC GRASS FED ANGUS BEEF TOP SIRLOIN STEAK" },
      { "name": "Ribeye Steak Boneless", "net_weight": "Variable / 10-14 OZ", "label_title": "ORGANIC GRASS FED ANGUS BEEF RIBEYE STEAK BONELESS" },
      { "name": "New York Strip Steak Boneless", "net_weight": "Variable / 10-12 OZ", "label_title": "ORGANIC GRASS FED ANGUS BEEF NY STRIP STEAK BONELESS" }
    ]
  },
  "geometry": {
    "exterior_color": "natural beige unbleached molded plant fiber pulp",
    "rim": "wide flat perimeter sealing flange with uniform thickness for hermetic film sealing",
    "base_panel": "rectangular recessed base panel with smooth rounded corner transitions",
    "wall_ribs": "vertical scalloped ribs and structural fluting on the walls providing rigidity",
    "corner_features": "distinctive pyramid features in the corners for stacking strength",
    "liner": "smooth glossy white liner inside the cavity protecting fiber from moisture"
  },
  "as_sold": {
    "film": "tight clear modified atmosphere packaging (MAP) barrier film heat-sealed flush to the perimeter rim flange",
    "contents": "raw fresh USDA Organic Grass Fed Angus Beef steak resting inside on an absorbent meat pad",
    "label_sleeve": "vertical paper label wrap-around sleeve positioned over the left side of the tray featuring 'TRADER JOE'S', 'ORGANIC GRASS FED ANGUS BEEF', cut title in bold red, steer silhouette, and USDA Organic seal, leaving right half exposed to show marbled steak"
  }
}
```

### Strict Anti-Drift Invariants
- **MUST KEEP**: Natural unbleached kraft-beige cellulose pulp texture, scalloped wall fluting, pyramid corners, white inner cavity liner, embossed Cirkla logo on bottom, vertical paper sleeve on left half, clear MAP wrap, proportioned human hands.
- **MUST NOT**: Styrofoam, shiny gray plastic, smooth walls without fluting, changed rim thickness, gibberish sleeve lettering, extra divider compartments, floating trays, six-fingered or distorted hands.

---

## 4. END-TO-END PIPELINE WORKFLOW (7 STAGES)

The core orchestrator is `src.pipeline.GenerationPipeline`. The workflow proceeds through 7 deterministic stages:

```
[User Short Prompt]
       │
       ▼
1. Prompt Expansion (src.expander)
   ├── Ingest knowledge/product.json
   ├── Infer shot type (front_chest, pov_hands, side_profile, cart_topdown, closeup_hands)
   ├── Select matching plate (Trader Joe's Shelf) & person reference
   └── Inject strict MUST KEEP block & 5 anti-drift rules
       │
       ▼
2. Reference Asset Selection (src.generator)
   ├── Select authentic tray cutouts matching camera angle (angles.json)
   └── Compute placement bounding box [ymin, xmin, ymax, xmax]
       │
       ▼
3. Multi-Reference Generation (src.generator)
   ├── Send plate + tray refs + person ref to Gemini Imagen 3
   ├── Region Lock Enforcement: strict zero-change check outside mask
   └── Perceptual difference check (rejects raw plate + copy-paste artifacts)
       │
       ▼
4. Dual-Pass Quality Assurance (src.qa)
   ├── Vision LLM evaluates 10-point checklist
   ├── Crop tray region and extract DINOv2 ViT-S/14 patch embeddings
   ├── Cosine similarity against ground-truth tray library (threshold >= 0.60)
   └── Verify Trader Joe's sleeve layout & typography (threshold >= 0.70)
       │
       ▼
5. Automated Repair & Anti-Drift Fallback (src.finish)
   ├── If hands distorted: fix_region with local prompt
   └── If tray drifted: perspective warp authentic cutout tray onto region
       │
       ▼
6. Candid Smartphone Optical Finishing (src.finish)
   ├── Measure plate background ISO noise, grain size, & sharpness
   ├── Synthesize matching monochromatic grain over generated subject
   ├── LAB color temperature balancing to match store refrigerated deck
   ├── Mild skin texture softening (recovers natural pores from AI airbrushing)
   └── Embed C2PA / EXIF provenance metadata (never stripped)
       │
       ▼
7. Batch Diversity, Ranking & Run Manifest (src.pipeline)
   ├── Calculate perceptual hash (pHash) pairwise distances (threshold >= 8)
   ├── Rank candidate variations: max checklist passes -> DINOv2 similarity -> QA score
   └── Write outputs/runs/{run_id}/manifest.json and img_{i}.json sidecars
```

---

## 5. COMPLETE PYTHON API CONTRACTS

### 5.1 Pipeline Execution (`src.pipeline.GenerationPipeline`)

```python
from src.pipeline import GenerationPipeline
from src.config import load_config

cfg = load_config()
pipeline = GenerationPipeline(config=cfg)

result = pipeline.run(
    short_prompt="me holding ribeye tray in meat aisle",
    shot_type="front_chest",             # "front_chest" | "pov_hands" | "side_profile" | "cart_topdown" | "closeup_hands"
    n=4,                                 # Number of candidate variations (1-6)
    person_ref=None,                     # Optional Path or PIL.Image for person reference
    use_composite=False,                 # Force cutout composition upfront
    finishing=True,                      # Apply plate-matched phone optical finishing
    manual_mode=False,                   # If True, returns reference bundle for Gemini Web app
    progress_callback=lambda pct, msg: print(f"[{int(pct*100)}%] {msg}"),
    strict_must_keep=True,               # Prepends strict invariant block to prompt
    auto_composite_fallback=True,        # Automatically composites real tray if model drifts
    realism_pass=True,                   # Injects candid phone optical instructions
)
```

#### Output Schema (`result` dict)
```python
{
    "run_id": "20261015_120000_front_chest",
    "status": "GENERATED",               # "GENERATED" | "NOT GENERATED" | "rejected"
    "manifest": { ... },                # Full manifest data saved to disk
    "best_image": {
        "variation_index": 0,
        "file": "outputs/runs/run_id/img_0_finished.png",
        "raw_image": "outputs/runs/run_id/img_0_raw.png",
        "finished_image": "outputs/runs/run_id/img_0_finished.png",
        "qa_score": 0.88,
        "passed_qa": True,
        "checklist_passes": 10,
        "similarity_max": 0.842,
        "notes": "PASS: authentic fiber texture, fluting verified"
    },
    "variations": [
        {
            "variation_index": 0,
            "raw_image": "outputs/runs/run_id/img_0_raw.png",
            "finished_image": "outputs/runs/run_id/img_0_finished.png",
            "passed_qa": True,
            "qa_score": 0.88,
            "checklist_passes": 10,
            "similarity_max": 0.842,
            "qa_feedback": "PASS: packaging geometry matches ground truth",
            "repaired": False,
            "repair_type": None,
            "pose_framing_choice": {
                "person_reference": "shopper_man_henley.png",
                "tilt_degrees": -3.5,
                "gaze": "looking at packaging"
            }
        }
    ]
}
```

### 5.2 Prompt Expansion (`src.expander.expand`)

```python
from src.expander import expand

expanded_data = expand(
    short_prompt="shopper inspecting packaging fluting",
    shot_type="side_profile",
    n=4,
    config=cfg,
    strict_must_keep=True,
    realism_pass=True,
)
```
Returns a dict containing:
- `final_prompt`: Exhaustive prompt string with physical tray geometry and Trader Joe's retail bunker descriptors.
- `plate`: Filename of the background plate (e.g. `Trader Joe's Shelf.jpeg`).
- `shot_type`: Standardized shot type token.
- `placement_region`: Bounding box `[ymin, xmin, ymax, xmax]` in normalized coordinates (0.0 to 1.0).
- `variation_choices`: Array of per-variation framing, angle tilts, and shopper references.

### 5.3 Quality Assurance (`src.qa.evaluate_image`)

```python
from src.qa import evaluate_image
from PIL import Image

candidate_img = Image.open("path/to/candidate.png")
qa_result = evaluate_image(
    image=candidate_img,
    config=cfg,
    client=pipeline.client,
)

print(qa_result.passed)            # bool: True if meets checklist & similarity thresholds
print(qa_result.overall_score)     # float: 0.0 - 1.0
print(qa_result.similarity_max)    # float: DINOv2 cosine similarity against ground truth
print(qa_result.checklist)         # dict: Detailed breakdown of 10 verification items
```

### 5.4 Optical Finishing (`src.finish.apply_plate_matched_finishing`)

```python
from src.finish import apply_plate_matched_finishing

finished_img = apply_plate_matched_finishing(
    generated_img=candidate_img,
    plate_img=Image.open("assets/plates/Trader Joe's Shelf.jpeg"),
    placement_bbox=[0.15, 0.15, 0.95, 0.85],
    config=cfg,
)
```

---

## 6. CRITICAL PIPELINE INVARIANTS & CONSTRAINTS

When designing or modifying a pipeline for CirklaGen, you **MUST NEVER VIOLATE** these engineering rules:

1. **Strict Region Locking**:
   - The background plate outside `placement_region` must be 100% byte-for-byte identical to the original background plate.
   - Any pixel modification outside the inpainting mask triggers run failure (`assert_pixels_outside_region_match`).
2. **Never Paste Tray Cutouts As Raw Outputs**:
   - The model must generate the tray in-scene with realistic lighting, shadows, and natural human finger occlusion.
   - If the output is near-identical to the background plate with only a pasted graphic, the run is rejected as `NOT GENERATED`.
3. **No Stylized/Studio Renders**:
   - The camera perspective must strictly look like an ordinary customer's phone camera (24-28mm focal length, non-commercial lighting, natural skin with pores, slight tilt).
4. **Secrets in `.env` Only**:
   - `GEMINI_API_KEY` is loaded via `src.config.get_api_key()`. Never hardcode keys in scripts or tests.
5. **Deterministic Mock Mode**:
   - Setting `mock_mode: true` in `config.yaml` or running with `--mock` allows the entire pipeline and test suite to execute offline without billable API calls.

---

## 7. HOW TO BUILD / EXTEND PIPELINES FOR THIS PROJECT

If you are asked to construct a new pipeline variant (e.g., ComfyUI Flux workflow, multi-angle batch runner, video motion generation, or custom diffusion server):

### Pattern A: Custom Generative Model Adapter
Implement the abstract interface in `src/adapters/`:
```python
from src.adapters import BaseAdapter

class CustomPipelineAdapter(BaseAdapter):
    def generate_variation(
        self,
        prompt: str,
        plate_image: Image.Image,
        tray_references: List[Image.Image],
        placement_region: List[float],
        seed: int,
    ) -> Image.Image:
        # 1. Prepare mask from placement_region
        # 2. Invoke model with reference conditioning
        # 3. Composite output strictly back onto original plate
        # 4. Return synthesized Image
        pass
```

### Pattern B: Batch Video Generation Pipeline
1. Take the best still candidate from `pipeline.run()`.
2. Crop the shopper and packaging.
3. Feed into image-to-video diffusion (e.g., Luma / Kling / SVD) with low motion bucket (0.15-0.25) simulating a customer shifting their weight in front of the refrigerated case.
4. Pass generated MP4 through optical stabilization and grain matching.

### Pattern C: Headless CLI Scripting
```python
import sys
from src.pipeline import GenerationPipeline

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "me holding tray"
    pipe = GenerationPipeline()
    res = pipe.run(short_prompt=prompt, n=4)
    print("Best image saved at:", res["best_image"]["finished_image"])

if __name__ == "__main__":
    main()
```

---

## 8. CLI COMMANDS CHEATSHEET

```powershell
# 1. Activate Environment
.\.venv\Scripts\Activate.ps1

# 2. Run Test Suite (AppTest, QA, Region Lock, Framing Diversity)
pytest tests/test_app.py tests/test_region_lock.py tests/test_framing_diversity.py -v -W ignore

# 3. Run Streamlit Interactive Studio
streamlit run app.py --server.port 8501

# 4. Run Headless Scenario Generation
python -m src.pipeline "candid shopper holding ribeye in meat aisle" --shot-type front_chest -n 3

# 5. Run Multi-Model Bakeoff
python scripts/bakeoff.py --prompt "holding organic tenderloin in aisle"

# 6. Verify Output Region Lock Integrity
python scripts/check_output.py outputs/runs/latest/img_0_finished.png
```

---

## 9. STREAMLIT CLOUD PRODUCTION DEPLOYMENT

1. Sign in to [share.streamlit.io](https://share.streamlit.io).
2. Connect repository: `Shlok2814/CirklaGen` (branch `main`, file `app.py`).
3. Under **Advanced Settings > Secrets**, configure:
   ```toml
   GEMINI_API_KEY = "your_google_gemini_api_key_here"
   ```
4. Streamlit Cloud automatically provisions Debian system packages via `packages.txt` (`libgl1`, `libglib2.0-0`) and installs Python wheels via `requirements.txt`.
5. Public App URL: `https://cirklagen.streamlit.app`

