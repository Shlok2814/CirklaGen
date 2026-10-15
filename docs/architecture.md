# CirklaGen Architecture Overview

## Core System Modules

1. **Expander (`src.expander`)**: Expands short user prompt into a photorealistic candid scene prompt, strictly grounded by `knowledge/product.json`.
2. **Generator (`src.generator`)**: Calls generative image models via unified adapters (`src.adapters`).
3. **Quality Assurance (`src.qa`)**: Audits output fidelity against tray specifications using vision models (threshold: 0.6).
4. **Finishing (`src.finish`)**: Adds smartphone camera optical artifacts (grain, vignette, softness, JPEG compression, tilt).
5. **Metadata & Provenance (`src.metadata`)**: Preserves all provenance, watermarks, and run parameters.
