# CirklaGen

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cirklagen.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Shlok2814%2FCirklaGen-blue?logo=github)](https://github.com/Shlok2814/CirklaGen)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**CirklaGen** is an AI-powered visual generation system that transforms short text prompts into photorealistic, candid smartphone-style photos of customers holding an authentic **Cirkla molded-fiber meat tray** inside a **Trader Joe's meat aisle**. Real aisle photos are reused as background plates to ensure the store environment stays consistent.

> 🌐 **Live Web App**: [https://cirklagen.streamlit.app](https://cirklagen.streamlit.app) (Streamlit Community Cloud)

---

## 🚀 Streamlit Community Cloud Deployment

To deploy this repository live on Streamlit Cloud:

1. Sign in to [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **Create app** (or **New app**).
3. Fill in the repository settings:
   - **Repository**: `Shlok2814/CirklaGen`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `cirklagen` (gives `https://cirklagen.streamlit.app`)
4. Click **Advanced settings...** -> **Secrets** and enter:
   ```toml
   GEMINI_API_KEY = "your_google_gemini_api_key_here"
   ```
5. Click **Deploy!** Streamlit Cloud installs `packages.txt` and `requirements.txt` automatically.

---

## Universal Project Rules

- **Python 3.11** virtual environment.
- **Streamlit UI** & **Google GenAI SDK** (`google-genai`).
- **Secrets in `.env` only**: API keys and secrets live exclusively in `.env`, never in code.
- **Configuration-Driven**: All model names, thresholds, finishing defaults, and paths are defined in `config.yaml`.
- **Modularity**: Small, testable modules. Every module has a `__main__` demo or a pytest.
- **Ground Truth**: The **only** source of truth about the tray is `knowledge/product.json` and the images in `assets/trays/`. Product details are never invented.
- **Provenance & Watermarks**: Every output is an AI-generated mockup. Provenance metadata and watermarks are never stripped.
- **Resilient APIs**: Exponential backoff retries and rate-limit handling with readable error messages.
- **No Silent Failures**: Logging to console and `outputs/logs/cirklagen.log`.

---

## Project Structure

```text
cirklagen/
  config.yaml
  .env.example
  requirements.txt
  knowledge/
    product.json
  prompts/
    expander_system.txt
  assets/
    raw_trays/
    trays/
    plates_raw/
    plates/
    persons/
    reference/
  src/
    __init__.py
    config.py
    client.py
    expander.py
    generator.py
    qa.py
    finish.py
    assets_prep.py
    pipeline.py
    metadata.py
    adapters/
      __init__.py
      gemini.py
      openai.py
      fal.py
  app.py
  scripts/
    run_examples.py
    build_report.py
    bakeoff.py
    check_output.py
  phase2/
  docs/
  tests/
  outputs/
    runs/
    examples/
    logs/
  README.md
  cc.md                        # Claude Code & Pipeline Architecture Blueprint
```

---

## Setup & Installation

### 1. Create and Activate Virtual Environment
Use Python 3.11:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Secrets
Copy the environment template and add your API key:
```powershell
copy .env.example .env
```
In `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## How to Run

### Run the Web Application
```powershell
streamlit run app.py
```
Or directly using the virtual environment:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

### Run Test Suite
```powershell
pytest -v
```

### Run Module Demos
Every module has an executable demo entrypoint:
```powershell
python src/config.py
python src/client.py
python src/expander.py
python src/generator.py
python src/qa.py
python src/finish.py
python src/assets_prep.py
python src/pipeline.py
python src/metadata.py
python scripts/run_examples.py
python scripts/build_report.py
python scripts/bakeoff.py
python scripts/check_output.py
```
