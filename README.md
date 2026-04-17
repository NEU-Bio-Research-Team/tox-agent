# ToxAgent (v0.0.6 Beta)

A multi-agent AI platform for molecular toxicity analysis from SMILES, combining:
- Screening mechanisms from graph/SMILES models
- Structural explanation (atom/bond attribution)
- Research context enrichment from external sources
- Structured report synthesis for direct R&D workflow integration

Production web app: https://tox-agent.web.app

---

## 1) Current Version and Updates

### Current Version
- Frontend: **0.0.6 (Beta)**
- Default workspace mode: **tox21_only**
- Workspace priority: **safety_first**

### What's New in 0.0.6
- Added new tox-type mechanism model options:
  - `tox21_ensemble_3_best` (ChemBERTa + MolFormer + Pretrained-GIN).
  - `tox21_pretrained_gin_model`.
- Updated analyze routing so explainer engine follows selected `tox_type_model`.
- For ensemble mode, explainer now routes to the designated best-member engine.
- Extended backend serving with pretrained-GIN Tox21 inference support.

### Upgrade Note
- If you have an old tab open, please hard refresh your browser to get the latest bundle.

### Current 2-Head Model Ranking (Web Selectable)

The following ranking covers the 2-head models currently exposed in the web UI model selectors
(`Binary Toxicity Model` and `Toxicity Type Model`), ranked by `joint_auc_beta3`.

Source of truth:
- `models/dualhead_model_ranking.csv`
- `frontend/src/app/components/hero-section.tsx`

| Rank | Model key (web option value) | UI label (short) | joint_auc_beta3 |
|---|---|---|---:|
| 1 | `dualhead_ensemble6_simple` | Ensemble-6 Simple (Recommended) | 0.8467 |
| 2 | `dualhead_ensemble3_weighted` | Ensemble-3 Weighted | 0.8466 |
| 3 | `dualhead_ensemble3_simple` | Ensemble-3 Simple | 0.8455 |
| 4 | `dualhead_ensemble5_simple` | Ensemble-5 Simple | 0.8451 |
| 5 | `pretrained_2head_herg_molformer_model` | MolFormer Dual-Head (Full) | 0.8270 |
| 6 | `pretrained_2head_herg_chemberta_model` | ChemBERTa Dual-Head (Full) | 0.8178 |
| 7 | `pretrained_2head_herg_pubchem_model` | PubChem Dual-Head (Full) | 0.8133 |
| 8 | `pretrained_2head_herg_pubchem_quick` | PubChem Dual-Head (Quick) | 0.7896 |
| 9 | `pretrained_2head_herg_molformer_quick` | MolFormer Dual-Head (Quick) | 0.7746 |
| 10 | `pretrained_2head_herg_chemberta_quick` | ChemBERTa Dual-Head (Quick) | 0.7725 |

Legacy compatibility note:
- `tox21_ensemble_3_best` is still available in the UI for backward compatibility and is served as the same ensemble spec as `dualhead_ensemble3_simple`.

---

## 2) ToxAgent Overview

ToxAgent is a multi-agent system with 3 main layers:

1. Presentation layer (Frontend)
   - React + Vite
   - Displays quick verdict, full report, heatmap, and molecule image

2. Agent orchestration layer
   - InputValidator Agent: input normalization/validation
   - Screening Agent: calls model pipeline for prediction + attribution
   - Researcher Agent: fetches research context (PubChem/PubMed)
   - Writer Agent: synthesizes results into a final structured report
   - Orchestrator Agent: coordinates, fallbacks, and state repair as needed

3. Model/API layer
   - FastAPI model server with endpoints `/health`, `/predict`, `/analyze`, `/agent/analyze`
   - Graph models + explainability
   - OOD guard and confidence information

ToxAgent aims to:
- Not just provide toxicity scores
- But also explain why the model made its assessment
- And provide literature context to support research decisions

---

## 3) ToxAgent Workflow (ASCII)

```text
+-------------------+
| User enters SMILES|
+---------+---------+
          |
          v
+-------------------+
| Frontend (React)  |
| POST /agent/analyze
+---------+---------+
          |
          v
+-----------------------------+
| Orchestrator Agent          |
+-----+-----------------+-----+
      |                 |
      v                 v
+-------------+   +----------------+
| Input       |   | Researcher     |
| Validator   |   | Agent          |
+------+------+   +-------+--------+
       |                  |
       v                  v
+----------------+   +----------------------+
| Screening Agent|   | PubChem / PubMed     |
+-------+--------+   +----------------------+
        |
        | tool call: analyze_molecule
        v
+----------------------------------------------+
| Model Server (/analyze)                      |
| - Canonicalize + validation                  |
| - Toxicity scoring                           |
| - Tox21 mechanism scores                     |
| - Structural explanation (heatmap + molecule)|
| - OOD assessment                             |
+----------------------+-----------------------+
                       |
                       v
              +----------------+
              | Writer Agent   |
              | final_report   |
              +--------+-------+
                       |
                       v
+-----------------------------------------------+
| Frontend Report                               |
| - Executive summary                           |
| - Clinical/mechanism sections                 |
| - Structural explanation images + top atoms   |
| - Literature context + recommendations        |
+-----------------------------------------------+
```

---

## 4) How to Use ToxAgent

### Easiest Way (Production)
1. Open https://tox-agent.web.app
2. Enter SMILES
3. Adjust threshold in Settings if needed
4. Click Analyze
5. View quick verdict and full report
6. Click version badge to view release notes anytime

### Run Local Backend (FastAPI)
```bash
conda env create -f environment.yml
conda activate drug-tox-env
pip install -r model_server/requirements.txt
uvicorn model_server.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Quick check:
```bash
curl -sS http://127.0.0.1:8080/health

curl -sS -X POST http://127.0.0.1:8080/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "smiles":"CC(=O)Oc1ccccc1C(=O)O",
    "clinical_threshold":0.6,
    "mechanism_threshold":0.6,
    "return_all_scores":true,
    "explain_only_if_alert":false,
    "explainer_epochs":80,
    "explainer_timeout_ms":30000
  }'
```

### Run Local Frontend
```bash
cd frontend
npm install
npm run dev
```

### Build UI and Deploy Hosting (from repo root)
```bash
npm run build
npm run deploy:hosting
```

Notes:
- `npm run build` now builds `frontend/` directly.
- `npm run deploy:hosting` will build first, then deploy only Firebase Hosting.

If backend is running on a different port, set env variable before running frontend:
```bash
export VITE_API_BASE_URL=http://127.0.0.1:8080
npm run dev
```

### Run Tox21 Script (current workspace mode)
```bash
python scripts/train_tox21_gatv2.py --device cuda --config config/tox21_gatv2_config.yaml
python scripts/predict_tox21.py --smiles "CCO" --device cuda
```

---

## 5) Other Important Details

### 5.1 API Quick Reference
- `GET /health`: model/runtime status
- `POST /predict`: quick prediction for a SMILES
- `POST /analyze`: returns clinical + mechanism + explanation
- `POST /agent/analyze`: full multi-agent workflow + final report
- `POST /extract-smiles-from-image`: upload PNG/JPG/JPEG/WebP and extract validated canonical SMILES

### 5.2 Workspace Mode and Guard Rails
Current workspace config:
- `mode: tox21_only`
- `primary_dataset: tox21`
- `clintox_enabled: false`
- `tox21_enabled: true`

Meaning:
- Pipeline and scripts prioritize Tox21
- ClinTox training/eval paths are restricted by guard rails

### 5.3 Reliability and Explanation
- GNNExplainer is an attribution tool, not an absolute certificate
- OOD flag should be prioritized in decision making
- Model results should be combined with expert review and experimental data

### 5.4 Quick Troubleshooting
- Symptom: heatmap present but no molecule image
  - Check drawing runtime libs in container (Cairo/font)
- Symptom: slow first request
  - Check warm instance/startup probe on Cloud Run
- Symptom: frontend calls localhost in production
  - Check `VITE_API_BASE_URL` and Firebase Hosting rewrite

### 5.5 Next Directions
- Add benchmark/telemetry for latency and quality report per session
- Enhance explainability with stronger chemical constraints
- Strengthen failure registry and feedback loop process

### 5.6 New Input Modes (MVP)
Frontend now supports 3 input modes in the main analysis panel:
- Type SMILES directly.
- Draw molecule with **Ketcher** and export SMILES into the main input.
- Upload structure image (PNG/JPG/JPEG/WebP) and extract SMILES from image OCR.

Backend image OCR flow:
- Validate MIME type and extension.
- Enforce max upload size (`5MB` by default).
- Preprocess image with Pillow (RGB + autocontrast).
- Run MolScribe image-to-SMILES model.
- Validate and canonicalize with RDKit.

Standardized OCR error codes:
- `unsupported_image_format`
- `image_too_large`
- `smiles_not_detected`
- `invalid_smiles_from_image`
- `extraction_service_unavailable`

Performance notes:
- Existing analysis pipeline (`/agent/analyze`) is unchanged.
- OCR is a separate endpoint and only runs when user uploads an image.
- MolScribe preload is disabled by default to avoid startup dependency on Hugging Face connectivity.

### 5.7 OCR Runtime Environment Variables
- `SMILES_IMAGE_MAX_BYTES` (default: `5242880`)
- `MOLSCRIBE_PRELOAD_ON_STARTUP` (default: `false`)
- `MOLSCRIBE_AUTO_DOWNLOAD` (default: `true`)
- `MOLSCRIBE_REPO_ID` (default: `yujieq/MolScribe`)
- `MOLSCRIBE_CHECKPOINT_NAME` (default: `swin_base_char_aux_1m.pth`)
- `MOLSCRIBE_MODEL_PATH` (optional explicit local checkpoint path)
- `MOLSCRIBE_DEVICE` (`cpu` or `cuda`, default follows runtime device)

---

## 6) Detailed Run Guide (Step-by-Step)

This guide is for re-running the project with the new Draw + Image OCR features.

### Step 1: Prepare Environment

Option A (recommended): Conda
```bash
conda env create -f environment.yml
conda activate drug-tox-env
```

Option B: Existing Python environment
- Ensure Python, torch stack, RDKit, and project dependencies are already available.

### Step 2: Install Backend Dependencies
```bash
pip install -r model_server/requirements.txt
```

Install MolScribe without dependency override of torch:
```bash
pip install --no-deps molscribe==1.1.1
```

### Step 3: Configure OCR (Optional but Recommended)

PowerShell example:
```powershell
$env:MOLSCRIBE_PRELOAD_ON_STARTUP="false"
$env:MOLSCRIBE_AUTO_DOWNLOAD="true"
$env:SMILES_IMAGE_MAX_BYTES="5242880"
```

For startup-sensitive deployments (for example FPT AI Factory), keep
`MOLSCRIBE_PRELOAD_ON_STARTUP=false` and provide a local checkpoint via
`MOLSCRIBE_MODEL_PATH` if outbound Hugging Face access is restricted.

If you already downloaded a checkpoint:
```powershell
$env:MOLSCRIBE_MODEL_PATH="D:/path/to/swin_base_char_aux_1m.pth"
```

### Step 4: Start Backend
```bash
uvicorn model_server.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Quick checks:
```bash
curl -sS http://127.0.0.1:8080/health

curl -sS -X POST http://127.0.0.1:8080/extract-smiles-from-image \
  -F "file=@test_data/example_molecule.png"
```

### Step 5: Start Frontend
```bash
cd frontend
npm install
npm run dev
```

If backend is custom host/port, set:
```bash
export VITE_API_BASE_URL=http://127.0.0.1:8080
```

### Step 6: Validate End-to-End
1. Open web UI.
2. Use **Type SMILES** tab and run analysis.
3. Use **Draw Molecule** tab, export SMILES, then analyze.
4. Use **Upload Image** tab, extract SMILES, then analyze.
5. Try invalid file type or oversized image and verify specific error code appears in UI.

### Step 7: Build and Deploy
```bash
npm run build
npm run deploy:hosting
```

For container builds (Cloud Run), Dockerfile is updated to:
- install OCR runtime dependencies,
- install MolScribe with `--no-deps`,
- skip OCR preload by default so startup is non-blocking.

---

## Project Structure (summary)

```text
tox-agent/
|- agents/              # Multi-agent orchestration
|- backend/             # Data/model/explainer core
|- model_server/        # FastAPI serving layer
|- frontend/            # React web app
|- scripts/             # Train/predict/evaluate utilities
|- config/              # Workspace + model config
|- models/              # Saved model artifacts
`- deploy/              # Cloud Run/Firebase deployment assets
```

---

## Citation

```bibtex
@inproceedings{nguyen2026smilesgnn,
  title     = {Advancing Clinical Toxicity Prediction Through Multimodal Fusion
               of SMILES Sequences and Molecular Graph Representation},
  author    = {Nguyen, Thuy-Quynh and Nguyen, Trong-Nghia and Nguyen, Quang-Minh
               and Le, Duc-Minh and Ho, Nhat-Minh Nguyen and Doan, Thanh-Long Dai},
  year      = {2026}
}
```

---

## License

This project is for research purposes. Tox21 and ClinTox are from MoleculeNet (MIT License).
