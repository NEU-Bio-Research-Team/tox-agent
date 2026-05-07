# ToxAgent

ToxAgent is a multi-agent toxicity analysis platform for SMILES-based compound screening. The current workspace combines a React/Vite frontend, a FastAPI model server, multi-agent orchestration, explainability tooling, and Firebase-backed persistence for analysis history and report chat sessions.

Production app: https://tox-agent.web.app

## Current Snapshot

- Frontend bundle version: `0.0.9`
- Root npm package version: `1.0.0`
- Checked-in workspace mode: `full`
- Primary dataset: `tox21`
- Threshold policy: `safety_first`
- Frontend deploy target: Firebase Hosting from `frontend/dist`
- Backend deploy target: Cloud Run service `tox-agent-cpu`
- Auth and persistence: Firebase Auth + Cloud Firestore

Version note:
- The UI bundle version and the root npm package version are different on purpose. The frontend release label currently tracks `0.0.9`, while the repo-level npm scripts package is `1.0.0`.

## What The System Does

ToxAgent currently supports these product flows:

- Accept SMILES from plain text input, molecular drawing, or image upload plus OCR.
- Run fast prediction with `/predict`, full structured analysis with `/analyze`, or multi-agent report generation with `/agent/analyze`.
- Produce toxicity reports that combine clinical signal, mechanism signal, structural explanation, OOD assessment, and follow-up recommendations.
- Support grounded report chat through `/agent/chat` and `/agent/chat/stream`.
- Persist authenticated user analyses and chat sessions in Firestore.
- Route hosted frontend API calls through Firebase Hosting rewrites to the Cloud Run backend.

## System Architecture

### 1. Frontend Layer

`frontend/` is the only active web client shipped to production.

- Stack: React 18, React Router 7, TypeScript, Vite 6, MUI, and Radix UI.
- Main app entrypoint: `frontend/src/main.tsx` -> `frontend/src/app/App.tsx`.
- Main routes:
  - `/` landing page
  - `/analyze` protected analysis workspace
  - `/report` protected report view
  - `/chat` follow-up chatbot view
  - `/settings`, `/documents`, `/about`, `/login`, `/register`
- The frontend uses a safe base URL resolver so hosted builds fall back to relative API paths when a localhost API URL is accidentally bundled.

### 2. API And Orchestration Layer

`model_server/main.py` is the FastAPI entrypoint for the backend runtime.

- Core route registration lives in `model_server/route_groups.py`.
- The backend can run either:
  - ADK-backed orchestration when Google ADK is available
  - deterministic fallback orchestration when ADK is unavailable or disabled
- The server exposes system, inference, analysis, streaming, and report-chat routes from one FastAPI app.

### 3. Agent Layer

`agents/` is no longer limited to the older four-agent description. The active runtime surface now includes:

- `orchestrator_agent.py`: request coordination, validation, routing, and fallback behavior
- `screening_agent.py`: screening pipeline execution and structured prediction packaging
- `researcher_agent.py`: research context lookup
- `evidence_qa_agent.py`: evidence curation and claim-support checks
- `writer_agent.py`: final report synthesis
- `report_chat_agent.py`: grounded report QA, report section lookup, analog comparison, rerun helpers, and mechanism explanation
- `adk_compat.py`: ADK availability and compatibility helpers

### 4. ML And Inference Layer

`backend/` contains the model and explainability runtime.

Current runtime capabilities include:

- xSMILES clinical inference paths
- pretrained dual-head hERG models
- Tox21 ensemble routing and pretrained-GIN support
- additional Tox21 backends such as AttentiveFP, GPS, and fingerprint-based models
- clinical head inference support
- GNNExplainer and gradient-based explanation flows
- OOD guard logic and inference context reporting
- workspace mode controls from `config/workspace_mode.yaml`

### 5. Data And Service Layer

`services/` and Firebase resources support persistence and knowledge retrieval.

- `services/firestore_client.py`: Firestore access helpers
- `services/genai_runtime.py`: GenAI runtime integration
- `services/knowledge_retriever.py`: knowledge retrieval utilities
- `services/molecule_retriever.py`: similar-molecule retrieval used by MolRAG-style flows
- `services/result_fusion.py`: fusion helpers between baseline model output and retrieval/context signals

Firestore rules currently cover these main surfaces:

- `molecules`
- `predictions`
- `users/{uid}`
- `users/{uid}/analyses`
- `users/{uid}/chatSessions`

## Current Deployment Topology

```text
Browser
  |
  v
Firebase Hosting
  |-- serves frontend/dist
  |
  |-- rewrites /health, /predict, /analyze, /agent/**,
  |           /smiles/**, /extract-smiles-from-image
  v
Cloud Run: tox-agent-cpu
  |
  +-- FastAPI model server
      |
      +-- agents/
      +-- backend/
      +-- services/
      |
      +-- external services
          - Firebase Auth / Firestore
          - PubChem / PubMed style retrieval
          - Google GenAI / ADK runtime
```

## End-To-End Runtime Flow

```text
User input
  |- type SMILES
  |- draw molecule
  `- upload image for OCR
          |
          v
Frontend (React + Vite)
          |
          +--> /smiles/preview
          +--> /extract-smiles-from-image
          +--> /predict
          +--> /analyze
          `--> /agent/analyze or /agent/analyze/stream
                      |
                      v
FastAPI model server
  |- validation and canonicalization
  |- screening and mechanism inference
  |- explanation and OOD assessment
  |- research enrichment and report synthesis
  `- chat session creation for follow-up QA
                      |
                      v
Structured report + optional Firestore persistence + report chat
```

## Repository Layout

The repo has grown beyond the older MVP structure. The directories below are the main active surfaces today:

- `frontend/`: production web app and build output for Firebase Hosting
- `model_server/`: FastAPI serving layer and Cloud Run container target
- `agents/`: orchestration, report generation, evidence QA, and report chat logic
- `backend/`: model loading, inference, explainers, OOD logic, and workspace controls
- `services/`: Firestore, retrieval, GenAI, and result-fusion helpers
- `firestore/`: Firestore-related utilities and scripts
- `config/`: model config and workspace mode config
- `models/`: local model artifacts expected by the backend runtime
- `tests/`: unit tests plus smoke-style coverage
- `scripts/`: training, prediction, and experiment utilities
- `deploy/`: deployment manifests and environment assets
- `docs/`: specs, runbooks, and archived documentation
- `legacy/`: archived or superseded surfaces not used in the current deploy path
- `src/`: compatibility wrappers retained for older scripts and integrations

## Local Development

### 1. Create The Python Environment

Use the checked-in conda environment name:

```bash
conda env create -f environment.yml
conda activate drug-tox-env
pip install -r model_server/requirements.txt
```

### 2. Install Frontend And Root Npm Dependencies

The root package provides convenience scripts. The frontend has its own dependency tree.

```bash
npm install
npm --prefix frontend install
```

### 3. Configure Environment Variables

Common local variables:

```bash
export MODELS_ROOT="$PWD/models"
export VITE_API_BASE_URL="http://127.0.0.1:8080"
```

Optional repo-level env files:

- `.env`
- `.env.local`

If you use OCR or hosted AI features, those env files are the right place for runtime-specific secrets and toggles.

### 4. Start The Backend

```bash
uvicorn model_server.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Health check:

```bash
curl -sS http://127.0.0.1:8080/health
```

Minimal analysis example:

```bash
curl -sS -X POST http://127.0.0.1:8080/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "smiles": "CC(=O)Oc1ccccc1C(=O)O"
  }'
```

### 5. Start The Frontend

From the repo root:

```bash
npm run dev
```

That command delegates to `frontend/` and starts the Vite dev server.

### 6. Run Tests And Build

```bash
npm run test
npm run test:smoke
npm run test:smoke:adk
npm run build
```

Notes:

- `npm run test` runs the Python unit test suite under `tests/`.
- `npm run build` builds `frontend/`.
- `npm run deploy:hosting` builds first, then deploys only Firebase Hosting.

## API Quick Reference

### System

- `GET /health`: health check

### Molecule Input And Inference

- `POST /extract-smiles-from-image`: image-to-SMILES OCR
- `POST /smiles/preview`: render or preview a molecule structure
- `POST /predict`: single-molecule toxicity prediction
- `POST /predict/batch`: batch toxicity prediction
- `POST /explain`: explanation for a molecule or target class
- `POST /analyze`: structured analysis with clinical, mechanism, explanation, and OOD outputs

### Agent And Report Flows

- `POST /agent/analyze`: full multi-agent report generation
- `POST /agent/analyze/stream`: streaming event version of report generation
- `POST /agent/chat`: grounded follow-up QA against a generated report
- `POST /agent/chat/stream`: streaming report-chat endpoint

## Current Workspace Configuration

The checked-in workspace config in `config/workspace_mode.yaml` is currently:

- `mode: full`
- `primary_dataset: tox21`
- `clintox_enabled: true`
- `tox21_enabled: true`
- `threshold_policy: safety_first`

This is a meaningful change from older README versions that described the workspace as `tox21_only`.

## OCR Runtime Notes

The image-upload path is handled separately from the normal analysis flow.

- Uploads are validated before OCR.
- MolScribe is used for image-to-SMILES extraction when available.
- Common runtime knobs include `SMILES_IMAGE_MAX_BYTES`, `MOLSCRIBE_PRELOAD_ON_STARTUP`, and `MOLSCRIBE_MODEL_PATH`.

## Deployment Notes

- `firebase.json` serves the SPA from `frontend/dist` and rewrites API requests to Cloud Run.
- `cloudbuild.tox-agent.yaml` builds the backend container from `model_server/Dockerfile`.
- The Cloud Run health route can be aliased through `AIP_HEALTH_ROUTE` when the environment requires a non-default health path.

## Research And Safety Notes

- This project is intended for research and decision-support workflows, not as a standalone medical or regulatory system.
- Structural explanations are supportive evidence, not proof of mechanism.
- OOD warnings should be treated as first-class reliability signals.

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

## License

This project is for research purposes. Tox21 and ClinTox are from MoleculeNet and follow their respective upstream licensing terms.