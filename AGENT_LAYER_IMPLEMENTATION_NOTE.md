# Agent Layer Implementation Note

## 1. Objective
- Implement full agentic layer for ToxAgent based on the approved architecture.
- Keep compatibility with current workspace and deployment setup:
  - FastAPI model server already deployed on Cloud Run.
  - Firebase Hosting rewrite already routes API paths to Cloud Run.
- Validate flow before deployment and redeploy with Docker + Cloud Run + Firebase Hosting.

## 2. Implementation Plan

### Step A - Agent Layer Scaffolding
- Create package `agents/` with:
  - `agents/__init__.py`
  - `agents/screening_agent.py`
  - `agents/researcher_agent.py`
  - `agents/writer_agent.py`
  - `agents/orchestrator_agent.py`

### Step B - Agent Runtime Behavior
- Add ADK-compatible agent declarations (`LlmAgent`, `ParallelAgent`, `SequentialAgent`).
- Add deterministic helper functions for each stage so flow can be tested without LLM uncertainty:
  - Input validation + health check
  - Screening tool pipeline
  - Research tool pipeline
  - Report synthesis
  - Orchestrator state assembly

### Step C - Smoke Test Entry
- Add script `scripts/test_agent_layer_flow.py` to run one full pipeline with a SMILES input and print key outputs.

### Step D - Local Validation
- Use conda env `drug-tox-env`.
- Run smoke test against deployed API URL from `.env` (`MODEL_SERVER_URL`).
- Validate expected state keys:
  - `validation_status`
  - `screening_result`
  - `research_result`
  - `final_report`

### Step E - Deployment
- Rebuild Docker image via Cloud Build.
- Deploy new image to Cloud Run service `tox-agent-cpu` in `asia-southeast1`.
- Rebuild frontend and redeploy Firebase Hosting for consistency.
- Verify:
  - `https://tox-agent-cpu-.../health`
  - `https://tox-agent.web.app/health`
  - `https://tox-agent.web.app/analyze`

## 3. Acceptance Criteria
- Agent files exist and are importable.
- Orchestrator flow runs successfully via test script.
- Cloud Run revision is updated and healthy.
- Firebase Hosting deploy succeeds and rewrites still function.

## 4. Status Tracker
- [x] Plan written
- [ ] Agent files implemented
- [ ] Smoke-test script implemented
- [ ] Agent flow tested
- [ ] Cloud Run redeployed
- [ ] Firebase Hosting redeployed
