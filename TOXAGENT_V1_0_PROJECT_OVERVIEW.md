# ToxAgent v1.0 - Project Overview and FE/BE Integration Plan

## 1. Muc tieu v1.0

Hoan thanh ket noi end-to-end giua frontend React/Vite va backend FastAPI + ADK Agent Layer de:

- Nhan SMILES tu giao dien.
- Goi 1 API phan tich chinh: POST /agent/analyze.
- Hien thi bao cao day du tu final_report (khong dung mock data).
- Hien thi tien trinh agent dua tren agent_events (co fallback).
- Van hanh on dinh local va deploy len Cloud Run + Firebase Hosting.

## 2. Tong quan kien truc hien tai

### Frontend (React + Vite)

- Trang chinh: IndexPage
  - HeroSection
  - AgentProgressPanel
  - QuickVerdictCard
- Trang bao cao: ReportPage
  - ReportHeader
  - ReportSidebar
  - ClinicalToxicitySection
  - MechanismProfilingSection
  - StructuralExplanationSection
  - LiteratureContextSection
  - AIRecommendationsSection

### Backend (FastAPI model_server)

- GET /health
- POST /predict
- POST /predict/batch
- POST /explain
- POST /analyze (pure ML)
- POST /agent/analyze (ADK full pipeline)

### Agent Layer (ADK)

- InputValidator
- Parallel:
  - ScreeningAgent
  - ResearcherAgent
- WriterAgent

## 3. Quyet dinh hop dong FE-BE cho v1.0

### 3.1 Endpoint chinh

Frontend v1.0 su dung duy nhat:

- POST /agent/analyze
- Request body:
  - smiles: string
  - include_agent_events: true

### 3.2 Du lieu nguon cho UI

Toan bo ReportPage va QuickVerdictCard su dung:

- response.final_report

Tien trinh pipeline su dung:

- response.agent_events

### 3.3 Environment

- Bien moi truong FE:
  - VITE_API_BASE_URL
- Local dev mac dinh:
  - frontend/.env.local => VITE_API_BASE_URL=http://localhost:8000

## 4. Contract mapping (Final Report -> UI)

### Clinical

- final_report.sections.clinical_toxicity.probability -> p_toxic
- final_report.sections.clinical_toxicity.confidence -> confidence
- final_report.sections.clinical_toxicity.verdict -> label
- final_report.sections.clinical_toxicity.interpretation -> narrative

### Mechanism

- final_report.sections.mechanism_toxicity.task_scores -> chart bars
- final_report.sections.mechanism_toxicity.highest_risk -> top risk task
- final_report.sections.mechanism_toxicity.assay_hits -> quick stats

### Structural

- final_report.sections.structural_explanation.top_atoms
- final_report.sections.structural_explanation.top_bonds
- final_report.sections.structural_explanation.heatmap_base64 -> img data URI

### Literature

- final_report.sections.literature_context.compound_id.cid
- final_report.sections.literature_context.compound_id.pubchem_url
- final_report.sections.literature_context.relevant_papers[]
- final_report.sections.literature_context.bioassay_evidence.active_assays[]

### Recommendations

- final_report.executive_summary
- final_report.sections.recommendations[]
- final_report.risk_level

## 5. Cac mismatch da duoc xu ly trong code

1. Batch schema mismatch:
- BatchPredictRequest.smile_list -> smiles_list
- predict_batch endpoint doc req.smiles_list

2. Literature snippet mismatch:
- research_tools tra ve snippet (thay cho abstract_snippet)

3. Authors mismatch:
- research_tools tra ve authors dang string (join tu list)

4. Batch tool parser mismatch:
- tools/tox_tools.analyze_molecules_batch da xu ly dung response object cua /predict/batch

5. Compatibility hardening:
- evidence_qa_agent _to_list co the parse chuoi authors tach bang dau phay

## 6. Kien truc state frontend v1.0

Them global context:

- frontend/src/lib/ReportContext.tsx

State trung tam:

- report: AgentAnalyzeResponse | null
- isLoading: boolean
- error: string | null

Flow:

1. IndexPage goi agentAnalyze(smiles)
2. Save ket qua vao ReportContext
3. QuickVerdictCard doc final_report
4. ReportPage doc final_report va truyen props xuong tung section

## 7. Danh sach file chinh da thay doi

### Backend

- model_server/schemas.py
- model_server/main.py
- tools/research_tools.py
- tools/tox_tools.py
- agents/evidence_qa_agent.py

### Frontend

- frontend/src/lib/api.ts
- frontend/src/lib/ReportContext.tsx
- frontend/src/app/App.tsx
- frontend/src/app/pages/index-page.tsx
- frontend/src/app/pages/report-page.tsx
- frontend/src/app/components/agent-progress-panel.tsx
- frontend/src/app/components/quick-verdict-card.tsx
- frontend/src/app/components/report-header.tsx
- frontend/src/app/components/report-sidebar.tsx
- frontend/src/app/components/report/clinical-toxicity-section.tsx
- frontend/src/app/components/report/mechanism-profiling-section.tsx
- frontend/src/app/components/report/structural-explanation-section.tsx
- frontend/src/app/components/report/literature-context-section.tsx
- frontend/src/app/components/report/ai-recommendations-section.tsx
- frontend/.env.local

## 8. Ke hoach test local

### 8.1 Khoi dong backend (conda env drug-tox-env)

1. Kich hoat moi truong:
- conda activate drug-tox-env

2. Chay API server local:
- uvicorn model_server.main:app --host 127.0.0.1 --port 8000

3. Smoke test endpoint:
- curl http://127.0.0.1:8000/health
- curl -X POST http://127.0.0.1:8000/agent/analyze -H "Content-Type: application/json" -d '{"smiles":"CCO","include_agent_events":true}'

### 8.2 Khoi dong frontend

1. Di chuyen frontend:
- cd frontend

2. Cai package va chay dev:
- npm ci
- npm run dev -- --host 127.0.0.1 --port 5173

3. Kiem thu UI:
- Nhap SMILES
- Bam Phan tich
- Kiem tra AgentProgressPanel
- Kiem tra QuickVerdictCard
- Mo ReportPage va doi chieu du lieu voi JSON API

### 8.3 Build test

- cd frontend
- npm run build

## 9. Ke hoach deploy Cloud Run + Firebase

Lam theo DEPLOY_FIREBASE_APP_RUNBOOK.md voi luong frontend-only hoac full stack:

### Neu chi thay doi frontend

- cd frontend
- npm ci
- npm run build
- cd ..
- npx -y firebase-tools@latest deploy --only hosting --project tox-agent

### Neu co thay doi backend

1. Build image Cloud Build
2. Deploy Cloud Run service tox-agent-cpu
3. Smoke test Cloud Run
4. Deploy Firebase Hosting neu can

Ghi chu:
- firebase.json da co rewrite /agent/** -> Cloud Run service tox-agent-cpu (asia-southeast1)

## 10. Non-goals cua v1.0

- Chua implement streaming realtime true cho agent_events trong khi request dang chay.
- AgentProgressPanel hien dung event-driven sau khi response ve + fallback simulate khi khong co events.
- Chua them PDF export nang cao (hien tai dung window.print()).

## 11. Tieu chi done v1.0

1. FE khong con hardcode mock cho report/chi so chinh.
2. Luong IndexPage -> API -> ReportPage hoat dong end-to-end.
3. Validation status loi duoc hien thi ro tren UI.
4. Build frontend thanh cong.
5. Smoke test /agent/analyze thanh cong local.
6. Deploy cloud thanh cong theo runbook.
