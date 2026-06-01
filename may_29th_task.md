Giờ tôi đã có đủ context chi tiết từ codebase thực tế. Tổng hợp checklist toàn diện:

***

# ToxAgent — Master Checklist: Audit Hiện Tại & Migration Roadmap

## 🔍 PHẦN 1 — AUDIT CHECKLIST: Dò Cấu Trúc Hiện Tại

Đây là danh sách các mục bạn phải dò qua **trước khi chạm vào bất kỳ dòng code nào**.

***

### A. Cloud Dependencies — Tìm Tất Cả Điểm Phụ Thuộc

**[A1] Google GenAI / Gemini SDK**

- [ ] `writer_agent.py` — `from google import genai` → dùng `client.models.generate_content()` với model `AGENT_MODEL_PRO=gemini-2.5-pro` và `AGENT_MODEL_FAST=gemini-2.5-flash` 
- [ ] `services/genai_runtime.py` — `build_genai_client_candidates()`, `call_with_retry()` → toàn bộ retry logic và auth management cho Gemini 
- [ ] `report_chat_agent.py` (35KB) — chưa đọc hết, nhiều khả năng có streaming Gemini call cho chat session
- [ ] `evidence_qa_agent.py` — xác nhận có dùng genai hay không
- [ ] `molrag_reasoner.py` (41KB) — file lớn nhất, cần kiểm tra có embed Gemini call không
- [ ] `agents/__init__.py` — xem export list để biết bao nhiêu agent phụ thuộc Gemini
- [ ] Tìm tất cả chỗ dùng env var: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `AGENT_MODEL_PRO`, `AGENT_MODEL_FAST`

**[A2] Google ADK (Agent Development Kit)**

- [ ] `main.py` import: `from google.adk.runners import Runner`, `from google.adk.sessions import InMemorySessionService` 
- [ ] `agents/adk_compat.py` — wrapper `LlmAgent` dùng trong `writer_agent.py` 
- [ ] Tất cả file dùng `ADK_AVAILABLE` flag — đây là graceful fallback hay hard dependency?
- [ ] Xác nhận: nếu ADK không có, flow có chạy không? (code hiện tại có `if ADK_AVAILABLE` guards không?)

**[A3] Firebase / Firestore**

- [ ] `services/` directory — tìm `firestore_client.py` hoặc `firebase_admin`
- [ ] Grep: `from firebase_admin`, `from google.cloud.firestore`, `db.collection()`
- [ ] Xác định: data nào đang được persist (chat sessions? analysis history? user data?)
- [ ] `main.py` — `create_chat_session()`, `get_session()` đang lưu state ở đâu?

**[A4] Google Cloud Storage (GCS)**

- [ ] `entrypoint.sh` — có `gsutil cp gs://tox-agent-models/...` không?
- [ ] `MODELS_ROOT` env var — hiện trỏ về đâu trên Cloud Run?
- [ ] Model file list cần download: `smilesgnn_model`, `tox21_gatv2_model`, `tox21_pretrained_gin_model`, `tox21_attentivefp_model`, `tox21_gps_model`, `tox21_fingerprint_model`, `clinical_head_model`, `pretrained_2head_herg_chemberta_model`, `pretrained_2head_herg_pubchem_model`, `pretrained_2head_herg_molformer_model` 
- [ ] Tổng dung lượng model artifacts là bao nhiêu GB?

**[A5] External HTTP APIs (giữ nguyên hay local cache?)**

- [ ] PubChem API — `researcher_agent.py`, `molrag_reasoner.py`
- [ ] PubMed/NCBI API — `researcher_agent.py`
- [ ] Quyết định: giữ external call (cần internet) hay local cache với SQLite?

***

### B. Runtime & Concurrency — Dò Bottleneck

**[B1] Uvicorn single-worker**

- [ ] Confirm `--workers 1` trong `entrypoint.sh` hoặc Cloud Run command
- [ ] `main.py` có dùng `asyncio` đúng cách không? GNN inference có được wrapped trong `run_in_executor` chưa hay còn blocking event loop?

**[B2] ThreadPoolExecutor usage**

- [ ] `orchestrator_agent.py` — `ThreadPoolExecutor(max_workers=2)` cho Screening ‖ Research parallel 
- [ ] Có bất kỳ chỗ nào khác dùng ThreadPoolExecutor?
- [ ] `FuturesTimeoutError` được handle đúng không? Nếu timeout, pipeline trả về gì?

**[B3] Timeout configuration**

- [ ] Cloud Run config: `MODEL_SERVER_TIMEOUT: "240"` — 4 phút
- [ ] Từng stage có per-stage timeout không? (screening, research, writer riêng biệt)
- [ ] Gemini call có timeout set không? `call_with_retry` retry bao nhiêu lần?

**[B4] State management**

- [ ] `InMemorySessionService` từ ADK — session sống trong memory của process, không persist qua restarts
- [ ] Nếu Cloud Run instance bị cold start hoặc restart, mọi active chat session mất hết
- [ ] Xác nhận: có user data nào cần migrate sang persistent storage không?

***

### C. Dependencies — Dọn Trước Khi Migration

**[C1] `requirements.txt` / `pyproject.toml`**

- [ ] List tất cả Google-specific packages: `google-adk`, `google-generativeai`, `google-cloud-firestore`, `google-cloud-storage`, `firebase-admin`
- [ ] Các package này có version pinned không? (quan trọng cho reproducibility)
- [ ] Sau migration, packages nào có thể xóa hoàn toàn?

**[C2] Import guards**

- [ ] `writer_agent.py` line: `if genai is None: return [], "google_genai_not_available"` — đây là fallback tốt 
- [ ] Tương tự, tìm tất cả `try/except` import guards cho Google packages
- [ ] Xác nhận: nếu tất cả Google packages không có, hệ thống có start được không?

***

### D. Frontend

- [ ] `frontend/` build output deploy lên Firebase Hosting — base URL là gì?
- [ ] CORS config trong `main.py` — whitelist origin nào?
- [ ] API URL trong frontend hardcode hay lấy từ env?
- [ ] Sau migration, Nginx sẽ serve static SPA từ `frontend/dist/`

***

## 🗺️ PHẦN 2 — MIGRATION ROADMAP: Từng Bước Làm Gì

### Phase 0 — Baseline Measurement (1 ngày)
*Không động code, chỉ đo.*

```
1. Chạy 1 request /agent/analyze với SMILES đơn giản (Aspirin: CC(=O)Oc1ccccc1C(=O)O)
2. Ghi lại: total latency, peak RAM, CPU %
3. Chạy 5 request sequential, ghi latency từng request
4. Ghi lại tổng model size trên disk
```

Mục tiêu: có baseline số thực để so sánh sau migration. Không có baseline → không biết migration có tốt hơn không.

***

### Phase 1 — Detach Cloud Storage: Local Models (2–3 ngày)

**Mục tiêu**: Không còn download model từ GCS lúc startup.

```bash
# Bước 1: Download toàn bộ model artifacts về local
gsutil -m cp -r gs://tox-agent-models/models ./models/

# Bước 2: Set env var
MODELS_ROOT=/absolute/path/to/models

# Bước 3: Sửa entrypoint.sh — xóa gsutil download step
# Bước 4: Test: docker build + run với MODELS_ROOT local
```

**Verify**: `docker run --env MODELS_ROOT=/models -v $(pwd)/models:/models tox-agent`

***

### Phase 2 — Replace Firestore: Local Database (2–3 ngày)

**Mục tiêu**: Chat session và analysis data lưu local.

Tìm tất cả Firestore calls trong codebase:
```bash
grep -r "firestore\|firebase\|db\.collection" --include="*.py" .
```

Replace với SQLite (đơn giản nhất, zero config):
```python
# services/db_client.py (mới)
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "./data/tox_agent.db"))

def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)
```

Schema tối thiểu:
```sql
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT,
    data TEXT  -- JSON blob
);

CREATE TABLE IF NOT EXISTS analysis_history (
    id TEXT PRIMARY KEY,
    smiles TEXT,
    created_at TEXT,
    result TEXT  -- JSON blob
);
```

***

### Phase 3 — Replace Gemini: Local LLM (3–5 ngày)

Đây là phase phức tạp nhất. `writer_agent.py` hiện dùng `client.models.generate_content()` từ Google GenAI SDK .

**Bước 3a — Setup Ollama (nếu không có GPU) hoặc vLLM (có GPU)**

```bash
# Option A: Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b        # CPU: ~4GB RAM
# ollama pull qwen2.5:14b     # CPU: ~9GB RAM

# Option B: vLLM (cần GPU)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --port 8000
```

**Bước 3b — Tạo adapter thay thế GenAI SDK**

```python
# services/llm_client.py (mới — thay services/genai_runtime.py)
import os
from openai import OpenAI

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")  # Ollama default
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")

_client = None

def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LOCAL_LLM_URL, api_key="local")
    return _client

def call_local_llm(prompt: str, temperature: float = 0.3) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LOCAL_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content
```

**Bước 3c — Sửa `writer_agent.py`**

Chỉ cần sửa `_maybe_llm_recommendations()` — toàn bộ logic còn lại (prompt building, parsing, fallback) giữ nguyên :

```python
# Thay thế _maybe_llm_recommendations():
from services.llm_client import call_local_llm

def _maybe_llm_recommendations(...) -> Tuple[List[Dict], str]:
    enabled = os.getenv("WRITER_ENABLE_LLM_RECOMMENDATIONS", "1")
    if enabled in {"0", "false", "no"}:
        return [], "llm_disabled_by_env"
    
    try:
        prompt = _build_llm_prompt(...)  # giữ nguyên hàm này
        text = call_local_llm(prompt, temperature=0.3)
        parsed = _parse_llm_recommendations(text)
        if parsed:
            return parsed, "local_llm_success"
        return [], "llm_parse_failed"
    except Exception as exc:
        return [], f"local_llm_error:{exc}"
```

**Bước 3d — Tương tự cho `report_chat_agent.py` và `evidence_qa_agent.py`**

***

### Phase 4 — Remove ADK Dependency (1–2 ngày)

Google ADK là orchestration framework — `LlmAgent` trong `writer_agent.py` wrap logic agent . Khi đã dùng local LLM, ADK không còn cần thiết.

```python
# agents/adk_compat.py — đơn giản hóa thành:
class LlmAgent:
    """Lightweight ADK-compatible shim. ADK-free local runtime."""
    def __init__(self, name, model, description, instruction, tools, output_key):
        self.name = name
        self.model = model
        self.instruction = instruction
        self.output_key = output_key
    
    async def run(self, state: dict) -> dict:
        # Gọi local LLM thay vì ADK runner
        ...
```

***

### Phase 5 — Multi-Worker + Nginx (2–3 ngày)

Sau khi toàn bộ Google dependency đã sạch:

**Bước 5a — Tăng workers**
```bash
# Dockerfile CMD hoặc entrypoint.sh
gunicorn model_server.main:app \
    -w 2 \
    -k uvicorn.workers.UvicornWorker \
    --preload \
    --timeout 300 \
    -b 0.0.0.0:8080
```

**Bước 5b — Nginx config**
```nginx
upstream tox_backend {
    least_conn;
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;  # nếu chạy instance thứ 2
    keepalive 16;
}

server {
    listen 80;
    root /path/to/frontend/dist;
    try_files $uri /index.html;
    
    location ~ ^/(health|predict|analyze|agent|smiles|extract) {
        proxy_pass http://tox_backend;
        proxy_read_timeout 300s;
        proxy_buffering on;
    }
}
```

**Bước 5c — Docker Compose local**
```yaml
services:
  nginx: { ... }
  tox-api-1: { ..., ports: ["8080:8080"] }
  tox-api-2: { ..., ports: ["8081:8080"] }
  llm-server: { ... }  # Ollama hoặc vLLM
  redis: { ... }       # prediction cache
```

***

### Phase 6 — Observability (1–2 ngày)

Chỉ cần thêm timing log vào `orchestrator_agent.py`:

```python
# orchestrator_agent.py
import time

async def run_orchestrator_flow(smiles, ...):
    t0 = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=2) as pool:
        screening_future = pool.submit(run_screening, smiles)
        research_future = pool.submit(run_research, smiles)
        
        t1 = time.perf_counter()
        screening = screening_future.result(timeout=120)
        t2 = time.perf_counter()
        research = research_future.result(timeout=60)
        t3 = time.perf_counter()
    
    report = build_final_report(...)
    t4 = time.perf_counter()
    
    logger.info(json.dumps({
        "stage": "orchestrator",
        "smiles": smiles[:20],
        "screening_ms": int((t2-t1)*1000),
        "research_ms": int((t3-t2)*1000),
        "writer_ms": int((t4-t3)*1000),
        "total_ms": int((t4-t0)*1000),
    }))
```

***

## 📊 PHẦN 3 — Priority Matrix

| Phase | Effort | Impact | Phụ thuộc | Làm trước? |
|---|---|---|---|---|
| **Phase 0** — Baseline measurement | 1 ngày | Không làm chỉ mù | Không có | ✅ Đầu tiên |
| **Phase 1** — Local models | 2 ngày | Xóa GCS dependency | Phase 0 | ✅ Ngay sau |
| **Phase 2** — SQLite thay Firestore | 2 ngày | Xóa Firebase | Phase 1 | ✅ Song song Phase 1 |
| **Phase 3** — Local LLM | 3–5 ngày | Core migration | Phase 1+2 | ⚡ Quan trọng nhất |
| **Phase 4** — Remove ADK | 1 ngày | Clean up | Phase 3 | Sau Phase 3 |
| **Phase 5** — Multi-worker + Nginx | 2 ngày | Performance | Phase 3+4 | Sau Phase 4 |
| **Phase 6** — Observability | 1 ngày | Visibility | Phase 5 | Cuối cùng |

**Tổng thời gian**: 12–17 ngày nếu làm sequential, 8–12 ngày nếu overlap Phase 1+2.