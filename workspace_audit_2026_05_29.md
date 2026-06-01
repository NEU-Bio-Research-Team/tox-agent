# ToxAgent - Báo cáo khai phá workspace và audit hiện trạng

Ngày tổng hợp: 2026-05-29

## 1. Cách tôi khai phá workspace

Tôi đã dùng CodeGraph đã tạo cho repo này để lấy ảnh chụp nhanh của codebase, sau đó đọc trực tiếp các file owner để xác nhận các phụ thuộc và runtime path quan trọng.

Thông tin CodeGraph hiện tại:

- Files được index: 259
- Files được parse thành node: 222
- Nodes: 3,354
- Edges: 7,031
- Backend: sqlite nội bộ của CodeGraph
- Trạng thái: index up to date

Mục tiêu của báo cáo này là trả lời hai câu hỏi:

1. Workspace hiện tại đang ở trạng thái nào về kiến trúc, dependency và runtime?
2. Checklist migration trong may_29th_task.md cần giữ nguyên, sửa lại hay bổ sung điểm nào?

## 2. Ảnh chụp nhanh workspace hiện tại

Repo hiện tại không còn là MVP nhỏ nữa. Đây là một workspace đã có đầy đủ các lớp:

- frontend/: React + Vite, deploy lên Firebase Hosting
- model_server/: FastAPI backend đang là runtime phục vụ chính
- agents/: orchestration, writer, report chat, research, evidence QA, MolRAG
- backend/: model loading, inference, explainer, OOD
- services/: Firestore, GenAI runtime, retriever, fusion
- tools/: các tool research, model-server router call, utility logic
- firestore/: script và utility liên quan Firestore
- deploy/: env và deployment assets
- models/: đã có sẵn artifacts local

Tính đến lúc audit, thư mục models/ đang có dữ liệu local với tổng dung lượng xấp xỉ 1.8G. Nghĩa là repo đã ở trạng thái nửa local, nửa cloud, không phải hoàn toàn phụ thuộc vào remote artifact nữa.

## 3. Kết luận tổng quan trước khi đi từng checklist

Có 5 nhận định quan trọng cần chốt ngay:

1. Google dependency vẫn là dependency thật, không phải chỉ còn dấu vết lịch sử.
2. ADK đã được guard để không làm app crash khi import fail, nhưng runtime vẫn có nhánh dùng ADK thật trong model_server/main.py.
3. Firestore đang được dùng ở hai vai trò khác nhau:
   - Backend services cho retrieval và availability check.
   - Frontend cho persistence của analyses và chatSessions của user.
4. Report chat session trong backend không persist bằng Firestore; nó đang nằm trong memory process.
5. Startup artifact sync hiện tại không dùng gsutil như checklist dự đoán; nó dùng Python script và hỗ trợ cả GCS lẫn S3.

## 4. Audit chi tiết theo checklist

### A. Cloud dependencies

#### A1. Google GenAI / Gemini SDK

Đã xác nhận có sự phụ thuộc thật vào Google GenAI ở nhiều điểm:

- services/genai_runtime.py:
  - import `from google import genai`
  - có `build_genai_client_candidates()`
  - có `call_with_retry()`
  - auth flow ưu tiên `GEMINI_API_KEY` / `GOOGLE_API_KEY`, sau đó mới tới Vertex ADC
- agents/writer_agent.py:
  - import `from google import genai`
  - `_maybe_llm_recommendations()` sử dụng GenAI khi `WRITER_ENABLE_LLM_RECOMMENDATIONS=1`
  - có fallback tốt: nếu genai unavailable thì trả về deterministic recommendations
- agents/molrag_reasoner.py:
  - có import genai
  - có `build_genai_client_candidates()` và `call_with_retry()`
  - có nhánh `client.models.generate_content(...)`
- tools/research_tools.py:
  - phần literature synthesis có thể gọi Gemini khi `RESEARCH_ENABLE_LLM_SYNTHESIS=1`
  - nếu genai unavailable thì rơi về deterministic synthesis
- model_server/main.py:
  - report chat runtime không nằm trong agents/report_chat_agent.py mà nằm trong server
  - có `_build_report_chat_client()` và sử dụng `google_genai.Client(...)`

Cần sửa lại checklist gốc ở điểm này:

- report_chat_agent.py không phải nơi gọi Gemini trực tiếp.
- evidence_qa_agent.py không gọi Gemini; đây là deterministic QA layer.
- researcher_agent.py không tự gọi Gemini, nhưng nó gọi tools/research_tools.py; chính file tools này mới có LLM synthesis phụ thuộc Gemini.

Env var liên quan đã xác nhận xuất hiện:

- GEMINI_API_KEY
- GOOGLE_API_KEY
- AGENT_MODEL_PRO
- AGENT_MODEL_FAST
- GEMINI_MODEL
- GOOGLE_CLOUD_PROJECT
- GOOGLE_CLOUD_LOCATION
- GEMINI_LOCATION

Kết luận A1:

- Migration Gemini không chỉ đụng vào writer_agent.py.
- Ít nhất phải chạy qua 5 surface: services/genai_runtime.py, agents/writer_agent.py, agents/molrag_reasoner.py, tools/research_tools.py, và phần report chat runtime trong model_server/main.py.

#### A2. Google ADK

Đã xác nhận cả import và runtime path:

- agents/adk_compat.py:
  - có `ADK_AVAILABLE = True` mặc định
  - import `LlmAgent`, `ParallelAgent`, `SequentialAgent` từ `google.adk.agents`
  - nếu import fail thì chuyển sang shim class nội bộ
- model_server/main.py:
  - import `Runner`
  - import `InMemorySessionService`
  - `_initialize_adk_runtime()` khởi tạo `InMemorySessionService()` và `Runner(...)` nếu available

Điều quan trọng là fallback đang tồn tại thật:

- Nếu ADK không available, startup không crash; server ghi log và bỏ qua khởi tạo ADK runtime.
- Trong route phân tích agent, có `_build_deterministic_response(...)` gọi `run_orchestrator_flow(...)` qua `asyncio.to_thread(...)` để fallback.

Kết luận A2:

- ADK hiện tại là optional runtime, không phải hard blocker cho startup.
- Tuy nhiên nó vẫn là dependency động vào flow sản phẩm thật, nên không thể xem là đã tách xong.

#### A3. Firebase / Firestore

Đã xác nhận Firestore có mặt ở nhiều lớp:

- services/firestore_client.py:
  - import `firebase_admin`, `credentials`, `firestore`
  - có probe client với collection `_molrag_probe`
  - có `fetch_collection_documents(...)`
- agents/report_chat_agent.py:
  - không dùng Firestore để lưu chat session backend
  - có dùng Firestore để đọc collection `molrag_knowledge`
- frontend/src/lib/firestore-history.ts:
  - lưu `users/{uid}/analyses`
- frontend/src/lib/chat-history.ts:
  - lưu `users/{uid}/chatSessions`
- firestore.rules:
  - có rules cho `molecules`, `predictions`, `users/{uid}/analyses`, `users/{uid}/chatSessions`

Kết luận quan trọng:

- Persistence user-facing đang nằm ở frontend Firebase SDK, không phải do backend model_server tự viết Firestore record.
- Backend report chat session lại đang nằm ở memory `_SESSION_STORE` trong agents/report_chat_agent.py.
- Như vậy hiện tại có hai lớp state riêng:
  - State user history persist trong Firestore qua frontend.
  - State chat grounding runtime trong memory process của backend.

Điều này có nghĩa migration Firestore không chỉ là thay services/firestore_client.py bằng SQLite. Nếu muốn cắt Firebase thật sự, cần thay cả các helper frontend đang ghi vào Firestore.

#### A4. Google Cloud Storage / model artifacts

Checklist gốc cần cập nhật lại ở đây.

Tình hình thực tế:

- model_server/scripts/entrypoint.sh không chạy `gsutil cp`
- Nó luôn gọi `python /app/model_server/scripts/download_model_artifacts.py`
- Script này:
  - đọc `MODELS_ROOT`
  - đọc `MODEL_ARTIFACTS_URI`
  - hỗ trợ cả `gs://` và `s3://`
  - dùng `google.cloud.storage` cho GCS
  - dùng `boto3` cho S3
  - nếu local artifacts đã tồn tại và không set URI thì bỏ qua remote sync

Deploy env hiện tại trong deploy/cloudrun-env.yaml:

- MODELS_ROOT=/app/models
- MODEL_ARTIFACTS_URI=gs://tox-agent-models/models
- MODEL_SERVER_TIMEOUT=240

Những model dir quan trọng đã được khai báo trong model_server/main.py:

- smilesgnn_model
- tox21_gatv2_model
- tox21_pretrained_gin_model
- tox21_attentivefp_model
- tox21_gps_model
- tox21_fingerprint_model
- clinical_head_model
- pretrained_2head_herg_chemberta_model
- pretrained_2head_herg_pubchem_model
- pretrained_2head_herg_molformer_model

Tình trạng local hiện tại:

- Thư mục models/ đã có sẵn artifact local
- Tổng dung lượng hiện tại: khoảng 1.8G

Kết luận A4:

- Repo đã có sẵn local model artifacts.
- Cloud dependency ở đây là startup sync policy, không phải bắt buộc kỹ thuật tuyệt đối.
- Tách GCS sẽ dễ hơn checklist ban đầu vì code đã có branch `using local artifacts`.

#### A5. External HTTP APIs

Đã xác nhận external research layer đang phụ thuộc internet thật:

- tools/research_tools.py:
  - PubChem: `https://pubchem.ncbi.nlm.nih.gov/rest/pug`
  - PubMed: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils`
  - Europe PMC fallback
  - Semantic Scholar fallback
  - retry logic bằng httpx + exponential backoff

researcher_agent.py chỉ là wrapper gọi các tool này.

Kết luận A5:

- Nếu migration hướng offline hoặc local-first, đây là một surface riêng, tách biệt với Firestore và Gemini.
- Hiện tại nó chưa có local cache bên trong code chính; phần lớn vẫn là live HTTP.

### B. Runtime và concurrency

#### B1. Uvicorn single-worker

Đã xác nhận trong model_server/scripts/entrypoint.sh:

- `uvicorn model_server.main:app`
- `--workers 1`

Tuy nhiên cần nói chỉnh checklist ở điểm blocking:

- model_server/main.py đã dùng `asyncio.to_thread(...)` ở nhiều chỗ cho các việc nặng
- Có `asyncio.to_thread(...)` cho model preload, image extraction, analyze sync path, report hydration, validation, screening, research, writer, report chat
- Có `asyncio.create_task(...)` + `asyncio.wait(...)` để chạy screening và research song song trong stream flow

Kết luận B1:

- Worker level vẫn là single-worker.
- Nhưng event loop không hoàn toàn bị block theo kiểu thuần sync; đã có effort tách CPU hoặc blocking work sang thread.
- Vẫn cần benchmark thực tế, nhưng nhận định "main.py còn blocking toàn bộ event loop" là quá mạnh nếu chỉ dựa vào đọc mã.

#### B2. ThreadPoolExecutor usage

Đã xác nhận:

- agents/orchestrator_agent.py:
  - `ThreadPoolExecutor(max_workers=2)`
  - chạy `run_screening` và `run_research` song song trong deterministic flow
- model_server/main.py:
  - import `ThreadPoolExecutor` và `FuturesTimeoutError`
  - dùng thêm một `ThreadPoolExecutor(max_workers=1)` cho timeout wrapper của explainer sync

Điểm cần lưu ý:

- deterministic orchestrator flow trong agents/orchestrator_agent.py không set timeout riêng cho `screening_future.result()` và `research_future.result()`
- timeout hiện tại rõ ràng nhất là cho explainer, không phải cho toàn bộ screening hoặc research pair ở deterministic path

Kết luận B2:

- Có concurrency cấp agent.
- Timeout handling hiện tại không đồng đều trên mọi stage.

#### B3. Timeout configuration

Đã xác nhận:

- deploy/cloudrun-env.yaml đặt `MODEL_SERVER_TIMEOUT: "240"`
- tools/tox_tools.py dùng giá trị này cho call tới model server
- model_server/main.py có timeout wrapper cho explainer với `explainer_timeout_ms`
- services/genai_runtime.py retry tối đa 4 lần với exponential backoff
- tools/research_tools.py retry PubMed và PubChem tối đa 3 lần

Nhưng hiện tại chưa thấy một timeout stage-level đơn giản và đồng bộ cho toàn bộ screening, research, writer trong deterministic orchestrator flow.

#### B4. State management

Đã xác nhận có ít nhất hai cơ chế state khác nhau:

- ADK runtime session: `InMemorySessionService` trong model_server/main.py
- Report chat backend session: `_SESSION_STORE` trong agents/report_chat_agent.py

Hai cơ chế này đều là in-memory process state.

Song song, frontend lại persist chat history và analysis history trên Firestore.

Kết luận B4:

- Nếu Cloud Run restart, state runtime trong backend có nguy cơ mất.
- User-visible history có thể vẫn tồn tại trong Firestore nếu frontend đã persist.
- Grounding state hoặc runtime session không đồng nhất với persisted history.

### C. Dependencies cần dọn trước migration

#### C1. requirements

Đã đọc hai file dependency chính:

- model_server/requirements.txt
- requirements.txt

Google-specific package đang tồn tại:

- google-adk>=0.4.0
- google-genai>=0.8.0
- google-cloud-storage>=2.16.0
- firebase-admin

Nhận xét:

- Nhiều package đang pin theo lower bound, không pin version đầy đủ.
- Việc reproducibility chưa thật chặt, đặc biệt nếu migration cần so sánh baseline trước và sau.

Package có khả năng xóa sau migration, tùy theo mức độ tách cloud:

- google-adk
- google-genai
- google-cloud-storage
- firebase-admin
- có thể cả boto3 nếu quyết định không hỗ trợ artifact từ S3

#### C2. Import guards và startup resilience

Đã xác nhận có nhiều import guard tốt:

- services/genai_runtime.py: nếu không import được genai thì trả về `[]` client candidates
- agents/writer_agent.py: nếu genai unavailable thì rơi về deterministic recommendations
- tools/research_tools.py: nếu genai unavailable thì rơi về deterministic literature synthesis
- agents/adk_compat.py: nếu không import được Google ADK thì dùng shim class
- model_server/main.py: nếu không import được Runner hoặc InMemorySessionService thì ghi `ADK_RUNTIME_IMPORT_ERROR` và tiếp tục startup
- services/firestore_client.py: nếu `firebase_admin` unavailable thì đánh dấu `firebase_admin_unavailable`

Kết luận C2:

- Hệ thống có khả năng startup được trong nhiều chế độ degraded.
- Nhưng startup được không đồng nghĩa feature parity được giữ nguyên.

### D. Frontend

Đã xác nhận:

- firebase.json đang deploy `frontend/dist`
- Firebase Hosting rewrite các route API chính sang Cloud Run service `tox-agent-cpu` ở `asia-southeast1`
- frontend/src/lib/api.ts có cơ chế base URL an toàn:
  - nếu production bundle lỡ chứa localhost URL thì sẽ rơi về relative path
  - cách này phù hợp với Firebase Hosting rewrite
- model_server/app_factory.py đang CORS `allow_origins=["*"]`

Ngoài ra, frontend vẫn phụ thuộc Firebase cho persistence history:

- analyses: frontend/src/lib/firestore-history.ts
- chat sessions: frontend/src/lib/chat-history.ts

Kết luận D:

- Nếu sau migration muốn Nginx serve SPA từ frontend/dist thì phần static hosting rất dễ chuyển.
- Nhưng nếu muốn bỏ Firebase thật sự, phải thay cả frontend persistence layer và auth story, không chỉ đổi hosting.

## 5. Những điểm checklist gốc cần sửa lại

Đây là phần quan trọng nhất để tránh đi sai hướng:

1. `report_chat_agent.py` không phải nơi gọi Gemini trực tiếp; runtime chat model nằm ở `model_server/main.py`.
2. `evidence_qa_agent.py` hiện tại là deterministic, không phải Gemini surface.
3. GCS startup không dùng `gsutil cp`; đang dùng Python downloader có hỗ trợ GCS và S3.
4. Firestore không chỉ ở backend; frontend mới là nơi persist user analyses và chat sessions.
5. Repo đã có model artifacts local sẵn, tổng khoảng 1.8G; migration local models sẽ dễ hơn dự tính ban đầu.
6. Event loop đã được giảm block một phần bằng `asyncio.to_thread(...)`; cần đo benchmark thay vì kết luận cảm tính.

## 6. Đánh giá mức độ sẵn sàng cho migration

### Những phần dễ tách nhất trước

- Phase local models: khá dễ, vì code đã sẵn branch local artifact
- Phase baseline measurement: rất nên làm ngay
- Phase observability: dễ thêm, impact cao

### Những phần dễ bị đánh giá thiếu scope nếu chỉ đọc checklist gốc

- Replace Firestore:
  - không chỉ là backend database client
  - còn là frontend history storage và có thể dính tới auth-adjacent flow
- Replace Gemini:
  - không chỉ là writer_agent
  - còn MolRAG, literature synthesis, report chat runtime trong server
- Remove ADK:
  - có thể dễ xóa package, nhưng cần giữ được fallback và shape event hoặc response hiện tại

## 7. Đề xuất roadmap đã được tinh chỉnh theo workspace hiện tại

### Phase 0 - Baseline measurement

Giữ nguyên. Đây là phase nên làm trước tất cả.

Ngoài latency và RAM, nên ghi thêm:

- thời gian startup lần đầu
- thời gian warm request lần 2-5
- thời gian tạo report chat session

### Phase 1 - Local models

Nên đổi tên từ "Detach Cloud Storage" thành "Chuẩn hóa local-first model artifact runtime".

Vì lý do:

- Repo đã có models/ local
- Script download hiện tại đã có branch skip remote sync khi local artifact tồn tại

Việc cần làm thật sự:

- quyết định `MODEL_ARTIFACTS_URI` có bị bỏ trong local và production hay không
- quyết định có còn muốn giữ fallback S3 hoặc GCS không
- benchmark startup với local-only so với remote-sync mode

### Phase 2 - Replace Firestore

Cần tách thành 2 sub-phase:

- Phase 2a: backend retrieval và diagnostics
- Phase 2b: frontend persistence cho analyses và chatSessions

Nếu không tách như vậy, rất dễ xóa được backend Firestore client nhưng vẫn còn Firebase dependency trong frontend.

### Phase 3 - Replace Gemini

Cần tách thành ít nhất 4 workstream:

- 3a. writer recommendations
- 3b. literature synthesis trong tools/research_tools.py
- 3c. MolRAG reasoning trong agents/molrag_reasoner.py
- 3d. report chat runtime trong model_server/main.py

Nếu không, migration sẽ mới xong writer nhưng report chat và MolRAG vẫn cloud-bound.

### Phase 4 - Remove ADK

Chỉ nên làm sau khi đã thay xong các surface gọi GenAI hoặc ADK runtime trong model_server/main.py.

Nếu làm sớm, bạn sẽ dễ mất fallback path đang hỗ trợ production hoặc degraded mode.

### Phase 5 - Multi-worker + reverse proxy

Có giá trị, nhưng nên làm sau khi đo baseline và sau khi chốt rõ model loading memory footprint.

Lý do:

- hiện tại models/ đã 1.8G
- tăng workers mà không đo memory sẽ rất dễ gặp OOM hoặc startup chậm hơn kỳ vọng

### Phase 6 - Observability

Nên đẩy lên sớm hơn, có thể làm ngay sau Phase 0 hoặc song song Phase 1.

Lý do:

- repo hiện có nhiều fallback branch
- nếu không có telemetry, rất khó biết request đang đi qua deterministic path, ADK path, hay genai fallback path

## 8. Mức độ ưu tiên thực dụng để tối ưu công sức

Nếu mục tiêu là "giảm cloud lock-in nhanh nhất mà ít vỡ nhất", thứ tự tôi khuyến nghị là:

1. Baseline + observability
2. Chuẩn hóa local-first models
3. Tách writer và literature synthesis khỏi Gemini
4. Tách report chat runtime khỏi Gemini
5. Tách MolRAG reasoning khỏi Gemini
6. Sau đó mới dọn ADK
7. Firestore migration tách riêng frontend và backend
8. Cuối cùng mới tới multi-worker và reverse proxy

Thứ tự này thực dụng hơn checklist gốc vì nó cắt đúng các surface có traffic thật trước, và tránh xóa sớm framework fallback khi chưa có runtime thay thế.

## 9. Kết luận cuối cùng

Workspace hiện tại đang ở trạng thái "hybrid cloud-local":

- local models đã tồn tại
- backend có deterministic fallback khá tốt
- nhưng cloud dependencies vẫn còn ở nhiều điểm có ý nghĩa sản phẩm thật

Nếu chỉ nhìn bề ngoài, có thể tưởng rằng việc migration chủ yếu là đổi writer_agent.py và thay Firestore client. Sau khi khai phá bằng CodeGraph và đọc file owner, kết luận đúng hơn là:

- migration Gemini là bài toán đa điểm, không phải một file
- migration Firestore là bài toán full-stack, không phải chỉ backend
- ADK hiện tại là optional runtime nhưng vẫn là phần của behavior thật
- model artifact runtime đã gần sẵn sàng cho local-first hơn dự tính ban đầu

Tóm lại: repo này chưa cloud-free, nhưng đã có sẵn khá nhiều hook fallback để migration theo từng phase mà không cần đập đi xây lại.