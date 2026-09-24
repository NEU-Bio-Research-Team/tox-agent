# ToxAgent — Kế hoạch endpoint selection, XAI artifacts, session naming, agent capabilities và production

> **Ngày lập:** 2026-09-07  
> **Trạng thái:** kế hoạch triển khai dựa trên code hiện tại và research bên ngoài  
> **Phạm vi:** `frontend/`, `toxpred/`, `toxagent-control/`, OpenCode runtime, PostgreSQL/Alembic và web/research providers  
> **Không phải:** báo cáo rằng các hạng mục bên dưới đã được triển khai xong

## 1. Kết luận điều hành

Repo hiện tại đã có phần lớn khung kiến trúc cần thiết, nhưng một số tính năng
đang ở trạng thái “có từng mảnh” chứ chưa thành một product flow hoàn chỉnh.

| Yêu cầu | Hiện trạng đọc trực tiếp từ repo | Quyết định đề xuất |
|---|---|---|
| Chọn predictor endpoint trong chat | `MessageComposer` đã có checkbox `herg`, `tox21`, `clintox`, nhưng danh sách hard-code và không đọc capability thật | Giữ lựa chọn theo **logical endpoint**, lấy inventory từ `/v1/predict/capabilities`, disable endpoint không phục vụ và lưu preference theo user/session |
| Predictor luôn đi cùng explainer artifact | Prediction và explanation là hai request/lifecycle khác nhau; XAI hiện on-demand | Tạo một `AnalysisArtifactBundle`: prediction + một explanation artifact cho mọi target đã khai báo; run chỉ `completed` khi bundle đủ hoặc đã có typed failure artifact |
| Highlight atom/bond | API chỉ trả atom importance; phần không map được gộp vào `unmapped_importance`; UI chỉ vẽ molecule thường và bar list | Mở contract XAI v2 có `atoms[]`, `bonds[]` và SVG highlight được tạo deterministically bằng RDKit; luôn giữ numeric payload là source of truth |
| Session hiện tên thay vì ID | DB/domain/API đã có `title`, sidebar ưu tiên title; session mới không truyền title nên fallback sang ID; full row vẫn in ID | Thêm auto-title deterministic ngay sau message đầu, rename thủ công, optional LLM refinement; ID chỉ hiện trong inspector/copy-details |
| Agent đọc predictor/explainer, search web và tài liệu | MCP đã có prediction slice, attribution, Europe PMC search/read và grounded answer; profile OpenCode deny web trực tiếp; UI còn ẩn research intent | Chuẩn hóa tool catalog, thêm bundle read, compound identity, compare/export; thêm safe web fetch/search qua control-plane; chỉ dùng Obscura cho dynamic-web tier |
| User tự OAuth provider cho OpenCode local | Local launcher đã copy `~/.local/share/opencode/auth.json` vào isolated OpenCode home; chưa có guided bootstrap trong `bin/toxagent` | Làm BYOC local flow qua `opencode auth login`; ToxAgent không giữ refresh token trong DB. Multi-user hosted OAuth là workstream riêng |
| SQLAlchemy/control plane production | SQLAlchemy Core async, PostgreSQL, Alembic, outbox và CI migration test đã có | Hardening pool/health/query/index, tách migration job khỏi app replica, distributed worker/notification, OIDC, object store, telemetry, load/restore drill |

Thứ tự nên làm là:

1. Chốt contract bundle/XAI/session title.
2. Hoàn thiện endpoint selector và session naming.
3. Xây XAI v2 + artifact bundle end-to-end.
4. Chuẩn hóa core agent tools, rồi mới mở safe web capability.
5. Làm local OpenCode BYOC/OAuth onboarding.
6. Hardening PostgreSQL và tách control plane thành web/worker/migration workloads.
7. Chạy alpha, load/failure/restore drill rồi mới production go/no-go.

## 2. Những gì repo đã có và không nên xây lại

### 2.1 Predictor và frontend

- ToxPred có `GET /v1/models`, `POST /v1/predictions`, batch prediction,
  token attribution và atom explanation.
- Control plane đã proxy các route stateless `/v1/predict*`, có auth và
  process-local concurrency limiter.
- `GET /v1/predict/capabilities` đã trả `served_endpoints`, model inventory và
  `blocked_reason`.
- Session message contract đã có `analysis_options.endpoints` và
  `include_attribution`.
- Frontend đã có endpoint checkbox trong advanced popover của chat composer.
- `ExplainPanel` đã gọi `/v1/predict/explain`, nhưng chỉ khi user bấm nút.
- `AtomHighlightDepiction` đang cố ý không highlight trực tiếp vì
  `smiles-drawer` không map positional `atom_index` an toàn; nó chỉ hiện bar
  ranking cạnh cấu trúc 2D thường.

### 2.2 Session và persistence

- `sessions.title` đã tồn tại ở domain, SQL schema, REST response và frontend.
- Session list dùng `row.title ?? row.session_id`; vì create flow không đặt title
  nên user vẫn thường thấy ID.
- SQLAlchemy 2 async Core, `asyncpg`, PostgreSQL 16, Alembic và transactional
  outbox đã tồn tại.
- Có optimistic versioning, session-scoped ownership, idempotency constraints,
  run/observation/answer audit và PostgreSQL migration test.

### 2.3 Agent/tool plane

Tool registry hiện có:

| Tool | Có thể làm gì | Profile hiện tại |
|---|---|---|
| `create_analysis_snapshot` | Gọi predictor và persist snapshot | `analysis` |
| `get_analysis_slice` | Đọc field cụ thể của predictor artifact, kèm field path/observation ID | `analysis`, `report_qa`, `evidence_research`, `audit_readonly` |
| `get_attribution` | Gọi token attribution cho đúng endpoint/Tox21 assay và persist observation | `report_qa` |
| `search_toxicology_evidence` | Search Europe PMC, normalize/filter rồi persist evidence | `evidence_research` |
| `get_evidence_record` | Đọc evidence đã persist; citation validator yêu cầu read trước cite | `evidence_research`, `audit_readonly` |
| `submit_grounded_answer` | Submit structured answer qua validator | `analysis`, `report_qa`, `evidence_research` |

Điểm đáng chú ý: source code hiện đã wire Europe PMC mặc định, nhưng comment ở
`MessageComposer` vẫn nói evidence search chưa được wire và do đó ẩn intent khỏi
UI. Comment/UI này đã stale. Ở default Compose dùng scripted runtime thì các
intent agentic vẫn không chạy; khi bật OpenCode overlay, registry có thể cung
cấp evidence tools thật.

### 2.4 OpenCode

- Runtime pin `1.17.11`, adapter HTTP/SSE và contract snapshot đã có.
- Profile `toxagent` là deny-all, chỉ allow namespace MCP `toxagent_*`.
- Local launcher cô lập `HOME`/XDG để global config, foreign MCP và permission
  không leak vào worker.
- Launcher đã biết copy riêng provider auth file vào isolated home.

## 3. Kiến trúc đích

```text
Browser
  ├─ endpoint/assay selector
  ├─ chat + session title
  └─ artifact viewer (prediction + XAI SVG + numeric details)
           │ public /v1 + OIDC
           ▼
toxagent-control
  ├─ API replicas: auth, validation, reads, SSE
  ├─ worker replicas: analysis, XAI bundle, research, runtime turns
  ├─ typed MCP tools + capability tokens + audit
  ├─ safe provider adapters
  │    ├─ ToxPred / ToxOCR
  │    ├─ Europe PMC / PubChem / EPA CTX
  │    ├─ search API + safe document fetch
  │    └─ optional Obscura dynamic-browser service
  ├─ PostgreSQL: product state + outbox + job leases
  └─ object store: images, sanitized SVG, export bundles, retained raw payload
           │ private network
           ├──────────────► ToxPred (prediction + numeric XAI + RDKit SVG)
           └──────────────► OpenCode runtime (model only sees run-scoped MCP)
```

Nguyên tắc giữ nguyên:

- Browser và model không gọi thẳng ToxPred, database hay arbitrary URL.
- Predictor number, explanation number và source document đều có provenance.
- hERG, ClinTox và từng Tox21 assay vẫn là phép đo riêng; không tạo aggregate
  toxicity/safety score.
- “Luôn có explainer artifact” không đồng nghĩa gradient luôn thành công. Khi
  explainer lỗi, bundle vẫn phải có một immutable `failed` artifact với error
  code/provenance; không được giả lập highlight.
- Không cho model shell/filesystem/direct OpenCode web tools chỉ để tăng
  capability. Tất cả đi qua control-plane tools có scope và audit.

## 4. Workstream A — endpoint/model selection trong chat

### 4.1 Làm rõ contract

UI nên cho user chọn **logical endpoint** (`herg`, `tox21`, `clintox`) và Tox21
assay, không cho nhập URL hoặc chọn một deployment/model artifact tùy ý.

Lý do:

- Model registry hiện quyết định model nào phục vụ endpoint.
- Một model `herg-tox21-chemberta-v1` đang phục vụ nhiều logical endpoints.
- Cho browser chọn arbitrary predictor URL phá private-network boundary, audit
  provenance và mở SSRF/data-exfiltration surface.
- `model_id` nên hiển thị read-only để reproducibility; model routing là config
  của deployment và chỉ admin mới thay đổi.

### 4.2 API contract đề xuất

Mở rộng `GET /v1/predict/capabilities`:

```json
{
  "capability_version": "predict-capabilities-v2",
  "default_endpoints": ["herg", "tox21"],
  "served_endpoints": ["herg", "tox21"],
  "endpoints": [
    {
      "id": "herg",
      "display_name": "hERG blockade",
      "enabled": true,
      "model_id": "herg-tox21-chemberta-v1",
      "supports_explanation": true,
      "explanation_target_required": false,
      "blocked_reason": null
    },
    {
      "id": "tox21",
      "enabled": true,
      "model_id": "herg-tox21-chemberta-v1",
      "supports_explanation": true,
      "tasks": ["NR-AR", "...", "SR-p53"],
      "explanation_target_required": true
    },
    {
      "id": "clintox",
      "enabled": false,
      "supports_explanation": false,
      "blocked_reason": "release artifact has no reproducible tokenizer"
    }
  ]
}
```

Mở rộng `AnalysisOptions`:

```json
{
  "endpoints": ["herg", "tox21"],
  "explanation_mode": "required",
  "explanation_targets": [
    {"endpoint": "herg"},
    {"endpoint": "tox21", "task": "SR-p53"}
  ]
}
```

`explanation_mode` nên có ba giá trị rõ nghĩa:

- `required`: bundle không complete cho tới khi mỗi target có artifact
  `completed`, `partial` hoặc `failed`.
- `on_demand`: backward-compatible, chỉ prediction được tạo ban đầu.
- `none`: dành cho batch/cost-sensitive use case; UI phải nói rõ không có XAI.

Product default theo yêu cầu mới: `required`. API cũ không gửi field có thể tạm
map sang `on_demand` trong một compatibility window, sau đó đổi default ở major
contract version.

Không tự chạy cả 12 backward pass của Tox21 một cách ngầm định. Khi chọn Tox21
và `required`, composer bắt buộc user chọn ít nhất một assay hoặc chọn rõ
“tất cả 12 assays” cùng cảnh báo latency/cost. Không có combined Tox21 explainer.

### 4.3 Frontend

Thay `ALL_ENDPOINTS` hard-code bằng query React Query dùng
`getPredictCapabilities()`:

- Render endpoint chips ngay trong thanh dưới chat; advanced popover giữ phần
  threshold và Tox21 assay.
- Disabled + tooltip từ `blocked_reason` cho endpoint unavailable.
- Hiện model ID trong details, không biến thành option thường.
- Không cho bỏ chọn tất cả endpoints.
- Khi Tox21 + required XAI, mở nested assay picker.
- Lưu last selection trong user preference; session draft có thể override.
- Gửi selection trong chính message để snapshot audit đúng cấu hình lúc chạy.
- Capability response là authority; local preference chỉ là default UI.

### 4.4 Files chính

- `frontend/src/components/workbench/MessageComposer.tsx`
- `frontend/src/lib/api/types.ts`
- `frontend/src/lib/api/endpoints.ts`
- `frontend/src/lib/preferences.ts`
- `toxagent-control/toxagent/api/schemas.py`
- `toxagent-control/toxagent/api/routes.py`
- `toxagent-control/toxagent/application/policy.py`
- `toxpred/api/routes.py`

### 4.5 Acceptance criteria

- UI không hard-code trạng thái served/unserved.
- ClinTox không được submit khi predictor báo unavailable.
- Refresh/session switch không làm selection của session A leak sang B.
- Payload và persisted policy snapshot ghi đúng endpoints, assays và explanation
  targets.
- Capability đổi giữa lúc mở trang và submit tạo typed conflict/unavailable,
  không silently substitute endpoint/model.

## 5. Workstream B — predictor + explainer thành một artifact bundle chắc chắn

### 5.1 Domain model

Thêm aggregate/read model `AnalysisArtifactBundle`, không nhét explanation vào
`predictor_response` cũ:

```text
AnalysisArtifactBundle
  analysis_id
  prediction_observation_id
  requested_explanation_targets[]
  explanation_observation_ids[]
  depiction_attachment_ids[]
  status: building | complete | complete_with_failures
  bundle_schema_version
  content_sha256
```

Prediction snapshot vẫn immutable và giữ lossless predictor response. Mỗi
endpoint/assay explanation là một immutable observation riêng để:

- retry đúng target lỗi mà không rewrite prediction;
- cache theo canonical SMILES + model artifact hash + method + target;
- cite/audit từng explanation;
- tránh một payload khổng lồ cho 12 Tox21 assays.

### 5.2 Completion semantics

“Output chắc chắn” được định nghĩa bằng invariant có thể test:

```text
bundle.status là terminal
⇔ prediction artifact tồn tại
∧ với mọi requested target có đúng một terminal explanation artifact
  (completed | partial | failed)
```

Run không emit `completed` trước khi invariant đúng. Nếu predictor prediction
lỗi thì run lỗi và không tạo bundle giả. Nếu prediction thành công nhưng một
explainer lỗi/hết timeout, bundle là `complete_with_failures`; UI vẫn thấy
prediction và failure card của explainer, không thấy highlight bịa.

### 5.3 Orchestration

Thay flow `CreateAnalysis` hiện tại bằng hai lớp:

1. `CreatePredictionSnapshot`: gọi forward prediction ngoài transaction.
2. `BuildAnalysisBundle`: fan-out bounded explanation targets, gom kết quả.
3. Một transaction ngắn persist prediction, terminal explanation observations,
   bundle index và outbox events.
4. Nếu process chết giữa compute và commit, retry dùng idempotency/cache key.

Để tránh giữ transaction trong lúc model backward pass, không persist trạng
thái nửa vời trừ một durable job record. Ở production nên dùng job/lease table
để worker khác recover; xem Workstream F.

Quick Predict không persist DB nhưng trả cùng wire shape:

```json
{
  "prediction": {"...": "AnalysisProjection"},
  "explanations": [{"...": "ExplanationV2"}],
  "bundle_status": "complete",
  "persisted": false
}
```

### 5.4 XAI contract v2 cho atom và bond

Hiện `token_atom_align_v1` chỉ map token span sang atom; dấu `=`, `#`, ring
closure và topology bị gộp vào một scalar. Để highlight bond có căn cứ, thêm
`token_structure_align_v2`:

```json
{
  "status": "completed",
  "endpoint": "tox21",
  "task": "SR-p53",
  "canonical_smiles": "...",
  "structure_order_version": "rdkit-structure-order-v2",
  "atoms": [
    {"atom_index": 0, "symbol": "C", "importance": 0.12,
     "relative_importance": 0.18, "source": "token_projection"}
  ],
  "bonds": [
    {"bond_index": 0, "begin_atom_index": 0, "end_atom_index": 1,
     "bond_type": "SINGLE", "importance": 0.05,
     "relative_importance": 0.07, "source": "explicit_token"}
  ],
  "unmapped_importance": 0.03,
  "tokens": [],
  "method": "grad_x_embedding_l2_v1+token_structure_align_v2",
  "limitations": ["attribution_not_causality"]
}
```

Quy tắc khoa học:

- Explicit bond token map vào RDKit bond index tương ứng.
- Ring closure phải resolve qua parser state, không map bằng regex đơn giản.
- Implicit bonds không có token thì **không được giả là direct bond
  attribution**. Nếu UI muốn màu liên tục cho chúng, dùng một field riêng
  `display_importance` suy ra từ hai atom kề và `source=adjacent_atom_derived`.
- Legend phân biệt direct token projection và display-derived bond color.
- Method hiện dùng L2 magnitude nên chỉ có độ lớn, không diễn giải tăng/giảm
  toxicity và không dùng red/green safe/toxic.
- Giữ full `tokens[]` và `unmapped_importance` để audit projection loss.

### 5.5 Rendering highlight

Khuyến nghị thay quyết định cũ “frontend tự vẽ bằng smiles-drawer” bằng một ADR
mới: ToxPred tạo **sanitized SVG depiction** bằng RDKit từ đúng
`canonical_smiles`, `atom_index` và `bond_index` vừa tính.

Lý do:

- Current UI đã chứng minh smiles-drawer không map positional atom index an toàn.
- RDKit chính thức hỗ trợ `highlightAtoms`, `highlightBonds`,
  `highlightAtomColors`, `highlightBondColors` và SVG output.
- ToxPred đã có RDKit và là nơi duy nhất nắm chắc index ordering.
- Frontend chỉ render artifact; không phải tái hiện cheminformatics mapping.

Contract rendering:

- Numeric explanation là canonical artifact; SVG là derived presentation.
- SVG ghi `numeric_content_sha256`, palette/version, width/height và renderer
  version.
- Sanitize SVG theo allowlist elements/attributes; không script, external URL,
  event handler, `foreignObject` hoặc embedded data ngoài policy.
- Dùng single-hue sequential palette, legend luôn hiện limitation.
- Khi SVG không load, UI fallback sang atom/bond table; không mất numeric XAI.
- Persist SVG trong object store cho durable session; Quick Predict có thể trả
  inline sanitized SVG với size cap.

RDKit documentation xác nhận drawing API hỗ trợ highlight cả atom/bond và trả
SVG: [RDKit Draw API](https://rdkit.org/docs/source/rdkit.Chem.Draw.html),
[MolDraw2D API](https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html).

### 5.6 Tool access

Thêm hai tool read-only thay vì bắt agent ghép nhiều primitive calls:

- `get_analysis_bundle(analysis_id, include=["prediction_summary", "explanation_summary", "provenance"])`
- `get_explanation_slice(analysis_id, endpoint, task?, fields?, top_k=12)`

Tool trả field paths/observation IDs, không trả raw SVG vào model context. Model
chỉ cần numeric summary; UI tự lấy depiction attachment bằng product API.

### 5.7 Acceptance criteria

- Mỗi requested target luôn có exactly one terminal explanation artifact.
- Probability trong explanation khớp probability của prediction trong tolerance
  đã chốt; mismatch fail bundle/contract test.
- Atom/bond index round-trip được test bằng canonical SMILES có branch, aromatic,
  ring closure, stereo, bracket atom và disconnected fragments.
- SVG snapshot test xác nhận class/index/color đúng; security test từ chối SVG
  active content.
- UI highlight atom và bond, có legend và `attribution_not_causality` luôn thấy.
- Reload/session recovery đọc lại toàn bộ bundle từ DB/object store.

## 6. Workstream C — session name thay cho session ID

### 6.1 Product behavior

Quy tắc hiển thị:

- Primary label luôn là `title`.
- ID không nằm trong normal sidebar/history row.
- ID chỉ hiện ở inspector, “Copy technical details”, URL route và audit export.
- User có thể rename; manual title không bao giờ bị auto-title overwrite.

### 6.2 Title lifecycle

Đề xuất hai tầng để không phụ thuộc LLM mới dùng được product:

1. **Deterministic immediate title** sau message có nội dung đầu tiên:
   - text: normalize whitespace, lấy semantic prefix tối đa 60–80 ký tự;
   - molecule: `Phân tích hERG + Tox21 · <short canonical SMILES>`;
   - OCR: `Phân tích cấu trúc từ ảnh · <short canonical SMILES>`;
   - batch: `Phân tích batch · N phân tử`.
2. **Optional background refinement** sau khi có first completed bundle/answer:
   model sinh 3–8 từ, cùng ngôn ngữ session; output qua length/control-character
   validator. Không chạy nếu title source là `manual`.

Thêm fields qua migration expand-compatible:

```text
sessions.title                 existing
sessions.title_source          null | deterministic | model | manual
sessions.title_status          pending | ready | failed
sessions.title_updated_at
```

### 6.3 API và concurrency

- `PATCH /v1/sessions/{session_id}` body `{title, expected_version}`.
- Trim/normalize; 1–120 chars; reject control chars/empty string.
- Ownership check như mọi session read/write.
- Optimistic version conflict trả 409.
- Emit `session.title_updated` event; frontend update query cache/SSE reducer.
- Auto-title dùng compare-and-set `title IS NULL` để không đè rename vừa xảy ra.

### 6.4 Query performance

`SessionService.list()` hiện tạo N+1 pattern: với từng session lại query runs và
messages, và `run_count` chỉ đếm tối đa 10 runs vì gọi `limit=10`. Trước
production cần một projection query duy nhất hoặc denormalized session summary:

- `last_message_preview`, `last_activity_at`, `run_count`, `active_run_id`;
- cursor pagination theo `(updated_at, id)`, không dùng offset cho history lớn;
- server-side search trên title/preview; không chỉ filter 25 rows đã load.

### 6.5 Acceptance criteria

- Session mới có title trước hoặc ngay sau durable first message event.
- Không có normal view nào fallback hiển thị raw ID lâu dài; trong vài trăm ms
  pending có thể dùng `Phiên mới`.
- Rename manual sống qua reload và không bị refinement ghi đè.
- Vietnamese/English title, emoji/control char/very long message đều được test.
- Session list không N+1 và `run_count` là count thật.

## 7. Workstream D — mở rộng agent capabilities

### 7.1 Capability matrix hiện tại

| Nhu cầu lõi | Hiện có? | Gap |
|---|---:|---|
| Lấy predictor result | Có | `get_analysis_slice` nhiều call; chưa có bundle summary |
| Lấy explainer result | Một phần | `get_attribution` vừa compute vừa read; chưa có read tool riêng cho persisted XAI v2 |
| Search tài liệu khoa học | Có | Europe PMC provider + tools đã có; UI intent stale/ẩn; coverage một provider |
| Search internet chung | Chưa | OpenCode `websearch`/`webfetch` bị deny đúng chủ đích |
| Đọc URL/document | Chưa tổng quát | Evidence record chỉ đọc payload đã lấy trong search |
| Chemical identity/property | Chưa | Nên thêm PubChem/DSSTox identity trước arbitrary web |
| So sánh nhiều analysis | Chưa | Agent phải tự đọc nhiều slices, tốn tool budget |
| Export/audit | Chưa | Có dữ liệu nền nhưng chưa có tool/product flow |

### 7.2 Tool catalog đề xuất

#### Tier 0 — product-owned, ưu tiên đầu tiên

1. `get_analysis_bundle` — prediction + explanation summary + provenance.
2. `get_explanation_slice` — persisted atoms/bonds/tokens top-k theo target.
3. `compare_analysis_results` — server tính exact delta theo declared field
   paths; model không tự trừ số.
4. `list_session_analyses` — compact list trong đúng current session.
5. `export_analysis_bundle` — tạo sanitized JSON/PDF/ZIP artifact; model chỉ
   nhận attachment metadata, không tự viết file.

#### Tier 1 — authoritative scientific sources

6. `resolve_compound_identity` qua PubChem PUG REST: CID, canonical/isomeric
   SMILES, InChIKey, synonyms và basic properties; tách “same connectivity” với
   exact stereochemical identity.
7. `search_toxicology_evidence` + `get_evidence_record` — giữ Europe PMC làm
   literature provider đầu tiên.
8. `lookup_regulatory_toxicology` qua EPA CTX/CompTox sau khi có API key và
   review terms; normalize DTXSID/assay/source provenance.
9. `get_source_document` — đọc full text/abstract từ provider allowlist, có size,
   MIME, redirect và license policy.

PubChem công bố PUG REST cho programmatic compound/property access và yêu cầu
không vượt khoảng 5 requests/giây; adapter cần cache, limiter và xử lý 503:
[PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest). EPA công bố
CTX APIs để truy cập dữ liệu computational toxicology và hiện yêu cầu đăng ký
API key miễn phí: [EPA Computational Toxicology APIs](https://www.epa.gov/comptox-tools/computational-toxicology-and-exposure-apis).
Europe PMC REST là nguồn phù hợp cho literature/search metadata:
[Europe PMC REST API](https://europepmc.org/RestfulWebService).

#### Tier 2 — safe general web

10. `web_search` — dùng một search provider có API/terms rõ, trả compact result
metadata; search result chưa phải evidence.
11. `fetch_web_document` — chỉ GET/HEAD, allow public Internet, resolve DNS và
kiểm IP trước/sau redirect, deny private/link-local/metadata networks, cap size,
MIME/time/redirect, extract text/markdown, store content hash/retrieved time.
12. `browse_dynamic_page` — optional Obscura tier cho trang cần JS/click/wait;
không bật mặc định cho mọi profile.

### 7.3 Obscura research và khuyến nghị

Repo được nhắc đến là
[`h4ckf0r0day/obscura`](https://github.com/h4ckf0r0day/obscura). Đây không phải
“không có browser engine”; nó là một headless browser engine độc lập viết bằng
Rust, nhúng V8, nói Chrome DevTools Protocol và không cần Chromium/Chrome/Node.
Release/Docker hiện hỗ trợ HTML/text/markdown/link extraction, JS evaluation,
screenshot/PDF, Puppeteer/Playwright CDP và chặn private network mặc định. Tác
giả cũng ghi rõ rendering engine còn đang phát triển; long-tail CSS, một số Web
APIs, media/compositor/font có thể khác Chromium.

Có community MCP adapter
[`Metadrama/obscura-mcp`](https://github.com/Metadrama/obscura-mcp) với page,
interact, persistent session và parallel scrape tools, nhưng tại thời điểm
research repo này còn rất nhỏ. Không nên đưa nguyên MCP đó vào production
allowlist trước security/contract review.

Quyết định đề xuất:

- Không thay Europe PMC/PubChem bằng Obscura.
- Dùng direct HTTP/provider API cho nguồn có API chuẩn: rẻ, deterministic, dễ
  provenance và ít prompt-injection surface hơn.
- Spike Obscura cho dynamic documents mà HTTP extractor không đọc được.
- Chạy Obscura thành service riêng, non-root, read-only filesystem, ephemeral
  profile/session, egress proxy và network policy; không mount product secrets.
- Control plane expose một tool hẹp `browse_dynamic_page`, không expose
  arbitrary `eval`/cookie APIs cho model ở MVP.
- Vẫn áp SSRF checks của ToxAgent dù Obscura có private-network deny mặc định;
  defense in depth và chống DNS rebinding/redirect.
- Sanitise/extract content, đánh dấu toàn bộ web content là untrusted data;
  không cho nội dung trang thay đổi tool policy/system instruction.
- Chạy paired benchmark HTTP extractor vs Obscura trên 30–50 nguồn thực tế:
  extraction success, fidelity, latency, memory, token count và prompt-injection
  rate. Chỉ promote nếu thắng rõ ở dynamic cohort.

### 7.4 Profiles và policy

Không có một “super agent” thấy tất cả tools. Dùng profile allowlist:

| Profile | Tools |
|---|---|
| `analysis` | create/read bundle, submit answer |
| `report_qa` | bundle/explanation/read/compare, submit answer |
| `evidence_research` | bundle read, scientific search/read, compound/regulatory lookup, submit answer |
| `web_research` | evidence tools + web search/fetch; dynamic browse khi deployment bật |
| `audit_readonly` | persisted read/export metadata; không external network |

Mọi external tool cần:

- capability token scope theo run/session/profile;
- per-tool timeout, retry, circuit breaker, rate/cost budget;
- normalized error taxonomy;
- tool call audit và observation/evidence ID;
- provenance gồm provider, query/URL, retrieval time, content hash;
- output size cap và compact model projection;
- injection classifier/markup stripping; citation chỉ từ accepted record.

### 7.5 Acceptance criteria

- Agent lấy được exact prediction và XAI từ persisted artifacts sau restart.
- Web search result không thể được cite trước `fetch/read` và acceptance policy.
- Cross-session artifact ID luôn indistinguishable với not-found.
- SSRF suite phủ loopback, RFC1918, IPv6 local, cloud metadata, redirect chain và
  DNS rebinding.
- Prompt injection trong page/evidence không thể mở thêm tool hoặc thay system
  policy.
- Profile live-surface test xác nhận shell/edit/subagent/direct OpenCode web
  vẫn bị deny.

## 8. Workstream E — user tự OAuth/provider cho OpenCode khi clone local

### 8.1 Phân biệt hai loại auth

Không trộn:

1. **ToxAgent product identity:** ai được đọc session/API; production dùng OIDC.
2. **OpenCode LLM provider credential:** credential để OpenCode gọi OpenAI,
   Anthropic, xAI…; local clone có thể dùng credential của chính user.

Mục này nói về loại 2. Refresh/access token của provider không được lưu trong
ToxAgent database, message, event, manifest hoặc log.

### 8.2 Local single-user BYOC flow — nên làm trước

OpenCode official docs cho biết `/connect`/`opencode auth login` lưu credential
ở `~/.local/share/opencode/auth.json`, và server API có provider/auth/OAuth
authorize/callback endpoints. Tham khảo:
[OpenCode providers](https://opencode.ai/docs/providers/),
[OpenCode server API](https://dev.opencode.ai/docs/server/).

Flow đề xuất:

```text
./bin/toxagent setup --agent
  1. kiểm opencode version đúng pin
  2. tạo isolated config/data dirs, chmod 0700
  3. chạy `opencode auth login` trong chính isolated XDG_DATA_HOME
     (hoặc import có xác nhận từ auth file hiện hữu)
  4. `opencode auth list` + `/provider` xác nhận connected provider
  5. user chọn provider/model từ inventory thật
  6. ghi provider/model ID không-secret vào .env local
  7. khởi động OpenCode loopback với deny-all toxagent profile
  8. live surface check + một provider smoke có xác nhận chi phí
```

Thay việc copy auth file mỗi lần start bằng một data volume/home cô lập bền vững:

```text
.data/opencode-auth/       mode 0700, gitignored, user-owned
.data/opencode-workspaces/ ephemeral child per product run
```

Auth root bền vững và run workspace phải là hai vùng khác nhau. Workspace per
run được xóa sau turn; auth store không được copy vào workspace.

Các lựa chọn onboarding:

- `--auth existing`: import/copy một lần từ path user xác nhận.
- `--auth login`: chạy OpenCode interactive OAuth/API-key flow trong isolated
  home; mặc định khuyến nghị.
- Environment API key: chỉ cho CI/advanced local; không ghi vào file nếu user
  chọn session-only.

### 8.3 Hosted multi-user — không dùng chung auth.json

Nếu ToxAgent sau này là SaaS nhiều user, không thể mount một shared personal
OAuth file vào OpenCode worker. Cần một trong hai topology được security/legal
duyệt:

- server-workload provider account do deployment owner quản lý; hoặc
- per-user BYOC vault: OAuth Authorization Code + PKCE, encrypted refresh token,
  KMS envelope encryption, per-user runtime isolation, revocation và provider
  terms cho delegated use.

Không proxy raw OAuth token qua browser vào message/MCP. OpenCode management API
chỉ ở private network và phải có server password/mTLS/service auth; official
server docs hỗ trợ `OPENCODE_SERVER_PASSWORD`, nhưng adapter hiện chưa gửi Basic
Auth nên đây là gap phải đóng trước remote production runtime.

### 8.4 Version caveat

Official OpenCode docs phản ánh bản hiện tại, còn repo pin `1.17.11`. Trước khi
dựa vào `/provider/{id}/oauth/*` phải snapshot OpenAPI của **binary pin thật** và
thêm adapter contract tests. Nếu pin cũ không có đủ API, local setup có thể gọi
CLI `opencode auth login`; không tự upgrade runtime trong cùng PR với product
feature.

### 8.5 Acceptance criteria

- Fresh clone có guided flow login provider mà không sửa source code.
- Restart giữ auth, nhưng tạo/reap workspace per run bình thường.
- `git status`, DB dump, event stream, runtime manifest và logs không chứa token.
- Two local OS users không đọc auth store của nhau.
- Revoked/expired OAuth trả typed `provider_auth_required` và UI hướng dẫn
  reconnect; không biến thành generic runtime failure.

## 9. Workstream F — SQLAlchemy và `toxagent-control` tới production

### 9.1 Đính chính hiện trạng

“Setup SQLAlchemy” phần cơ bản đã xong:

- `sqlalchemy>=2.0`, `asyncpg`, `psycopg`, `alembic` đã khai báo.
- Application dùng `create_async_engine` và SQLAlchemy Core repositories.
- Compose có PostgreSQL 16.
- Alembic có baseline `0001` và usage events `0002`.
- CI đã có PostgreSQL migration contract theo progress doc.

Việc còn lại là productionize, không phải cài package lại.

### 9.2 Gap cần sửa ngay

1. **Migration ownership mâu thuẫn:** runbook yêu cầu một migration writer,
   nhưng container entrypoint mặc định chạy `alembic upgrade head` trước mỗi
   app replica. Khi scale nhiều replica, nhiều migration writer có thể chạy
   đồng thời.
2. **DB pool chưa cấu hình:** engine dùng defaults; chưa có `pool_size`,
   `max_overflow`, `pool_timeout`, `pool_recycle`, `pool_pre_ping` theo capacity.
3. **Readiness chưa probe DB:** `/health/ready` kiểm predictor/runtime nhưng chưa
   xác nhận database query và Alembic revision.
4. **In-process scheduler:** background run task thuộc API process; startup coi
   mọi non-terminal run là orphan. Topology này khó scale/rolling deploy.
5. **Cross-instance SSE notifier:** notifier in-memory chỉ wake cùng process;
   instance khác phải poll. Chưa có PostgreSQL LISTEN/NOTIFY hoặc broker.
6. **N+1 session list và một số count sai như §6.4.**
7. **Object store production chưa có:** filesystem attachment adapter không phù
   hợp nhiều replica.
8. **Observability package gần như trống:** chưa có traces/metrics/dashboard.
9. **Auth production chưa hoàn tất:** production đã từ chối static token nhưng
   chưa có đầy đủ OIDC/JWKS flow và frontend login.

### 9.3 Database engine configuration

Thêm settings có bounds và validation:

```text
TOXAGENT_DB_POOL_SIZE
TOXAGENT_DB_MAX_OVERFLOW
TOXAGENT_DB_POOL_TIMEOUT_S
TOXAGENT_DB_POOL_RECYCLE_S
TOXAGENT_DB_COMMAND_TIMEOUT_S
TOXAGENT_DB_STATEMENT_TIMEOUT_MS
TOXAGENT_DB_LOCK_TIMEOUT_MS
TOXAGENT_DB_ECHO=false
```

Engine production bật `pool_pre_ping=True`; pool capacity tính theo:

```text
(web replicas × pool max) + (worker replicas × pool max) + migration/admin
< managed PostgreSQL max_connections với headroom
```

Không retry từng statement trong transaction một cách mù. Nếu connection chết
giữa transaction, rollback toàn UoW và retry ở application boundary chỉ cho
operation idempotent. SQLAlchemy docs mô tả `pool_pre_ping` để kiểm connection
khi checkout và các tùy chọn pool/recycle:
[SQLAlchemy engine configuration](https://docs.sqlalchemy.org/en/20/core/engines.html),
[SQLAlchemy connection pooling](https://docs.sqlalchemy.org/en/21/core/pooling.html).

### 9.4 Migration topology

Tách image command/workload:

```text
toxagent-migrate: alembic upgrade head       # one-shot protected job
toxagent-api:     uvicorn ...                # không migrate
toxagent-worker:  durable job consumer       # không migrate
```

- Production đặt `TOXAGENT_SKIP_MIGRATIONS=1` cho app/worker hoặc bỏ logic migrate
  khỏi app entrypoint hoàn toàn.
- Deploy gate: backup verified → one migration job → revision check → API/worker
  rollout.
- Dùng expand/backfill/switch/contract như runbook hiện có.
- Không dùng `metadata.create_all()` ngoài test/dev.
- Thêm migration lock/advisory lock như safety net, nhưng không xem đó là lý do
  cho phép mọi replica tự migrate.
- Baseline migration dùng `metadata.create_all()` thuận tiện cho fresh DB nhưng
  các revision tiếp theo phải explicit/reviewable; thêm autogenerate drift test.

### 9.5 Durable job execution

Tách `RunScheduler` khỏi API process:

- `jobs` table hoặc external broker; với phạm vi hiện tại có thể bắt đầu bằng
  PostgreSQL-backed queue (`FOR UPDATE SKIP LOCKED`).
- Fields: job ID, run ID, kind, status, attempt, available_at, lease_owner,
  lease_expires_at, idempotency key, last_error.
- API transaction tạo message + run + job + outbox cùng lúc.
- Worker claim lease, heartbeat, execute, commit artifacts; worker chết thì lease
  hết hạn và job được reclaim.
- Per-session concurrency vẫn enforce trong DB.
- Startup không còn fail tất cả non-terminal runs chỉ vì một API replica restart.
- CPU/GPU predictor vẫn là service riêng; control worker không import model code.

### 9.6 Events và SSE nhiều replica

- PostgreSQL outbox tiếp tục là source of truth.
- Dùng LISTEN/NOTIFY chỉ làm wake-up hint; client vẫn đọc theo durable sequence.
- Poll fallback khi miss notification/reconnect.
- Một dispatcher hoặc mỗi API replica scoped-read; đo outbox lag.
- Event retention/compaction không được tạo gap mà client không reconcile được.

### 9.7 Identity/security

- Implement OIDC/JWKS verifier: exact issuer/audience, allowed algorithms,
  `exp`/`nbf`, JWKS cache/rotation, subject and role mapping.
- Frontend dùng Authorization Code + PKCE hoặc BFF secure session; bỏ pasted
  static token ở production.
- MCP capability signing secret tách khỏi OIDC keys.
- OpenCode management endpoint private + Basic Auth/mTLS; MCP endpoint private,
  short-lived run token, revocation.
- Secret manager cho DB/provider/object-store credentials; không `.env` trong
  production image.
- Network policies: frontend→control only; control→DB/predictor/OCR/runtime/
  allowed providers; runtime→run-scoped MCP và LLM provider only.

### 9.8 Object store và retention

Thêm S3/GCS-compatible adapter:

- server-side encryption/KMS, bucket private, no public ACL;
- content-addressed key + owner/session prefix;
- signed URL chỉ do authorized API phát, TTL ngắn;
- lifecycle cho transient upload, XAI SVG và export;
- DB row + object write dùng staged/finalize workflow và cleanup reconciliation;
- backup/restore drill phải cover cả DB lẫn object store references.

### 9.9 Observability và SLO inputs

Instrument OpenTelemetry/metrics:

- request rate/error/latency theo route và typed error;
- DB checkout wait, pool used/overflow, query latency, conflict/deadlock;
- job queue depth/age/lease reclaim/attempt;
- outbox lag, SSE clients/reconnect/gap;
- predictor/OCR/XAI latency, explanation partial/failure ratio;
- tool calls, provider 429/circuit state, runtime steps/tokens/cost;
- answer correction/fallback/citation validation;
- redact SMILES/text theo data policy; không high-cardinality IDs trong metric
  labels.

Không chốt SLO bằng số đoán. Chạy alpha/soak ít nhất một tuần rồi đặt SLO và
alert có owner/runbook.

### 9.10 Production verification

- Migration from previous release và fresh DB.
- PostgreSQL integration cho toàn repository/E2E suite, không chỉ schema check.
- Load mixed traffic: session reads/SSE + prediction + XAI + research.
- Failure injection: DB restart/latency/deadlock, worker kill, predictor 503,
  XAI timeout, Europe PMC 429, OpenCode disconnect, object-store outage.
- Backup restore sang environment mới; compare counts/hashes/outbox sequence và
  open artifacts.
- Canary new runtime/model/tool schema with manifest diff and rollback.
- Security review cho IDOR, OAuth/token leakage, MCP replay, SSRF, prompt
  injection, SVG/XSS, upload MIME, CORS/CSRF và supply chain/SBOM.

## 10. Chuỗi PR triển khai đề xuất

| PR | Nội dung | Phụ thuộc | Ước lượng |
|---|---|---|---:|
| P0 | ADR: endpoint vs model, bundle terminal semantics, XAI v2/SVG ownership | — | 1–2 ngày |
| P1 | Capability-driven endpoint + Tox21 assay selector, preferences, tests | P0 | 2–4 ngày |
| P2 | Session rename API, deterministic auto-title, hide ID, list query fix | — | 3–5 ngày |
| P3 | `token_structure_align_v2`, bond contract và scientific tests | P0 | 4–7 ngày |
| P4 | RDKit sanitized SVG rendering + FE atom/bond viewer | P3 | 3–5 ngày |
| P5 | AnalysisArtifactBundle orchestration/persistence/recovery | P3 | 5–8 ngày |
| P6 | Bundle/explanation/compare MCP tools + tool evals | P5 | 3–5 ngày |
| P7 | Enable/fix evidence intent UI; PubChem identity adapter | P6 | 3–5 ngày |
| P8 | Safe web search/fetch provider + SSRF/injection/citation tests | P7 | 5–8 ngày |
| P9 | Obscura isolated spike + paired benchmark; ADR promote/reject | P8 | 3–5 ngày |
| P10 | `bin/toxagent setup --agent` local BYOC/OAuth flow | — | 3–5 ngày |
| P11 | DB pool/readiness/query indexes + separate migration command/job | — | 3–5 ngày |
| P12 | Durable worker/job leases + cross-instance event notify | P11 | 6–10 ngày |
| P13 | OIDC, object store, secrets/network policies | P11 | 7–12 ngày |
| P14 | Telemetry, load/soak/failure/restore drills, release gates | P4–P13 | 7–12 ngày + thời gian quan sát |

Một engineer tuần tự: khoảng **53–93 engineer-days**, tùy độ rộng của web/OIDC/
object-store deployment. Hai track có thể song song:

- Product/science: P0 → P1/P3 → P4/P5 → P6/P7/P8/P9.
- Platform: P2/P10/P11 → P12/P13 → P14.

Production không nên được tuyên bố xong chỉ vì merge P14; cần alpha telemetry,
restore drill và security/scientific sign-off.

## 11. Release gates

### Gate 1 — product contract

- Endpoint/assay selection lấy từ capability thật.
- Session title usable, rename/manual precedence đúng.
- Prediction + requested explanations reload được thành một terminal bundle.
- Atom/bond highlight có index/provenance/limitation đúng.

### Gate 2 — agent capability

- Tool eval chứng minh agent đọc exact predictor/XAI artifacts.
- Europe PMC/PubChem evidence có accepted provenance và read-before-cite.
- Safe web path qua SSRF/prompt-injection suite.
- Obscura chỉ được promote nếu benchmark và security review đạt.

### Gate 3 — local distribution

- Fresh clone setup OpenCode provider bằng credential của chính user.
- Auth/config/workspace isolation và reconnect flow được test.
- Không secret trong git/log/DB/event/audit artifact.

### Gate 4 — production candidate

- Một migration writer, pool sizing, durable worker và multi-instance SSE chạy
  thật trên PostgreSQL.
- OIDC, object store, secret manager và network policy hoạt động.
- Load/soak/failure injection/backup restore/canary rollback đạt.
- Critical eval/numeric/citation gates của agentic rebuild plan vẫn xanh.
- Engineering, security, product và toxicology SME ký go/no-go.

## 12. Quyết định cần owner chốt

| Quyết định | Khuyến nghị mặc định | Owner cần có |
|---|---|---|
| Required XAI targets cho Tox21 | User chọn assay; không ngầm chạy cả 12 | Product + SME |
| Lưu SVG bao lâu | Cùng retention với analysis; numeric artifact là canonical | Product + Security |
| Auto-title có dùng LLM không | Deterministic trước; model refinement optional | Product |
| General web có cần login/click không | MVP read-only search/fetch; Obscura dynamic read-only | Security + Product |
| Search provider cụ thể | Chọn API có terms/SLA; không scrape search engine UI | Legal/Platform |
| Local BYOC provider support matrix | Chỉ provider đã smoke với pin OpenCode | Platform |
| Hosted credential topology | Server-workload trước; per-user vault chỉ khi thật sự cần | Security + Legal |
| Managed Postgres/object store/cloud | S3-compatible abstraction + PostgreSQL 16 trước | Platform |
| Retention/deletion/RPO/RTO/SLO | Chốt từ policy và alpha data, không đoán | Product + Security + Ops |

## 13. Definition of Done tổng

- User chọn đúng endpoint/assay từ chat và thấy rõ model/provenance thực tế.
- Mọi analysis yêu cầu XAI tạo đủ prediction + terminal explanation artifacts;
  failure được biểu diễn trung thực.
- Plot highlight đúng atom/bond trên canonical structure, có fallback và
  limitation không thể bị ẩn.
- User làm việc bằng tên session; ID chỉ còn là chi tiết kỹ thuật/audit.
- Agent lấy lại predictor/XAI, search/read literature và safe web qua typed,
  scoped, audited tools.
- Local clone có guided OpenCode OAuth/BYOC mà ToxAgent không sở hữu credential.
- Control plane chạy PostgreSQL/object store với migration/worker/event/auth/
  telemetry production topology và đã qua restore/failure/security/science gates.

