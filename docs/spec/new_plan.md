Có. Sau khi đọc lại branch hiện tại, xem đúng hai screenshot bạn gửi, đọc thẳng `RunBlock`, event stream, `QuickPredictPage`, `SettingsPage`, predictor registry và harness provider contract, mình sẽ **đổi plan trước đó theo hướng giữ Agent Layer làm phần cốt lõi của ToxAgent**, chứ không coi nó là thứ sẽ loại bỏ.

Một điểm cần nói rõ về phần “trải nghiệm thật ChatGPT/Perplexity”: mình không thể đăng nhập và thao tác trong tài khoản riêng của hai sản phẩm như một người dùng độc lập, nhưng mình đã đối chiếu **UI/behavior hiện hành từ tài liệu sản phẩm chính thức**, cộng với code frontend hiện tại và 2 screenshot của bạn. ChatGPT hiện tách progress/activity khỏi answer, cho phép theo dõi tiến trình real-time và có activity history/sources riêng; Search dùng inline citation + Sources panel. Perplexity Advanced Deep Research cũng chuyển sang progress hiển thị “đang đọc nguồn nào / đang học được gì”, có key findings xuất hiện dần, thay vì show raw tool trace. ([OpenAI Help Center][1]) ([Perplexity AI][2])

Và mình đồng ý với assessment của bạn: **UI hiện tại không phải chỉ cần polish CSS; interaction model đang sai.**

---

# 0. Target architecture mới của ToxAgent

Mục tiêu cuối mình đề xuất:

```text
tox-agent/
│
├── frontend/
│   ├── src/
│   │   ├── features/
│   │   │   ├── chat/
│   │   │   ├── predict/
│   │   │   ├── models/
│   │   │   ├── providers/
│   │   │   ├── sources/
│   │   │   └── settings/
│   │   ├── components/
│   │   ├── lib/
│   │   └── styles/
│   └── ...
│
├── backend/
│   │
│   ├── control/
│   │   ├── src/toxagent/
│   │   │   ├── api/
│   │   │   ├── application/
│   │   │   ├── agent/
│   │   │   ├── harness/
│   │   │   │   ├── adapters/
│   │   │   │   │   ├── opencode.py
│   │   │   │   │   ├── dsh.py
│   │   │   │   │   └── scripted.py
│   │   │   │   ├── provider.py
│   │   │   │   ├── gateway.py
│   │   │   │   └── context.py
│   │   │   ├── integrations/
│   │   │   │   ├── predictor/
│   │   │   │   └── ocr/
│   │   │   ├── activities/
│   │   │   ├── persistence/
│   │   │   └── domain/
│   │   ├── migrations/
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   ├── predictor/
│   │   ├── src/toxpred/
│   │   │   ├── api/
│   │   │   ├── application/
│   │   │   ├── domain/
│   │   │   └── scientific/
│   │   ├── registry/
│   │   │   ├── models/
│   │   │   └── profiles/
│   │   ├── configs/
│   │   ├── evals/
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   └── ocr/
│       ├── src/toxocr/
│       ├── tests/
│       └── pyproject.toml
│
├── devops/
│   ├── docker/
│   ├── compose/
│   ├── cloud/
│   ├── scripts/
│   └── observability/
│
├── docs/
│   ├── architecture/
│   ├── agent/
│   ├── predictor/
│   ├── deployment/
│   └── development/
│
├── .github/
│   └── workflows/
│
├── README.md
├── LICENSE
├── VERSION
└── .env.example
```

Root lúc đó thực tế chỉ còn:

```text
frontend
backend
devops
docs
.github
```

Nhưng **agent hoàn toàn không bị bỏ**. Ngược lại, nó có boundary rất rõ:

```text
Frontend
   │
   ▼
Control Plane
   │
   ├──── Agent Kernel
   │       │
   │       ▼
   │    Harness
   │       ├── OpenCode
   │       ├── DeepSeek Harness
   │       └── future runtime
   │
   ├──── Predictor Service
   │
   ├──── OCR Service
   │
   └──── Evidence/Search tools
```

Điều này thực ra hợp với code hiện tại. `AgentRuntimeProvider` đã được thiết kế thành abstraction, với `provider_id`, `model_id`, runtime events và lifecycle riêng; comment trong code còn nói rõ future OpenCode/DSH adapters đều implement contract này.

Hiện adapters đã có `opencode_v1.py` và deterministic `scripted.py`, nên **không nên đập harness viết lại**. Ta chỉ clean layout và phát triển tiếp abstraction đang đúng.

---

# 1. Git refactor: giữ Agent Layer nhưng làm repo product-centric

Cấu trúc hiện tại có `frontend`, `toxagent-control`, `toxpred`, `toxocr`, `backend`, `models`, `config`, `deploy`, `infra`, `scripts`, `benchmarks`, `artifacts`... song song ở root.

Mình sẽ mapping như sau:

| Current                    | Target                                  |
| -------------------------- | --------------------------------------- |
| `frontend/`                | `frontend/`                             |
| `toxagent-control/`        | `backend/control/`                      |
| `toxpred/`                 | `backend/predictor/src/toxpred/`        |
| `toxocr/`                  | `backend/ocr/src/toxocr/`               |
| `artifacts/`               | `backend/predictor/registry/`           |
| relevant `config/`         | `backend/predictor/configs/`            |
| `benchmarks/`              | `backend/predictor/evals/`              |
| deployment scripts         | `devops/`                               |
| `infra/compose/`           | `devops/compose/`                       |
| root deployment            | `devops/docker/`, `devops/cloud/`       |
| ML scripts                 | predictor `evals/` / research migration |
| agent scripts              | `devops/scripts/agent/`                 |
| historical root `backend/` | migrate/remove after parity             |
| `.github/`                 | giữ nguyên                              |

`toxagent-control/toxagent/agent/` hiện đã có `budget.py` và `kernel.py`; `harness/` có gateway/context/provider/adapters. Đây là code đáng giữ và phát triển tiếp.

## Không giữ `research/` lẫn trong control plane

Hiện `toxagent-control/toxagent/` còn có cả:

```text
agent/
answer/
api/
application/
capabilities/
connections/
domain/
harness/
persistence/
predictor/
research/
```

Target nên là:

```text
control/src/toxagent/
├── agent/
├── harness/
├── api/
├── application/
├── domain/
├── persistence/
├── activities/
└── integrations/
    ├── predictor/
    └── ocr/
```

`predictor/` bên control chỉ được là **client/integration**, không chứa model logic.

---

# 2. Các `.pt` checkpoint: đừng biến chúng trực tiếp thành dropdown

Đây là chỗ rất quan trọng.

Bạn có nhiều `.pt` trong workspace là tốt, nhưng UI không nên đơn giản scan:

```text
*.pt
```

rồi đưa hết vào:

```text
Choose model
├── abc.pt
├── final2.pt
├── best_model.pt
└── new_final_really.pt
```

Vì `.pt` ≠ deployable predictor.

Repo hiện tại đã có case thực tế chứng minh điều đó: ClinTox checkpoint tồn tại nhưng bị block vì tokenizer chính xác bị mất; embedding vocabulary không khớp, nên registry cố tình không serve model đó.

### Model binary và Model Definition phải tách nhau

Mình đề xuất:

```text
backend/predictor/registry/models/
├── herg-chemberta-v1.yaml
├── herg-attentivefp-v2.yaml
├── tox21-chemberta-v1.yaml
├── tox21-gatv2-v3.yaml
└── clintox-smilesgnn-v1.yaml
```

Ví dụ:

```yaml
model_id: herg-chemberta-v1
display_name: ChemBERTa hERG v1

provider: chemberta

capabilities:
  - herg

artifact:
  path: ${TOXAGENT_MODELS_DIR}/herg/chemberta-v1/best_model.pt
  sha256: ...

feature_schema: chemberta-smiles-v1

tokenizer:
  path: tokenizer/

thresholds:
  herg: 0.4133

runtime:
  device: auto
  batch_size: 32

status: admitted
```

Checkpoint thực tế:

```text
.data/models/
```

hoặc:

```text
/workspace/models/
```

và **không cần nằm trong Git source tree**.

Git chỉ giữ:

```text
manifest
config
hash
metadata
calibration
evaluation metadata
```

---

# 3. Model admission pipeline cho hàng loạt checkpoint hiện tại

Mình muốn làm thêm command:

```bash
./bin/toxagent models scan
```

Output kiểu:

```text
Found 17 checkpoints

✓ herg/chemberta/best_model.pt
  detected: ChemBERTa dual-head
  status: ready for validation

△ clintox/best_model.pt
  tokenizer missing

△ gatv2/final.pt
  model config missing

✓ tox21/attentivefp.pt
  config found
```

Sau đó:

```bash
./bin/toxagent models inspect <path>
./bin/toxagent models validate <model-id>
./bin/toxagent models admit <model-id>
```

Lifecycle:

```text
discovered
    ↓
draft
    ↓
validated
    ↓
evaluated
    ↓
admitted
    ↓
servable
```

Chỉ `admitted` mới hiện mặc định trong UI.

Và khi inspect `.pt`, ưu tiên:

```python
torch.load(..., weights_only=True)
```

không tùy tiện unpickle checkpoint lạ.

---

# 4. Tách rõ “Endpoint” và “Model”

Hiện user chọn:

```text
hERG
Tox21
ClinTox
```

nhưng đó là **toxicity endpoints**, không phải model.

Ta cần hai tầng:

```text
Endpoint
   ↓
Compatible Models
```

Ví dụ:

```text
hERG
├── ChemBERTa dual-head v1
├── MolFormer v2
└── AttentiveFP v3

Tox21
├── ChemBERTa dual-head v1
├── GATv2
└── AttentiveFP
```

Current `ModelRegistry` đã gần support điều này. Nó cho phép nhiều provider có cùng capability, nhưng **cố tình throw error khi capability trở nên ambiguous**, tức là code hiện tại đã đặt nền móng để chuyển sang explicit model selection.

Thay vì:

```python
registry.for_capability("herg")
```

ta chuyển thành:

```python
registry.resolve(
    capability="herg",
    model_id="herg-chemberta-v1",
)
```

Nếu model không support endpoint:

```text
422 incompatible_model
```

---

# 5. Predictor selection model

Mình đề xuất domain object:

```python
PredictionSelection(
    bindings={
        "herg": "herg-chemberta-v1",
        "tox21": "herg-tox21-chemberta-v1",
    }
)
```

Và support 3 mode:

```text
Auto
Manual
Compare        # phase sau
```

### Auto

ToxAgent chọn default/admitted model.

### Manual

User quyết định:

```text
hERG   → ChemBERTa v1
Tox21  → GATv2 v3
```

### Compare

Expert/research workflow:

```text
hERG →
    ChemBERTa
    MolFormer
    AttentiveFP
```

và trả side-by-side.

V1 nên implement **Auto + Manual trước**. Compare là Phase 2.

---

# 6. Predictor selection trong Quick Predict

`QuickPredictPage` hiện đã giữ state:

```tsx
const [endpoints, setEndpoints] = ...
```

và request hiện chỉ gửi:

```tsx
quickPredict({
    smiles,
    endpoints,
    threshold_overrides
})
```

Ta mở rộng thành:

```ts
{
  smiles,
  endpoints: ["herg", "tox21"],

  model_selection: {
    herg: "herg-chemberta-v1",
    tox21: "herg-tox21-chemberta-v1"
  }
}
```

UI:

```text
Endpoints & models

✓ hERG
  ChemBERTa hERG/Tox21 v1                 ▾

✓ Tox21
  ChemBERTa hERG/Tox21 v1                 ▾

○ ClinTox
  Model unavailable
```

Nếu cùng một dual-head model phục vụ hERG + Tox21:

```text
⚡ Hai endpoint dùng chung một model — chỉ cần 1 inference pass
```

Điều này cũng tận dụng được optimization hiện có: `ToxicityPredictor` đang deduplicate provider, nên dual-head hERG + Tox21 không chạy backbone hai lần.

---

# 7. Predictor selection trong Session

Đây mới là phần quan trọng hơn.

Mỗi Session có:

```text
Agent configuration
Predictor configuration
```

Ví dụ:

```json
{
  "ai_profile_id": "openai-gpt56",
  "predictor_bindings": {
    "herg": "herg-chemberta-v1",
    "tox21": "tox21-gatv2-v3"
  }
}
```

Khi user chat:

> Phân tích hERG của aspirin

Agent gọi:

```text
predict_toxicity
```

nhưng LLM **không được tự chọn model**.

Control plane inject:

```text
session predictor configuration
        ↓
run context
        ↓
predict tool
        ↓
selected predictor
```

Tức:

```text
User decides model
Agent decides when to use predictor
```

chứ không phải:

```text
Agent decides model
```

Đây là boundary mình rất muốn giữ.

---

# 8. Pin configuration theo từng Run

Ví dụ user đổi model giữa conversation:

```text
Turn 1 → ChemBERTa
Turn 2 → ChemBERTa
[user switches]
Turn 3 → GATv2
```

Ta phải lưu:

```text
Run #1
predictor_snapshot = ChemBERTa

Run #2
predictor_snapshot = ChemBERTa

Run #3
predictor_snapshot = GATv2
```

Không được query lại current session config khi review historical result.

Run provenance sẽ có:

```json
{
  "ai": {
    "runtime": "opencode",
    "provider": "openai",
    "model": "..."
  },

  "predictors": {
    "herg": {
      "model_id": "...",
      "artifact_sha256": "...",
      "threshold": 0.4133
    }
  }
}
```

Predictor hiện đã lưu model ID, weights SHA, tokenizer SHA, feature schema và base-model metadata trong provenance, nên phần này chỉ cần nối qua control run snapshot.

---

# 9. API target cho model selection

Predictor service:

```http
GET /v1/models
GET /v1/models/{model_id}

POST /v1/predictions
POST /v1/predictions:batch
```

Current `/v1/models` đã tồn tại.

Request mới:

```json
{
  "smiles": "...",

  "endpoints": [
    "herg",
    "tox21"
  ],

  "model_selection": {
    "herg": "herg-tox21-chemberta-v1",
    "tox21": "herg-tox21-chemberta-v1"
  }
}
```

Control plane product API:

```http
GET  /api/predictors/catalog
GET  /api/predictor-profiles

GET  /api/sessions/{id}/settings
PATCH /api/sessions/{id}/settings

POST /api/predict
```

---

# 10. AI Provider configuration: tách 2 khái niệm

Đây là chỗ mình sẽ tránh một architectural mistake lớn.

**Runtime provider** không phải **AI provider**.

Ví dụ:

```text
Runtime adapter:
    OpenCode

AI provider:
    OpenAI

Model:
    GPT-x
```

hoặc:

```text
Runtime adapter:
    DSH

AI provider:
    Anthropic

Model:
    Claude ...
```

Vì vậy architecture:

```text
Agent Kernel
     │
     ▼
Runtime abstraction
     │
     ├── OpenCode Adapter
     ├── DeepSeek Harness Adapter
     └── future...
              │
              ▼
        AI Provider Profile
              │
       ┌──────┼────────┐
       ▼      ▼        ▼
    OpenAI Anthropic Gemini ...
```

Contract hiện tại đã có:

```python
provider_id
model_id
```

trong `RuntimeSessionSpec`, nên đây là extension tự nhiên chứ không phải rewrite.

---

# 11. Provider Profile

Entity:

```text
AIProviderProfile
```

```json
{
  "id": "openai-main",

  "display_name": "My OpenAI",

  "provider": "openai",

  "model": "...",

  "base_url": null,

  "runtime_adapter": "opencode",

  "secret_ref": "secret://providers/...",

  "capabilities": {
    "tools": true,
    "vision": true,
    "structured_output": true
  },

  "status": "healthy"
}
```

Providers v1:

```text
OpenAI
Anthropic
Gemini
OpenRouter
OpenAI-compatible endpoint
```

OpenAI-compatible đặc biệt hữu ích cho:

```text
local inference
vLLM
LiteLLM
DeepSeek endpoint
self-hosted APIs
```

---

# 12. Settings UI mới

`SettingsPage` hiện tại về cơ bản mới chỉ có Control Plane connection + Expert Mode.

Target:

```text
Settings

General

AI Providers
──────────────────────────
OpenAI
GPT ...
Connected ✓
Default

Anthropic
Claude ...
Connected ✓

+ Add provider


Agent Runtime
──────────────────────────
OpenCode                     Default
DeepSeek Harness             Experimental


Predictor Models
──────────────────────────
14 admitted
3 unavailable
2 need validation

Manage models →


Advanced
──────────────────────────
Expert mode
Developer diagnostics
```

Add provider flow:

```text
Choose provider
      ↓
API key / sign in
      ↓
Choose model
      ↓
Test connection
      ↓
Capabilities detected
      ↓
Save
```

API key **không bao giờ GET ngược về frontend**.

UI chỉ nhận:

```text
sk-••••••••7xP2
```

hoặc đơn giản:

```text
Credential saved ✓
```

---

# 13. Session AI selector

Trong session không nên nhồi tất cả vào composer.

Top bar:

```text
Phân tích cấu trúc từ ảnh          GPT ... ▾
```

Click model:

```text
Agent model

✓ GPT ...          My OpenAI
  Claude ...       Anthropic
  DeepSeek ...     OpenRouter

Manage providers
```

Predictor configuration có thể nằm trong một compact controls popover:

```text
Predictors: Auto ▾
```

hoặc:

```text
hERG: ChemBERTa · Tox21: GATv2
```

Như vậy user luôn biết:

> “AI nào đang reasoning?”
> “predictor khoa học nào đang tạo score?”

Hai thứ này tuyệt đối không nên trộn.

---

# 14. Vấn đề UI hiện tại mình thấy trong screenshot

Code xác nhận đúng những gì screenshot đang thể hiện.

Current `RunBlock.tsx` render trực tiếp:

```text
run · intent · lane
```

sau đó:

```text
search_toxicology_evidence 1194ms
get_evidence_record 17ms
...
```

và còn render raw:

```text
failure_code
run ID
potential billing
xem chi tiết run
```

Đối với developer console thì tốt.

Đối với end-user chat thì sai abstraction.

Hiện event layer cũng trực tiếp expose:

```text
tool.started
tool.completed
tool.failed
```

cho frontend reducer.

Đó là root cause.

Không phải:

> “RunBlock CSS xấu.”

Mà là:

> **transport/debug event đang bị dùng làm product UX event.**

---

# 15. Xóa `RunBlock` khỏi main transcript

Current:

```text
┌──────────────────────────────┐
│ run: tìm evidence · agentic  │
│ ✓ tool                       │
│ ✓ tool                       │
│ ✓ tool                       │
│ ✓ tool                       │
│ error                        │
│ xem chi tiết run             │
└──────────────────────────────┘
```

Target:

```text
● Đang tìm các nghiên cứu liên quan…
```

sau vài giây:

```text
● Đang đọc và đối chiếu 12 nguồn…
```

sau đó:

```text
● Đang tổng hợp bằng chứng…
```

sau đó answer bắt đầu stream.

Không box.

Không border.

Không milliseconds.

Không internal tool name.

Không run UUID.

---

# 16. Semantic Activity Layer

Backend phải thêm layer:

```text
RAW EXECUTION EVENT

tool.started
search_toxicology_evidence
```

↓

```text
SEMANTIC ACTIVITY

activity.started
kind=literature_search
```

↓

Frontend:

```text
Đang tìm các nghiên cứu liên quan…
```

Schema:

```ts
type ActivityEvent = {
  activity_id: string
  run_id: string

  phase:
    | "planning"
    | "retrieval"
    | "reading"
    | "prediction"
    | "analysis"
    | "synthesis"

  kind:
    | "literature_search"
    | "evidence_reading"
    | "evidence_validation"
    | "toxicity_prediction"
    | "structure_recognition"
    | "cross_check"
    | "answer_synthesis"

  status:
    | "started"
    | "progress"
    | "completed"
    | "failed"

  label_key: string

  progress?: {
    current?: number
    total?: number
  }

  visibility:
    | "user"
    | "details"
    | "debug"
}
```

---

# 17. Cùng một tool nhưng label khác nhau

Đúng như bạn muốn.

Internal tool:

```text
search_toxicology_evidence
```

không có nghĩa lúc nào UI cũng phải ghi:

> Search toxicology evidence.

Context A:

```text
user: Có research nào về chất này?
```

UI:

```text
Đang tìm các nghiên cứu liên quan…
```

Context B:

```text
user: Kết luận này có đáng tin không?
```

UI:

```text
Đang kiểm tra bằng chứng…
```

Context C:

```text
agent đang verify disagreement
```

UI:

```text
Đang đối chiếu các nguồn…
```

Context D:

```text
systematic literature-style search
```

UI:

```text
Đang mở rộng phạm vi tìm kiếm…
```

Backend tool vẫn đúng một cái.

---

# 18. Nhưng đừng để LLM tự viết status tùy ý

Mình không khuyên:

```python
display_text = llm.generate_status()
```

vì có thể nói sai việc hệ thống thực sự đang làm.

Thay vào đó:

```text
tool
+
run intent
+
current phase
+
orchestration context
```

→ deterministic presentation registry.

Ví dụ:

```python
ToolPresentationRegistry = {
    ("search_toxicology_evidence", "discover"):
        "activity.searching_literature",

    ("search_toxicology_evidence", "verify"):
        "activity.verifying_evidence",

    ("search_toxicology_evidence", "crosscheck"):
        "activity.crosschecking_sources",
}
```

Frontend chỉ localize:

```text
activity.searching_literature
→ Đang tìm các nghiên cứu liên quan…
```

---

# 19. Gộp nhiều tool calls

Screenshot của bạn có:

```text
search
get
search
search
search
search
search
search
get
get
get
...
```

Không bao giờ render 15 dòng đó.

7 search calls:

```text
search × 7
```

được aggregate thành:

```text
● Đang tìm kiếm trên nhiều nguồn…
```

rồi:

```text
✓ Đã tìm thấy 63 kết quả
```

4 evidence fetch:

```text
● Đang đọc các nghiên cứu phù hợp…
```

rồi:

```text
✓ Đã xem kỹ 12 nguồn
```

Đây là sự khác biệt giữa **observability UI** và **end-user UI**.

---

# 20. Interaction pattern mình chọn: ChatGPT × Perplexity hybrid

ChatGPT Search hiện dùng inline citations và một Sources surface riêng; Deep Research cho phép xem progress real-time và có activity history riêng thay vì trộn debug traces vào câu trả lời. ([OpenAI Help Center][3])

Perplexity Advanced Deep Research hiện còn đi xa hơn: progress cho thấy sources đang được đọc, nội dung đang được học, và key findings có thể xuất hiện trước khi final report hoàn tất. ([Perplexity AI][2])

ToxAgent nên lấy:

```text
ChatGPT:
conversation cleanliness
inline progress
inline citations

+

Perplexity:
research progress
source-first UX
intermediate findings

+

ToxAgent:
predictor provenance
scientific model result
toxicity evidence
```

---

# 21. UI trạng thái session mình muốn

### 0 ms

User send:

```text
Có những research gì về chất này?
```

### 100–300 ms

Dưới message:

```text
✦ Đang phân tích yêu cầu…
```

Subtle pulse.

### 500 ms

Crossfade:

```text
⌕ Đang tìm các nghiên cứu liên quan…
```

### 2 s

```text
⌕ Đang rà soát 18 nguồn…
```

### 4 s

```text
◌ Đang đọc những nguồn phù hợp nhất…
```

### 7 s

```text
✦ Đang tổng hợp bằng chứng…
```

### Sau đó

Status row nhẹ nhàng collapse.

Answer stream trực tiếp:

```text
Các tài liệu hiện có cho thấy...
```

Bên dưới answer:

```text
12 nguồn   ·   8.4 giây
```

Không có bordered assistant card lớn.

---

# 22. Animation spec

Mình sẽ dùng motion khá tiết chế:

```text
Status enter:
opacity 0 → 1
translateY 3px → 0
180 ms

Status change:
crossfade
150–220 ms

Current icon:
soft pulse 1.0 → 0.65 → 1.0
~1.4 s

Text loading:
subtle gradient shimmer
không chạy quá nhanh

Completed:
check icon
fade after ~800 ms

Answer:
stream normally
```

Nếu dùng React:

```text
motion/react
AnimatePresence
layout
```

là đủ.

Không cần animation lòe loẹt.

Và phải respect:

```css
@media (prefers-reduced-motion: reduce)
```

---

# 23. `RunBlock.tsx` nên được thay thế

Current:

```text
RunBlock
```

Target architecture:

```text
Transcript
├── UserTurn
├── AssistantTurn
│
├── ActivityPresence        ← lightweight
│
├── AnswerContent
│
└── TurnActions
```

Components:

```text
ActivityPresence.tsx
ActivityLine.tsx
ActivityHistoryPopover.tsx
CitationChip.tsx
SourcePreview.tsx
RunDetailsDrawer.tsx
```

`RunDetailsDrawer` mới là nơi chứa:

```text
search_toxicology_evidence
1194ms
call_id
run_id
billing
raw events
```

---

# 24. Developer Mode

Bạn vẫn cần những thông tin hiện tại để debug agent.

Do đó **không xóa observability**.

Chỉ chuyển nó khỏi conversation.

```text
Settings
→ Developer mode
```

Khi bật:

```text
⋯
→ Run details
```

Drawer:

```text
Run
─────────────────
Runtime       OpenCode
Provider      OpenAI
Model         ...

Steps         17
Tools         14

search_toxicology_evidence
1.194s

get_evidence_record
17ms

...

Raw events
Request
Response
Usage
```

Đây là nơi current UI nên đi tới.

---

# 25. Right panel `Artifacts` hiện tại cũng nên đổi

Screenshot đang duplicate tool calls:

Main conversation:

```text
tools...
```

Right panel:

```text
tools...
```

Đây là thông tin hai lần.

Mình sẽ bỏ `Artifacts` generic và làm:

```text
Sources | Results
```

### Sources

```text
12 sources

1  Ciprofloxacin alters...
   Nature · 2025
   Academic

2  AI optimization...
   ...
```

Click → evidence details.

### Results

```text
Molecule
Canonical SMILES

Predictor results
hERG
Tox21

Structure
Attribution
Charts
```

Developer trace không nằm ở đây.

---

# 26. Citation UX phải sửa ngay

Screenshot đang show literal:

```text
▤cite▤evd_750a...
```

Cái này không nên tồn tại trong product UI.

Target:

```text
... tính nhạy cảm với ciprofloxacin. ¹
```

Hover `¹`:

```text
┌──────────────────────────────┐
│ Study title                  │
│ Journal · 2025               │
│ Academic                     │
│                              │
│ Relevant passage...          │
│                              │
│ Open source →                │
└──────────────────────────────┘
```

Hoặc small pill:

```text
[1]
```

Perplexity hiện cũng hỗ trợ hover/select source để xem source details và source type labels như Academic/Government/Trusted. Đây đặc biệt phù hợp với ToxAgent vì scientific provenance là core feature. ([Perplexity AI][4])

---

# 27. Runtime failure UX cũng phải đổi

Screenshot hiện tại:

```text
Runtime agent hiện không khả dụng.
(runtime_unavailable)

run_a806...
recovery run...
potentially billed...
```

User không cần biết mấy thứ đó trong conversation.

Khi mất runtime:

```text
↻ Mất kết nối với agent — đang khôi phục…
```

Nếu recovery thành công:

```text
✓ Đã kết nối lại
```

fade out.

Nếu không:

```text
Không thể tiếp tục phản hồi.

Thử lại    Chi tiết
```

Chỉ click `Chi tiết` mới thấy:

```text
runtime_session_lost
run_id
provider
usage
```

---

# 28. Frontend folder cũng nên reorganize theo feature

Hiện frontend đã tách:

```text
components/
hooks/
lib/
pages/
```

và component directories `answer`, `artifacts`, `inspector`, `transcript`, `workbench`...

Sau redesign mình muốn:

```text
frontend/src/features/

chat/
├── components/
│   ├── Transcript.tsx
│   ├── AssistantTurn.tsx
│   ├── ActivityPresence.tsx
│   ├── CitationChip.tsx
│   └── Composer.tsx
├── hooks/
├── state/
└── api/

predict/
├── QuickPredictPage.tsx
├── PredictorSelector.tsx
└── ResultView.tsx

models/
├── ModelPicker.tsx
├── ModelCatalog.tsx
└── ModelHealthBadge.tsx

providers/
├── ProviderPicker.tsx
├── ProviderSetupDialog.tsx
└── ProviderStatus.tsx

sources/
├── SourcesPanel.tsx
├── SourceCard.tsx
└── SourcePreview.tsx
```

Hiện `frontend/src/components/transcript/` có `RunBlock`, `AnswerBlock`, `RecoveryBanner`, `SystemEventCard`, `AnalysisSystemCard`... nên refactor này có target rất rõ.

---

# 29. Backend activity architecture

Không sửa raw event contract.

Giữ:

```text
tool.started
tool.completed
tool.failed
runtime.*
```

cho audit.

Thêm projection:

```text
Raw event stream
       │
       ▼
Activity Projector
       │
       ├── aggregate
       ├── classify
       ├── sanitize
       └── localize key
       │
       ▼
activity.started
activity.progress
activity.completed
```

Frontend normal mode subscribe:

```text
activity.*
answer.*
message.*
```

Developer mode subscribe thêm:

```text
tool.*
runtime.*
```

---

# 30. Điều này còn giúp agent layer scale tốt hơn

Sau này nếu đổi:

```text
OpenCode
→ DSH
```

raw event có thể hoàn toàn khác.

Nhưng UI vẫn chỉ biết:

```text
activity.searching_sources
activity.running_predictor
activity.synthesizing
```

Vậy frontend không bị coupled với OpenCode.

Đây là một architecture win rất lớn.

---

# 31. Các Epic/Issue mình sẽ tạo

| Epic                | Issue                                   | Priority | Kết quả                         |
| ------------------- | --------------------------------------- | -------: | ------------------------------- |
| **E0 Repo**         | R0 Baseline tests before migration      |       P0 | khóa behavior hiện tại          |
|                     | R1 Move control → `backend/control`     |       P0 | agent giữ nguyên                |
|                     | R2 Move predictor → `backend/predictor` |       P0 | clean service boundary          |
|                     | R3 Move OCR → `backend/ocr`             |       P0 | clean service boundary          |
|                     | R4 Consolidate devops                   |       P0 | root gọn                        |
|                     | R5 Retire legacy backend after parity   |       P1 | bỏ duplication                  |
| **E1 Models**       | M1 Introduce per-model manifests        |       P0 | scalable checkpoint registry    |
|                     | M2 External models root                 |       P0 | `.pt` không làm bẩn Git         |
|                     | M3 Checkpoint scanner                   |       P1 | phát hiện checkpoint hiện có    |
|                     | M4 Admission validator                  |       P0 | chỉ serve model reproducible    |
|                     | M5 Model catalog API                    |       P0 | UI discover models              |
|                     | M6 Explicit capability→model resolver   |       P0 | nhiều model/endpoint            |
| **E2 Predict**      | P1 Add `model_selection` request        |       P0 | manual model choice             |
|                     | P2 Quick Predict selector               |       P0 | endpoint + model                |
|                     | P3 Session predictor settings           |       P0 | model persistent per session    |
|                     | P4 Run config snapshots                 |       P0 | reproducibility                 |
|                     | P5 Compare mode                         |       P2 | model comparison                |
| **E3 AI Providers** | A1 Provider profile domain              |       P0 | reusable config                 |
|                     | A2 Secure secret store                  |       P0 | BYOC safe                       |
|                     | A3 Provider CRUD API                    |       P0 | setup from product              |
|                     | A4 Connection/capability test           |       P0 | validate before session         |
|                     | A5 Settings UI                          |       P0 | add/edit/remove provider        |
|                     | A6 Session model selector               |       P0 | per-session AI choice           |
|                     | A7 DSH adapter                          |       P1 | runtime expansion               |
| **E4 Activity UX**  | U1 Semantic activity schema             |       P0 | decouple raw tools              |
|                     | U2 Activity projector                   |       P0 | context labels                  |
|                     | U3 Aggregate repeated tool calls        |       P0 | no trace spam                   |
|                     | U4 Replace `RunBlock`                   |       P0 | ChatGPT-style status            |
|                     | U5 Smooth motion system                 |       P1 | polished transitions            |
|                     | U6 Recovery UX                          |       P0 | hide raw runtime errors         |
|                     | U7 Run Details drawer                   |       P1 | preserve diagnostics            |
| **E5 Evidence UX**  | S1 Structured citation payload          |       P0 | no raw cite tokens              |
|                     | S2 Citation chips + preview             |       P0 | ChatGPT/Perplexity style        |
|                     | S3 Sources panel                        |       P0 | scientific evidence surface     |
|                     | S4 Results panel                        |       P1 | predictor/artifact outputs      |
| **E6 Quality**      | Q1 Event-contract tests                 |       P0 | stable streaming                |
|                     | Q2 Predictor selection tests            |       P0 | wrong model never silently used |
|                     | Q3 Provider integration tests           |       P0 | runtime safety                  |
|                     | Q4 Playwright session UX                |       P0 | validate transitions            |
|                     | Q5 reduced-motion/accessibility         |       P1 | accessibility                   |

---

# 32. Implementation order mình khuyên

Không làm UI trước backend protocol.

```text
Phase 0
Baseline + git refactor

        ↓

Phase 1
Predictor registry
model admission
explicit model selection

        ↓

Phase 2
Session configuration
AI provider profiles

        ↓

Phase 3
Semantic activity protocol

        ↓

Phase 4
Full chat UI redesign

        ↓

Phase 5
Sources / citations / Results panel

        ↓

Phase 6
DSH + compare models
```

Nếu đổi UI trước Activity Protocol, rất dễ kết thúc bằng một `RunBlock` đẹp hơn nhưng vẫn architecture cũ.

---

# 33. Chia thành PR thực tế

Mình khuyên khoảng **8 PR nhỏ hơn thay vì một mega-refactor**:

```text
PR-01
refactor: establish product monorepo layout

PR-02
refactor: introduce scalable predictor artifact registry

PR-03
feat: support explicit predictor model selection

PR-04
feat: persist session AI and predictor configuration

PR-05
feat: add configurable AI provider profiles

PR-06
feat: introduce semantic agent activity stream

PR-07
feat: redesign session transcript and progress UX

PR-08
feat: redesign citations sources and result inspector
```

Sau khi tất cả ổn:

```text
PR-09
chore: retire legacy predictor and obsolete repository paths
```

---

# 34. Definition of Done cuối cùng

Khi hoàn tất, flow phải thành:

```text
User opens ToxAgent

        ↓

Configure AI
OpenAI / Anthropic / ...
        ↓

Configure predictor
hERG → ChemBERTa
Tox21 → GATv2

        ↓

Start session

        ↓

Ask:
"research gì về chất này?"

        ↓

✦ Đang phân tích yêu cầu…
⌕ Đang tìm các nghiên cứu liên quan…
◌ Đang đọc những nguồn phù hợp…
✦ Đang tổng hợp bằng chứng…

        ↓

Answer streams naturally
with proper citations

        ↓

Sources
[1] ...
[2] ...
[3] ...

        ↓

No raw tool names
No raw run IDs
No ms timing
No giant run cards
No duplicated trace panel
```

Nhưng khi bật developer mode:

```text
Run details
→ full exact tool trace
→ runtime
→ request
→ latency
→ usage
→ recovery
→ provenance
```

Tức là **không hy sinh observability để lấy UX đẹp**; chỉ đặt mỗi loại thông tin đúng tầng của nó.

---

## Điểm mình coi là quan trọng nhất

Nếu triển khai đúng plan này thì ToxAgent sẽ có ba hệ độc lập nhưng phối hợp sạch:

```text
Agent intelligence
    = AI Provider + Harness

Scientific inference
    = Endpoint + Explicit Predictor Model

Product experience
    = Semantic Activity + Conversation UI
```

Hiện tại ba lớp này còn bị leak vào nhau: runtime details leak lên UI, model registry mới chỉ phù hợp với một model/capability, và provider config còn thiên về setup OpenCode ngoài product.

**Đây là refactor đáng làm trước khi tiếp tục thêm nhiều feature**, vì sau đó bạn có thể thêm DSH, thêm 20 checkpoint, thêm agent tools hay thêm provider mà UI và architecture không phình ra tương ứng.

[1]: https://help.openai.com/en/articles/10500283-deep-research-faq?utm_source=chatgpt.com "Deep research in ChatGPT | OpenAI Help Center"
[2]: https://www.perplexity.ai/help-center/en/articles/13600190-what-s-new-in-advanced-deep-research?utm_source=chatgpt.com "What's New in Advanced Deep Research | Perplexity Help Center"
[3]: https://help.openai.com/en/articles/9237897?utm_source=chatgpt.com "Searching the web with ChatGPT | OpenAI Help Center"
[4]: https://www.perplexity.ai/help-center/en/articles/20260806-understanding-source-labels?utm_source=chatgpt.com "Understanding source labels | Perplexity Help Center"
