# ToxAgent harness — bản thiết kế kiến trúc

Tài liệu tiếp nối `DESIGN.md`. `DESIGN.md` trả lời **có nên harness-hoá không, và ở đâu**;
tài liệu này trả lời **nó trông cụ thể như thế nào** — module, contract, bất biến, điều kiện thoát
của từng giai đoạn.

Tài liệu có hai phần:

- **[Phần 1](#phần-1--kiến-trúc-đề-xuất-đọc-từ-dsh) (mục 0–8)** — kiến trúc đề xuất, đọc từ DeepSeek Harness.
- **[Phần 2](#phần-2--opencode-cùng-bài-toán-harness-lời-giải-khác) (mục 9–14)** — đọc lại cùng bài toán qua OpenCode, và
  những chỗ Phần 1 phải sửa. **Mục 14 là bảng thay đổi**; ai đã đọc Phần 1 rồi thì đọc thẳng mục đó.

Nguồn: ghi chú DSH trong `~/dsh-plugin/projects/deepseek_harness/` (đặc biệt `source.md`,
`g1-foundation`, `g2-raw`, `g3a-core`, `experiment.md`); bản cài `opencode 1.17.11` trên máy này
cùng docs tại `opencode.ai/docs`; đối chiếu với nhánh `agent_test` của
[NEU-Bio-Research-Team/tox-agent](https://github.com/NEU-Bio-Research-Team/tox-agent) tại commit `12c3aa5`.

> **Trạng thái:** đề xuất kiến trúc, chưa phải quyết định. Mọi khẳng định về code hiện tại đều
> kiểm chứng lại được bằng hai phụ lục ở cuối. Tài liệu này **không mâu thuẫn** với `DESIGN.md`
> ở bất kỳ điểm nào; nó bổ sung một thay đổi thứ tự giai đoạn (mục 6) và bảy rủi ro mới (mục 7, 12.2).

---

> **Bản sao.** Bản gốc nằm ở `~/dsh-plugin/projects/toxagent_harness/ARCHITECTURE.md`, cạnh
> `DESIGN.md` và bộ ghi chú DSH mà nó trích dẫn. Sửa bản gốc trước, rồi copy lại sang đây.

---

# PHẦN 1 — Kiến trúc đề xuất, đọc từ DSH

## 0. Đọc DSH ra được gì

Định nghĩa của DeepSeek gọn đến mức dễ bị đọc lướt:

> **Agent = Model + Harness.** Model sinh reasoning, text và tool call. Harness quyết định tám thứ.

Bảng dưới là toàn bộ giá trị của việc đọc DSH: nó cho ta **danh sách các quyết định phải có chủ sở
hữu**. Cột thứ ba cho thấy ToxAgent hôm nay vẫn ra đủ tám quyết định đó — chỉ là rải rác, ẩn danh,
và phần lớn nằm trong `model_server/main.py`.

| Harness quyết định | DSH đặt ở đâu | ToxAgent hôm nay đặt ở đâu | Sau khi thiết kế lại |
| --- | --- | --- | --- |
| Model thấy context nào | `system-prompt/assemble` + `deriveMessages()` | `_build_report_chat_prompt` nối một chuỗi phẳng `"USER: …\nASSISTANT: …"` | `harness/prompt` |
| Có những tool nào | `ctx.tools` registry có scope + `ToolRestriction` | Danh sách tool viết tay trong prompt planning, thực thi bằng chuỗi `if/elif` trong `_execute_report_chat_tool` | `harness/tools` |
| Tool được chạy hay phải xin phép | `tools/pre-execute` + `ctx.approval` | Không có | pre-execute hook (không cần approval; cần validate, quota, dedupe) |
| Command chạy sandbox nào | bubblewrap → Landlock / Seatbelt | Không chạy code người dùng | **bỏ hẳn** |
| Lịch sử lưu / replay / compact ra sao | Append-only `SessionEvent` log | `ReportChatSession` in-memory (`_SESSION_STORE` là một dict module-level) + `_trim_history_to_budget` cắt cặp lượt cũ nhất | `harness/session` |
| Khi nào agent tiếp tục hay dừng | `agent/turn-stopping` + `concludesTurn` trên tool result | `max_tool_calls: int = 3`, một vòng `for` | `harness/loop` |
| Có subagent / workflow không | Capability seam cho cả hai | Không | Workflow = Làn A. Subagent: không |
| Kết quả hiển thị ở đâu | ~35 plugin `ui-*` | SSE + frontend React | Giữ nguyên |

### 0.1 Cái KHÔNG mang sang — nói trước để khỏi mất thời gian

**Cordis và "Everything is a Plugin".** Đây là phần hay nhất của DSH về mặt kỹ thuật và là phần
**không nên** chép. Lý do là quy mô: DSH có 247 package `@deepseek-ai/dsh-*`, ~7.903 file, ~1.350
cạnh peer-dependency. Chính ghi chú DSH thừa nhận cái giá: *"DSH loại bỏ core cứng nhưng không loại
bỏ complexity. Complexity chuyển thành dependency graph, profile/bundle ordering, service injection,
version compatibility, plugin conflict, preset drift, debugging composition."* ToxAgent có 12.265
dòng Python trong ba gói (`agents/`, `tools/`, `model_server/`). Ở quy mô đó, một plugin runtime là
chi phí thuần: bạn nhận về một kiểu hỏng mới (`inject` một service không ai cung cấp thì fiber treo
ở `PENDING` **im lặng, không log gì**) mà không nhận về khả năng thay thế nào bạn thật sự cần.

Cái đáng lấy từ Cordis là **kỷ luật capability seam**, không phải cái loader: mỗi năng lực có một
*Definition* (Protocol), một *Provider* mặc định, và các *Consumer* chỉ nói chuyện qua Definition.
Trong Python đó là `typing.Protocol` + một module cung cấp, không cần framework.

**Sandbox.** Không chạy code người dùng. Đây là phần khó nhất của DSH và ToxAgent được miễn phí.

**Code Mode / PTC, subagent, model-authored workflow, Creator Mode.** Không có bài toán nào trong
ToxAgent cần chúng. Nhắc lại để về sau không ai mở lại.

**Mọi con số cache của DSH.** `experiment.md` đo trên route Claude/Codex qua pi-ai với
`cache_control` tường minh. ToxAgent chạy Gemini. Nguyên lý chuyển được, con số thì không —
xem mục 4.

### 0.2 Cái đáng mang sang, xếp theo giá trị

1. **Bất biến reconstructability.** DSH: *"Model-visible means logged — anything that reaches a model
   request must be reconstructable from the log."* Cụ thể hoá bằng event `request/header` ghi cả
   system prompt đã render lẫn tool schema đã lắp, khiến **mọi request là một hàm thuần của log**.
   ToxAgent cần một phiên bản mạnh hơn (mục 3), vì output của nó là con số khoa học.
2. **Surface ≠ transcript.** DSH có *hai* projection trên cùng một log: `deriveMessages()` dựng lịch
   sử gửi cho model (surface — cố ý **che** các đoạn đã bị compaction thay thế), còn transcript cho
   người đọc lại các event append-origin. Một log, hai cách đọc. Đây chính là thứ cho phép audit
   trail và compaction cùng tồn tại mà không đánh nhau.
3. **Pipeline tool có điểm mở rộng cố định và bất biến rõ.** `pre-execute → guards → execute →
   post-execute → result`. Các bất biến đáng chép nguyên: arguments parse **một lần** rồi deep-freeze
   trước khi chính sách chạy; post-execute được thay content **hoặc** value, không bao giờ cả hai;
   `tools/result` là vùng no-transform.
4. **Tách seam đo lường khỏi seam chính sách.** DSH tách `ctx.tokenMeter` khỏi `ctx.compaction` với
   một câu: *"the seam owns no pricing API."* Nghĩa là chính sách nén không được tự đoán chi phí.
5. **Fail fast khi cấu hình sai.** `dsh-compaction-basic` từ chối load nếu có setting lạ, hoặc nếu
   `retainRatio` không nhỏ hơn ngưỡng. Ngược hẳn với ToxAgent hôm nay, nơi hàng chục `os.getenv(...)`
   nằm rải rác và một biến gõ sai chỉ im lặng rơi về default.

---

## 1. Kiến trúc mục tiêu

```
harness/
  session/      event log append-only, surface, hai projection
  prompt/       context assembly: [static] rồi [volatile]
  tools/        registry + pipeline (pre / execute / post)
  loop/         driver turn/step
  provenance/   observation store + numeric validator
  llm/          adapter seam (gemini, local, chuỗi fallback)
  budget/       token meter (countTokens) + chính sách nén
  router.py     phân làn A/B, deterministic
```

Ánh xạ code hiện có:

| Code hôm nay | Đi về đâu |
| --- | --- |
| `agents/orchestrator_agent.py::run_orchestrator_flow` | Giữ nguyên, thành **Làn A**, được bọc bởi một tool duy nhất |
| `tools/tox_tools.py`, `tools/research_tools.py` | Thành body của `execute()` trong `harness/tools` |
| `model_server/main.py::_execute_report_chat_tool` (chuỗi `if/elif`) | Xoá, thay bằng registry |
| `model_server/main.py::_plan_report_chat_tool_call` + `_heuristic_report_chat_tool_plan` | Xoá, thay bằng native function calling |
| `model_server/main.py::_make_report_chat_llm_caller` | Xoá, thay bằng `harness/loop` |
| `agents/report_chat_agent.py::ReportChatSession`, `_SESSION_STORE` | Thay bằng `harness/session` |
| `estimate_context_tokens`, `validate_context_budget`, `_trim_history_to_budget` | Thay bằng `harness/budget` |
| `_compact_tool_result(max_chars=1800)` | Thay bằng observation policy khai báo theo từng tool |
| `_normalize_report_chat_*` (vá chuỗi sau khi sinh) | Thay bằng validator ở mục 3 |
| `_resolve_inference_backend`, `_resolve_*_model_key`, `_load_*_bundle_sync`, `_blend_member_scores_*` | Một provider registry có tên trong `harness/llm` (hoặc `model_server/inference/`) |

**Một luật phụ thuộc, viết ra để CI cưỡng chế được:** `harness/` **không được** import
`model_server/*` hay `agents/*`. Chiều ngược lại thì được. `main.py` dài 6.278 dòng chính vì hôm nay
không có luật nào như vậy.

---

## 2. Các contract cốt lõi

### 2.1 Session event

DSH có 12 variant lõi. ToxAgent cần 10 — bảng dưới, cột cuối ghi rõ chỗ **cố ý lệch** khỏi DSH.

| Event | Payload | Ghi chú |
| --- | --- | --- |
| `turn/start` | `{turn}` | Mở trước khi claim input |
| `turn/end` | `{turn, reason}` | `completed \| aborted \| error \| max_steps` |
| `step/start` | `{turn, step}` | Một lời gọi model + các tool nó yêu cầu |
| `step/end` | `{turn, step}` | |
| `user/message` | `{content, source}` | `source`: `user \| injected \| lane_a_result` |
| `assistant/message` | `{content, usage, provider, model}` | |
| `tool/call` | `{call_id, name, arguments}` | `arguments` là **chuỗi JSON thô model sinh ra**, chưa parse |
| `tool/result` | `{call_id, obs_id, content, is_error}` | `obs_id` trỏ vào observation store |
| `request/header` | `{system, tools, config}` | Chỉ ghi khi **thay đổi** — cùng ý tưởng `reason: initial\|change` của DSH |
| `validation/verdict` | `{ok, violations}` | **Mới, không có trong DSH.** Kết quả validator provenance |

Cố ý **bỏ** `assistant/chunk`. DSH giữ nó để replay ở mức token cho UI; ToxAgent chỉ cần audit,
và giữ chunk sẽ làm log phình gấp nhiều lần mà không phục vụ mục đích nào. SSE vẫn stream token
bình thường — nó chỉ không được persist.

Ba luật của DSH nên chép nguyên vì chúng rẻ và cứu rất nhiều lỗi về sau:

- `seq = len(log)` — hợp đồng liên tục, không bao giờ lọc bớt event khỏi log chuẩn.
- Mọi `event.data` phải JSON-serialize được **không mất mát**; kiểm ngay tại `append()`, không đợi
  lúc ghi đĩa. (DSH từ chối `BigInt`, hàm, `Date`, class instance, mảng thưa, tham chiếu vòng.)
- Event đã accept thì **đóng băng**. Không ai được sửa lịch sử bền vững.

### 2.2 Tool definition

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str            # ngắn — xem mục 4 về chi phí prefix
    parameters: dict            # JSON Schema
    execute: Callable[[dict, ToolExec], dict]
    timeout_s: float
    retries: int = 0
    concurrency_safe: bool = False
    observation: ObservationPolicy = ...   # mục 2.3
```

Pipeline, và các bất biến đi kèm:

```
tool/call (ghi log TRƯỚC khi chạy)
  → pre-execute   validate SMILES · dedupe · quota · deadline còn lại
  → execute       timeout · retry · circuit breaker
  → post-execute  tách blob · rút numeric index · đóng dấu obs_id · chiếu xuống text
  → tool/result   (đóng băng, vùng no-transform)
```

- `validate_smiles` là **hook ở pre-execute**, không phải tool. Nó không mang thông tin gì để model
  suy luận; làm tool chỉ tốn một round-trip và một chỗ trong schema.
- `check_claim_support` là **validator ở post-execute**, không phải tool. Việc kiểm chứng không nên
  là quyết định của model.
- Arguments parse một lần rồi freeze trước khi bất kỳ chính sách nào chạy.
- Signal/deadline được truyền suốt pipeline; wrapper được thay nhưng không được bỏ (xem rủi ro 7.6).

### 2.3 Observation store — và một chỗ phải lệch khỏi DSH

Mỗi tool result sinh ra một bản ghi:

```python
@dataclass(frozen=True)
class Observation:
    obs_id: str                  # "obs_7"
    tool: str
    args_hash: str
    numeric_index: dict[str, float]   # {"clinical.probability": 0.82, ...} — có cấu trúc, không phải text
    blobs: dict[str, bytes]           # heatmap_base64, molecule_png_base64 — KHÔNG vào context
    projection: str                   # đúng cái model được nhìn thấy
    raw: dict                         # đầy đủ, để audit
```

Blob là chuyện dứt khoát: `explain_prediction` hôm nay trả `heatmap_base64` và
`molecule_png_base64`. Ảnh base64 **không được vào context của model** — vừa vô dụng vừa đắt.
Post-execute tách ra, giữ trong store, để lại một reference id.

**Chỗ phải lệch khỏi DSH.** Tool-result pruner của DSH cắt theo ký tự: giữ 4096 đầu + marker +
1024 cuối, ngưỡng 8192. Docs của chính họ ghi giới hạn: *"Pruning is syntactic"* (cắt mù đầu/cuối).
Với DSH thì hợp lý — thứ bị cắt là output bash, văn bản thuần. **Với ToxAgent thì sai**: mọi
observation là JSON, và cắt mù đầu/đuôi một JSON tạo ra JSON hỏng — tệ hơn là không cắt, vì model
sẽ đọc một cấu trúc cụt và tưởng nó đầy đủ.

Nên: **chiếu theo trường, khai báo trong từng tool definition.** Ví dụ:

| Tool | Vào context | Chỉ vào store |
| --- | --- | --- |
| `predict_toxicity` | `final_verdict`, `clinical.probability`, top-3 mechanism task + score | toàn bộ 12 task Tox21, metadata model |
| `explain_prediction` | `top_atoms[:8]`, `top_bonds[:8]` | `heatmap_base64`, `molecule_png_base64` |
| `search_literature` | mỗi bài: pmid, title, năm, 240 ký tự đầu abstract | abstract đầy đủ |
| `find_analogs` | top-5: smiles, tanimoto, nhãn độc tính | toàn bộ top_k, fingerprint |

Đây là bảng phải viết tay một lần cho 10 tool. Nó rẻ, và nó là toàn bộ khác biệt giữa một budget
context ổn định và một `find_analogs(top_k=10)` nuốt trọn cửa sổ.

### 2.4 Router — deterministic, không để LLM chọn

```
input                                                        → làn
─────────────────────────────────────────────────────────────────────
SMILES hợp lệ, không kèm câu hỏi                             → A
ảnh                          → resolve (MolScribe)           → A
tên hợp chất phân giải được  → resolve_compound (PubChem)    → A
batch nhiều phân tử / benchmark                              → A  (không bao giờ B)
đang có report mở + câu hỏi                                  → B
còn lại                                                      → B
```

Làn B **gọi được** Làn A qua tool `run_full_analysis(smiles)`. Đó là chỗ hai làn gặp nhau.

Điểm cần sửa ngay: hôm nay `_looks_like_smiles` đang đóng vai bộ phân loại ý định, và nó **từ chối
mọi từ viết thường** — `aspirin`, `caffeine`, `paracetamol` đều trả `False`. Nó nên chỉ còn làm đúng
một việc: nhận diện chuỗi SMILES. Việc phân giải tên là của `resolve_compound`, việc phân làn là của
router, và cả hai đều **test được bằng unit test** — điều mà một prompt không làm được.

---

## 3. Bất biến provenance — đặc tả cưỡng chế

Đây là bất biến khiến kiến trúc này đáng tồn tại. DSH có *"model-visible means logged"*.
ToxAgent cần cái mạnh hơn, vì nó xuất bản con số:

> **Mọi con số xuất hiện trong output đều phải truy được về một observation cụ thể của lượt đó.**

Nhắc lại điều `DESIGN.md` đã chỉ ra: hệ thống hôm nay **an toàn một cách tình cờ**, vì LLM gần như
không được phép sinh số — `_deterministic_reasoning` sinh toàn bộ kết quả rồi LLM chỉ `result.update()`
đè lên. Chuyển sang agent loop là mở lại cửa đó. Nên bất biến này không phải tính năng thêm vào sau;
nó là **điều kiện để việc chuyển đổi không làm hệ thống tệ đi**.

### 3.1 Phát biểu chính xác

Một token số trong câu trả lời là hợp lệ nếu thoả một trong ba:

- **(a)** khớp một giá trị trong `numeric_index` của observation nào đó thuộc lượt này, trong dung sai;
- **(b)** là kết quả của một phép biến đổi **khai báo được** trên các giá trị đó: làm tròn, đổi đơn vị,
  đổi thang (0.82 → 82%), hiệu hoặc tỉ số của hai giá trị đã có;
- **(c)** nằm trong danh sách trắng: PMID, năm xuất bản, số thứ tự mục/danh sách, số xuất hiện nguyên
  văn trong `projection` của một observation (trích dẫn abstract).

### 3.2 Cơ chế

1. `post-execute` đóng dấu `obs_id` và rút `numeric_index` **có cấu trúc** — không regex trên text
   sau này, vì lúc đó đã mất kiểu.
2. Sau khi sinh, validator trích số từ câu trả lời.
3. So khớp với dung sai làm tròn: khớp nếu `|x − y| ≤ 0.5 × 10^(−d)`, với `d` là số chữ số thập phân
   model in ra. Nghĩa là model in "0,8" từ giá trị 0,82 thì **hợp lệ**; in "0,9" thì **không**.
4. Vi phạm → **reject và retry**, tối đa 1 lần, đưa danh sách số vi phạm vào prompt retry.
5. Retry vẫn hỏng → **không vá chuỗi**. Trả về câu trả lời deterministic (đã có sẵn:
   `_format_report_chat_fallback_reply`) và ghi `validation/verdict` vào log.

Bước 5 là chỗ phân biệt thiết kế này với hệ thống hôm nay. `_normalize_report_chat_out_of_scope_reply`
và `_normalize_report_chat_citation_confidence` đang vá chuỗi *sau khi* model đã sinh sai — nghĩa là
lỗi vẫn xảy ra, chỉ bị che đi.

### 3.3 Bốn chỗ sẽ đau, và cách xử

| Vấn đề | Xử lý |
| --- | --- |
| **Dấu thập phân tiếng Việt.** Model trả lời tiếng Việt sẽ in `0,82`; `numeric_index` giữ `0.82`. Không chuẩn hoá thì validator false-positive **hàng loạt** ngay ngày đầu | Chuẩn hoá cả hai phía trước khi so; regex phải nhận cả `,` lẫn `.` làm dấu thập phân, và nhận `.` làm dấu phân nhóm nghìn |
| Số nằm trong trích dẫn nguyên văn abstract | Luật (c): cho phép mọi số xuất hiện trong `projection`, không chỉ trong `numeric_index` |
| Model tự tính "3/5 bài báo → 60%" | Luật (b), transform khai báo trước: `ratio`, `percent`, `diff`, `round`, `unit_convert`. Ngoài danh sách thì không hợp lệ |
| PMID, năm, số thứ tự | Whitelist theo observation của `search_literature` |

### 3.4 Chi phí, và cách bật

Validator là một pass regex + tra bảng: mili-giây. Retry mới đắt — một lời gọi LLM nữa cho mỗi lượt
vi phạm. **Đừng bật `strict` ngay.** Giai đoạn đầu chạy **shadow mode**: validator chạy, ghi
`validation/verdict` vào log, nhưng không reject. Sau hai tuần bạn có tỉ lệ vi phạm thật, và
tỉ lệ đó mới quyết định được `strict` đáng giá bao nhiêu.

---

## 4. Context assembly và ngân sách token

### 4.1 Nguyên tắc: tĩnh trước, biến động sau

```
[static]   luật hệ thống · tool schema · report schema · skill đang bật
[volatile] phân tử · điểm số · analog · observation của lượt này · ngôn ngữ đầu ra
```

Repo hôm nay vô tình làm đúng một nửa (`system_prompt` lên đầu, observation nối vào cuối). Cái sai
nằm chỗ khác: **mọi thứ là một chuỗi**, nên không có breakpoint để đặt cache, và bốn lời gọi plan
thì tách rời hoàn toàn khỏi lời gọi chính.

### 4.2 Đếm token: đừng chép nợ kỹ thuật của DSH

`estimate_context_tokens = len(text) // 4` giả định ~4 ký tự/token. Tiếng Việt tokenize tệ hơn nhiều,
nên `validate_context_budget(max_tokens=8000)` đang hụt đáng kể trên nội dung tiếng Việt.

Điểm đáng chú ý: **DSH mắc đúng lỗi này và đã tự ghi nhận nó**. Dev note của `compaction-basic`:
*"the token meter's four-characters-per-token heuristic underprices CJK text and JSON schemas."*
ToxAgent gặp đúng cả hai thứ đó — tiếng Việt và JSON observation. Dùng `count_tokens` của provider.
Đây là một trong số ít chỗ mà đọc DSH cho ta biết **cái gì đừng chép**.

### 4.3 Nén: chiếu trước, summarize sau

Thứ tự bắt buộc, mượn nguyên của DSH (*pruner chạy trước range selection, và "can advance the surface
without a summary"*):

1. Chiếu theo trường (mục 2.3) — miễn phí, không gọi LLM.
2. Đẩy observation cũ ra khỏi context nhưng giữ trong store, để lại `obs_id` để model gọi lại được.
3. Chỉ khi vẫn vượt ngưỡng: summarize.

Với ToxAgent, thứ cần nén là **tool observation**, không phải lịch sử chat — ngược với DSH. Lịch sử
chat của một phiên hỏi đáp về một report vốn ngắn; một `search_literature` đầy abstract thì không.

### 4.4 Cache: phải đo trước khi tối ưu

Hai điều chuyển được từ `experiment.md` ở dạng **nguyên lý**:

- **Tool schema chiếm 80,3% prefix tĩnh** trong session được đo (`toolsTokens 6475` / `systemTokens 1588`).
  Với 10 tool của ToxAgent, giữ `description` ngắn là đòn bẩy lớn nhất lên chi phí cold-start.
- **Đổi tập tool giữa session là thứ nguy hiểm nhất** — prefix mất hiệu lực kể từ token schema đầu
  tiên bị đổi. Nên tập tool phải cố định trong một session, kể cả khi bật/tắt skill.

Còn con số thì phải đo lại trên Gemini: cơ chế implicit caching và ngưỡng token tối thiểu khác hẳn
route Anthropic có `cache_control` tường minh. Cách đo rẻ nhất: log
`usage_metadata.cached_content_token_count` mỗi lượt vào `assistant/message`, chạy hai tuần, rồi mới
quyết có cần explicit context caching hay không. `experiment.md` cũng ghi một bài học đắt: cache
sập về 0% mà **không có thay đổi cấu trúc nào** — prefix caching của provider là best-effort, nên
đừng thiết kế thứ gì phụ thuộc vào nó.

---

## 5. Audit trail: một log, hai projection

```
                 ┌── surface projection ──→ lịch sử gửi cho LLM (che đoạn đã nén)
session log ─────┤
   (JSONL)       └── transcript projection ─→ audit trail / phụ lục báo cáo
```

SSE là **projection thứ ba**, và nó phải được **dẫn xuất từ log**, không phải một nguồn song song.
Đây chính là lỗi cần tránh khi viết lại: `agent_analyze_stream` hôm nay đã phát đủ các event
`{token, reasoning, tool_call, tool_result, agent_event, done, error}` — nhưng chúng **không được ghi
lại ở đâu cả**. Hạ tầng đã có một nửa; nửa còn thiếu là độ bền.

Lưu trữ: JSONL mỗi session một file. Repo đã có `firestore/` với rules — session log là thứ đáng đưa
lên Firestore, không phải object `ReportChatSession`. Hôm nay `_SESSION_STORE` là một dict
module-level: **restart là mất sạch**, và với một sản phẩm sinh báo cáo độc tính thì đó là mất
audit trail.

---

## 6. Lộ trình — điều kiện thoát thay cho danh sách việc

`DESIGN.md` đã có bốn giai đoạn. Bảng dưới thêm **điều kiện thoát đo được** cho mỗi giai đoạn, và
**tách một giai đoạn mới 1.5**.

| GĐ | Làm gì | "Xong" nghĩa là | Không được làm |
| --- | --- | --- | --- |
| **0** | Dọn theo mục 8 của `DESIGN.md` | `grep -c LlmAgent agents/` = 0 · `main.py` < 4.000 dòng · benchmark cho kết quả **giống hệt baseline chụp trước khi dọn** | Thêm bất kỳ tính năng nào |
| **1** | Chuẩn hoá 10 tool, mỗi cái một schema, timeout/retry khai báo. Bọc thành MCP server | Cắm vào DSH (hoặc Claude Code / Codex) và gõ "aspirin có độc không" thì chạy được | Viết loop |
| **1.5** | **Mới.** Observation store + event log, bọc quanh Làn A | `run_orchestrator_flow` ghi ra session log JSONL; dựng lại được toàn bộ một lần phân tích từ log | Vẫn chưa viết loop |
| **2** | Harness riêng: loop + tool pipeline + context assembly (~1.500 dòng Python) | Validator provenance chạy **shadow mode**, có số vi phạm thật | Bật `strict` khi chưa có số |
| **3** | Skills, permission, compaction | | |

**Vì sao tách giai đoạn 1.5.** `DESIGN.md` xếp audit trail vào giai đoạn 2, ngầm định nó cần agent
loop. Nó không cần. Làn A hôm nay đã có đủ mọi thứ để ghi một event log tử tế: các bước cố định,
các tool call rõ ràng, kết quả có cấu trúc. Ghi log cho Làn A **trước** cho ba cái lợi: audit trail
có ngay cho phần đang thật sự chạy production; observation store được thử trên dữ liệu thật trước
khi loop phụ thuộc vào nó; và nếu giai đoạn 2 bị hoãn vô thời hạn, bạn vẫn giữ được phần giá trị
lớn nhất.

Nhắc lại lối tắt của `DESIGN.md`, vì nó vẫn là khuyến nghị mạnh nhất trong cả hai tài liệu:
**xong giai đoạn 1, bạn đã có agent chạy thật mà chưa viết dòng harness nào.** Giá trị thật của nó
không phải tiết kiệm công — là **kiểm chứng tool surface trước khi đóng đinh**.

Và giữ nguyên khuyến nghị dứt khoát: **đừng chuyển sang LangGraph hay pydantic-ai.** Chúng là thư
viện graph/agent, không phải harness. Bạn sẽ lặp lại đúng cái bẫy ADK đã gây ra trong repo này —
vòng lặp của người khác, rồi ~700 dòng đọc lại state họ để lại.

---

## 7. Bảy rủi ro chưa nêu trong DESIGN.md

**7.1 Tool gọi ngược HTTP vào chính process.** `analyze_molecule` POST `/analyze` tới
`MODEL_SERVER_URL`, mặc định `http://127.0.0.1:8000` — tức **chính server đang phục vụ
`/agent/analyze`**. Hôm nay không deadlock vì mọi handler đều `await asyncio.to_thread(...)`, nhưng
mỗi lần phân tích chiếm hai thread của anyio pool và **hai lớp timeout 240s lồng nhau**. Harness phải
gọi thẳng `_analyze_request_sync` in-process khi URL trỏ về chính nó. Đây cũng là lý do đừng để
`analyze_molecule` thành async naive: một `httpx` đồng bộ gọi vào event loop của chính mình là
deadlock cứng.

**7.2 SSE và event log thành hai nguồn sự thật.** Xem mục 5. Nếu viết loop mà vẫn emit SSE độc lập,
bạn sẽ có hai lịch sử không khớp nhau và không cách nào biết cái nào đúng.

**7.3 Ngôn ngữ đi vào prefix tĩnh.** `agents/language.py` + `choose_text` chọn chuỗi VI/EN ở tầng
deterministic — chỗ đó giữ nguyên. Nhưng ngôn ngữ đầu ra **không được** đi vào system prompt tĩnh:
một người dùng đổi ngôn ngữ giữa session sẽ làm mất toàn bộ prefix cache. Đặt nó ở phần volatile.
Ngoài ra validator ở mục 3 phải nhận diện được cả hai ngôn ngữ (dấu thập phân, cách viết phần trăm).

**7.4 Benchmark baseline phải chụp TRƯỚC giai đoạn 0.** `scripts/eval_e2e_benchmark.py` và
`tests/smoke/agent_layer_flow_smoke.py` đều gọi thẳng `run_orchestrator_flow`. Không chụp baseline
trước khi dọn thì không có cách nào chứng minh việc dọn không làm đổi kết quả — và đó là toàn bộ
lý do giai đoạn 0 an toàn.

**7.5 Evidence QA trùng hai nơi — chọn bản nào.** `agents/evidence_qa_agent.py` (291 dòng, đã viết
xong, chưa nối vào orchestrator) và `main.py::_build_evidence_qa_result`. Chọn bản trong `agents/`
làm chính, vì nó đã tách khỏi `main.py` — đúng chiều phụ thuộc ở mục 1.

**7.6 Timeout lồng nhau không có ngân sách chung.** `MODEL_SERVER_TIMEOUT = 240s`;
`_pubmed_get_with_retry` 3 lần × 15s; `_pubchem_get_with_retry` 3 lần × 12s; cộng chuỗi fallback
Gemini. Một lượt Làn B tệ nhất vượt xa mọi timeout của Cloud Run. Harness cần **một deadline cho cả
lượt**, truyền xuống mọi tool qua `ToolExec` — đúng cách DSH giữ `signal` xuyên suốt pipeline và cho
phép wrapper thay nhưng không cho phép bỏ.

**7.7 Kho blob bị dọn trong khi con trỏ vẫn còn.** Xem [mục 12.2](#12-ba-thứ-đừng-chép) — kiểu hỏng
này đo được trên bản OpenCode cài ở máy này, và với ToxAgent nó có nghĩa là **mất bằng chứng**, không
chỉ mất tiện nghi.

---

## 8. Bốn câu hỏi của DESIGN.md — mặc định khuyến nghị

`DESIGN.md` để mở bốn câu. Dưới đây là mặc định để công việc không bị chặn; nếu bạn không phản đối,
cứ coi như đã chốt.

| Câu hỏi | Mặc định | Hệ quả |
| --- | --- | --- |
| Report có cần tái lập bit-exact? | **Có** | Làn A deterministic vĩnh viễn; câu hỏi "agent-hoá pipeline" đóng lại |
| Người dùng chính là ai? | Nghiên cứu viên, từng phân tử một | Batch không bao giờ đi qua Làn B |
| Có cần audit trail? | **Có** | Giai đoạn 1.5 lên trước giai đoạn 2 |
| Ngân sách token/tháng? | Chưa biết | Chạy token meter + shadow mode 2 tuần rồi quyết. Đến lúc đó, compaction là *tính năng*, không phải bắt buộc |

---

# PHẦN 2 — OpenCode: cùng bài toán harness, lời giải khác

## 9. Vì sao đọc OpenCode sau DSH

DSH là một *agent runtime* mở nguồn được 3 tuần, còn ở developer preview. OpenCode là một
**sản phẩm đã chạy 16 tháng**: repo tạo 30/04/2025, MIT, TypeScript, 203.216 sao, 26.477 fork,
5.591 issue mở, push gần nhất đúng hôm nay (02/09/2026).

Khác biệt đó quyết định giá trị của mỗi bên đối với ToxAgent:

> **DSH cho ta từ vựng và bất biến. OpenCode cho ta những mặc định đã sống sót qua va chạm với
> người dùng thật.**

Cách làm giống hệt phần DSH: **ưu tiên bản cài trên máy hơn tài liệu.** Máy này có
`opencode 1.17.11`, và kho dữ liệu của nó là bằng chứng tốt hơn mọi trang docs:
`~/.local/share/opencode/opencode.db` — SQLite 344 MB, **323 session, 6.907 message,
31.588 part, 41.341 event**, cộng `tool-output/` và `snapshot/`. Mọi con số dưới đây đều đọc
ra từ đó và kiểm chứng lại được bằng Phụ lục B.

---

## 10. Bảy chỗ OpenCode làm khác DSH — và chỗ nào hợp ToxAgent hơn

### 10.1 State-sourced + change feed, thay vì event-sourced

DSH: log append-only **là** sự thật duy nhất, state là projection dẫn xuất.

OpenCode: ngược lại. `message` và `part` là những row **có `time_updated`, sửa được tại chỗ**.
Bảng `event` chỉ là một *change feed*: đếm 41.341 event nhưng chỉ có **6 loại**, và tất cả đều là
created/updated —

```
message.part.updated.1   28.241      session.created.1              121
message.updated.1        10.046      session.next.model.switched.1   72
session.updated.1         2.797      session.next.agent.switched.1   64
```

Không có event ngữ nghĩa nào. Lịch sử ngữ nghĩa nằm trong `message`/`part`; `event` chỉ để đánh
thức subscriber (SSE, TUI, IDE).

Hệ quả thực dụng: một tool đang stream thì cập nhật **tại chỗ** vào part của nó; đọc "trạng thái
hiện tại" là một câu SQL, không phải fold cả log. Đánh đổi: mất replay ở mức token.

Chú ý chi tiết rẻ mà đáng chép: **loại event mang số phiên bản trong tên** — `.1` ở cuối
`message.part.updated.1`. Đổi payload thì tăng lên `.2`, consumer cũ vẫn đọc được cái cũ.

> **Khuyến nghị cho ToxAgent: lấy mô hình OpenCode.** ToxAgent cần audit ("con số này từ đâu ra"),
> không cần replay ở mức token — và mục 2.1 của tài liệu này đã bỏ `assistant/chunk` vì đúng lý do
> đó. OpenCode cho thấy nếu đã bỏ chunk thì nên đi hết đường: bỏ luôn event sourcing, lưu thành
> bảng. Xem mục 14 để biết mục 2.1 thay đổi thế nào.

### 10.2 Message/Part — và tool result là một part có máy trạng thái

Phân bố 31.588 part trên máy này cho thấy toàn bộ mô hình dữ liệu:

```
tool 8.458 · reasoning 6.585 · step-start 6.470 · step-finish 6.406
text 2.592 · patch 911 · file 161 · compaction 5
```

Vai trò message chỉ có `user` và `assistant` (6.543 assistant / 364 user). **Tool result không phải
message** — nó là một part của assistant message. Ranh giới step cũng là part (`step-start` /
`step-finish`), không phải event riêng.

Hình dạng một tool part, đọc nguyên từ DB:

```json
{ "type": "tool", "tool": "glob", "callID": "call_TwysQWzk4Pna...",
  "state": {
    "status": "completed",
    "input":  { "pattern": "**/chuong_2.md" },
    "output": "/home/minhquang/office-auto/chuong_2.md",
    "metadata": { "count": 1, "truncated": false },
    "title":  "",
    "time":   { "start": 1779945017602, "end": 1779945017742 }
  },
  "metadata": { "openai": { "itemId": "WoDYfbuh..." } } }
```

Đọc kỹ ba trường trong `state`:

| Trường | Là gì | Tương ứng ở mục 2.3 |
| --- | --- | --- |
| `state.output` | **đúng cái model nhìn thấy** | `Observation.projection` |
| `state.metadata` | dữ liệu có cấu trúc, riêng của tool, không nhất thiết vào context | `numeric_index` + `raw` |
| `state.title` | nhãn cho UI | (chưa có — nên thêm) |
| `callID` | định danh | `obs_id` |

> **Mục 2.3 không phải phát minh — nó là hội tụ.** Một hệ thống đang chạy production với 8.458 tool
> call trên máy này đã tách đúng ba trường ấy. Điều đó hạ rủi ro thiết kế của mục 2.3 xuống gần
> bằng không, và cho ta chữ ký để chép thay vì tự nghĩ.

Hai thứ nên chép thêm ngay:

- `state.status` ∈ `pending | running | completed | error` — trên máy này: 8.307 completed,
  151 error. Có máy trạng thái thì UI stream được, và tỉ lệ lỗi theo tool là một câu SQL.
- `state.time.start/end` — **độ trễ từng tool call, miễn phí**. Bottleneck của Làn A là `/analyze`
  và PubMed; hôm nay không ai đo được cái nào tốn bao lâu vì không có chỗ nào ghi.

### 10.3 Tool trả về ba trường — cộng attachments

Chữ ký trong `@opencode-ai/plugin`:

```ts
type ToolResult = string | { title?, output, metadata?, attachments? }
type ToolAttachment = { type: "file"; mime: string; url: string; filename?: string }
```

> **Đây là lời giải cho `heatmap_base64`.** Mục 2.3 nói "tách blob khỏi context, để lại reference
> id". OpenCode đã chuẩn hoá nó thành `attachment{mime, url}`. ToxAgent nên chép nguyên chữ ký này
> thay vì tự đặt tên: `explain_prediction` trả `output` (top_atoms/top_bonds), `metadata`
> (numeric index), và `attachments: [{mime:"image/png", url:"/obs/obs_7/heatmap.png"}]`.

`ToolContext` còn hai thứ đáng lấy:

- `metadata({title, metadata})` — tool **báo tiến độ giữa chừng**. Với `/analyze` mất 5–30 giây,
  đây là khác biệt giữa một spinner câm và một thanh tiến trình có nghĩa.
- `abort: AbortSignal` — truyền xuống mọi tool. Chính là deadline chung mà rủi ro 7.6 đòi.

### 10.4 Bề mặt hook: 15 hook, một chữ ký duy nhất

Toàn bộ khả năng mở rộng của OpenCode nằm trong một interface `Hooks`, và **mọi hook đều cùng một
chữ ký**: `(input, output) => Promise<void>`, sửa `output` tại chỗ.

| Hook | Làm được gì | Dùng ở ToxAgent |
| --- | --- | --- |
| `tool.execute.before(input{tool,sessionID,callID}, output{args})` | ghi đè tham số trước khi chạy | **pre-execute**: validate + canonical hoá SMILES |
| `tool.execute.after(input{...,args}, output{title,output,metadata})` | viết lại kết quả model thấy | **post-execute**: chính là ObservationPolicy ở mục 2.3 |
| `tool.definition(input{toolID}, output{description,parameters})` | sửa schema gửi cho LLM | cắt schema động (mục 4.4) |
| `chat.params(…, output{temperature,topP,topK,maxOutputTokens,options})` | tham số mỗi lời gọi | temperature 0 cho Làn A |
| `chat.headers` | header HTTP | routing/quota |
| `experimental.chat.system.transform(…, output{system: string[]})` | **system prompt là MẢNG chuỗi** | đúng cái static/volatile ở mục 4.1 |
| `experimental.chat.messages.transform(…, output{messages})` | sửa lịch sử trước khi gửi | nén observation |
| `experimental.session.compacting(…, output{context, prompt})` | tuỳ biến prompt nén | prompt nén riêng cho observation |
| `experimental.compaction.autocontinue(…, output{enabled})` | có tự "continue" sau khi nén không | tắt — Làn A không tự chạy tiếp |
| `experimental.text.complete(…, output{text})` | sửa text sau khi sinh | **không dùng** — đây đúng là chỗ vá chuỗi mà mục 3 cấm |
| `permission.ask(input: Permission, output{status})` | allow/deny/ask | tắt bề mặt tool (mục 10.7) |
| `event`, `config`, `tool`, `auth`, `provider`, `command.execute.before`, `shell.env` | | |

So với DSH: `agent/pre-step` là waterfall có **15 listener** tranh nhau, quyết định trả về là
*authoritative*, guard đơn điệu không ai undo được deny, ba chế độ dispatch (waterfall / serial /
parallel), `next()` lồng nhau. OpenCode không có gì trong số đó — chỉ là mutate một object.

Cái giá: khi hai plugin cùng sửa một `output`, **thứ tự nạp quyết định**, và không có cách nào diễn
đạt "deny thắng allow". Với một hệ nhiều bên như DSH thì đó là lỗ hổng thật. Với ToxAgent — một
đội, khoảng 5 hook, không có plugin bên thứ ba — cái giá đó bằng **không**.

> **Khuyến nghị: lấy chữ ký của OpenCode.** Đừng xây bộ máy waterfall của DSH cho 5 hook.

Một điểm cần nhớ riêng: `experimental.text.complete` cho phép sửa text **sau khi model đã sinh**.
Đó chính xác là `_normalize_report_chat_out_of_scope_reply` của ToxAgent hôm nay, và mục 3 đã kết
luận không dùng. OpenCode có hook đó không có nghĩa ToxAgent nên dùng — với ToxAgent, sai số là
sai kết quả khoa học, không phải sai chính tả.

### 10.5 Compaction bằng con trỏ, không bằng surface replacement

Một `compaction` part, nguyên văn từ DB:

```json
{"type":"compaction","auto":true,"overflow":true,"tail_start_id":"msg_e95a43521001PGX4X4uIAxPshu"}
```

Cộng với một bảng:

```sql
CREATE TABLE session_context_epoch (
  session_id text PRIMARY KEY, baseline text NOT NULL,
  snapshot text NOT NULL, baseline_seq integer NOT NULL );
```

Nghĩa là: **lịch sử gốc không bị đụng đến**. Model view = `baseline` (bản tóm tắt) + các message từ
`tail_start_id` trở đi. Nén là **ghi một con trỏ**, không phải viết lại lịch sử.

So sánh với DSH, nơi cùng bài toán tốn: `SurfaceOp{op:'replace',start,end}`, ràng buộc
`sourceEventSeqs` phải liệt kê *mọi* node bị che, `replaceGeneration` đơn điệu, `SurfaceManager`
fold theo cửa sổ và **fail** nếu một replacement vượt qua đầu cửa sổ, cộng
`toolPairingBalancedBefore/After` kiểm tra hai mép.

> **ToxAgent lấy cách của OpenCode.** Nó *chính là* "một log, hai projection" ở mục 5, cài đặt bằng
> một con trỏ: transcript đọc toàn bộ message; surface đọc `baseline + tail`. Audit trail được giữ
> **theo cấu trúc**, không phải nhờ kỷ luật.

Chép luôn hai cờ `auto` và `overflow` — chúng ghi *vì sao* nén xảy ra (tự động hay thủ công; do áp
lực hay do tràn cửa sổ). Rẻ, và đó là dữ liệu để về sau chỉnh ngưỡng bằng số thay vì bằng cảm giác.

### 10.6 Kế toán token và chi phí là cột hạng nhất

Một `step-finish` part, nguyên văn:

```json
{"type":"step-finish","reason":"tool-calls","snapshot":"bf19a40bb916...",
 "tokens":{"total":8647,"input":255,"output":174,"reasoning":26,
           "cache":{"write":0,"read":8192}},"cost":0}
```

Và bảng `session` có sẵn các cột `cost`, `tokens_input`, `tokens_output`, `tokens_reasoning`,
`tokens_cache_read`, `tokens_cache_write`.

> **Đây đúng là dụng cụ đo mà mục 4.4 đòi.** Mục 4.4 nói "log `cached_content_token_count` mỗi lượt
> rồi mới tối ưu" nhưng chưa nói lưu ở đâu. OpenCode trả lời: **ở hai mức**. Chi tiết trên từng
> step (part), và **denormalize lên row session** để câu hỏi "phiên này tốn bao nhiêu" là một câu
> SQL chứ không phải fold cả log. Chép cả hai mức.

Ghi chú: `snapshot` trong step-finish là một sha của trạng thái filesystem, cho phép revert về bất
kỳ step nào. ToxAgent không sửa filesystem nên **bỏ** — nhưng cái tương đương thì có thật: chụp
trạng thái *report state* ở mỗi step sẽ cho phép "quay lại trước khi rerun_screening". Xếp vào GĐ3.

### 10.7 Permission là dữ liệu bền vững, có pattern — và đó là cách xoá bề mặt tool

Ba tầng, đọc được từ máy này:

```sql
-- bền vững theo project
CREATE TABLE permission (project_id, action, resource, …);
CREATE UNIQUE INDEX permission_project_action_resource_idx ON permission(project_id, action, resource);
```

```json
// override theo từng session, cột session.permission
[{"permission":"question","pattern":"*","action":"deny"},
 {"permission":"plan_enter","pattern":"*","action":"deny"},
 {"permission":"task","pattern":"*","action":"deny"}]
```

Cộng cấu hình tĩnh, theo docs: giá trị `allow | ask | deny`, wildcard `*` và `?`,
**luật khớp cuối cùng thắng**, override theo từng agent, và `--auto` duyệt mọi thứ chưa bị deny.
Tool tự xin quyền giữa chừng qua `ToolContext.ask({permission, patterns, always, metadata})`, với
`always` để ghi nhớ vĩnh viễn.

Đáng chú ý là **cách nó được dùng thật trên máy này**: 72 session deny `question`/`plan_enter`/
`plan_exit`, 9 session deny `task`, 4 session deny `todowrite`. Tức là permission ở đây không dùng
để chặn hành động nguy hiểm — nó dùng để **tắt bớt bề mặt tool**.

ToxAgent không cần approval của người dùng (không chạy code người dùng). Nhưng nó cần đúng cơ chế
đó cho việc khác, và có một bằng chứng đắt giá ngay trong `~/.config/opencode/opencode.jsonc` của
bạn — ghi chú giải thích vì sao MCP server `officecli` bị tắt hẳn từ 03/08/2026:

> Server đó bọc cả CLI thành một tool, kéo theo `officecli load_skill` — registry skill riêng của
> nó. Đo hai lần ngày 31/07/2026: agent tuyên bố sẽ dùng skill này, rồi chạy `load_skill word`
> thay vào đó, và không tạo ra file hợp đồng template nào. **Một luật trong prompt không sửa được:
> model không bao giờ gọi tool `skill`, nên `permission.skill` không bao giờ áp dụng.** Tắt server
> là *xoá bề mặt* thay vì đi xin model tránh nó.

> **Bài học, áp thẳng vào ToxAgent:** luật viết trong prompt không phải cơ chế cưỡng chế. Mục 2.2
> đã đề xuất bỏ `rerun_screening` — công cụ tồn tại hôm nay *chỉ vì* một luật prompt nói "chỉ chạy
> khi user đưa SMILES mới". Ghi chú trên là bằng chứng đo được, trên chính máy này, rằng cách làm
> đó hỏng. Xoá khỏi catalog, đừng viết luật.

---

## 11. Hai thứ nên chép nguyên

### 11.1 Lõi là HTTP server headless; mọi giao diện chỉ là client

`opencode serve` chạy một HTTP server headless, publish **OpenAPI 3.1 tại `/doc`**, phát SSE ở
`/event` và `/global/event`, và SDK được **sinh từ chính spec đó**. TUI, web, IDE extension đều chỉ
là client. Docs nói thẳng: *"The TUI is the client that talks to the server."*

> Đây là lập luận mạnh nhất trong cả Phần 2: **ToxAgent gần như đã ở sẵn kiến trúc này.** Đã có
> FastAPI, đã có SSE ở `/agent/analyze/stream`, đã có frontend rời. Thiếu đúng ba thứ:
> (1) OpenAPI cho phần agent — hôm nay `schemas.py` mới mô tả `/analyze` và `/agent/analyze`,
> chưa mô tả session/message/part; (2) SSE **dẫn xuất từ log** thay vì phát song song (mục 5);
> (3) một client sinh tự động thay vì gọi tay.
>
> Nghĩa là con đường rẻ nhất không phải "dựng một harness", mà là **làm cho cái server đang có
> nói đúng ba thứ đó**.

### 11.2 Agent định nghĩa bằng markdown + frontmatter

`~/.config/opencode/agents/<tên>.md`, tên file thành tên agent. Frontmatter: `description` (bắt
buộc), `model`, `prompt`, `permission`, `mode`, `temperature`, `top_p`, và **`steps`** — trần số
vòng lặp. Có primary agent (Tab để đổi) và subagent (gọi bằng `@`, `hidden: true` để ẩn).

> Đây là cách rẻ nhất để làm mục 7 của `DESIGN.md` ("skills: playbook hERG / DILI / Ames dạng
> `.md`"). Không cần viết code: một thư mục markdown. Và `steps` chính là thứ thay cho
> `max_tool_calls: int = 3` đang hard-code trong `main.py` — trần vòng lặp phải là **cấu hình khai
> báo được**, không phải hằng số trong hàm.

---

## 12. Ba thứ đừng chép

**12.1 `--auto`.** Cờ duyệt tự động mọi thứ chưa bị deny. Với coding agent trên máy cá nhân thì hợp
lý. ToxAgent không có hành động nguy hiểm nào cần duyệt — nên đừng dựng cả bộ máy approval rồi tắt
nó đi. Lấy phần *pattern matching* để tắt bề mặt tool, bỏ phần *approval*.

**12.2 Spill store bị dọn trong khi con trỏ vẫn còn.** Output lớn được ghi ra
`~/.local/share/opencode/tool-output/tool_<callID>`, và trong hội thoại chỉ còn
`"...output truncated... Full output saved to <path>"`. Đo trên máy này: **622/8.458 tool call bị
cắt (7,4%)**, **189 part đang trỏ vào `tool-output/`**, nhưng thư mục chỉ còn **29 file** —
và 8/8 đường dẫn lấy mẫu đều **không còn tồn tại**.

> Với coding agent thì đó chỉ là phiền. **Với ToxAgent thì đó là mất bằng chứng.** Nếu observation
> store giữ `raw` và blob, nó phải có **cùng vòng đời với session**, không được là cache. Đây là
> rủi ro 7.7 ở mục 14.

**12.3 Policies.** `provider.use` allow/deny theo provider, global đè project. Chưa giải quyết vấn
đề nào của ToxAgent (một Gemini + chuỗi fallback local). Ghi lại để sau khỏi bàn lại.

---

## 13. Chốt: ba lựa chọn cạnh nhau

| | DSH | OpenCode | **ToxAgent nên** |
| --- | --- | --- | --- |
| Mô hình lưu trữ | Event-sourced, log là sự thật | State-sourced + change feed | **OpenCode** |
| Đơn vị lịch sử | 12 loại `SessionEvent` phẳng | `message` + `part` có `time_updated` | **OpenCode** |
| Ranh giới step | Event riêng | Part `step-start` / `step-finish` | **OpenCode** |
| Bề mặt mở rộng | ~20 event, 3 chế độ dispatch, waterfall có thẩm quyền | 15 hook, một chữ ký, mutate `output` | **OpenCode** |
| Compaction | Surface replacement + shadowedSeqs + generation | Con trỏ `tail_start_id` + `baseline` | **OpenCode** |
| Permission | `ask \| never`, leo thang một lần | Bảng bền vững + pattern + "luật cuối thắng" | **OpenCode** (chỉ phần pattern) |
| Kế toán token | Projection `tokenUsage`, seam riêng | Cột hạng nhất trên step **và** session | **OpenCode** |
| Kết quả tool | `content` / `value`, `meta` riêng | `title` / `output` / `metadata` / `attachments` | **OpenCode** |
| Transport | Cordis event + Web UI plugin | HTTP headless + OpenAPI + SSE | **OpenCode** |
| **Bất biến "logged"** | *"model-visible means logged"*, `request/header` khiến mọi request là hàm thuần của log | không phát biểu | **DSH** |
| **Kỷ luật seam** | Definition / Provider / Consumer, tách seam đo khỏi seam chính sách | không phát biểu | **DSH** |
| **Bất biến pipeline tool** | args freeze một lần; post-execute thay content **hoặc** value; `tools/result` là vùng no-transform | không phát biểu | **DSH** |
| Quy mô phải nuôi | 247 package | một server + một schema | — |

Gói lại trong một câu:

> **Lấy từ vựng và bất biến của DSH; lấy cấu trúc dữ liệu và hook surface của OpenCode.**
> DSH dạy *phải bảo đảm cái gì*. OpenCode dạy *cách rẻ nhất để bảo đảm nó*.

---

## 14. Những mục ở Phần 1 thay đổi thế nào

| Mục | Thay đổi |
| --- | --- |
| **2.1** Session event | Taxonomy giữ nguyên, nhưng **lưu thành bảng `message` + `part`**, không phải JSONL append-only. `turn/step` thành part `step-start`/`step-finish`. `tool/call` + `tool/result` gộp thành **một** part `tool` có `state.status`. Tên loại mang hậu tố phiên bản (`.1`) |
| **2.2** ToolDefinition | Thêm `title` và `attachments` vào kiểu trả về. Hook lấy chữ ký `(input, output) => void` của OpenCode, không dựng waterfall |
| **2.3** Observation | **Được xác nhận** bởi hình dạng tool part. Đổi tên cho khớp: `projection` → `output`, `raw`/`numeric_index` → `metadata`, `obs_id` → `callID`. Blob thành `attachments:[{mime,url}]` |
| **4.4** Đo cache | Có dụng cụ cụ thể: ghi `tokens{input,output,reasoning,cache{read,write}}` + `cost` trên mỗi step, **và** denormalize lên row session |
| **5** Audit trail | Compaction bằng **con trỏ** (`tail_start_id` + `baseline` + `baseline_seq`), không phải surface replacement. Lịch sử gốc bất biến theo cấu trúc |
| **6** Lộ trình | GĐ 1.5 nay có sơ đồ bảng cụ thể để chép: `session`, `message`, `part`, `session_context_epoch`, cộng `permission` nếu cần tắt bề mặt tool |
| **7** Rủi ro | Thêm **7.7 — spill store dangling**: nơi giữ `raw`/blob phải cùng vòng đời với session, không được là cache bị GC. Đo được trên máy này: 8/8 con trỏ mẫu đã chết |
| **11.2** (mục mới) | Trần vòng lặp (`max_tool_calls`) phải là cấu hình khai báo được (`steps` trong frontmatter), không phải hằng số trong hàm |


---

## Phụ lục A — kiểm chứng các khẳng định mới của Phần 1

`DESIGN.md` đã có phần kiểm chứng riêng; dưới đây chỉ là các khẳng định **mới** của Phần 1.

```bash
cd tox-agent   # nhánh agent_test, commit 12c3aa5

# 7.1 — tool gọi ngược HTTP vào chính process
grep -n 'MODEL_SERVER_URL\|MODEL_SERVER_TIMEOUT' tools/tox_tools.py   # dòng 34, 37: 127.0.0.1:$PORT, 240s
grep -n '_router.post("/analyze"' tools/tox_tools.py                  # dòng 255
grep -n 'async def analyze\b' -A 10 model_server/main.py              # dòng 4998; 5006 to_thread

# 7.6 — timeout lồng nhau
grep -n 'def _pubmed_get_with_retry\|def _pubchem_get_with_retry' tools/research_tools.py  # 46, 69

# mục 5 — session store in-memory
grep -n '_SESSION_STORE' agents/report_chat_agent.py                  # dòng 71: dict module-level

# mục 4.2 — đếm token bằng len//4
grep -n 'def estimate_context_tokens' -A 3 agents/report_chat_agent.py  # dòng 917

# mục 0 — vòng lặp chat: 3 tool call, prompt phẳng, plan tách rời
grep -n 'max_tool_calls' model_server/main.py                         # mặc định 3
grep -n 'def _build_report_chat_prompt' -A 10 model_server/main.py    # dòng 812

# 7.4 — coupling của benchmark
grep -rn 'run_orchestrator_flow' --include=*.py scripts/ tests/

# quy mô, để đối chiếu với lập luận ở mục 0.1
wc -l model_server/*.py agents/*.py tools/*.py | tail -1              # 12.265 dòng
```

---


---

## Phụ lục B — kiểm chứng các khẳng định về OpenCode

Mọi con số ở Phần 2 đọc từ bản cài trên máy này (`opencode 1.17.11`), không phải từ docs.

```bash
DB=~/.local/share/opencode/opencode.db

# 9 — quy mô kho dữ liệu
sqlite3 "$DB" "select 'sessions',count(*) from session
  union all select 'messages',count(*) from message
  union all select 'parts',count(*) from part
  union all select 'events',count(*) from event;"          # 323 / 6907 / 31588 / 41341

# 10.1 — event chỉ là change feed, và có hậu tố phiên bản
sqlite3 "$DB" "select type, count(*) from event group by 1 order by 2 desc;"

# 10.2 — phân bố part, vai trò message, hình dạng một tool part
sqlite3 "$DB" "select json_extract(data,'\$.type'), count(*) from part group by 1 order by 2 desc;"
sqlite3 "$DB" "select json_extract(data,'\$.role'), count(*) from message group by 1;"
sqlite3 "$DB" "select data from part where json_extract(data,'\$.type')='tool' limit 1;"
sqlite3 "$DB" "select json_extract(data,'\$.state.status'), count(*) from part
  where json_extract(data,'\$.type')='tool' group by 1;"   # completed 8307 / error 151

# 10.3, 10.4 — chữ ký ToolResult, ToolContext và toàn bộ 15 hook
cat ~/.config/opencode/node_modules/@opencode-ai/plugin/dist/tool.d.ts
sed -n '173,320p' ~/.config/opencode/node_modules/@opencode-ai/plugin/dist/index.d.ts

# 10.5 — compaction bằng con trỏ
sqlite3 "$DB" "select data from part where json_extract(data,'\$.type')='compaction' limit 1;"
sqlite3 "$DB" ".schema session_context_epoch"

# 10.6 — kế toán token hai mức
sqlite3 "$DB" "select data from part where json_extract(data,'\$.type')='step-finish' limit 1;"
sqlite3 "$DB" ".schema session" | grep -o 'tokens_[a-z_]*\|cost'

# 10.7 — permission bền vững + override theo session
sqlite3 "$DB" ".schema permission"
sqlite3 "$DB" "select permission, count(*) from session group by 1 order by 2 desc limit 5;"

# 12.2 — spill store dangling (con số sẽ khác theo máy, kiểu hỏng thì không)
sqlite3 "$DB" "select json_extract(data,'\$.state.metadata.truncated'), count(*) from part
  where json_extract(data,'\$.type')='tool' group by 1;"   # 622 bị cắt / 7685 không
sqlite3 "$DB" "select count(*) from part where data like '%tool-output%';"  # 189 con trỏ
ls ~/.local/share/opencode/tool-output | wc -l                              # 29 file còn lại
sqlite3 "$DB" "select data from part where data like '%tool-output%' limit 40;" \
  | grep -o '/[^ \"]*tool-output/tool_[A-Za-z0-9]*' | sort -u | head -8 \
  | while read -r p; do [ -f "$p" ] && echo "OK   $p" || echo "GONE $p"; done   # 8/8 GONE
```

Metadata repo (03/09/2026): `https://api.github.com/repos/anomalyco/opencode` — 203.216 sao,
26.477 fork, MIT, TypeScript, tạo 30/04/2025, 5.591 issue mở.
Docs đã đọc: `opencode.ai/docs/{,agents,permissions,policies,tools,server}`.

---

*Phần 1 viết ngày 02/09/2026; Phần 2 bổ sung ngày 03/09/2026. Dựa trên commit `12c3aa5` của nhánh
`agent_test`, bộ ghi chú DSH `@deepseek-ai/dsh@0.1.1-rc.2`, và bản cài `opencode 1.17.11`.*
