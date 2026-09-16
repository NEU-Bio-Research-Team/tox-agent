# Review kiến trúc agent và kế hoạch đưa ToxAgent vào benchmark

> Ngày review: 2026-09-16  
> Revision được đọc: `f723487`  
> Vai trò của tài liệu: companion implementation review cho
> [`TOXAGENT_AGENT_BENCHMARK_REVIEW_AND_PROPOSAL_VI.md`](./TOXAGENT_AGENT_BENCHMARK_REVIEW_AND_PROPOSAL_VI.md),
> không thay thế thiết kế TAB-Suite v3 trong tài liệu đó.

## 1. Kết luận điều hành

ToxAgent không cần đổi framework agent hoặc chuyển sang multi-agent để chạy được benchmark. Nền móng đúng đã có: backend giữ quyền điều phối, model bị giới hạn bởi capability và tool profile, dữ liệu khoa học có canonical source, output đi qua typed submit tool, scheduler có lease/fencing, và report v2 đã chuyển phần lớn workflow về server.

Khoảng trống lớn nhất hiện nay không phải là “agent chưa đủ thông minh”, mà là **implementation truth, evaluation truth và release truth chưa khép thành một vòng kín**:

1. Eval runner chỉ khám phá `evals/tasks`, còn regression task mới của Adaptive Decision Support (ADS) nằm trong `evals/regression/tasks`. Task này hiện không tham gia suite hash, run hoặc release gate.
2. Task schema v1 và sáu category cũ chưa biểu diễn các năng lực mới như planning, coverage/stopping, development posture, report orchestration, OCR và provider portability.
3. Nhiều thay đổi kiến trúc quan trọng vẫn nằm sau rollout flag mặc định tắt. Một manifest benchmark hiện chưa chụp effective flags, runtime profile, tool registry, budget và topology, nên hai run có cùng suite hash vẫn có thể đang chấm hai sản phẩm khác nhau.
4. Deterministic graders đã khá tốt cho contract, số liệu và ownership, nhưng semantic rubric và SME grader vẫn chỉ là `deferred`. Vì vậy hệ thống có thể chứng minh “đúng schema” tốt hơn “đúng lập luận khoa học”.
5. Decision-support đang chạy qua `AgentRuntimeGateway`, còn `ScientificAgentKernel`, plan/case schema và một số budget code tồn tại nhưng không phải live path. Nếu không chốt ranh giới, repo sẽ tiếp tục có hai kiến trúc cùng trông như canonical.
6. State dài hạn của agent còn yếu: context lấy một cửa sổ message gần đây, còn checkpoint làm việc chưa chứa summary/plan thực. Đây là điểm yếu trực tiếp cho multi-turn, resume và long-session benchmark.

Khuyến nghị tổng thể là **giữ kiến trúc control-plane hiện tại**, bổ sung một evaluation spine có thể tái lập và một `DecisionSupportState` nhỏ do server sở hữu. Không kích hoạt `ScientificAgentKernel` chỉ vì nó đã tồn tại; không thêm raw web/shell; không chia thành nhiều agent trước khi paired eval chứng minh lợi ích.

## 2. Scope và phương pháp review

Review này đọc cả code, test, eval, deployment, tài liệu kiến trúc và lịch sử remediation. Trọng tâm là đường chạy thật từ request đến artifact, không suy luận kiến trúc từ tên thư mục.

Các vùng đã kiểm tra:

- Product topology: `frontend`, `backend/control`, `backend/predictor`, `backend/ocr`, PostgreSQL và các compose overlay.
- Control flow: submit message, router, scheduler, runtime gateway, OpenCode adapter, MCP tool registry, typed answer/report submission.
- Scientific flow: prediction, explanation/attribution, evidence search/read, evidence relation, report fact bundle và development posture.
- State and safety: session ownership, idempotency, capability token, profile allowlist, retry/correction/fallback, cancellation và fencing.
- Evaluation: task/schema/fixtures/graders/runner/manifests, CI workflows và runtime-provider matrix.
- Previous audit/remediation: audit ngày 2026-09-13, implementation plan, progress report và ADR 0009/0010.

Các lệnh kiểm tra mục tiêu trong môi trường review:

- Documentation check: đạt (`documentation OK`).
- 187 unit/contract tests thuần cho eval, router, context, tool registry, report semantic consistency và report synthesis: đạt.
- Eval inventory: 50 canonical tasks; runner liệt kê 6 deterministic/scripted tasks và 44 task cần agent runtime.
- Regression inventory: có 1 file ADS nhưng runner hiện không khám phá file đó.
- Full DB-backed suite không hoàn tất trong sandbox này: async SQLite dừng ở `Database.create_schema()`. Đây là giới hạn môi trường review, không được diễn giải thành lỗi chức năng của product.
- Không chạy live OpenCode/model, external worker drill, PostgreSQL integration, load test hoặc SME review trong lần review này.

## 3. Workspace truth: code nào là sản phẩm hiện tại?

### 3.1 Canonical source tree

Theo [`WORKSPACE_LAYOUT.md`](./WORKSPACE_LAYOUT.md), source được track và deploy nằm ở:

| Vùng | Trách nhiệm |
|---|---|
| `frontend/` | Chat/report UI và trạng thái tiến độ |
| `backend/control/` | API, persistence, orchestration, agent runtime gateway, MCP tools, eval |
| `backend/predictor/` | ToxPred inference và endpoint semantics |
| `backend/ocr/` | MolScribe structure recognition |
| `devops/` | Compose overlays, deployment và operational scripts |
| `docs/` | Product/architecture/audit/benchmark documentation |

Workspace cục bộ còn có nhiều thư mục gốc mang tên source cũ hoặc generic (ví dụ các vùng agent, service, tool, script và model-server đời trước). Chúng không có tracked file tại revision được review và phần lớn bị `.gitignore` bỏ qua. Đây là residual/local state, không phải source of truth của sản phẩm.

Rủi ro thực tế: một người hoặc agent mới có thể đọc nhầm code ignored này, sửa nhầm path hoặc đưa kết luận audit từ code không ship. Cần biến “tracked deployable tree” thành rule máy kiểm tra được, không chỉ là kiến thức trong tài liệu.

### 3.2 Documentation truth còn phân mảnh

Repo có cả `docs/ARCHITECTURE.md` và `docs/architecture.md` với scope khác nhau; trên filesystem không phân biệt hoa/thường đây là một rủi ro packaging. [`CONFIGURATION.md`](./CONFIGURATION.md) còn mô tả trace/metrics theo trạng thái cũ và có path compose đã đổi. [`REMEDIATION_PROGRESS_2026-09-13.md`](./audit/REMEDIATION_PROGRESS_2026-09-13.md) chứa cả trạng thái đã merged lẫn đoạn mô tả cũ mâu thuẫn.

Đề xuất: tạo một generated `architecture-inventory.json` từ code/config và dùng documentation check đối chiếu intents, flags, profiles, tools, providers, queues và eval packs. Tài liệu vẫn giải thích “tại sao”; inventory máy sinh trả lời “hiện đang là gì”.

## 4. Agent flow hiện tại

### 4.1 Luồng tổng quát

```text
HTTP message
  -> SubmitMessage
     -> validate envelope / ownership / idempotency / concurrency
     -> route intent + lane
     -> persist message, run, config and runtime envelope
     -> enqueue durable run_job
        -> in-process scheduler OR external worker pool
           -> deterministic handler
              -> predictor / OCR / read-only operation
           OR
           -> agentic handler
              -> AgentRuntimeGateway
              -> OpenCode adapter + run-scoped MCP capability
              -> closed ToxAgent tool profile
              -> typed submit tool
              -> server validation / ID resolution / persistence / rendering
```

[`SubmitMessage`](../backend/control/src/toxagent/application/submit_message.py) là transaction boundary hợp lý. [`RunScheduler`](../backend/control/src/toxagent/application/run_scheduler.py) có durable job, lease, fencing, claim/adopt/cancel/handoff và queue class. External worker topology đã có trong [`external-workers.yaml`](../devops/compose/external-workers.yaml), nhưng flag mặc định vẫn chạy worker trong API process.

### 4.2 Deterministic lanes

- Structure/SMILES analysis gọi predictor service và lưu canonical analysis artifacts.
- OCR gọi MolScribe, sau đó đi qua analysis thay vì cho model tự diễn giải ảnh thành kết quả cuối.
- Các operation cần dữ liệu chính xác có thể dùng deterministic/read-only handler.

Đây là design nên giữ. Agent benchmark không nên bắt model làm lại toán, suy ra ID hoặc copy số liệu khi server đã biết đáp án.

### 4.3 Decision-support lane

Router mới hội tụ report QA, evidence và attribution về intent `decision_support`. Agent nhận context có giới hạn, inventory, analysis/report references và closed tool set. Tool profile cung cấp các thao tác đọc inventory, analysis bundle/slice, explanation, evidence search/read và typed `submit_answer`.

Điểm đúng:

- Model không có shell, filesystem hoặc raw network tool.
- Capability được bind theo session/run/profile/tool allowlist và expiry.
- `ToolOutput` tách canonical data, model view và UI view.
- Model không cấp database ID; server resolve refs và cấp identity.
- Citation phải đi qua search/read lifecycle trước khi dùng.
- Answer được kiểm tra numeric/classification/citation/limitation/safety trước khi accept.

Điểm còn thiếu:

- Planning chỉ tồn tại ngầm trong model turn/tool sequence; không có persisted plan/sufficiency/stop state để resume hoặc grade.
- Context dùng một cửa sổ message gần đây; `SessionCheckpoint` hiện chưa mang working summary thực.
- Accepted evidence được đưa vào context theo session với giới hạn số lượng, chưa có selection policy đủ rõ theo compound/question/proposition.
- Compound identity ở ADS có thể bắt đầu từ canonical SMILES và model-supplied names; retrieval quality phụ thuộc mạnh vào tên truy vấn này.
- `agent_synthesis` được source validator coi là source tồn tại dù không có artifact độc lập để resolve. Benchmark cần ngăn nó trở thành “lối thoát” cho unsupported claim.

### 4.4 Report lane

Hiện tồn tại hai đường:

1. Legacy/generic runtime path: model có nhiều quyền điều khiển report workflow hơn.
2. `report_orchestrator_v2`: server chạy stage machine `prepare -> compound -> predictions -> explanations -> evidence -> synthesis -> validation -> render`; model chỉ làm một synthesis boundary và submit typed synthesis.

Đường thứ hai phù hợp với ADR 0009 hơn và dễ benchmark hơn, nhưng rollout flag mặc định tắt. Vì vậy “code mới đã merged” chưa đồng nghĩa “sản phẩm mặc định đang chạy kiến trúc mới”. Mọi benchmark report phải ghi effective flag và chạy paired old/new cho đến khi cutover.

### 4.5 Live path và dormant path

ADR 0010 xác nhận ADS chạy trên `AgentRuntimeGateway`/OpenCode. [`ScientificAgentKernel`](../backend/control/src/toxagent/agent/kernel.py), case/investigation plan persistence và một phần capability/budget abstraction khác không phải live execution path.

Đây không nhất thiết là lỗi, nhưng là architecture debt. Cần chọn một trong hai:

- Harvest các data type hữu ích vào live control-plane rồi archive/xóa kernel; hoặc
- Chứng minh bằng paired benchmark rằng kernel thay thế gateway path tốt hơn, sau đó migration có ADR và cutover rõ.

Không nên để hai implementation cùng mang tên “agent architecture” vô thời hạn.

## 5. Đối chiếu với practice uy tín trong industry

### 5.1 Workflow trước, autonomy sau

Anthropic khuyến nghị bắt đầu bằng thiết kế đơn giản nhất, phân biệt workflow được định nghĩa trước với agent tự quyết động, và chỉ tăng complexity khi eval chứng minh giá trị. Họ cũng nhấn mạnh orchestration rõ, feedback từ environment và stopping condition ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

ToxAgent đang đi đúng hướng ở deterministic lanes và report orchestrator. Khoảng thiếu là decision-support chưa materialize plan/coverage/stop state, nên khó quan sát agent dừng vì “đã đủ bằng chứng” hay chỉ vì hết turn/budget.

### 5.2 Eval-driven development và trace-level grading

OpenAI khuyến nghị eval theo task thực tế, log đầy đủ, continuous evaluation và hiệu chỉnh automated grader bằng con người; đối với single-agent cần chấm cả tool selection lẫn arguments, không chỉ final answer ([Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)). Anthropic mô tả một agent eval bằng task, nhiều trial, grader và trace; các grader nên kết hợp deterministic, model và human thay vì ép một loại grader làm mọi việc ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

TAB-Suite v3 đã chọn đúng triết lý, nhưng implementation hiện mới mạnh ở deterministic final-state graders. Trace projection, semantic grader thật và SME calibration vẫn cần được nối vào runner.

### 5.3 Structured data flow và untrusted content

OpenAI khuyến nghị structured outputs giữa các node/tool để hạn chế dữ liệu không tin cậy điều khiển hành vi, cộng với tool approvals/guardrails và trace eval ([Safety in building agents](https://developers.openai.com/api/docs/guides/agent-builder-safety)). MCP Authorization yêu cầu resource server validate token audience và cấm token passthrough để tránh confused-deputy class ([MCP authorization specification](https://modelcontextprotocol.io/specification/draft/basic/authorization)).

ToxAgent có closed tool surface, typed submission và run-scoped capability — tốt hơn nhiều agent prototype. Phần cần làm thêm là gắn trust/provenance label vào abstract/snippet/provider payload, test indirect prompt injection ở tool output, và xuất security decision vào trace để grader kiểm tra được.

### 5.4 Durable state và idempotency

LangGraph dùng shared state, discrete nodes và checkpoint để pause/resume; side effect trước interrupt/retry phải idempotent ([Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph), [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)). Đây là pattern tham khảo, không phải lý do để đổi framework.

ToxAgent đã có persistence, run jobs, checkpoint cho report, idempotency và fencing. Phần cần bổ sung là working state nhỏ cho ADS để các thuộc tính plan, coverage, contradiction và stop reason trở thành dữ liệu bền vững.

### 5.5 Security regression là release gate

OWASP khuyến nghị abuse-case matrix, repeatable security regression và release gate cho tool misuse, prompt injection, memory poisoning, exfiltration và excessive agency ([AI Agent Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html)).

ToxAgent có một số task adversarial tốt nhưng task bank chưa phủ memory poisoning, provenance spoof, cross-run capability reuse, expired token, audience mismatch, evidence-to-tool instruction và partial-output leakage một cách hệ thống.

### 5.6 Hạ tầng là biến số của benchmark

Anthropic cho thấy giới hạn tài nguyên hạ tầng có thể làm điểm benchmark agent chênh đáng kể, nên environment, timeout và resource budget phải được kiểm soát và ghi lại ([Infrastructure noise in agent evaluations](https://www.anthropic.com/engineering/infrastructure-noise)).

Manifest ToxAgent đã ghi commit/runtime/trials, nhưng chưa đủ để phân biệt in-process với external worker, flag set, provider auth mode, retry policy, queue pressure, CPU/memory và evidence snapshot.

## 6. Finding và đề xuất theo ưu tiên

### P0-01 — Nối regression tree vào eval runner

**Bằng chứng:** [`runner.py`](../backend/control/evals/runner.py) khám phá `evals/tasks` và suite hash cũng dựa trên canonical task tree đó. [`ads-00-benzene-drug-decision-baseline-vi.json`](../backend/control/evals/regression/tasks/ads-00-benzene-drug-decision-baseline-vi.json) nằm ngoài đường khám phá.

**Hệ quả:** ADS regression có thể tồn tại trong repo nhưng không được chạy, không làm đổi hash và không chặn release.

**Thay đổi đề xuất:**

- Thêm task-set registry: `core`, `regression`, `security`, `report`, `ocr`, `sealed`.
- `--packs` chọn pack; default PR suite phải gồm `core,regression`.
- Suite hash gồm schema, selected task files, fixtures, grader code/config và pack manifest.
- Runner fail nếu có JSON task dưới một declared pack nhưng không load/validate được.
- Manifest ghi `selected_packs`, `discovered`, `executed`, `skipped`, `invalid`, với invariant tổng số phải khớp.

**Gate:** test ADS-00 phải xuất hiện trong `--list`, làm suite hash thay đổi và có result row ở mọi run chọn `regression`.

### P0-02 — Nâng task/result schema thành executable contract v3

Task schema hiện chưa biểu diễn đầy đủ các intent/capability mới. Không nên thay toàn bộ task cũ ngay; nên có loader adapter v1 -> v3 để giữ history.

Tối thiểu bổ sung:

- `capability_pack`, `intent`, `lane`, `risk_tier`, `language`, `runtime_requirement`.
- `initial_state`, `turns`, `fault_injection`, `resource_budget`, `expected_artifacts`.
- `required_graders`, `hard_gates`, `semantic_rubric`, `trial_policy`.
- `feature_requirements`: flags/profile/provider/topology.
- `data_provenance`, `fixture_snapshot`, `sealed_id` thay vì sealed content trong repo.

Result schema cần phân biệt `pass`, `fail`, `invalid`, `skipped`, `infra_error`; không được gộp infra failure thành product failure hoặc silently skip.

### P0-03 — Manifest phải chụp effective product

Mỗi eval run cần ghi:

- Commit/image digest và dirty-worktree indicator.
- Effective rollout flags và expiry status.
- Intent -> lane -> runtime profile -> tool allowlist mapping.
- Runtime/provider/model/version/auth mode; không ghi secret.
- In-process/external worker topology, queue class và concurrency caps.
- Prompt/tool/schema hashes.
- Budget snapshot: time, steps, tool calls, searches, reads, retries, correction turns.
- Predictor/OCR/provider versions và evidence fixture/live snapshot.
- Host resource envelope và timeout policy.

Không có các trường này thì kết quả vẫn hữu ích để debug, nhưng chưa đủ làm release evidence.

### P0-04 — Biến semantic grader và SME protocol thành code chạy được

[`rubric.py`](../backend/control/evals/graders/rubric.py) mới khai báo dimension; model/SME grading vẫn deferred. Tách trách nhiệm:

- Deterministic: schema, exact numeric fidelity, IDs, source existence, ownership, tool allowlist, citation resolution.
- Semantic judge: claim support, conflict handling, uncertainty calibration, quality của synthesis và development posture.
- SME: scientific acceptability, mechanistic reasoning, misuse/risk review.

Judge phải trả structured output kèm evidence span/reference, versioned rubric và abstain. Trước khi thành gate, đo agreement với dual-SME adjudicated set và publish confusion matrix theo dimension/language/risk tier.

### P0-05 — Credentialed runtime CI không được “xanh vì không chạy”

Live OpenCode matrix hiện phụ thuộc secret/flag môi trường. Khi credential không có, workflow nên phát hành trạng thái `not_evaluated` và chặn release evidence, thay vì tạo cảm giác suite đã pass.

Đề xuất ba mức:

- PR: scripted + deterministic packs bắt buộc.
- Nightly: credentialed live pack bắt buộc; thiếu credential là infra failure có alert.
- Release candidate: repeated live trials, provider matrix bắt buộc theo support claim, sealed set và signed manifest.

### P0-06 — Chốt rollout truth trước khi so điểm

Các flag `answer_draft_v2`, `evidence_pipeline_v2`, `report_orchestrator_v2`, `router_v2`, `external_worker_mode` mặc định tắt; ngày `remove_by` là 2026-12-12. Benchmark cần paired run flag-off/flag-on cho từng cutover, sau đó xóa old path khi removal condition đạt.

Đặc biệt, report benchmark phải dùng `report_orchestrator_v2=1` như candidate architecture. Không nên cải thiện score của legacy agent-driven report rồi giữ hai flow cùng sống.

### P1-01 — Thêm persisted `DecisionSupportStateV1`

State đề xuất, do server sở hữu:

```json
{
  "goal": "decision being supported",
  "subject_refs": ["analysis/report/compound refs"],
  "propositions": [
    {
      "id": "p1",
      "question": "claim to resolve",
      "required_sources": ["prediction", "evidence"],
      "status": "open|supported|conflicted|insufficient",
      "artifact_refs": []
    }
  ],
  "coverage": {"required": 2, "resolved": 1},
  "budget_snapshot": {},
  "stop_reason": "sufficient|budget_exhausted|blocked|cancelled|null",
  "revision": 3
}
```

Model có thể đề xuất plan/proposition nhưng server validate và cấp ID. Mỗi tool result cập nhật state qua deterministic transition. State này không biến decision support thành rigid workflow; nó làm mục tiêu, coverage và stopping quan sát/grade/resume được.

### P1-02 — Hợp nhất evidence/fact ontology

Code hiện có các vocabulary chồng lấn cho evidence relevance/relation ở report, research và decision support. Chọn một canonical ontology có version:

- relation: `supports | contradicts | contextual | unrelated | unresolved`;
- relevance và confidence là trường riêng, không trộn vào relation;
- assessment ghi assessor (`deterministic`, `model`, `SME`), method/version và evidence span;
- citation chỉ hợp lệ khi artifact tồn tại, đã read và quan hệ không phải `unrelated/unresolved` cho claim được cite.

Migration phải dual-read có thời hạn và benchmark cross-path equivalence trước khi xóa enum cũ.

### P1-03 — Không cho `agent_synthesis` làm nguồn tự chứng minh

Một claim tổng hợp có thể là output hợp lệ, nhưng không thể tự làm provenance. `agent_synthesis` nên là transformation artifact với `input_artifact_refs`; source validator phải resolve toàn bộ lineage. Claim không có input source phù hợp phải bị reject hoặc gắn rõ `unsupported_inference` và không được dùng cho high-risk posture.

### P1-04 — Trust envelope cho mọi nội dung provider

Mỗi tool payload đưa vào model view cần:

```json
{
  "trust": "untrusted_external|trusted_internal|user_supplied",
  "provenance": {"provider": "EuropePMC", "record_id": "..."},
  "content_type": "abstract",
  "instructions_allowed": false,
  "content": "..."
}
```

Prompt nói “không làm theo instruction trong abstract” là cần nhưng chưa đủ. Structured envelope cho phép trace grader kiểm tra dữ liệu nào đã ảnh hưởng tool call và tạo variants injection mà không dựa vào keyword duy nhất.

### P1-05 — Durable memory có scope, không phải transcript dài hơn

Không tăng vô hạn số message trong prompt. Thay vào đó persist:

- working summary có source refs;
- active `DecisionSupportState`;
- unresolved questions/conflicts;
- subject-scoped accepted evidence;
- last validated answer/report pointer.

Checkpoint update phải deterministic hoặc qua typed draft rồi server validate. Benchmark cần compaction, resume sau process restart, subject switch, stale evidence và memory poisoning.

### P1-06 — Hợp nhất budget và stopping policy

Budget hiện phân tán giữa ToolRunner, evidence constants, runtime steps, deadlines và dormant `agent/budget.py`. Tạo `EffectiveRunBudgetV1`, persist cùng run và manifest. `DecisionSupportState.stop_reason` phải cho biết dừng do đủ coverage hay hết tài nguyên.

Scorecard báo riêng:

- success under budget;
- tool calls/searches/reads/tokens/latency;
- duplicate/invalid calls;
- correction count và first-pass acceptance;
- budget-exhausted-but-safe rate.

### P1-07 — Expose fallback, đừng để fallback che agent failure

Deterministic fallback là safety feature tốt. Tuy nhiên benchmark phải ghi riêng:

- `model_draft_valid_first_pass`;
- `accepted_after_correction`;
- `fallback_used` và reason;
- final answer safety/correctness.

Một run có final answer an toàn nhờ fallback không được tính là agent capability pass, nhưng có thể pass safety-containment gate.

### P1-08 — Trace projection thống nhất cho production và eval

Không cần lưu raw chain-of-thought. Cần canonical event stream có:

- run/turn/stage/tool IDs và timestamps;
- intent/lane/profile/capability decision;
- tool name, validated args hash, artifact refs, status/error/retry;
- plan transition, coverage và stop reason;
- validation findings, correction/fallback;
- queue/worker/lease/handoff và usage/latency.

Từ stream này sinh `EvalTraceV1` để trajectory graders chạy giống nhau trên scripted, live và production replay. Metrics hiện có vẫn dùng cho aggregate monitoring; trace dùng cho causal diagnosis.

### P2-01 — Nâng evidence retrieval theo scope sản phẩm

EuropePMC-only và lexical relevance phù hợp baseline nhưng chưa đủ cho “drug development decision support”. Trước khi thêm provider, phải chốt product claims và capability packs:

- Literature pack: EuropePMC/PubMed-like evidence.
- Regulatory pack: chỉ bật khi có authoritative regulatory source và freshness contract.
- Compound/bioactivity pack: chỉ bật khi resolver/provider có deterministic identity mapping.
- Live-web pack: không bật mặc định và không cấp raw browser cho model.

Mỗi provider adapter cần snapshot fixtures, availability/freshness semantics, retry/circuit breaker và provenance. Benchmark chấm từng pack, không dùng provider thiếu để hạ điểm core agent.

### P2-02 — Chốt external worker topology bằng drill

Code đã hỗ trợ worker pools và DB concurrency caps nhưng default vẫn in-process. Trước cutover cần test:

- API restart không hủy active run;
- lease expiry/adoption không tạo duplicate side effect;
- report burst không starvation interactive queue;
- cancellation xuyên process;
- provider rate-limit/backpressure;
- poisoned job/dead-letter policy;
- rolling deploy với mixed worker version.

Các test này thuộc Control-Plane Conformance suite, không phải semantic agent score.

### P2-03 — Workspace và docs hygiene thành CI gate

- Fail docs check nếu canonical path không tồn tại.
- Case-collision check cho filename.
- Generated inventory diff phải được review khi tool/profile/flag/intent/provider đổi.
- Developer bootstrap cảnh báo ignored source-like roots.
- Archive hoặc ghi nhãn rõ tài liệu superseded; progress docs không được dùng như current architecture spec.

## 7. Kiến trúc đích tối thiểu

```text
                         +-----------------------------+
User/API ----------------> SubmitMessage + typed router |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         | Durable RunJob + Budget V1  |
                         +------+----------------------+
                                |
             +------------------+------------------+
             |                                     |
     deterministic lane                      agentic lane
  predictor / OCR / reads             AgentRuntimeGateway
             |                         closed capability
             |                                |
             +----------> Artifact/Facts <----+
                              Graph V1         |
                                 |             v
                                 |   DecisionSupportState V1
                                 |   plan/coverage/conflict/stop
                                 |             |
                                 +------> typed draft
                                             |
                                  server validation/render
                                             |
                                   Canonical EvalTrace V1
                                             |
                         deterministic + semantic + SME graders
```

Các invariants:

1. Server sở hữu workflow state, identity, validation, persistence và rendering.
2. Model chỉ đọc qua scoped tools và ghi qua typed proposal/submit tools.
3. Dữ liệu ngoài hệ thống luôn có provenance/trust label.
4. Mọi claim số liệu/citation/posture truy được về artifact lineage.
5. Retry/resume không nhân đôi side effect.
6. Fallback không được che first-pass failure trong score.
7. Mỗi benchmark result xác định được đúng product configuration đã chạy.

## 8. Ánh xạ đề xuất vào TAB-Suite v3

| Đề xuất | Suite chính | Task/regression cần thêm | Grader/gate |
|---|---|---|---|
| Task-set registry | Tất cả | `eval-pack-discovery` | inventory conservation + suite hash |
| Effective manifest | Control | same commit, khác flag/topology | manifest completeness hard gate |
| ADS persisted state | Capability | plan, conflict, stopping, resume | state transition + coverage grader |
| Evidence ontology | Scientific | support/contradict/contextual/unrelated | lineage + semantic relation |
| `agent_synthesis` lineage | Scientific/Safety | fabricated synthesis source | artifact resolution hard gate |
| Trust envelope | Safety | injection trong abstract/title/metadata | no prohibited tool/action + trace grader |
| Durable memory | Capability/Safety | compaction, subject switch, poison | state isolation + recovery |
| Unified budget | Capability/Control | low budget, provider timeout | success-under-budget + safe stop |
| Fallback visibility | Reliability | invalid first draft, correction fail | separate capability/safety outcomes |
| Report v2 cutover | Scientific/Control | partial stage failure, resume | checkpoint + fact consistency |
| External workers | Control | restart, lease loss, burst, cancel | exactly-once effect + no starvation |
| Provider matrix | Capability | same task/auth/runtime variants | paired delta + critical gates |
| OCR pack | Scientific | image -> structure -> prediction | structure identity + downstream fidelity |
| SME calibration | Scientific communication | conflict/posture/mechanism cases | agreement/confusion/abstention |

### Regression families nên bổ sung ngay

1. `ads-plan-*`: agent chọn đủ nguồn, không gọi tool vô ích, dừng đúng lý do.
2. `ads-conflict-*`: predictor và literature mâu thuẫn; không ép thành kết luận chắc chắn.
3. `ads-posture-*`: development posture bám fact, không nâng thành clinical/safety decision.
4. `memory-*`: compaction/restart/subject switch; không kéo evidence sai compound.
5. `security-evidence-*`: indirect injection ở title, abstract, author field và provider error.
6. `capability-*`: token hết hạn, sai audience, cross-run reuse, profile/tool mismatch.
7. `report-v2-*`: resume mỗi stage, partial provider failure, contradiction và fallback.
8. `worker-*`: lease fencing, duplicate claim, cancellation, queue fairness.
9. `ocr-*`: invalid image, ambiguous structure, confidence/identity propagation.
10. `infra-*`: provider outage, model timeout, DB transient, resource pressure; kết quả phải là `invalid/infra_error` đúng loại.

## 9. Kế hoạch triển khai theo wave

### Wave 0 — Truth and discovery (P0, 3–5 ngày)

- Task-set registry và loader v1/v3.
- Include regression pack trong list/hash/run.
- Manifest effective config v2.
- Sửa doc/path/case drift và tạo architecture inventory.
- CI conservation test: discovered = executed + skipped + invalid.

**Exit:** ADS-00 thực sự chạy; mọi run tái dựng được configuration; không còn silent task omission.

### Wave 1 — Evaluation spine (P0, 1–2 tuần)

- Result status taxonomy và canonical `EvalTraceV1` projection.
- Grader registry với declared inputs/versions.
- Tách capability outcome khỏi safety fallback outcome.
- Nightly credentialed run phát `not_evaluated/infra_error` rõ.
- Baseline 3+ trials cho stochastic tasks.

**Exit:** TAB-Suite có thể chấm final state, trajectory, budget và infra validity trong cùng manifest.

### Wave 2 — Decision-support state and provenance (P1, 1–2 tuần)

- `DecisionSupportStateV1` + transitions.
- Unified evidence relation và artifact lineage.
- `agent_synthesis` không còn tự chứng minh source.
- Subject-scoped checkpoint/memory và budget snapshot.

**Exit:** plan/coverage/conflict/stop/resume đều quan sát và grade được; không unsupported high-risk claim.

### Wave 3 — Safety and evidence hardening (P1/P2, 1–2 tuần)

- Trust envelope cho provider content.
- Injection/memory/capability abuse packs.
- Provider snapshots và fault injection.
- Security gates theo risk tier.

**Exit:** toàn bộ critical security tasks pass worst-of-n; không raw external content nào có thể tự cấp instruction authority.

### Wave 4 — Cutover and cleanup (P1/P2, 1–2 tuần + canary)

- Paired benchmark report v1/v2, router v1/v2 và in-process/external workers.
- Cutover theo removal condition; xóa old path/flag.
- Chọn archive hoặc migrate phần hữu ích của dormant kernel.

**Exit:** một canonical runtime path cho mỗi intent; architecture inventory không còn split-brain.

### Wave 5 — Semantic and release governance (P0/P2, liên tục)

- Dual-SME gold set và adjudication.
- Judge calibration, abstention và drift monitoring.
- Sealed release service ngoài repo.
- Production failure -> sanitized regression loop.

**Exit:** release decision có signed manifest, critical gates, uncertainty, judge agreement và SME audit trail.

## 10. Các thay đổi file/module dự kiến

| Vùng | Thay đổi |
|---|---|
| `backend/control/evals/runner.py` | pack discovery, status taxonomy, manifest v2, trace input |
| `backend/control/evals/schema/` | task v3, result v3, manifest v2, adapters |
| `backend/control/evals/packs/` | declared pack manifests và ownership |
| `backend/control/evals/graders/` | trace, semantic judge, judge calibration, infra validity |
| `backend/control/src/toxagent/domain/` | decision state, canonical evidence relation, artifact lineage |
| `backend/control/src/toxagent/application/` | state transitions, budget snapshot, trace projection |
| `backend/control/src/toxagent/harness/` | trust envelope, effective profile/prompt/tool hashes |
| `backend/control/src/toxagent/tools/` | lineage enforcement và canonical relation contract |
| `backend/control/src/toxagent/metrics.py` | first-pass/correction/fallback/stop reason metrics |
| `devops/` | benchmark topology/resource declaration và credentialed gate |
| `docs/` | generated inventory, canonical architecture entry point, superseded labels |

Database changes nên additive trước: new state/artifact/trace tables hoặc JSON versioned columns; dual-write chỉ có thời hạn; migration rollback không được xóa evidence cũ.

## 11. Release gates đề xuất

### PR gate

- Schema/fixture/task validation 100%.
- Core + regression deterministic packs chạy.
- Critical contract/security task không fail.
- Task inventory conservation và suite hash reproducible.
- Architecture inventory không drift.

### Nightly gate

- Live runtime 3 trials cho stochastic core.
- No silent skip; credential/infrastructure problems là visible invalid run.
- First-pass, correction, fallback, tool efficiency và budget metrics.
- Security worst-of-n và long-session recovery.
- So sánh với moving baseline bằng paired task IDs, không chỉ aggregate rate.

### Release-candidate gate

- Tất cả supported runtime/provider/auth combinations trong matrix.
- Sealed tasks ngoài repo và contamination controls.
- Critical hard gates 100% ở mọi required trial.
- Semantic judge đã calibrated; critical scientific sample có SME sign-off.
- External worker/restart/fencing/cancellation drills nếu topology đó được support.
- Signed manifest chứa effective product/config/environment.

Không nên có một “ToxAgent score” duy nhất. Release scorecard phải tách control conformance, capability, scientific communication và safety/reliability như tài liệu TAB-Suite v3 đã đề xuất.

## 12. Những việc không nên làm

1. Không đổi sang LangGraph/OpenAI Agents SDK chỉ để có tên framework phổ biến. Patterns của chúng hữu ích; migration chỉ hợp lý nếu paired eval chứng minh correctness/operability tốt hơn tổng migration cost.
2. Không chuyển ngay sang multi-agent. Với tool surface nhỏ và server-owned workflow, thêm agent làm tăng coordination failure, latency và grading complexity.
3. Không bật dormant `ScientificAgentKernel` song song với live gateway mà không có ADR/cutover.
4. Không cấp shell/filesystem/raw web cho scientific agent để tăng “khả năng”. Thêm provider adapter có provenance và snapshot tốt hơn.
5. Không coi deterministic fallback là agent pass.
6. Không dùng regex/keyword grader để thay semantic judge cho conflict, mechanism hoặc development posture.
7. Không giữ rollout flag sau ngày hết hạn; compatibility fork sẽ trở thành kiến trúc thật.
8. Không quảng bá regulatory/clinical safety capability trước khi có authoritative provider, scope, freshness và SME gate tương ứng.

## 13. Definition of Done

Chương trình này được xem là hoàn thành khi:

- [ ] Tất cả declared task packs được discover/hash/run; không task mồ côi.
- [ ] ADS/report/OCR/security/worker capabilities có pack và owner rõ.
- [ ] Manifest đủ để tái dựng effective product configuration và environment.
- [ ] Mọi run phân biệt product failure, infra error, invalid và skip.
- [ ] Decision support có persisted goal/proposition/coverage/conflict/stop state.
- [ ] Mọi scientific claim/posture có artifact lineage; synthesis không tự làm source.
- [ ] External content có trust/provenance envelope và injection regression.
- [ ] First-pass, correction và fallback được báo riêng.
- [ ] Semantic judge được calibrate với adjudicated SME set; có abstention.
- [ ] Critical gates pass trên required trials và supported provider matrix.
- [ ] Report v2 và external worker topology đã cutover hoặc được ghi rõ là unsupported.
- [ ] Dormant/legacy agent architecture được archive/xóa hoặc chính thức migrate.
- [ ] Benchmark result có thể dùng trực tiếp làm release evidence cho TAB-Suite v3.

## 14. Quyết định nên chốt ngay

1. Chọn `backend/*` là canonical product tree và CI-enforce source-of-truth boundary.
2. Chọn TAB-Suite v3 làm evaluation contract; tài liệu này là implementation bridge.
3. Sửa task discovery/hash trước khi viết thêm nhiều task.
4. Chọn `AgentRuntimeGateway + server-owned state/workflow` là live architecture; không có hai control planes.
5. Chọn report orchestrator v2 là candidate path để benchmark/cutover.
6. Chọn artifact lineage và trust envelope là điều kiện bắt buộc cho evidence-derived claim.
7. Chọn “simple single-agent + typed tools” làm mặc định; complexity phải được eval chứng minh.
8. Chọn multi-dimensional release scorecard; không có aggregate score che critical failure.

## 15. Tài liệu tham khảo

### Nội bộ repo

- [Benchmark review và TAB-Suite v3](./TOXAGENT_AGENT_BENCHMARK_REVIEW_AND_PROPOSAL_VI.md)
- [Agentic flow audit 2026-09-13](./audit/AGENTIC_FLOW_AUDIT_2026-09-13_VI.md)
- [Remediation implementation plan](./audit/AGENTIC_FLOW_REMEDIATION_IMPLEMENTATION_PLAN_2026-09-13_VI.md)
- [Remediation progress](./audit/REMEDIATION_PROGRESS_2026-09-13.md)
- [ADR 0009 — agentic workflow boundary](../backend/control/docs/adr/0009-agentic-workflow-boundary.md)
- [ADR 0010 — adaptive decision-support boundary](../backend/control/docs/adr/0010-adaptive-decision-support-boundary.md)
- [Metrics contract](./observability/METRICS.md)
- [Runtime/provider matrix](../backend/control/evals/manifests/runtime-provider-matrix-v1.json)

### Industry/primary sources

- OpenAI, [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- OpenAI, [Safety in building agents](https://developers.openai.com/api/docs/guides/agent-builder-safety)
- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Anthropic, [Infrastructure noise in agent evaluations](https://www.anthropic.com/engineering/infrastructure-noise)
- LangGraph, [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- LangGraph, [Interrupts and durable execution](https://docs.langchain.com/oss/python/langgraph/interrupts)
- OWASP, [AI Agent Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html)
- Model Context Protocol, [Authorization specification](https://modelcontextprotocol.io/specification/draft/basic/authorization)
