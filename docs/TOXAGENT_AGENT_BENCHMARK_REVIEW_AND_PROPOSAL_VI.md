# Review benchmark agent và đề xuất TAB-Suite v3 cho ToxAgent

> **Ngày nghiên cứu:** 2026-09-16  
> **Phạm vi:** agent layer của `toxagent-control`, gồm routing, tool use, state, evidence, grounded answer, report orchestration, security và reliability.  
> **Tài liệu được review:** `agent_benchmark_v1.md`, `agent_benchmark_v2.md`, implementation trong `backend/control/evals/`, spec và audit agentic flow hiện hành.  
> **Kết luận ngắn:** Không nên triển khai nguyên xi v1 hoặc v2. Nên giữ các quyết định tốt của hai bản này, nhưng thay bằng một TAB-Suite v3 dựa trên outcome/state, có benchmark packs theo capability, task thật do SME xác nhận, claim-level grading, release set được bảo vệ bên ngoài repo và một protocol thống kê đúng nghĩa.

---

## 1. Câu trả lời trực tiếp

### Benchmark hiện tại có literature review từ benchmark SOTA/industry không?

**Chưa đạt.**

- v1 hầu như không có literature review. Nó là một bản thiết kế nội bộ tốt về mặt kiến trúc, nhưng không chỉ ra quyết định nào được vay mượn từ benchmark nào, không có bibliography, không so sánh coverage hoặc phương pháp chấm với benchmark bên ngoài.
- v2 có nhắc một số ý đã xuất hiện trong literature như bias của LLM judge, held-out set, multi-turn và dual-use, nhưng chỉ dẫn nguồn cụ thể cho case study Urbina. Không có search protocol, bảng đối chiếu benchmark, tiêu chí chọn/bỏ nguồn, hoặc mapping từ từng benchmark bên ngoài vào ToxAgent.
- Cả hai bản gọi bộ tự thiết kế 120/130 task là “benchmark suite” nhưng chưa chứng minh task authenticity, độ khó, độ đại diện, inter-rater reliability, khả năng phân biệt hai agent version, hoặc tương quan với lỗi production.

### Có benchmark SOTA/industry phù hợp không?

**Có, nhưng không có một benchmark công khai nào có thể dùng nguyên xi cho ToxAgent.** ToxAgent cần một benchmark nội bộ theo domain, lấy pattern đã được kiểm chứng từ nhiều họ benchmark:

1. **Tool calling và orchestration:** BFCL, API-Bank.
2. **Multi-turn + policy + final database state:** τ-bench/τ²/τ³-bench.
3. **Phân tích trajectory và partial progress:** AgentBoard.
4. **Prompt injection và tool security:** AgentDojo, ToolEmu.
5. **Misuse/dual-use:** AgentHarm, ChemSafetyBench, case study Urbina.
6. **Evidence và citation:** ALCE, RAGChecker, RAGTruth.
7. **Scientific-agent task authenticity:** ScienceAgentBench, LAB-Bench/LABBench2.
8. **Toxicology reasoning:** ToxReason và OECD (Q)SAR Assessment Framework.
9. **Expert-written rubric và worst-case reliability:** HealthBench.
10. **Industry implementation practice:** outcome + transcript + mixed graders, nhiều trial, production-failure replay theo hướng dẫn thực hành của Anthropic; final-response/single-step/trajectory split theo LangSmith.

### ToxAgent nên benchmark agent layer như thế nào?

Không dùng một điểm tổng duy nhất. Cần đánh giá theo bốn lớp độc lập:

1. **Control-plane conformance:** auth, state, provenance, budget, recovery, deterministic validators.
2. **Agent capability:** chọn tool, lập kế hoạch, hỏi làm rõ, evidence research, report synthesis.
3. **Scientific communication:** numeric fidelity, endpoint semantics, uncertainty, citation support, semantic consistency.
4. **Safety and reliability:** injection, cross-session isolation, harmful optimization, false refusal, fault recovery và repeatability.

Kết quả release phải là một **scorecard nhiều chiều với hard gates**, không phải trung bình để lỗi an toàn bị che bởi điểm viết hay.

---

## 2. Phương pháp nghiên cứu và giới hạn

Đây là một **targeted literature review**, không phải systematic review theo PRISMA. Quy trình:

1. Đọc toàn bộ hai plan và phần evaluation của spec gốc.
2. Kiểm tra implementation thật: task schema, task bank, fixtures, hard gates, runner, manifests và audit live.
3. Tìm nguồn sơ cấp: paper, proceedings, official benchmark site, official product/engineering documentation và OECD guidance.
4. Chỉ giữ benchmark có ít nhất một pattern hữu ích trực tiếp cho ToxAgent: tool use, stateful multi-turn, evidence/citation, scientific work, toxicology, safety hoặc eval methodology.
5. Phân biệt rõ benchmark cho base model, benchmark cho agent/harness, benchmark domain khoa học và standard/guidance.

Giới hạn:

- Không chạy lại live model vì review này không được yêu cầu tiêu thêm provider credit và baseline đã có trong repo.
- Không đánh giá lại chất lượng predictor; model benchmark và agent benchmark phải tách riêng.
- SOTA thay đổi nhanh. Danh mục nguồn và benchmark nên được review lại mỗi 6 tháng.

---

## 3. Hiện trạng có bằng chứng trong repo

### 3.1 Những gì đang chạy thật

Tại thời điểm review:

| Hạng mục | Hiện trạng kiểm tra từ repo |
|---|---:|
| Task JSON | **50** |
| Nhóm task | 6 nhóm: numeric 12, endpoint 8, report-QA 10, evidence 8, failure 6, adversarial 6 |
| Critical task | **17** |
| Ngôn ngữ | **44 EN, 6 VI** |
| Số user turn/task | 6 task có 1 turn, 43 task có 2 turn, 1 task có 3 turn; không có task dài hạn |
| Fixture được task tham chiếu | **9** |
| Hard gate có code | **10** |
| Model-rubric grader chạy thật | **Không**; kết quả được ghi `deferred` |
| SME grader chạy thật | **Không**; kết quả được ghi `deferred` |
| Sealed release set | **Không có** |
| OCR/bioactivity/report-build task trong bank 50 task | **Không có coverage đầy đủ** |

Nguồn repo: [eval README](../backend/control/evals/README.md), [task schema](../backend/control/evals/schema/task.schema.json), [hard gates](../backend/control/evals/graders/hard_gates.py), [grader registry](../backend/control/evals/graders/__init__.py), [task generator](../backend/control/evals/build_tasks.py).

### 3.2 Baseline live có thể dùng

Full sweep gần nhất được ghi rõ trong progress document thực thi **35/50 task live-compatible**:

- pass@1 theo bốn sweep: 65,71% → 77,14% → 85,71% → 82,86%;
- sweep cuối: 29/35 pass;
- critical: 10/11 pass;
- nhiều failure ban đầu là lỗi task/grader/prompt/schema, không chỉ lỗi agent;
- chưa đạt `pass^3=100%` cho critical set và chưa có full baseline đồng nhất cho toàn bộ capability mới.

Nguồn repo: [Agentic layer progress §14](spec/TOXAGENT_AGENTIC_LAYER_PROGRESS_VI.md) và các manifest dưới `backend/control/evals/manifests/`.

Audit ngày 2026-09-13 còn phát hiện các failure class chưa nằm trong task bank hiện tại:

- report được accept dù executive summary mâu thuẫn với explanation;
- evidence retrieval persist tài liệu không liên quan;
- profile/budget được ghi một kiểu nhưng runtime dùng kiểu khác;
- report latency/token/tool-error quá cao;
- first-pass validation thấp;
- telemetry usage bị lặp;
- eval environment chưa hermetic/portable.

Nguồn repo: [Agentic flow audit](audit/AGENTIC_FLOW_AUDIT_2026-09-13_VI.md).

### 3.3 Hệ quả

Benchmark mới không được bắt đầu từ con số 120 hay 130. Nó phải bắt đầu từ **failure taxonomy đã quan sát**, sau đó mới mở rộng bằng task SME và benchmark-inspired variants. Nếu không, suite sẽ có coverage rộng trên giấy nhưng bỏ sót đúng lỗi production mà ToxAgent đang có.

---

## 4. Review `agent_benchmark_v1.md`

### 4.1 Điểm tốt nên giữ

1. Tách **predictor benchmark** khỏi **agent benchmark**.
2. Giữ three-boundary topology và xem control plane là source of truth.
3. Dùng frozen, integration và live modes.
4. Hard gate đứng trước quality scoring.
5. Đo numeric fidelity, provenance, cross-session isolation, denied tool, latency/cost và first-candidate rate.
6. Không dùng `pass@k` làm headline cho user-facing path; quan tâm consistency qua nhiều trial.

### 4.2 Vấn đề nghiêm trọng

| Vấn đề | Bằng chứng | Tác động |
|---|---|---|
| Không có literature review | Không có bibliography hay benchmark mapping | Không biết thiết kế đã kế thừa gì, bỏ sót gì, hoặc đang lặp lại lỗi benchmark cũ nào |
| Mô tả proposed state như implemented state | Nói 120 task/12 gate/file tree hoàn chỉnh; repo có 50 task/10 gate | Roadmap và estimate sai baseline |
| JSON mẫu không hợp schema | Dùng `description`, `text`, thiếu `schema_version`, `title`; category/gate mới chưa có trong enum | Không thể nạp bằng runner hiện tại |
| Tier-2 rubric thiếu 60% trọng số | Hai weighted dimensions chỉ cộng 40% | Không thể tính score được mô tả |
| URL validity giao cho LLM judge | URL existence/resolution là phần có thể kiểm deterministic | Tăng chi phí và giảm độ tin cậy không cần thiết |
| `must_mention` dễ chấm sai | Audit live đã cho thấy model diễn đạt đúng bằng từ đồng nghĩa nhưng fail string match | Benchmark đo phrasing thay vì capability |
| Trộn component quality vào agent quality | OCR recognition accuracy và predictor/bioactivity quality nằm chung suite | Không xác định lỗi do agent, predictor hay OCR |
| Không có task lifecycle/SME sign-off | `build_tasks.py` được xem như nguồn tạo task chính | Dễ có answer key sai và grader bug |
| Ngưỡng release không có căn cứ | 80/85/95/98%, 3/5 trial được đưa ra không power analysis | Green/red có vẻ chính xác nhưng không có ý nghĩa thống kê rõ |
| Thiếu production feedback loop | Không có sampling từ trace thật, incident replay, drift | Suite nhanh lỗi thời |

### 4.3 Kết luận cho v1

V1 là **architecture sketch hữu ích**, không phải benchmark specification có thể triển khai trực tiếp. Có thể giữ nhiều tư tưởng kiến trúc, nhưng task bank, schema, grader và release statistics cần thiết kế lại.

---

## 5. Review `agent_benchmark_v2.md`

### 5.1 Cải thiện đúng so với v1

1. Thêm traceability invariant ↔ gate ↔ task.
2. Nhận diện judge bias và yêu cầu human calibration.
3. Thêm held-out set, grader/schema versioning, baseline snapshot và risk register.
4. Bổ sung multi-turn, misuse resistance, false refusal và regression gate.
5. Ghi nhận nhu cầu stratified fixtures và fingerprint rõ ràng cho activity cliff.
6. Sửa dependency order: cần seed labels của SME trước khi calibrate judge.

Các hướng này phù hợp với literature hiện đại về agent eval.

### 5.2 Những điểm vẫn sai hoặc chưa đủ

#### A. Không code-aware

V2 tự ghi rằng không truy cập được repo. Vì vậy nó tiếp tục giả định 120-task v1 đã tồn tại và xây 130 task trên baseline không có thật. File tree, schema v2, 16 gate, judge ensemble, sealed set và baselines trong tài liệu đều chưa tồn tại.

Mẫu task v2 vẫn không hợp schema hiện tại: dùng `$schema`, `text`, `assistant_turn_ref`, category mới và field mới mà runner không hiểu.

#### B. “Deterministic hard gate” bị dùng quá phạm vi

Các gate như `no_causal_attribution_overclaim` hoặc `no_malicious_optimization_assistance` không thể đáng tin nếu chỉ là regex. Chúng phụ thuộc phủ định, ngữ cảnh, mục đích, quan hệ giữa nhiều turn và target. Nên tách:

- structural checks deterministic;
- policy/outcome checks dựa trên state và tool side effects;
- semantic checks bằng expert rubric đã calibrate.

#### C. Diễn giải sai applicability domain hiện tại

V2 mô tả `applicability: ok` là “nằm trong phạm vi huấn luyện”. Nhưng invariant gốc nói `element_rules_v1` chỉ là kiểm tra nguyên tố cơ bản, **không phải learned OOD và không chứng minh in-distribution**. Benchmark không được ép agent nói một điều chính hệ thống không biết.

Gold behavior đúng là: mô tả chính xác rule đã chạy, không nâng nó thành bằng chứng về safety hoặc model reliability.

#### D. Judge ensemble chưa nhất quán

- Tài liệu yêu cầu “tối thiểu 2 judge” nhưng blocking verdict lại yêu cầu “2/3 judge”. Cần chọn tối thiểu 3 hoặc định nghĩa majority cho từng số judge.
- Dùng ensemble mọi task rất tốn chi phí. Cách tốt hơn là một judge đã calibrate cho bulk scoring, deterministic graders nơi có thể, và second/third judge cho disagreement/high-risk sample.
- `url_validity` không nên là LLM blocking dimension; resolver và evidence store phải chấm.
- Position swap chỉ có ý nghĩa với pairwise judging; nhiều rubric trong v2 là pointwise.

#### E. Protocol SME và agreement metric chưa chặt

- Hai reviewer phù hợp với Cohen's κ; Fleiss' κ phù hợp với nhiều rater cố định hơn. Với prevalence rất lệch như hard safety failure, κ có thể gây hiểu lầm.
- Nên báo cả raw agreement, sensitivity/specificity trên positive/negative controls, confusion matrix và Gwet's AC1 hoặc Krippendorff's α khi phù hợp.
- “Double blind” nên đổi thành **model-blinded independent review**. Benchmark operator vẫn biết task và output có thể tiết lộ style/model.

#### F. Lỗi thống kê trong release gate

Mệnh đề “pass 5 lần nên đủ chặt” chỉ tính `0.5^5 ≈ 3,1%`: xác suất đạt 5/5 nếu success rate thật là 50%. Nó **không** chứng minh success rate thật ≥90%. Với 5/5, khoảng tin cậy vẫn rất rộng.

Các vấn đề khác:

- Wilson CI trên 130 task chỉ có diễn giải sampling nếu task là mẫu đại diện; task bank curated không phải random sample từ production distribution.
- Sealed set 26 task cho 9 nhóm chỉ còn khoảng 2–4 task/nhóm, quá ít để kết luận held-out delta theo category.
- M9 chỉ có 10 task nhưng đặt false-refusal ≤2%; độ phân giải task-level đã là 10%, chưa kể chỉ một phần là benign controls.
- Nhiều rollouts của cùng một task không độc lập như 300 task khác nhau; CI phải cluster theo task/family.

#### G. “Sealed trong repo” không thật sự sealed

Nếu `tasks/sealed/` nằm trong cùng repo mà agent-dev truy cập được thì nó chỉ là test split, không phải secret release set. Sealed set cần ở access-controlled store hoặc eval service, với canary, audit log và rotation.

#### H. Misuse policy quá rộng

“Tăng ái lực với target sinh học nhạy cảm” có thể là câu hỏi nghiên cứu hợp lệ. Gate phải xét intent, target class, requested action, mức độ hành động hóa và policy version; đồng thời có matched benign pairs để đo over-refusal.

#### I. Chưa có mechanistic toxicology gold standard

V2 kiểm tra cách nói về hERG/Tox21/bioactivity nhưng chưa đánh giá chuỗi cơ chế, chất lượng evidence hoặc distinction giữa association và AOP-level causal support. ToxReason và OECD QAF chỉ ra đây là lớp riêng cần đánh giá nếu sản phẩm muốn sinh mechanistic interpretation.

### 5.3 Kết luận cho v2

V2 tốt hơn rõ rệt về governance và nhận diện failure mode, nhưng vẫn là **proposal chưa được literature-grounded và chưa executable**. Không nên gọi nó là “revised canonical architecture” trước khi schema, task, gold labels và grader meta-eval tồn tại.

### 5.4 Scorecard so sánh nhanh

| Tiêu chí | v1 | v2 | Nhận định |
|---|---|---|---|
| Kiến trúc boundary/model-vs-agent | Tốt | Tốt | Nên giữ |
| Fidelity với repo hiện tại | Yếu | Yếu | Cả hai mô tả nhiều thành phần chưa tồn tại |
| Literature grounding | Yếu | Một phần | V2 có đúng hướng nhưng thiếu review/mapping/citation đầy đủ |
| Task authenticity và SME validation | Yếu | Khá hơn | V2 có sign-off nhưng chưa có protocol/data thật |
| Deterministic grading | Khá | Khá | Cần thu hẹp gate về phần thật sự deterministic |
| Semantic grading | Chưa hoàn chỉnh | Thiết kế tốt hơn | Chưa có judge meta-eval hoặc implementation |
| Statistical validity | Yếu | Chưa đạt | V2 thêm CI nhưng sample design và diễn giải `pass^k` còn sai |
| Security/misuse | Một phần | Tốt về coverage | V2 cần policy contextual và benign controls |
| Executable với code hiện tại | Không | Không | Cần schema migration và backward-compatible loader |
| Khuyến nghị | Không triển khai nguyên xi | Dùng làm input cho v3 | TAB-Suite v3 nên là canonical proposal mới |

---

## 6. Literature map: benchmark nào hữu ích cho ToxAgent?

### 6.1 Agent/tool benchmarks

| Benchmark | Đo gì | Pattern nên lấy cho ToxAgent | Không nên copy nguyên xi |
|---|---|---|---|
| [AgentBench](https://arxiv.org/abs/2308.03688) | Agent trong 8 interactive environments | Đánh giá multi-turn, decision making, instruction following | Domain quá rộng; final metrics không đủ cho toxicology safety |
| [API-Bank](https://arxiv.org/abs/2304.08244) | Planning, API retrieval và calling trên runnable tools | Unit test tool selection/arguments và task decomposition | Dialogue/API không phản ánh product state của ToxAgent |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard) | Function calling, relevance/irrelevance, parallel/multiple calls, multi-turn, format sensitivity, cost/latency | Tool-selection pack; no-tool-needed; malformed args; multi-turn state transitions; prompt-format variants | BFCL chấm model/tool-call layer, không đủ để chứng nhận final scientific answer |
| [τ-bench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b126cc38b8638e07bef37e7b2bb72bf-Abstract-Conference.html) | Tool-agent-user interaction, domain policy và final DB state; đề xuất `pass^k` | User simulator có goal; chấm final state/policy; consistency qua trial | Retail/airline policy không phải scientific policy; task/answer-key cần audit kỹ |
| [AgentBoard](https://arxiv.org/abs/2401.13178) | Multi-turn agents và fine-grained progress rate | Subgoal progress cho report/evidence workflows; failure diagnosis | Progress không được bù cho hard safety violation |
| [GAIA](https://arxiv.org/abs/2311.12983) | Real-world tool use, reasoning, browsing, multimodal | Thiết kế question dễ kiểm chứng nhưng cần nhiều bước; hidden answers | Không đủ domain depth cho toxicology |

### 6.2 Security và misuse

| Benchmark/source | Bài học chính cho ToxAgent |
|---|---|
| [AgentDojo](https://arxiv.org/abs/2406.13352) | Đo đồng thời utility dưới benign condition và attack success dưới indirect prompt injection; dùng environment thật và state checks. Đây là khung phù hợp nhất cho evidence abstract chứa instruction độc hại. |
| [ToolEmu](https://arxiv.org/abs/2309.15817) | Scenario-based long-tail risk testing có ích khi tool thật đắt/nguy hiểm, nhưng LM-emulated tool/judge không được xem là ground truth tuyệt đối. |
| [AgentHarm](https://arxiv.org/abs/2410.09024) | Safety của agent phải đo harmful **multi-step completion**, không chỉ refusal text. ToxAgent cần chấm cả việc agent có gọi tool/tiết lộ actionable optimization hay không. |
| [ChemSafetyBench](https://arxiv.org/abs/2411.16736) | Chemistry safety cần matched scenarios và jailbreak variants. Dùng để thiết kế taxonomy, không nhập task synthesis nguy hiểm vào repo public. |
| [Urbina et al.](https://www.nature.com/articles/s42256-022-00465-9) | Chứng minh mục tiêu drug-discovery có thể bị đảo sang tối ưu độc tính; hỗ trợ việc có misuse pack. Nó không tự định nghĩa refusal policy cho mọi câu hỏi affinity. |

### 6.3 Evidence, RAG và citation

| Benchmark | Pattern nên lấy |
|---|---|
| [ALCE](https://aclanthology.org/2023.emnlp-main.398/) | Tách answer correctness, citation precision/correctness và citation completeness/recall. ToxAgent hiện mới kiểm citation ID tồn tại, chưa đủ kiểm nguồn có hỗ trợ claim. |
| [RAGChecker](https://arxiv.org/abs/2408.08067) | Tách retrieval metrics khỏi generation metrics để biết lỗi do search, promotion, reading hay synthesis. |
| [RAGTruth](https://arxiv.org/abs/2401.00396) | Claim/span-level annotation cho unsupported hoặc contradictory content; phù hợp với report semantic consistency. |
| [CRAG](https://arxiv.org/abs/2406.04744) | Test dynamic/long-tail facts và explicit abstention; live-evidence run không được trộn với deterministic CI score. |

### 6.4 Scientific và toxicology benchmarks

| Benchmark/standard | Relevance với ToxAgent |
|---|---|
| [ScienceAgentBench](https://arxiv.org/abs/2410.05080) | Task lấy từ 44 publication, SME validation, chấm executable result và cost. Bài học: task authenticity và output artifact quan trọng hơn prompt tự sinh hàng loạt. |
| [LAB-Bench](https://arxiv.org/abs/2407.10362) / [LABBench2](https://arxiv.org/abs/2604.09554) | Đo công việc nghiên cứu thực tế như literature search, figure/table/document interpretation. Dùng làm external diagnostic, không làm release gate trực tiếp vì chủ yếu model capability. |
| [ChemBench](https://arxiv.org/abs/2404.01475) | Kiểm tra chemistry knowledge/reasoning và overconfidence; phù hợp làm diagnostic cho backbone, không phải agent E2E. |
| [ToxReason](https://aclanthology.org/2026.findings-acl.977/) | Tách toxicity prediction khỏi mechanistic reasoning dựa trên Adverse Outcome Pathway. Nếu ToxAgent đưa ra cơ chế, cần một mechanistic pack dựa trên evidence/AOP, không suy cơ chế từ attribution. |
| [OECD QAF](https://www.oecd.org/en/publications/q-sar-assessment-framework-guidance-for-the-regulatory-assessment-of-quantitative-structure-activity-relationship-models-and-predictions_d96118f6-en.html) | Guidance chính thống để viết rubric về endpoint, applicability domain, uncertainty, model/prediction assessment và regulatory scope. Đây là standard, không phải agent benchmark. |

### 6.5 Rubric, judge và industry practice

| Nguồn | Pattern nên lấy |
|---|---|
| [HealthBench](https://openai.com/index/healthbench/) | Scenario thực tế, expert-written per-case rubric, consensus subset, multilingual coverage, grader meta-evaluation và worst-of-n reliability. Đây là pattern tốt cho high-stakes scientific communication. |
| [LLM-as-a-Judge/MT-Bench](https://arxiv.org/abs/2306.05685) | Position, verbosity và self-enhancement bias là lỗi đã biết; judge không được dùng mà không meta-eval. |
| [JudgeBench](https://arxiv.org/abs/2410.12784) | Judge mạnh vẫn có thể gần random trên response pairs khó; phải benchmark judge trên các lỗi ToxAgent thật. |
| [Self-Preference Bias](https://arxiv.org/abs/2410.21819) | Không dùng cùng lineage agent/judge như một bảo đảm độc lập duy nhất. |
| [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Phân biệt task, trial, grader, transcript, outcome, eval harness và agent harness; ưu tiên outcome; kết hợp code/model/human graders; xây task từ failure thật. |
| [LangSmith agent evaluation](https://docs.langchain.com/langsmith/evaluate-complex-agent) | Chấm riêng final response, trajectory và single step; không bắt exact trajectory nếu nhiều đường đi đều đúng. |

---

## 7. Thiết kế đề xuất: TAB-Suite v3

### 7.1 Nguyên tắc

1. **Outcome-first:** final product state và scientific claims quan trọng hơn việc agent đi đúng một trajectory mẫu.
2. **Hard invariants không được average.** Một câu trả lời hay không bù được cross-session leak hoặc số sai.
3. **Chấm đúng layer.** Predictor, OCR, retrieval provider, control plane và agent có score riêng.
4. **Gold facts có cấu trúc, không gold prose.** Không bắt model nói đúng một từ khóa khi có nhiều cách diễn đạt đúng.
5. **Mỗi gate phải có failure model và test cho chính grader.** Grader cũng là software cần benchmark.
6. **Capability và regression là hai suite khác nhau.** Capability có thể khó và pass rate thấp; regression phải gần tuyệt đối.
7. **Live evidence là observability/scheduled eval, không phải hermetic merge gate.**
8. **Mọi score phải kèm denominator, skipped/infra error và confidence interval phù hợp.**
9. **Không tuyên bố “safe” từ zero observed failures.** Chỉ báo upper confidence bound trên distribution đã test.
10. **Production failures phải trở thành regression tasks sau khi được de-identify.**

### 7.2 Bốn suite độc lập

| Suite | Câu hỏi | Grader chính | CI cadence |
|---|---|---|---|
| `control-conformance` | Auth/state/provenance/recovery/tool policy có đúng không? | Deterministic code/state | Mọi PR |
| `agent-capability` | Agent có hoàn thành scientific user goal không? | Outcome + claim metrics + calibrated rubric | Nightly/release |
| `agent-safety` | Có leak, injection success, harmful action, over-refusal không? | State/action policy + expert labels | Nightly/release |
| `agent-regression` | Failure đã từng sửa có tái xuất hiện không? | Deterministic hoặc case-specific | Mọi PR nếu frozen |

Không gộp AUROC/PR-AUC/ECE của ToxPred vào bất kỳ score nào ở đây.

### 7.3 Capability packs

Thay vì một bank monolithic cố định 130 task, dùng **core pack + capability packs**. Một pack chỉ active khi `/v1/models` và runtime profile khai capability tương ứng.

#### Core pack đề xuất

| Pack | Initial task families | Nội dung |
|---|---:|---|
| C1 Grounded numeric & endpoint semantics | 20 | Number transform, threshold/model provenance, endpoint independence, unavailable endpoint, applicability wording, no aggregate/clinical overreach |
| C2 Tool selection & planning | 16 | Right tool, right args, no-tool-needed, smallest slice, parallel vs sequential, clarification, budget |
| C3 Evidence retrieval & citation | 20 | Query quality, retrieval recall, relevance/promotion precision, read-before-cite, support/completeness, conflict, no-evidence abstention |
| C4 Multi-turn context & memory | 12 | Follow-up, correction, long-context invariant retention, explicit confirmation, session resume/compaction |
| C5 Report orchestration & consistency | 20 | Required sections, fact graph, cross-section consistency, evidence scope, figures, versioning, immutable artifact |
| C6 Fault tolerance & recovery | 12 | Predictor/provider/runtime failure, restart, timeout, duplicate delivery, cancellation, recovery run |
| C7 Security & prompt injection | 12 | Indirect injection, denied tool, cross-session access, data exfiltration, foreign analysis IDs, untrusted attachments |
| C8 Misuse & calibrated refusal | 8 families × generated variants | Harmful optimization paired với benign safety research; multi-turn reframing; helpful safe alternative; policy logging |

Con số trên là **starting coverage budget**, không phải tuyên bố statistical representativeness. Mỗi family nên sinh nhiều controlled variants; unit đánh giá chính là family và scenario, không phải chỉ số lượng file JSON.

#### Optional capability packs

| Pack | Chỉ bật khi | Agent đo gì | Component đo riêng gì |
|---|---|---|---|
| OCR handoff | `structure_recognition` available | Yêu cầu xác nhận khi ambiguity, không tự sửa cấu trúc, provenance attachment → SMILES | Exact/graph similarity, stereochemistry accuracy của ToxOCR |
| Target bioactivity | Predictor phục vụ target panel | Phân biệt Ki/Kd/IC50/EC50, activity cliff wording, selectivity và limitation | MAE/rank/cliff detection của predictor |
| Attribution/XAI | Endpoint có explainer | Đọc đúng endpoint, top contributors, non-causal wording, special/unmapped mass | Faithfulness/stability của explainer |
| Mechanistic toxicology | Có curated AOP/evidence source | Nối claim với MIE/KE/KER/AO có nguồn và uncertainty | Retrieval/ontology coverage |
| Multilingual | Product hỗ trợ VI/EN | Semantic parity, numeric locale, safety parity | N/A |

### 7.4 Cross-cutting coverage requirements

Mỗi release set phải có:

- ít nhất 30% task bằng tiếng Việt và phần còn lại tiếng Anh hoặc language pair;
- ít nhất 30% task có ≥5 user-agent exchanges thật, không chỉ “analyze rồi ask”;
- matched benign/adversarial pairs cho security và misuse;
- paraphrase, formatting và locale variants để tránh chấm prompt memorization;
- balanced cases: positive, negative, unavailable, ambiguous, contradictory và no-evidence;
- scaffold diversity cho molecule fixtures;
- explicit provenance của task: production failure, SME-authored, literature-derived hoặc generated variant;
- difficulty calibration bằng baseline agents, không chỉ cảm nhận của tác giả.

### 7.5 Task schema v3 tối thiểu

```yaml
schema_version: eval-task-v3
task_id: report-semantic-contradiction-001
suite: agent-capability
capability_pack: report_consistency
risk_tier: critical
source:
  kind: production_failure
  reference: audit-2026-09-13/P0-2
languages: [vi, en]
setup:
  fixture_id: report-contradictory-attribution-v1
  fixture_hash: sha256:...
  initial_state: ...
conversation:
  driver: scripted_user
  goal: ...
  turns: ...
outcome_contract:
  required_state_predicates: [...]
  forbidden_state_predicates: [...]
  allowed_side_effects: [...]
claim_contract:
  required_fact_ids: [...]
  forbidden_inferences: [...]
  citation_requirements: [...]
grading:
  deterministic: [...]
  semantic_rubric_id: report-consistency-v2
  human_required: true
variants:
  metamorphic_family: report-consistency-001
  hidden_parameters: [...]
governance:
  authored_by: ...
  sme_approved_by: ...
  approved_at: ...
  expires_at: ...
```

Task schema phải validate cả **task** lẫn **grader configuration**. Task không có SME approval không được vào release set.

---

## 8. Grading architecture

### Tier 0 — Infrastructure validity

Một trial không được tính pass/fail capability nếu:

- fixture hash sai;
- provider outage ngoài fault scenario;
- runner không thu đủ trace/state;
- wrong model/profile/schema version;
- timeout do eval infrastructure;
- grader error.

Kết quả là `invalid_trial`, không phải `fail` và tuyệt đối không được loại khỏi denominator âm thầm.

### Tier 1 — Deterministic hard gates

Chỉ dùng cho điều kiện có thể xác minh chắc chắn:

- numeric/classification claim khớp canonical field;
- cited ID tồn tại, thuộc session và đã read/promote;
- denied tool không execute;
- no cross-session read/write;
- final source graph reconstructable;
- invalid input không tạo analysis giả;
- budget/state machine/cancellation/recovery đúng;
- URL không do model tự tạo và resolve qua accepted provider record;
- report section/fact IDs nhất quán ở representation có cấu trúc.

Không dùng regex đơn lẻ để kết luận causal overclaim hoặc malicious intent.

### Tier 2 — Outcome và component metrics

- final-state predicates theo pattern τ-bench;
- subgoal progress theo pattern AgentBoard;
- tool-call correctness/arguments theo BFCL;
- retrieval recall@k, evidence precision@k, promotion precision;
- claim precision/recall;
- citation correctness và completeness theo ALCE;
- latency, tokens, cost và correction count.

Trajectory chỉ bị bắt exact khi đó là invariant. Nếu nhiều đường đi hợp lệ, chấm outcome và side effects.

### Tier 3 — Semantic rubric

Rubric phải **task-specific**, do SME viết hoặc duyệt. Các trục chung:

1. Scientific correctness.
2. Claim–evidence support.
3. Completeness/coverage.
4. Uncertainty calibration.
5. Endpoint/applicability interpretation.
6. Cross-section consistency.
7. Actionability trong scope decision-support.
8. Misuse/over-refusal khi applicable.

Model judge chỉ được dùng sau meta-eval. Với bulk run, dùng một judge version đã pin; gọi second judge cho disagreement/critical sample. Không yêu cầu ensemble ba model cho mọi output.

### Tier 4 — SME audit

- Hai SME độc lập cho critical semantic cases.
- Model/runtime name bị che.
- Disagreement có adjudicator.
- Báo confusion matrix, raw agreement, class-specific recall/precision và agreement coefficient phù hợp.
- Mỗi release audit 100% critical semantic failures, toàn bộ judge disagreement và một mẫu stratified của pass cases.
- Mọi major correction sinh regression task mới.

### Meta-eval cho judge

Gold set phải chứa controlled perturbations từ failure thật:

- đổi đúng một chữ số;
- đổi endpoint;
- citation có thật nhưng không support claim;
- thêm citation giả/authority cue;
- phủ định biến thành khẳng định;
- report summary mâu thuẫn body;
- bỏ limitation;
- thêm prose dài nhưng không thêm chất lượng;
- hoán đổi vị trí pairwise;
- VI/EN equivalents.

Judge gate đề xuất trước khi dùng blocking:

- sensitivity ≥0,90 trên critical-error controls;
- specificity ≥0,90 trên correct controls;
- không category critical nào <0,85;
- position/verbosity/citation-cue flip rate được báo và nằm dưới ngưỡng được phê duyệt;
- confidence interval và confusion matrix được lưu cùng `judge_version`.

Nếu không đạt, Tier 3 chỉ là advisory và SME quyết định.

---

## 9. Metrics và cách diễn giải

### 9.1 Scorecard bắt buộc

| Nhóm | Metric |
|---|---|
| Outcome | pass@1, pass^k, worst-of-n, final-state success |
| Scientific | numeric fidelity, grounded-claim precision/recall, unsupported critical claims, endpoint semantics |
| Evidence | retrieval recall@k, promotion precision, citation correctness, citation completeness, contradiction handling |
| Tool/trajectory | valid tool/argument rate, redundant calls, denied calls, subgoal progress, correction loops |
| Reliability | infra-valid trial rate, recovery success, restart reconstruction, timeout/cancel behavior |
| Safety | attack success rate, harmful action completion, data leak rate, policy compliance, false-refusal rate |
| Product | first-candidate acceptance, p50/p95 latency, tokens, provider cost, report completion with gaps |
| Fairness/parity | VI–EN paired delta, locale numeric errors, judge parity |
| Eval health | judge–SME agreement, grader false positive/negative, task ambiguity rate, invalid-trial rate |

### 9.2 `pass@1`, `pass^k` và worst-of-n

- `pass@1`: xác suất task pass trong một attempt; headline cho trải nghiệm người dùng.
- `pass^k`: tỷ lệ task mà **tất cả k trial** đều pass; đo reliability theo τ-bench.
- `worst-of-n`: điểm thấp nhất trong n response; phù hợp high-stakes behavior theo HealthBench.

Không dùng `pass^5=100%` như bằng chứng agent “≥90% reliable”. Đây là một release acceptance rule, không phải lower confidence bound.

### 9.3 Confidence interval và sample size

- Với task pass rate: Wilson interval hoặc paired bootstrap theo **task family**.
- So sánh hai version trên cùng task/trial: McNemar hoặc paired bootstrap, không dùng hai CI độc lập.
- Multiple variants trong cùng family phải cluster; không giả định độc lập.
- Với zero critical failures, báo one-sided upper bound. Quy tắc gần đúng “rule of three”: 0 failure trong 300 trial độc lập cho upper 95% khoảng 1%. Đây vẫn chỉ là distribution đã test.
- Một threshold 2% không thể kiểm đáng tin bằng 10 task. Cần hàng trăm benign-control trials hoặc đổi thành report-only cho tới khi đủ mẫu.

### 9.4 Không có aggregate toxicity score, cũng không có aggregate benchmark score để che lỗi

Dashboard có thể có summary, nhưng release decision phải giữ vector metric và hard gates. Không công bố một con số “TAB score” duy nhất.

---

## 10. Release protocol đề xuất

### Mọi PR

- `control-conformance` frozen/hermetic;
- toàn bộ regression tasks deterministic;
- task/schema/grader unit tests;
- không critical hard-gate regression;
- eval artifacts có suite hash, grader hash và environment metadata.

### Nightly

- core capability sample, 1–3 trial;
- agent safety sample;
- model/runtime pinned;
- trend pass@1, first-pass acceptance, latency/cost;
- invalid trials được quarantine và điều tra.

### Release candidate

1. Frozen full suite.
2. Predictor integration suite.
3. Live evidence suite riêng, timestamped.
4. Critical cases tối thiểu 5 trials/task; normal cases tối thiểu 3.
5. Không hard invariant failure.
6. Không cross-session leak, denied-tool execution hoặc source fabrication.
7. Không regression có ý nghĩa so với release trước trên paired analysis.
8. SME audit và judge meta-eval còn hiệu lực.
9. Latency/cost đạt product SLO được duyệt.
10. Sealed release result đạt gate nhưng không dùng để prompt-tune sau khi mở.

Ngưỡng 85% overall/80% per category của plan cũ có thể giữ **tạm thời** cho internal alpha để có continuity, nhưng phải được thay sau hai baseline release bằng target dựa trên distribution, risk tier và user impact. Critical scientific/safety invariants vẫn là zero-tolerance acceptance rule.

---

## 11. Dataset governance và chống benchmark overfitting

### 11.1 Nguồn task

Mục tiêu composition ban đầu:

- 40% de-identified production/audit failures;
- 30% SME-authored scientific scenarios;
- 20% adversarial/metamorphic variants;
- 10% external-benchmark-inspired diagnostics.

Không copy nguyên văn test item có license hạn chế. Chỉ áp dụng pattern hoặc dùng public dataset theo license.

### 11.2 Dev set và sealed release set

- Dev set có thể ở repo.
- Sealed set phải nằm ở access-controlled eval service/object store, không trong checkout của agent-dev.
- Task plaintext chỉ được materialize cho runner lúc chạy.
- Có immutable version, access log, canary string và rotation policy.
- Khi một sealed task bị dùng để debug/prompt-tune, task đó chuyển sang regression/dev và được thay bằng task mới.

### 11.3 Parametric variants

Thay vì tăng số file bằng prompt gần giống nhau, sinh variants có kiểm soát:

- molecule/scaffold;
- endpoint và availability;
- decimal locale;
- evidence supported/contradictory/absent;
- injection placement;
- language/paraphrase;
- tool error point;
- multi-turn reframing.

Gold state được sinh từ fixture facts; generated prompt phải qua validator và sampled SME review.

---

## 12. Mapping failure hiện tại → regression task bắt buộc

| Failure đã quan sát | Task family phải thêm | Grader |
|---|---|---|
| Executive summary phủ định explanation data | `report_cross_section_consistency` | Structured semantic consistency + SME |
| Evidence ethanol–hERG trả về 5 paper không liên quan | `evidence_promotion_precision` | Entity/endpoint relevance gold + promotion state |
| Runtime profile 64 step nhưng chạy profile 32 step | `effective_runtime_profile_truth` | Request/manifest contract |
| Claim ID do model tự bịa và va chạm | `server_owned_identity` | State/outcome; model không được chịu trách nhiệm DB ID |
| Regex hiểu sai phủ định | `negated_safety_statement` | Minimal pairs + grader meta-test |
| URL model tự viết | `provider_owned_citation` | Deterministic citation origin |
| Numeric rounding nhưng transform identity | `numeric_transform_contract` | Deterministic value/transform |
| Attribution special/unmapped mass bị bỏ qua | `attribution_mass_accounting` | Structured fact coverage |
| Report tool ordering sai, nhiều retry | `report_subgoal_progress_and_efficiency` | Subgoal state + redundant/error calls |
| Usage snapshot bị lưu lặp | `usage_event_normalization` | Deterministic event identity |
| Live evidence provider trả rỗng | `no_evidence_honest_abstention` | Answer behavior; provider outage tách khỏi agent failure |
| Recovery/cancel/restart | `durable_recovery` | Final DB/outbox/run state |

Đây là minimum regression pack có giá trị cao hơn việc tự sinh thêm hàng chục prompt chưa gắn với failure model.

---

## 13. Roadmap triển khai khả thi

### P0 — Freeze và sửa sự thật tài liệu (1 tuần)

- Gắn version cho current 50-task suite.
- Sửa README sai hiện trạng và tạo một reproducible test image/target.
- Thêm command xuất task inventory, gate coverage, language, turns, fixtures và skipped reasons.
- Định nghĩa rõ component vs agent metrics.
- Không đổi 50 task cũ cho tới khi có baseline archive.

**Exit:** fresh clone chạy được frozen eval; manifest đầy đủ; tài liệu khớp code.

### P1 — Regression-first v3 (2 tuần)

- Implement schema v3 tối thiểu và adapter đọc v1 task.
- Chuyển 12 failure families ở §12 thành task.
- Thêm final-state predicates, invalid-trial state và grader self-tests.
- Thêm report semantic consistency grader trên structured facts.

**Exit:** mọi finding audit quan trọng có regression coverage.

### P2 — Evidence và scientific grading (2–3 tuần)

- Claim segmentation và claim-level gold.
- Retrieval vs promotion vs citation metrics tách riêng.
- SME viết/duyệt rubric cho endpoint/applicability/report.
- Tạo VI/EN paired set.

**Exit:** đo được lỗi do retrieval, synthesis hay scientific interpretation.

### P3 — Multi-turn, security và misuse (2 tuần)

- Validated user simulator theo pattern τ-bench.
- AgentDojo-style indirect injection scenarios.
- Matched harmful/benign pairs và false-refusal metrics.
- 5+ turn invariant-retention cases.

**Exit:** utility-under-attack và safety-utility trade-off đều đo được.

### P4 — Judge calibration và sealed release service (2–3 tuần)

- SME seed set và controlled perturbations.
- Meta-eval judge; pin judge/version/prompt.
- Access-controlled sealed store; canary/rotation/audit log.
- Paired comparison và confidence reporting.

**Exit:** judge đạt gate; sealed task không lộ cho development path.

### P5 — Production feedback loop (liên tục)

- Stratified sample trace production đã de-identify.
- User/SME correction taxonomy.
- Failure → task SLA.
- Drift dashboard theo model/runtime/provider/profile.

---

## 14. Quyết định nên chốt

1. **Không chọn v1 hay v2 làm canonical implementation spec.** Dùng tài liệu này làm review và viết một schema/implementation RFC v3 riêng.
2. **Giữ 50-task suite như `eval-task-v1` regression baseline**, không gọi nó là full agent benchmark.
3. **Không triển khai 130 task trước grader và task lifecycle.** Quality của task/gold quan trọng hơn count.
4. **Ưu tiên report/evidence failures hiện có** trước OCR/bioactivity expansion.
5. **Dùng outcome/state graders làm nền**, LLM judge chỉ cho semantic residual.
6. **Sealed set ở ngoài repo.**
7. **Misuse pack phải balanced với benign controls**, không dùng keyword gate.
8. **Áp dụng OECD QAF cho scientific rubric** và ToxReason-style mechanistic pack chỉ khi hệ thống thực sự có AOP/evidence capability.
9. **Mọi release report phải công khai denominator, skip, infra error và CI**, không chỉ pass rate.
10. **Không dùng một aggregate TAB score.**

---

## 15. Definition of Done cho benchmark agent layer

TAB-Suite v3 chỉ được coi là sẵn sàng khi:

- [ ] schema v3 và backward-compatible loader tồn tại;
- [ ] task inventory được sinh tự động từ repo;
- [ ] mọi scientific/product invariant có owner, grader, positive và negative grader test;
- [ ] 12 failure families ở §12 có regression task;
- [ ] report build, evidence research, Q&A, attribution và recovery có E2E coverage;
- [ ] VI/EN parity set tồn tại;
- [ ] multi-turn set có hội thoại dài thật;
- [ ] retrieval, promotion, citation và synthesis được chấm riêng;
- [ ] judge có meta-eval với SME gold và controlled perturbations;
- [ ] sealed set không nằm trong repo;
- [ ] release run reproducible, versioned và lưu artifact;
- [ ] statistical report phân biệt task, family, trial và invalid trial;
- [ ] production failure intake đã hoạt động;
- [ ] SME và security owner đã sign-off policy/rubric;
- [ ] predictor/OCR scores không bị gộp vào agent score.

---

## 16. Tài liệu tham khảo chính

### Agent và tool use

1. Liu et al., [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688), 2023/ICLR 2024.
2. Li et al., [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://arxiv.org/abs/2304.08244), EMNLP 2023.
3. Patil et al., [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard), ICML 2025/current leaderboard.
4. Yao et al., [τ-bench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b126cc38b8638e07bef37e7b2bb72bf-Abstract-Conference.html), ICLR 2025.
5. Ma et al., [AgentBoard](https://arxiv.org/abs/2401.13178), NeurIPS 2024.
6. Mialon et al., [GAIA](https://arxiv.org/abs/2311.12983), ICLR 2024.

### Security và misuse

7. Debenedetti et al., [AgentDojo](https://arxiv.org/abs/2406.13352), 2024.
8. Ruan et al., [ToolEmu](https://arxiv.org/abs/2309.15817), ICLR 2024.
9. Andriushchenko et al., [AgentHarm](https://arxiv.org/abs/2410.09024), 2024.
10. Zhao et al., [ChemSafetyBench](https://arxiv.org/abs/2411.16736), 2024.
11. Urbina et al., [Dual use of AI-powered drug discovery](https://www.nature.com/articles/s42256-022-00465-9), Nature Machine Intelligence 2022.

### Evidence và scientific domain

12. Gao et al., [ALCE](https://aclanthology.org/2023.emnlp-main.398/), EMNLP 2023.
13. Ru et al., [RAGChecker](https://arxiv.org/abs/2408.08067), 2024.
14. Niu et al., [RAGTruth](https://arxiv.org/abs/2401.00396), ACL 2024.
15. Chen et al., [ScienceAgentBench](https://arxiv.org/abs/2410.05080), 2024.
16. Laurent et al., [LAB-Bench](https://arxiv.org/abs/2407.10362), 2024.
17. Laurent et al., [LABBench2](https://arxiv.org/abs/2604.09554), 2026.
18. Mirza et al., [ChemBench](https://arxiv.org/abs/2404.01475), 2024.
19. Park et al., [ToxReason](https://aclanthology.org/2026.findings-acl.977/), Findings of ACL 2026.
20. OECD, [(Q)SAR Assessment Framework](https://www.oecd.org/en/publications/q-sar-assessment-framework-guidance-for-the-regulatory-assessment-of-quantitative-structure-activity-relationship-models-and-predictions_d96118f6-en.html), 2023.

### Evaluation methodology

21. OpenAI, [HealthBench](https://openai.com/index/healthbench/), 2025.
22. Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), NeurIPS 2023.
23. Tan et al., [JudgeBench](https://arxiv.org/abs/2410.12784), 2024.
24. Wataoka et al., [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/abs/2410.21819), 2024.
25. Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), 2026.
26. LangChain, [Evaluate a complex agent](https://docs.langchain.com/langsmith/evaluate-complex-agent), current documentation.
