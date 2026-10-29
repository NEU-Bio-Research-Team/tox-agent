# Đề xuất kiến trúc agent cho ToxAgent: brainstorm và quyết định thiết kế

Ngày: **28/10/2026**.  
Căn cứ repository: **feat/agent-benchmark-master-plan**, head kiểm tra lại vẫn là **2f3c337463986d6eeed229f32f16b95403172259**.  
Tài liệu nền: **ToxAgent_Literature_Design_Benchmark_Review_VI_2026-10-28.md**.

Đây là bản đề xuất, chưa phải implementation hay kết quả benchmark. Các cấu hình số minh họa phải được tune trên dev. Không thay source repository trong lượt brainstorm này.

## 1. Quyết định mình đề xuất

Xây **một investigator làm việc trên Scientific Case, được đặt trong workflow thực thi xác định, với các scientific procedures có kiểu và worker theo nhu cầu**.

Tách rõ:

- **Workflow thực thi** giữ quyền chạy predictor, policy, stage, snapshot và release.
- **Investigator** quyết định câu hỏi còn thiếu gì, đọc gì tiếp theo, hiểu các bằng chứng và đề xuất kết luận.
- **Case** là trạng thái khoa học bền vững: claims, measurement scope, sources, assumptions, dependencies và revisions.
- **Procedures** xử lý việc xác định bằng code: chuẩn hóa, unit conversion, xử lý comparator, truy vấn batch, deduplicate và phép tính.
- **Verifier** kiểm field/rule bằng code; dùng một bước semantic review có giới hạn cho suy luận cần phán đoán.
- **Workers** chỉ mở khi có phần tìm kiếm/đọc tài liệu độc lập; trả evidence packet, không tự sửa kết luận chung.

Số lượng module không phải số lượng agent. MVP chỉ cần một vai trò LLM chính; reviewer là một call/role có mục tiêu. Workers là phần mở rộng có thể bật theo loại tác vụ.

Đối với sản phẩm, kiến trúc này nhắm vào **thời gian để nhà nghiên cứu kiểm chứng được một assessment**. Đối với paper, phần đáng kiểm tra là **measurement-aware evidence acquisition và cập nhật kết luận theo dependencies**.

## 2. Các phương án brainstorm

Đánh giá dưới đây là judgment thiết kế, chưa phải kết quả so sánh thực nghiệm.

| Phương án | Điểm mạnh | Điều cần kiểm chứng/chi phí | Vai trò mình chọn |
|---|---|---|---|
| Workflow RAG mạnh | Dễ tối ưu, dễ debug; phù hợp report lặp lại và task đã rõ | Bị hạn chế khi câu hỏi và nguồn làm thay đổi kế hoạch | Baseline B và đường phục vụ những task ổn định |
| Generic agent với công cụ và code | Linh hoạt, tác vụ mới không cần viết workflow riêng | Có thể xử lý nguồn tốt nhưng vẫn cần contract cho claims và revisions | Baseline C mạnh; học thiết kế môi trường từ Biomni |
| Team chuyên gia nhiều agent | Tìm nhiều hướng độc lập, cô lập context từng task | Overhead; cùng sai hoặc diễn giải scope khác nhau; khó quy nguyên nhân | Chỉ dùng cho breadth/batch đã đo được lợi ích |
| Case-based investigator trong durable workflow | Tận dụng repo hiện có; hỗ trợ provenance, revision, bounded actions | Cần chứng minh Case giúp quyết định thay vì chỉ ghi log | **Phương án chính** |
| Agent được huấn luyện SFT/RL theo miền | Có thể giảm prompting và học policy từ trajectories | Cần môi trường, reward, gold, data và compute đáng tin | Giai đoạn sau, sau khi benchmark ổn |

Hai loại baseline đầu vẫn rất đáng đầu tư. Nếu generic agent với môi trường tốt đã đủ, đó là lựa chọn sản phẩm hợp lý. Không nên cố thiết kế D phức tạp để tạo khác biệt hình thức.

## 3. Học gì từ industry và các scientific agents?

### 3.1. Các nguồn mới bổ sung

| Nguồn | Kỹ thuật/cách làm có mô tả công khai | Áp dụng cho ToxAgent |
|---|---|---|
| [NovoScribe, case study với Novo Nordisk](https://claude.com/customers/novo-nordisk) [I1] | Nội dung chuyên gia duyệt, RAG, case-specific variables, người dùng review | Tách nội dung chuẩn khỏi observation từng case và assessment được đề xuất |
| [Replit Agent, case study của LangChain](https://www.langchain.com/breakoutagents/replit) [I2] | Role có phạm vi; dynamic context; restricted DSL và tracing | Procedures rõ input/output; worker có hợp đồng; trace hỗ trợ sửa lỗi |
| [Anthropic: context engineering](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) [I3] | Quản lý những gì vào context, tool outputs ngắn và retrieval theo nhu cầu | Compile context từ Case và evidence refs, hạn chế nạp lại transcript/raw records |
| [Anthropic: advanced tool use](https://www.anthropic.com/research/advanced-tool-use) [I4] | Tool search, programmatic calling và usage examples | Gộp xử lý batch trong code; nạp capability phù hợp; ví dụ cho tool phức tạp |
| [Temporal: agent memory/state](https://docs.temporal.io/ai/featured/memory-state-stripe) [I5] | Memory khác execution state; resume/retry cần checkpoint và idempotency | Persist từng bước và kết quả; không dùng chat summary để đoán bước đã hoàn tất |
| [LangChain: runtime behind production deep agents](https://www.langchain.com/blog/runtime-behind-production-deep-agents) [I6] | Harness và runtime là hai trách nhiệm khác nhau | Cải thiện loop không bắt buộc thay control plane hiện có |
| [Biomni, Science 2026](https://pubmed.ncbi.nlm.nih.gov/42424436/) [A1] | Retrieval-augmented planning, môi trường công cụ và code execution | Chọn capability theo goal và compose hành động; thêm scope/revision của ToxAgent |
| [Biomni-R0, model card](https://huggingface.co/biomni/Biomni-R0-32B-Preview) [A2] | RL từ tương tác trong biomedical environment | Hướng học policy về sau, không thay thế việc xây evaluator |
| [OpenScholar, Nature 2026](https://www.nature.com/articles/s41586-025-10072-4) [A3] | Retrieval bổ sung từ feedback và citation verification | Kiểm những required claims chưa có căn cứ, tránh search lại toàn bộ |
| [Doyle, truth maintenance](https://groups.csail.mit.edu/medg/people/doyle/publications/) [A4] | Ghi lý do và dependencies để duy trì/cập nhật beliefs | Căn cứ lịch sử cho revision; không tuyên bố dependency tracking là ý tưởng mới |

Các nguồn commercial là public deployment/vendor case studies, không phải kiểm chứng độc lập về ROI hoặc về tác dụng của từng thành phần kiến trúc. Những lựa chọn áp dụng ở cột cuối là **suy luận của bản đề xuất**.

NovoScribe cho một ví dụ sản phẩm quan trọng: output dùng nội dung đã được chuyên gia duyệt cộng với biến của từng case, rồi đưa vào review. Case study báo giảm 90% writing time, nhưng số đó không phải benchmark toxicology agent và không được chuyển thành lời hứa cho ToxAgent. [I1]

Replit cho thấy một sản phẩm có thể có nhiều roles với công cụ được giới hạn và trace rõ; case study không chứng minh mọi bài toán đều cần multi-agent. [I2]

### 3.2. Cách phối hợp các tiền lệ

- Từ **Biomni**: đầu tư action environment, chọn tài nguyên và thực thi chương trình.
- Từ **OpenScholar/PaperQA2**: retrieval là một vòng điều tra có mục đích và có kiểm citation.
- Từ **industry harnesses**: context có cấu trúc, procedures, checkpoints, cache, observability.
- Từ **ToxAgent/domain**: measurement scope, quan hệ evidence, uncertainty, provenance và revision.
- Từ **truth maintenance**: conclusion phụ thuộc vào inputs và assumptions; thay premise thì kiểm lại phần liên quan.

Không cần tái lập toàn bộ một general-purpose biomedical platform. ToxAgent nên có môi trường nhỏ hơn, được kiểm kỹ cho những câu hỏi activity/toxicity đã chọn.

## 4. Kiến trúc tổng thể

~~~mermaid
flowchart TD
    Q["Câu hỏi hoặc bằng chứng mới"] --> R{"Task và subject"}
    R -->|Tra cứu| F["Đọc artifact và dựng câu trả lời"]
    R -->|Điều tra hoặc cập nhật| I["Investigator"]
    I <-->|Context và patch| C["Scientific Case"]
    I --> P["Scientific procedures và evidence tools"]
    I -.-> W["Worker độc lập khi cần"]
    W --> E["Evidence packets"]
    P --> E
    E --> N["Normalize, scope và provenance"]
    N --> C
    I --> D["Candidate claims"]
    D --> V["Field, rule và semantic checks"]
    F --> V
    V -->|Sửa có giới hạn| I
    V --> O["Assessment và phần chưa giải quyết"]
~~~

Vòng agent chỉ điều phối phần điều tra. Activity → policy → optional toxicity vẫn do workflow/server thực thi. Kết luận khoa học không tự thành quyền override policy hoặc thay prediction.

### 4.1. Bốn route phục vụ

| Route | Ví dụ | Cách chạy |
|---|---|---|
| Read/explain | “pIC50 và stage status là gì?” | Đọc artifact; template hoặc một bước giải thích ngắn |
| Investigate | “Nguồn thực nghiệm có phản bác prediction không?” | Vòng investigator trên Case |
| Update | “Tôi bổ sung assay/exposure; đánh giá đổi gì?” | Tính affected claims rồi điều tra phần liên quan |
| Compare/batch | “So sánh các ứng viên trong context này” | Procedures batch; Case/subject riêng; tổng hợp theo tiêu chí đã khai báo |

Router kiểm explicit subject và artifact trước. Với câu hỏi mơ hồ, có thể dùng classifier nhỏ nhưng phải có route “chưa xác định”. Không dùng keyword “safe” để tự quyết toàn bộ câu hỏi nghiên cứu. Đo routing errors và cost riêng.

Read/explain không bắt buộc mở hypotheses hay tải skills. Batch không tự tạo một fusion score activity–toxicity.

## 5. Scientific Case: bộ nhớ có cấu trúc dùng được cho hành động

### 5.1. Bốn nhóm dữ liệu

1. **Observations:** prediction snapshot, assay record, stage status và giá trị người dùng cung cấp, mỗi loại có producer/provenance.
2. **Evidence assessments:** claim nào được hỗ trợ/phản bác/trong context nào; passage và scope tương ứng.
3. **Derived conclusions:** input claims, inference rule, assumptions, trạng thái đề xuất/phát hành và version.
4. **Investigation state:** goal, required claims, gaps, hành động đã thử, reason for stop và budget còn lại.

Một hypothesis “supported” chưa đồng nghĩa một chân lý đã được xác minh. Trạng thái phải nói rõ được hỗ trợ bởi nguồn nào và trong phạm vi nào.

### 5.2. Measurement scope và numeric constraints

Tách ba lớp thay vì so chuỗi tất cả fields:

| Lớp | Câu hỏi |
|---|---|
| Identity/context | Cùng ligand, target/variant, endpoint, method và context cần thiết không? |
| Representation | Hai giá trị có khác unit/alias nhưng tương đương sau chuẩn hóa không? |
| Numeric constraint | Exact/range/censored value cho phép kết luận gì theo task rule? |

Giữ raw value và canonical value cùng conversion. Missing metadata là unknown, không phải wildcard. Một alias được normalize không tự làm hai assay tương đương. Hai số khác nhau cũng không tự thành scientific contradiction; phải xét measurement uncertainty và context.

### 5.3. Dependencies để cập nhật đúng phần liên quan

Ví dụ một margin claim dựa vào IC50, free Cmax, unit compatibility và công thức đã định nghĩa:

~~~json
{
  "claim": "exposure_margin",
  "inputs": ["assay_measurement_v1", "user_exposure_v1"],
  "rule": "ratio_same_concentration_unit",
  "assumptions": ["exposure_context_compatible"],
  "status": "proposed"
}
~~~

Khi Cmax được sửa:
- Tạo observation version mới, giữ lịch sử.
- Đánh dấu margin và những kết luận dùng margin cần tính/kiểm lại.
- Giữ nguyên predictor observation.
- Không chạy lại literature search trừ khi context/exposure mới tạo gap mới.

Nếu một claim còn một justification khác hợp lệ, việc mất một premise không được tự động xóa claim. MVP có thể đánh dấu “needs_reassessment”; giai đoạn sau hỗ trợ nhiều justification sets. Claim cycles và unresolved references phải được phát hiện.

Đây là **truth maintenance có phạm vi**, không phải hệ thống chứng minh mọi kết luận sinh học. Rule numeric/identity có thể kiểm bằng code; scientific interpretation vẫn cần evidence và đánh giá.

### 5.4. Memory nào vào context?

Context view gửi cho investigator chỉ gồm:
- Task/subject đang khóa và dữ kiện bắt buộc.
- Observations liên quan cùng identifiers.
- Kết luận hiện hành, những gì vừa thay đổi.
- Evidence liên quan, counter-evidence và missing metadata.
- Gaps chưa giải quyết, lần tìm trước và budget.
- Chỉ dẫn/skills cần cho task đó.

Raw full text, hàng trăm assay rows và lịch sử đầy đủ nằm ngoài prompt; tool trả refs để đọc thêm. Summary có thể giúp navigation nhưng không thay raw evidence trong verification.

Tách procedural memory (skills đã review), episodic memory (Case history) và execution state (step results). Không tự chuyển một thành công tình cờ thành skill được dùng ở mọi case.

## 6. Investigator: vòng điều tra có mục tiêu

### 6.1. Lập goal contract ngắn

Đầu vào user được chuyển thành:
- Subject và endpoint/context.
- Những câu hỏi phải trả lời.
- Dữ kiện nào cần có để trả lời.
- Các hành động được phép.
- Budget và stop conditions.

Required claims xuất phát từ câu hỏi, không được lấy từ hidden benchmark gold. Không ép mọi lookup thành tám giả thuyết.

### 6.2. Vòng quyết định

~~~text
read the current case view
identify the unresolved requirement that matters most
choose an admissible action, or answer/ask/stop
execute with a bounded procedure
validate and normalize returned artifacts
commit a small case patch with provenance
repeat only if a required question remains unresolved
~~~

Những gap chính:
- Identity/scope chưa xác định.
- Thiếu evidence phù hợp.
- Có record nhưng chưa đọc decisive span.
- Measurement extraction/units chưa giải quyết.
- Có bất đồng cần kiểm context.
- Thiếu input chỉ người dùng có.
- Provider/corpus không cung cấp được dữ liệu.

Ưu tiên ban đầu dùng rules minh bạch và phán đoán của investigator. Chưa gọi là tối ưu expected value of information khi chưa có utility/predictive model được kiểm chứng.

### 6.3. Counter-evidence và stopping

Tìm phản chứng khi đang chuẩn bị đưa ra claim quyết định và có nguồn hợp lý để kiểm. Không bắt lookup đơn giản tìm thêm paper cho đủ “research”.

Stop reasons:
- Câu hỏi đã được giải quyết trong scope.
- Cần thông tin người dùng.
- Nguồn có thể truy cập không giải quyết được.
- Provider failure ngăn tiếp tục.
- Budget exhausted.

Provider failure và thiếu bằng chứng khoa học là hai kết quả khác nhau. “Insufficient” phải nói thiếu gì và việc đó hạn chế claim nào.


### 6.4. Một cơ chế research có thể làm cụ thể hơn: chọn phép phân biệt

Khi có bất đồng, investigator giữ các explanations cạnh tranh, ví dụ:
- Prediction không khớp measurement trong đúng scope.
- Assay/readout khác nên hai nguồn chưa so trực tiếp được.
- Identifier hoặc variant đã bị ánh xạ sai.
- Nguồn thứ hai chỉ lặp lại cùng một thí nghiệm.

Mỗi candidate action phải ghi **gap hoặc explanation mà kết quả có thể phân biệt**. Đọc assay metadata có thể phân biệt khác readout; đọc thêm một review chung có thể không giải quyết được gap đó.

Bản đầu có thể rank actions theo thứ tự:
1. Có giải quyết identity/scope blocker không?
2. Có nhắm vào required claim quyết định chưa có căn cứ không?
3. Nguồn có khả năng truy cập và chứa dữ liệu cần thiết không?
4. Có tái dùng observation đã có trước khi search mới không?
5. Chi phí còn phù hợp budget không?

Khả năng truy cập có thể dựa trên metadata và lịch sử provider; khả năng chứa dữ liệu là estimate cần kiểm trên dev. Khi chưa học xác suất các outcomes, đây là heuristic có hướng đích, không phải Bayesian information gain.

Benchmark có thể thiết kế các case mà mỗi action mở một evidence packet, mỗi packet có cost và scope rõ. So policy với fixed search, random admissible action và generic ReAct trên cùng environment. Một hidden oracle dùng để phân tích upper bound không được cung cấp cho agent.

Nếu policy thắng chỉ do được thêm metadata, cấp metadata đó cho các controls. Nếu không thắng sau control, giữ phần representation/revision có ích và bỏ claim acquisition mới.

## 7. Procedures, tool surface và code execution

### 7.1. Hai tầng công cụ

**Primitives** là những tool hiện có: đọc artifact, search, read record/passages, case update và submit answer.

**Procedures** là tổ hợp có kiểu của những thao tác xác định, ví dụ:
- Resolve subject context.
- Thu assay records từ providers được chỉ định.
- Chuẩn hóa measurements, unit và comparators.
- Deduplicate theo nguồn gốc thí nghiệm.
- Tính derived quantities từ observation refs.
- Kiểm dependency và stale claims.

Investigator vẫn chọn procedure, parameters và hành động tiếp theo. Bên trong procedure là code ổn định. Đây không phải một fixed scientific workflow bắt mọi câu hỏi đi cùng một chuỗi.

### 7.2. Khi nào dùng programmatic orchestration?

Dùng cho batch queries, filtering, joins, conversions và xử lý intermediate records. Kết quả vào context gồm dữ kiện liên quan, counts, exclusions và refs để drill down.

Không để một summary che counter-evidence: procedure phải khai các tiêu chí lọc, rejected records/reasons và những trường còn thiếu.

MVP nên dùng một allowlisted procedure runner hoặc DSL có kiểu trên backend. Khi cần phân tích dữ liệu mở rộng mới thêm sandbox code execution với SDK tool access giới hạn và artifact outputs có provenance. Không cần cấp general shell cho mọi assessment.

Nếu một procedure gọi mười tools nội bộ, benchmark phải ghi mười sub-operations và toàn bộ usage. Không dùng một API call để che ngân sách; tất cả arms có cùng procedure access.

### 7.3. Tool/skill discovery có điều kiện

Giữ core capability nhỏ; capability hiếm mới tải schema/hướng dẫn khi cần. Nếu toolset hiện đã ngắn và dễ chọn, thêm một bước tool search có thể tăng latency mà không có lợi.

Skills nên chứa:
- Khi nào áp dụng.
- Input cần có.
- Những thao tác/rules có thể dùng.
- Failure modes.
- Một vài ví dụ canonical.
- Những suy diễn không được phép từ dữ liệu đó.

Ví dụ skill “compare assays” hướng dẫn tìm context quyết định và cách xử lý unknown; nó không được quyền ghi một claim là đúng chỉ vì workflow đã chạy hết.

Giữ nội dung domain tương đương cho C trong nghiên cứu để tách giá trị kiến thức khỏi cơ chế agent.

## 8. Workers: tách công việc, giữ kết luận tập trung

Worker có ích khi có hai câu hỏi có thể giải độc lập, ví dụ:
- Truy assay records.
- Đọc primary paper và passage thực nghiệm.

Bắt đầu với fan-out nhỏ, ví dụ tối đa hai workers như một cấu hình thử trên dev. Lookup không cần worker. Chỉ thêm worker thứ ba khi có lợi ích đo được.

Mỗi worker nhận goal, subject, source scope, output schema và budget; trả:
- Records/passages đã đọc.
- Measurements extracted và uncertainty.
- Relevance/scope assessment đề xuất.
- Access failures, exclusions và provenance.

Workers không tự cập nhật shared Case hay đưa final development posture. Investigator kiểm packet và commit qua một revision. Concurrent database reads có thể chạy song song; write coordination cần kiểm version.

Hai workers đọc lại cùng một paper không tạo hai nguồn độc lập. Có diversity nguồn cũng chưa đủ chứng minh independence; phải theo origin/measurement lineage.

## 9. Verification và answer compilation

### 9.1. Kiểm theo loại claim

| Loại | Kiểm bằng gì? |
|---|---|
| Artifact fact | Field value, identity, version và source resolver |
| Extracted numeric fact | Locator, table context, unit/comparator và normalization |
| Derived numeric claim | Approved calculation, input refs và dimensional checks |
| Scope relation | Context/identity rules; semantic review khi cần interpret |
| Scientific assessment | Premise set, evidence relevance, counter-evidence và assumptions |
| Recommendation | Basis, scope, điều kiện thay đổi và phần chưa biết |

Bước LLM review không phải một expert độc lập chỉ vì đổi prompt hay đổi model. Nó là bộ kiểm cần đo false accept và false block trên human gold.

### 9.2. Chữa vấn đề reviewer hiện tại

Phân biệt:
- Claim quan sát: “model báo X”.
- Claim được trích: “assay báo Y”.
- Claim suy ra: “X và Y bất đồng theo rule/context Z”.

Reviewer phải thấy cùng lúc X, Y, rule và scope. Không đòi một paper viết “model ToxAgent đang sai”. Numeric calculations do code kiểm; reviewer không tính nhẩm lại.

Review tập trung vào claims quyết định và derived relations; field lookups đơn giản có thể chỉ cần hard checks. Trong experimental arms, reviewer policy phải được giữ chung để không quy lợi ích riêng của gate cho investigator.

### 9.3. Phát hành phần có căn cứ

Hard errors về subject, nonexistent sources hoặc numeric transformation phải sửa. Với phần diễn giải chưa đủ:
- Sửa hoặc thu hẹp claim trong số vòng cho phép.
- Phát hành những facts đã xác minh, nói rõ assessment nào chưa hoàn tất.
- Nếu claim thiết yếu không được hỗ trợ, ghi thiếu input/evidence.
- Đo thông tin mất sau repair, không chỉ tỷ lệ gate pass.

Answer được dựng từ claims đã phân loại và refs; LLM có thể diễn đạt nhưng renderer giữ đúng số, đơn vị và status. Claim compiler là interface kiểm tra được, không phải bằng chứng rằng hệ thống không bao giờ hallucinate.

## 10. Runtime và chi phí

Giữ control plane, database, gateway và runtime adapter hiện có trước. Áp dụng những nguyên tắc durable execution vào các khoảng trống đã tìm được; không cần migrate sang LangGraph/Temporal chỉ để có tên framework.

### 10.1. Trạng thái thực thi

Mỗi step lưu request, result/ref, status, attempt và usage. Case patch có revision check. Resume dùng completed-step results thay vì đoán từ transcript. Đối với step dang dở, retry phải có idempotency theo loại thao tác; không hứa exactly-once cho external calls.

Awaiting user input phải giải phóng worker. Provider lỗi có bounded retry/backoff và được ghi là access failure, không thành “không có evidence”.

### 10.2. Cache

Tách:
- Provider/record cache.
- Frozen passage/parser cache.
- Procedure result cache.
- Prompt prefix cache nếu provider hỗ trợ.

Cache key cần identity, parameters, source version, model/policy/procedure version liên quan. Answer cũ không tự hợp lệ khi có exposure hoặc evidence mới. Cache reuse cần xuất hiện trong trace.

### 10.3. Model routing

Dùng code cho exact operations. Có thể dùng model rẻ cho extraction/routing đã được kiểm chứng, model mạnh cho integration khó. Escalation dựa trên schema failures, scope ambiguity, unresolved critical claims và dev-measured performance; không chỉ dựa self-confidence.

Main architecture comparison dùng cùng backbone. Mixed-model configuration là một product efficiency experiment riêng.

### 10.4. Đo cái gì để vận hành?

Một trace cần trả lời được:
- Chi phí nằm ở acquisition, case/skill operations, synthesis hay review?
- Nguồn quyết định đã tìm được chưa?
- Failed task do retrieval, interpretation, input schema hay release?
- Retry/resume có lặp action hoặc mất context không?
- User cần bao lâu để xác minh đúng assessment?

Không cần lưu chuỗi suy luận nội bộ dài; action rationale ngắn, inputs/outputs, artifacts và state deltas là các đối tượng kiểm tra chính.

## 11. Một case chạy qua kiến trúc

Đây là dữ liệu giả định để minh họa contract, không phải đánh giá một hợp chất thực.

### Lượt 1: đối chiếu prediction

User: “Model báo non-blocker, nhưng tôi thấy một paper đo IC50 0,3 µM. Hai kết quả có mâu thuẫn không?”

1. Router gắn đúng frozen analysis và tạo goal đối chiếu.
2. Investigator đọc prediction, threshold/model scope, xác minh paper và measurement.
3. Procedure chuẩn hóa IC50, giữ raw unit và locator; kiểm ligand/target/assay context.
4. Một record khác target được giữ như contextual; bản ghi database trích lại chính paper không được đếm thành phép đo độc lập.
5. Nếu context tương thích và task rule phân loại khiến hai phía bất đồng, đề xuất claim “prediction và measurement bất đồng theo rule này”.
6. Nếu assay context chưa đủ, kết luận comparability chưa xác định thay vì ép một phía thắng.
7. Verifier nhận prediction, measurement, rule và scope cùng lúc.
8. Output giữ riêng model observation, experimental observation, assessment và missing context.

Threshold trong một benchmark phải được khai báo. Một ngưỡng như 10 µM trong fixture không tự trở thành chuẩn safety phổ quát.

### Lượt 2: thêm exposure

User: “Free Cmax trong context này là 0,01 µM; tính margin giúp tôi.”

1. Persist user input thành observation với provenance và verification status.
2. Kiểm exposure context/unit có tương thích với calculation đã chọn.
3. Procedure tính 0,3 / 0,01 = 30.
4. Derived claim liên kết hai input versions và rule.
5. Những assessment phụ thuộc exposure được xem xét lại; predictor snapshot giữ nguyên.

Kết quả 30 là một phép tính theo inputs, không tự chứng minh nguy cơ lâm sàng chấp nhận được. Nếu chưa kiểm được comparability, hệ thống có thể trình bày calculation có điều kiện thay vì khẳng định interpretation.

### Lượt 3: thay đổi context

Nếu người dùng chuyển sang target khác, hệ thống xác định subject mới và tái dùng phần identity/general context phù hợp. Measurement hERG không được biến thành direct evidence về target mới.

Nếu người dùng sửa exposure từ 0,01 thành 0,02 µM trên cùng context, chỉ các downstream claims liên quan cần tính/kiểm lại. Không xóa lịch sử hoặc tự chạy lại predictor.

Giá trị sản phẩm là user thấy rõ **đã thay đổi gì, giữ nguyên gì, và vì sao**.

## 12. Ánh xạ vào repository hiện có

Các paths dưới đây là nơi phát triển đề xuất; không hàm ý các chức năng mới đã tồn tại đầy đủ.

| Nền hiện có | Thay đổi đề xuất |
|---|---|
| domain/scientific_case/ | Thêm claim justifications/dependencies, revision status; tái sử dụng event log |
| domain/evidence_relation.py | Tách compatibility khỏi raw unit/comparator equality; giữ unknown |
| research/measurement.py và normalization.py | Numeric constraints, canonical units và origin references |
| application/investigation/ | Compile Case view, affected-claim resolver, gap selection và review packets |
| harness/context.py | Ghép task/Case view vào context; chỉ dẫn ngắn; procedural knowledge tải có điều kiện |
| harness/gateway/ | Route và bounded investigation loop; resume và typed stop |
| tools/definitions/ và registry | Procedures có kiểu, version và permissions; adapters tới tools hiện có |
| persistence/ và report/ | Versioned claim lineage, output từ claims đã được chấp nhận |
| evals/investigation/ | Evidence-given, update pairs, cost accounting và matched-information controls |

Giữ một nguồn trạng thái chính. Không tạo một agent kernel/state machine khoa học song song rồi đồng bộ ngược về Case sau. Không cần thêm graph database: nodes/edges có thể là relational records và typed projections.

Một logical interface tối thiểu:
- Read current Case view.
- Read bound artifacts.
- Search/read evidence.
- Execute approved scientific procedure.
- Propose a Case patch.
- Submit grounded answer.
- Delegate bounded evidence task, khi feature đó bật.

Đây là interface logic, không phải yêu cầu rename tất cả MCP tools đã có.

## 13. Phần nào là research contribution?

### H1: measurement semantics

Representation tách scope, units và censored constraints giúp giảm false contradiction và sai áp dụng evidence.

Kiểm bằng paired cases:
- nM/µM tương đương.
- Same ligand nhưng khác target/variant.
- Exact/censored records.
- Thiếu assay context.
- Duplicate source có cùng experimental origin.

Giữ tools/knowledge giống nhau; ablate phần representation/reasoning cần xét. Common identity guards vẫn giữ cho mọi arm.

### H2: revision có dependencies

Structured dependencies giúp update đúng claim và bảo toàn facts không liên quan, đồng thời giảm chi phí so với tái tổng hợp từ history.

Controls: generic ReAct với transcript; ReAct với summary tốt; D2 cũ; phiên bản D mới. Cùng observations và nội dung hướng dẫn.

Metrics: required-update accuracy, preservation accuracy, stale-claim rate, task success và total cost. Khi chỉ một premise thay đổi nhưng claim có một justification khác, gold phải mô tả trường hợp đó.

### H3: acquisition theo gaps

Explicit information needs giúp dùng budget vào nguồn có thể thay đổi câu trả lời.

So adaptive policy với fixed acquisition schedule trên cùng Case/procedures. Tách evidence-given khỏi frozen retrieval. Đo acquisition cost, decisive evidence recall và task success, không chỉ số paper đọc.

**Novelty của H1–H3 chưa được chứng minh.** Claim dependencies/truth maintenance, adaptive retrieval và tool orchestration đều có tiền lệ. Đóng góp có thể nằm ở formalization cho measurement evidence, cơ chế phối hợp cụ thể và bằng chứng đối chứng.

### Controls về engineering

Context compilation, batching, caching và durable execution cần đo hiệu quả vận hành; chúng không tự là scientific reasoning novelty. Nếu bật một cải tiến chung, cho B/C cùng quyền dùng để không tạo baseline yếu.

Đối với bài mới, khóa primary D mới so với C mạnh trước test. D so với B kiểm nhu cầu agent. Giữ kết quả/protocol cũ như lịch sử riêng, không sửa mục tiêu sau khi đã thấy output.

## 14. Thứ tự triển khai mình chọn

| Đợt | Phạm vi | Kết quả cần nhìn thấy |
|---|---|---|
| 1 | Sửa F1–F4, user observations và subject-bound fallback; review baseline B | Case reference đi qua product; baseline không thua vì lỗi plumbing |
| 2 | Case view gọn, normalized measurements và claim dependencies | Một update thay đúng downstream claims; context không cần nạp cả history |
| 3 | Goal/gap contract và scientific procedures | Investigator có thể chọn đường đọc phù hợp, dừng đúng và log rõ lý do |
| 4 | Claim-set verification, output compilation và budget accounting | False block/false accept được đo; partial answer vẫn hữu ích |
| 5 | Optional workers, caching/model routing theo traces | Có lợi ích chất lượng/latency/cost so với cấu hình không bật |

Mỗi đợt có feature flag và diagnostic cases; scientific study dùng cấu hình đã freeze. Không coi unit/e2e pass là bằng chứng chất lượng khoa học.

Bắt đầu pilot real-source với SME trước khi huấn luyện controller. Chỉ train khi logs cho thấy failure có thể học được và reward đáng tin:
- Đầu tiên có thể train/rerank evidence relevance hoặc measurement extraction.
- Sau đó distill action selection từ trajectories được kiểm.
- RL là hướng nghiên cứu riêng khi có môi trường và split đủ tốt; reward không chỉ là tự đánh giá của agent.
- Giữ unseen compounds/measurements/documents và task families ngoài training.

## 15. Những phần mình để sau

- **Swarm chuyên gia cố định:** chỉ thêm khi độc lập context hoặc breadth thật sự tạo lợi ích.
- **Bayesian fusion score activity–toxicity:** cần định nghĩa predictive target, data và calibration riêng.
- **Autonomous self-improving skills:** kỹ năng có thể được đề xuất từ failures, nhưng phải review/version/evaluate trước dùng rộng.
- **Full RL agent:** Biomni-R0 là tiền lệ đáng theo dõi, không chứng minh training luôn là bước tiếp theo phù hợp.
- **Thay runtime framework:** ưu tiên chức năng thiếu và compatibility thay vì một migration lớn.
- **General-purpose biomedical environment:** mở rộng theo use cases đã kiểm, không dựa vào số lượng tools.

Đây là thứ tự đầu tư cho phạm vi hiện tại, không phải kết luận rằng các hướng trên không có giá trị.

## 16. Sản phẩm cuối nên đem lại gì?

Một assessment nên hiển thị:
1. Model và policy đã thực sự báo gì.
2. Evidence thực nghiệm nào liên quan trực tiếp.
3. Điểm thống nhất/bất đồng và điều kiện so sánh.
4. Phần chưa biết và dữ liệu nào có thể giải quyết.
5. Thay đổi so với assessment trước.
6. Nguồn và phép tính có thể mở để kiểm.

Các nhãn “proposed”, “supported within scope”, “needs reassessment” cần phục vụ việc hiểu nội dung, không đẩy thuật ngữ implementation vào flow của nhà nghiên cứu.

Thước đo sản phẩm nên gồm time-to-verified-answer, số thao tác kiểm nguồn, lỗi nghiêm trọng và cost per successful task. “Agent viết dài và nghe như chuyên gia” không phải tiêu chí đủ.

## 17. Nguồn và mức độ đọc

### Industry

- **[I1] Novo Nordisk / Anthropic.** [Novo Nordisk accelerates clinical documentation and drug development with Claude](https://claude.com/customers/novo-nordisk). Đọc case study; các lợi ích là customer/vendor-reported. Không dùng headline draft generation như thời gian toàn bộ approval.
- **[I2] LangChain / Replit.** [Replit Agent case study](https://www.langchain.com/breakoutagents/replit). Đọc cognitive architecture, context, restricted DSL và evaluation. Là case study về một phiên bản sản phẩm, không khẳng định kiến trúc hiện tại hoàn toàn không đổi.
- **[I3] Anthropic.** [Effective context engineering for AI agents](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents). Đọc nội dung context/tool design; không dùng các số tiết kiệm làm forecast cho ToxAgent.
- **[I4] Anthropic.** [Introducing advanced tool use on the Claude Developer Platform](https://www.anthropic.com/research/advanced-tool-use). Đọc tool discovery, programmatic calls, examples và applicability. Features vendor-specific có thể được hiện thực bằng backend abstraction khác.
- **[I5] Temporal.** [Agent memory and state](https://docs.temporal.io/ai/featured/memory-state-stripe). Đọc phân biệt context/memory/state, retry và idempotency. Không mặc định ToxAgent cần Temporal.
- **[I6] LangChain.** [The Runtime Behind Production Deep Agents](https://www.langchain.com/blog/runtime-behind-production-deep-agents). Dùng khái niệm harness/runtime và runtime capabilities, không xem là benchmark agent intelligence.

### Academic và scientific systems

- **[A1] Huang et al., Science (2026).** *Autonomous biomedical research with an artificial intelligence agent.* DOI: [10.1126/science.adz4351](https://doi.org/10.1126/science.adz4351). [Publication metadata/abstract](https://pubmed.ncbi.nlm.nih.gov/42424436/). Method details được đối chiếu với [preprint 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12157518/), [author PDF](https://www-cs-faculty.stanford.edu/people/jure/pubs/biomni-biorxiv25.pdf) và [official code](https://github.com/snap-stanford/Biomni). Publisher full text bị chặn; không khẳng định mọi chi tiết implementation của preprint trùng bản journal.
- **[A2] Li et al., technical report/model preview (2025).** *Biomni-R0.* [Model card](https://huggingface.co/biomni/Biomni-R0-32B-Preview), [report link](https://biomni.stanford.edu/blog/biomni-r0-technical-report). Model card/official README truy xuất được; report page không đọc được. Chỉ dùng thông tin RL/environment ở mức xác minh đó, không viện dẫn SOTA như kết quả đã tái lập.
- **[A3] Asai et al., Nature (2026).** *Synthesizing scientific literature with retrieval-augmented language models.* DOI: [10.1038/s41586-025-10072-4](https://www.nature.com/articles/s41586-025-10072-4). Đọc phương pháp ở review trước và truy xuất lại self-feedback/citation verification.
- **[A4] Doyle, Artificial Intelligence (1979).** *A truth maintenance system.* [Author bibliography](https://groups.csail.mit.edu/medg/people/doyle/publications/), [publisher](https://www.sciencedirect.com/science/article/abs/pii/0004370279900080). Dùng làm tiền lệ về lý do/dependencies cho belief maintenance; không tuyên bố ToxAgent tái lập đầy đủ TMS.
- **[A5] Skarlinski et al., preprint (2024).** [Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/html/2409.13740v2). PaperQA2/ContraCrow đã được đọc ở review trước; iterative retrieval và contradiction detection là tiền lệ.
- **[A6] Wadden et al., EMNLP (2020).** [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/). Căn cứ cho claim–evidence–rationale.
- **[A7] Combs et al., Frontiers in Toxicology (2026).** [ToxPipe: integrating AI into toxicoinformatics](https://www.frontiersin.org/journals/toxicology/articles/10.3389/ftox.2026.1907845/full). Đã đọc ở review trước. Kết quả lợi ích không nhất quán trong thiết lập của họ củng cố nhu cầu kiểm chứng từng cơ chế.

Toàn bộ đề xuất lựa chọn route, schema mở rộng, procedures, worker contracts và rollout là **tổng hợp thiết kế cho ToxAgent**, không được gán thành kiến trúc nguyên bản của bất kỳ nguồn nào.



