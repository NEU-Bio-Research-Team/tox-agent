# Kế hoạch triển khai P1/P2 cho Report, Explainer và UI

Ngày lập: 2026-09-09  
Tài liệu nguồn: `docs/REPORT_EXPLAINER_UI_HANDOFF_VI.md`  
Phạm vi: các hạng mục P1/P2 còn lại sau khi XAI-01, XAI-02, REP-01,
REP-02, REP-03 và UI-01 đã hoàn tất.

## 1. Kết luận điều hành

P1 và P2 nên được triển khai thành hai release nối tiếp, trên cùng một contract
artifact mới:

- **Release P1 — Scientific report v3:** bổ sung ma trận screening theo endpoint,
  applicability panel, concordance/conflict có cấu trúc, threshold context,
  đề xuất thí nghiệm có basis và UX điều hướng report dài.
- **Release P2 — Report governance:** bổ sung lineage/history, semantic diff,
  export manifest, telemetry có kiểm soát cardinality, retention/deletion và bộ
  regression fixture đóng băng.

Ước lượng tổng là **64–77 person-days**, tương đương khoảng **7–9 tuần lịch với
hai kỹ sư** làm song song theo các lane backend/report và frontend/platform,
chưa tính một tuần thu telemetry alpha trước khi khóa SLO. Không nên gộp thành
một PR lớn; plan dưới đây chia thành 13 PR có exit gate riêng.

Quyết định kiến trúc quan trọng nhất: các bảng quan trọng về khoa học không được
chỉ tồn tại dưới dạng `ReportTable.rows` do agent viết. Chúng phải là typed data
trong immutable `ReportArtifact`, được compiler dựng hoặc validator kiểm, rồi
mọi renderer cùng đọc. Nếu không, history/diff của P2 sẽ chỉ so sánh chuỗi và
không thể chứng minh numeric fidelity.

## 2. Hiện trạng và phạm vi thực tế

### 2.1 Nền đã có và sẽ tái sử dụng

- `ReportArtifact` là immutable document, có `content_sha256`, `version` và
  `supersedes_report_id`.
- `report_artifacts`, `report_renderings`, `report_figures`, claim/evidence link
  tables và object store đã tồn tại.
- Artifact v2 đã snapshot references và đưa snapshot vào content hash.
- Renderers Markdown, Markdown bundle, HTML và PDF dùng cùng artifact; figure có
  hash và endpoint delivery đã kiểm scope.
- UI đã có toggle summary/full, copy/open citation, badge gap có deep link,
  figure skeleton, top contributors, method text, evidence quality metadata,
  human-readable provenance và print rules ở mức nền.
- Predictor chỉ có `element_rules_v1`: đây là applicability guard theo element,
  **không phải learned OOD detector**. Payload cũng đã nói rõ
  `similarity_domain: null` và `uncertainty: null`.

### 2.2 Những gì còn thiếu thật sự

| ID | Hạng mục | Thiếu hiện tại |
|---|---|---|
| P1-SCI-01 | Endpoint screening matrix | Chưa có typed summary dựng từ observation |
| P1-SCI-02 | Applicability panel | Chưa có presentation riêng; dễ bị gọi nhầm là OOD score |
| P1-SCI-03 | Concordance | `EvidenceSynthesis` có relation nhưng chưa có summary model-vs-literature theo target và chưa có `mixed` |
| P1-SCI-04 | Conflict | Có `conflict_id` dạng string nhưng chưa có conflict object, grouping rule hoặc section renderer |
| P1-SCI-05 | Threshold/method context | Có dữ liệu rải rác nhưng chưa đóng thành typed, human-readable contract |
| P1-SCI-06 | Next experiments | Recommendation đã có nhưng category/priority còn free text và chưa có scientific constraints đủ mạnh |
| P1-UX-01 | Điều hướng dài | TOC hiện là danh sách link, chưa sticky/active section/mobile behavior |
| P1-UX-02 | Summary/full semantics | Toggle hiện chủ yếu ẩn technical detail; chưa định nghĩa tập nội dung bắt buộc của summary |
| P1-UX-03 | Back to claim | Reference chưa chỉ ngược tới mọi claim/synthesis đã dùng nó |
| P2-GOV-01 | History/lineage | Storage đã có link/version, chưa có chain validation và API/UI history |
| P2-GOV-02 | Semantic diff | Chưa có diff schema/service/API/UI |
| P2-GOV-03 | Export manifest | Chưa có manifest chuẩn và checksum list trong bundle |
| P2-OPS-01 | Telemetry | Có event/outbox và tool-call record, chưa có metric contract/exporter/dashboard |
| P2-OPS-02 | Retention | Có retention class/`expires_at`, chưa có policy mapping, expiry worker, tombstone/audit và restore/delete test |
| P2-QA-01 | Regression fixture | Chưa có một fixture report xuyên suốt đủ signed XAI, evidence conflict và gap |

### 2.3 Ngoài phạm vi

- Không tạo score hoặc verdict tổng hợp “safe/unsafe”.
- Không cộng số assay Tox21 dương thành severity.
- Không tuyên bố molecule “in distribution” khi element guard trả `ok`.
- Không xây learned OOD model trong epic này. Nếu làm sau, nó là model artifact,
  eval và release riêng; P1 chỉ dành chỗ cho `similarity_domain` nullable.
- Không re-fetch nguồn khi mở report, xem history hoặc diff.
- Không cho phép diff hay retention làm thay đổi artifact cũ.
- Không đặt SLO/alert threshold cuối cùng trước khi có dữ liệu alpha.

## 3. Các quyết định cần khóa trước khi code

Plan dùng các mặc định dưới đây để không chặn thiết kế. Product/security có thể
đổi ở PR-RP0, nhưng sau khi merge contract v3 thì thay đổi phải đi qua schema
version mới.

| Mã | Quyết định đề xuất | Lý do |
|---|---|---|
| D-P12-01 | Tên UI là **Endpoint screening matrix**, không phải “overall risk” | Tránh biến nhiều endpoint độc lập thành một verdict |
| D-P12-02 | Panel ghi **Applicability (rule-based)**; “OOD” chỉ xuất hiện trong note giải thích giới hạn | Hệ thống hiện chưa có learned OOD detector |
| D-P12-03 | Bổ sung `mixed`, giữ `contextualizes` | `mixed` là tổng hợp có cả support và contradiction; `contextualizes` là nguồn liên quan nhưng không kiểm trực tiếp prediction |
| D-P12-04 | P1 bump artifact thành `toxagent-report-v3`; v1/v2 tiếp tục read-only | Các field mới nằm trong content hash và cần reader compatibility rõ ràng |
| D-P12-05 | Diff là derived artifact có schema/hash riêng, không sửa hai report nguồn | Giữ tính bất biến và cho phép nâng diff algorithm độc lập |
| D-P12-06 | Summary và Full dùng đúng một artifact; summary không gọi LLM và không có câu kết luận riêng | Tránh hai phiên bản nội dung bất đồng |
| D-P12-07 | Retention duration là config theo environment; bảng ở mục 8 là default đề xuất | Thời hạn cuối cần owner pháp lý/security xác nhận |
| D-P12-08 | Telemetry không chứa SMILES, prose, title, URL, evidence excerpt, atom list hoặc user id thô | Giảm rủi ro dữ liệu nhạy cảm và cardinality |

## 4. Contract đích: `toxagent-report-v3`

### 4.1 Nguyên tắc migration

- Writer chỉ ghi v3 sau khi rollout; reader nhận v1/v2/v3.
- Không backfill hoặc mutate JSON của artifact cũ.
- UI có adapter: field v3 vắng trên v1/v2 thì render nội dung cũ, kèm nhãn
  “không có structured summary ở phiên bản artifact này”.
- `content_sha256` v3 bao gồm mọi field khoa học mới; `renderings`, history và
  diff vẫn nằm ngoài hash của report nguồn.
- Bump đồng bộ `SCHEMA_VERSION`, `SCHEMA_VERSIONS`, compiler version, wire
  schema, API TypeScript union, golden fixtures và renderer versions khi output
  thay đổi.

### 4.2 Các type mới

Tên field cuối cùng được khóa ở PR-RP0; shape mục tiêu:

```json
{
  "schema_version": "toxagent-report-v3",
  "endpoint_assessments": [
    {
      "endpoint": "herg",
      "task": null,
      "prediction_observation_id": "obs_...",
      "probability_field_path": "predictions.herg.probability",
      "probability": 0.82,
      "rendered_probability": "82.0%",
      "threshold": 0.50,
      "threshold_source": "artifact",
      "label": "blocker",
      "applicability_status": "ok",
      "explanation_status": "completed",
      "explanation_id": "exp_...",
      "evidence_relation": "mixed",
      "evidence_synthesis_ids": ["syn_..."]
    }
  ],
  "applicability_summary": {
    "status": "ok",
    "method": "element_rules_v1",
    "reasons": ["..."],
    "is_learned_ood": false,
    "similarity_domain": null,
    "uncertainty": null,
    "observation_id": "obs_..."
  },
  "concordance": [
    {
      "endpoint": "herg",
      "task": null,
      "relation_to_prediction": "mixed",
      "supporting_synthesis_ids": ["syn_..."],
      "contradicting_synthesis_ids": ["syn_..."],
      "contextualizing_synthesis_ids": [],
      "evidence_ids": ["evd_..."],
      "summary": "..."
    }
  ],
  "evidence_conflicts": [
    {
      "conflict_id": "cnf_...",
      "endpoint": "herg",
      "task": null,
      "summary": "...",
      "supporting_synthesis_ids": ["syn_..."],
      "contradicting_synthesis_ids": ["syn_..."],
      "differentiators": [
        {"dimension": "dose", "detail": "..."},
        {"dimension": "organism", "detail": "..."}
      ],
      "resolved": false
    }
  ]
}
```

Không lưu `overall_risk`, `risk_score`, `assay_hit_count` hoặc `safe` trong
contract.

### 4.3 Ownership của dữ liệu

| Dữ liệu | Owner tạo | Validator |
|---|---|---|
| Probability, threshold, label, threshold source | Compiler đọc lại observation | Exact field-path/numeric/classification fidelity |
| Applicability status/method/reasons | Compiler đọc prediction observation | Method allowlist và wording limitation |
| Explanation status/id/method metadata | Compiler resolve explanation observation | Endpoint/task/model/artifact/alignment linkage |
| Evidence item relation và context | Agent draft | Read-before-cite, accepted evidence, target/context validity |
| Concordance aggregate | Compiler derive từ accepted synthesis; agent chỉ viết summary có basis | Relation truth table và complete target coverage |
| Conflict differentiators/summary | Agent draft | Hai phía tồn tại, IDs resolve, context không bị bỏ |
| Recommendations/next experiments | Agent draft | Enum, basis claims, wording/scope policy |

### 4.4 Relation truth table

Cho từng `(endpoint, task)`:

| Evidence đã validate | `relation_to_prediction` |
|---|---|
| Có support, không contradiction | `supports` |
| Có contradiction, không support | `contradicts` |
| Có cả hai | `mixed` |
| Chỉ có contextual evidence | `contextualizes` |
| Không có evidence trực tiếp phù hợp | `insufficient` |

`mixed` bắt buộc có ít nhất một `EvidenceConflict`. `insufficient` không được
đổi thành support chỉ vì có nguồn cùng tên compound nhưng khác endpoint.

## 5. P1 — Scientific quality

### P1-SCI-01 — Endpoint screening matrix

**Implementation**

1. Thêm `EndpointAssessment` vào domain và artifact v3.
2. Viết deterministic projector từ analysis observations theo từng endpoint và
   từng Tox21 task được chọn. Không nhận probability/threshold/label từ draft.
3. Resolve explanation status và concordance sau khi synthesis đã validate.
4. Tạo table/card từ typed assessments trong Markdown/HTML/PDF/UI. Các renderer
   không tự tính lại label hoặc relation.
5. Sắp xếp theo request order; Tox21 theo `task_order_version`, không theo score.
6. Threshold override phải có badge cảnh báo và hiển thị cả source; không gọi
   override là model default.

**Acceptance**

- Mỗi selected-and-served target có đúng một row; selected-but-unserved là gap.
- Probability, threshold và label khớp observation 100%.
- Không có aggregate row/verdict và không sort theo “nguy hiểm nhất”.
- Hỗ trợ partial/failed explanation và insufficient evidence như giá trị thật.
- Bốn renderer có cùng số row, thứ tự và canonical values.

**Tests**

- Unit projector cho ClinTox, hERG, nhiều Tox21 task và threshold override.
- Validator test duplicate/missing target và altered value.
- Renderer golden + React accessibility table test.
- Property test: đổi thứ tự input không đổi task order chuẩn.

**Estimate:** 5–6 person-days. Phụ thuộc PR-RP0.

### P1-SCI-02 — Applicability panel đúng nghĩa

**Implementation**

1. Thêm `ApplicabilitySummary` do compiler đọc từ prediction observation.
2. Panel nằm ngay sau matrix, hiển thị status, method và toàn bộ reasons.
3. Với `element_rules_v1`, luôn hiển thị câu cố định: kết quả `ok` chỉ cho biết
   không có element bị rule đánh dấu; không chứng minh similarity với training
   distribution.
4. `similarity_domain` và `uncertainty` hiện “không được model cung cấp”, không
   biến mất và không mặc định thành low risk.
5. Dành discriminator cho learned method tương lai nhưng không mở UI claim trước
   khi có artifact/model card và eval tương ứng.

**Acceptance**

- `ok`, `limited`, `out_of_domain` render khác nhau nhưng không ánh xạ sang
  “safe/unsafe”.
- Không dùng nhãn “Low OOD risk” cho element rule.
- Missing method/reasons ở v3 bị validator từ chối.
- Panel nhất quán trong UI/Markdown/HTML/PDF.

**Tests:** three-status golden, missing/unknown method, wording policy eval,
screen-reader label.  
**Estimate:** 3 person-days. Phụ thuộc P1-SCI-01.

### P1-SCI-03 — Concordance model vs literature

**Implementation**

1. Làm rõ relation là relation của external evidence với prediction tại đúng
   `(endpoint, task)`, không phải “paper tốt/xấu”.
2. Mở wire contract cho evidence item có target và
   `relation_to_prediction`; reader v2 map relation cũ ở chế độ compatibility.
3. Compiler group các item đã validate và áp truth table mục 4.4.
4. Bảng concordance hiển thị relation, organism, assay, dose/exposure,
   source-quality tier, peer-review status nếu provider thật sự có field, và
   retrieval date. Giá trị không biết phải là `unknown`, không suy đoán.
5. Citation marker trong từng row dẫn tới reference snapshot nội bộ.

**Acceptance**

- Mọi selected target có row, kể cả `insufficient`.
- Một source khác endpoint/assay không thể trực tiếp support/contradict target.
- `mixed` được compiler derive, agent không thể tự gắn khi chỉ có một phía.
- Quality badge là metadata, không tự động đổi trọng số hoặc relation.

**Tests:** support-only, contradiction-only, mixed, contextual-only,
insufficient, context mismatch, rejected/unread evidence, v2 adapter.  
**Estimate:** 5–6 person-days. Phụ thuộc PR-RP0 và P1-SCI-01.

### P1-SCI-04 — Conflict section

**Implementation**

1. Thêm `EvidenceConflictCandidate` và compiled `EvidenceConflict`.
2. Conflict phải group ít nhất một synthesis support và một synthesis
   contradict cùng target.
3. Differentiator dùng enum đóng: `assay`, `endpoint_definition`, `organism`,
   `dose`, `exposure_duration`, `route`, `study_design`, `quality`, `other`.
4. Renderer trình bày hai phía song song, sau đó mới tới differentiators và
   trạng thái unresolved. Không tạo đoạn văn “chọn bên thắng” nếu basis không
   đủ.
5. Nếu concordance là `mixed` nhưng draft thiếu conflict, trả typed violation
   cho correction attempt.

**Acceptance**

- Không có conflict orphan hoặc conflict chỉ có một phía.
- Mọi evidence/synthesis id resolve trong cùng session/report.
- Assay/dose/species mismatch không bị rút gọn mất trong summary.
- Report với mixed evidence vẫn có endpoint-level conclusion thận trọng hoặc
  gap/limitation rõ ràng.

**Tests:** grouping validator, cross-target/cross-session rejection, renderer
golden, prompt-injection content retained as inert text.  
**Estimate:** 4–5 person-days. Phụ thuộc P1-SCI-03.

### P1-SCI-05 — Method note và threshold context

**Implementation**

1. Enrich `ExplanationPackage` v3 với server-owned metadata: target class,
   `model_id`, model artifact hashes, attribution method/version,
   alignment/atom-order version, numeric payload hash, renderer/palette version
   và completion time/duration nếu có.
2. Compiler lấy metadata từ explanation observation/provenance; draft chỉ tham
   chiếu `explanation_id`.
3. Matrix và explanation card hiển thị threshold value + source. Override dùng
   badge/callout nổi bật ở mọi renderer.
4. Human-readable note đứng trước raw hashes; raw data vẫn trong `<details>` ở
   UI/HTML và appendix trong Markdown/PDF.

**Acceptance**

- Không thể gắn method note của model/task khác.
- Override không thể trông giống artifact default.
- Numeric attribution thành công nhưng figure lỗi vẫn giữ method/numeric hash.
- UI và export nêu đúng target class đang được giải thích.

**Tests:** linkage/hash mismatch, override rendering, partial figure, legacy
package fallback.  
**Estimate:** 3–4 person-days. Phụ thuộc XAI-01 hiện có và PR-RP0.

### P1-SCI-06 — Recommended next experiments

**Implementation**

1. Đóng enum `action_category`: `confirmatory_assay`, `counter_screen`,
   `dose_response`, `replicate`, `orthogonal_method`, `applicability_check`,
   `evidence_review`.
2. Đóng enum priority `high|medium|low`; priority thể hiện thứ tự xác minh, không
   phải severity của compound.
3. Bắt buộc `basis_claim_ids`, `rationale`, conditions và target endpoint/task
   khi recommendation liên quan một endpoint.
4. Preflight skill yêu cầu mỗi recommendation giải thích uncertainty/gap/conflict
   nào nó xử lý; validator từ chối diagnosis, clinical dose hoặc safety promise.
5. UI group theo priority, có back-link tới basis claims.

**Acceptance**

- Không recommendation không basis hoặc category tùy ý.
- Conflict quan trọng có follow-up phù hợp hoặc lý do rõ vì sao chưa đề xuất.
- Không suy ra liều thực nghiệm cụ thể nếu evidence không cung cấp basis.
- Summary chỉ hiển thị recommendation high/medium; Full hiển thị tất cả.

**Tests:** enum/schema, invalid basis, unsafe wording eval, endpoint linkage,
summary/full parity.  
**Estimate:** 3–4 person-days. Phụ thuộc P1-SCI-03/04.

## 6. P1 — UX

### P1-UX-01 — Sticky TOC và active section

- Tách `ReportNavigation` khỏi `ReportBlock`.
- Desktop: sticky trong biên report, không sticky theo toàn transcript; đặt
  `top` theo app header và giới hạn chiều cao/overflow.
- Mobile/tablet: collapsible “Mục lục”; không chiếm cột nội dung cố định.
- Dùng `IntersectionObserver` với root phù hợp scroll container, cập nhật active
  section và `aria-current="location"`.
- Hash/deep-link phải giữ nguyên stable section ID; `scroll-margin-top` tính cả
  sticky header.
- Nếu `IntersectionObserver` không có, navigation link vẫn hoạt động.

**Acceptance/tests:** desktop/tablet/mobile component tests, keyboard navigation,
history/hash behavior, active-section integration test và Playwright visual khi
server thật ổn định.  
**Estimate:** 3 person-days.

### P1-UX-02 — Summary/Full có semantic rõ

Summary không phải “ẩn ngẫu nhiên phần technical”. Định nghĩa:

- luôn có title/status/gaps, endpoint matrix, applicability, executive summary,
  concordance summary, conclusions, high/medium recommendations, limitations và
  citations đang được nội dung summary dùng;
- ẩn contributor detail, claim field paths, raw provenance, low-priority
  follow-up và full conflict context;
- Full chứa toàn bộ 11 section và mọi structured item;
- chuyển view không fetch/rebuild report và không đổi content hash;
- query parameter `report_view=summary|full` cho deep link, default theo product
  hiện tại là Full để không bất ngờ làm mất chi tiết.

**Acceptance/tests:** cùng artifact/canonical value ở hai view, không citation
orphan trong summary, state qua reload/back-forward, accessibility của toggle.  
**Estimate:** 2–3 person-days. Phụ thuộc các component P1-SCI.

### P1-UX-03 — Back to claim và trạng thái tải

- Gắn stable DOM id cho claim và synthesis.
- Reference snapshot tính danh sách backlinks từ artifact tại render time;
  một reference có thể quay lại nhiều claim, không chỉ một.
- Thêm “Quay lại phát biểu” cạnh source, hỗ trợ keyboard và focus restoration.
- Giữ figure skeleton hiện có. Reference nằm trong report JSON nên không tạo
  network skeleton giả; thay vào đó dùng report-shell skeleton khi artifact
  đang tải và inline warning khi reference không resolve.

**Acceptance/tests:** one-to-many backlinks, inline-token and claim citation,
focus restored, no-JS/hash fallback, unresolved reference visible.  
**Estimate:** 2 person-days.

### P1 items đã có — chỉ harden, không làm lại

- Evidence quality badge/copy/open citation.
- Top contributor table và signed legend.
- Basic method text và human-readable provenance.
- Gap badge có link.
- Figure skeleton.
- Print stylesheet/page-break nền.

Trong P1 chỉ bổ sung test parity và sửa nếu contract v3 làm chúng regress.

## 7. P2 — Versioning, diff và export integrity

### P2-GOV-01 — Lineage/history

**Storage invariants**

- Một report có tối đa một parent trực tiếp.
- Parent phải cùng session và cùng logical subject; analysis có thể giống hoặc
  mới hơn theo rebuild request.
- `version = parent.version + 1`; report gốc là 1.
- Không cho cycle, self-link, fork ngầm hoặc version trùng trong cùng lineage.
- Version allocation phải atomic; thêm lineage/root id hoặc unique constraint
  phù hợp để tránh hai build đồng thời cùng nhận version.

Migration đề xuất thêm `lineage_root_id` và `subject_key_sha256` vào
`report_artifacts`, backfill mỗi artifact cũ thành root của chính nó nếu không
có chuỗi parent đủ tin cậy, rồi thêm unique constraint
`(session_id, lineage_root_id, version)`. Rebuild endpoint resolve root/subject
server-side từ parent; client và agent không được tự khai hai field này.

Hiện `latest_version_for_analysis()` chỉ lấy `max(version)` và
`supersedes_report_id` lấy từ `stage_state`; cách này chưa đủ cho rebuild dựa
trên analysis mới và có race. PR này thay bằng repository operation khóa/allocate
lineage trong transaction.

**API**

```text
GET /v1/sessions/{session_id}/reports/{report_id}/history
POST /v1/sessions/{session_id}/reports/{report_id}:rebuild
```

History trả metadata gọn, newest-first, có report/artifact/model hashes, analysis
id, status, created time và changed-dimensions summary. Rebuild request ghi parent
explicit, không nhận parent từ agent draft.

**Acceptance/tests:** linear chain, rebuild với analysis mới, concurrent rebuild,
cross-session parent, deleted/expired parent, authorization và pagination.  
**Estimate:** 5–6 person-days.

### P2-GOV-02 — Semantic diff

Diff service so sánh structured artifacts, không diff HTML/PDF hoặc prose thuần.

```text
GET /v1/sessions/{session_id}/reports/{report_id}/diff?base_report_id=rpt_...
```

`ReportDiff v1` gồm:

- identity: base/target id, version, artifact hash, schema;
- analysis/model: analysis hash, model id/artifact hashes, predictor/compiler
  version thay đổi;
- endpoint: added/removed target, probability old/new/delta, threshold/source,
  label, applicability, explanation status/hash;
- evidence: reference added/removed/metadata-changed, relation/concordance/conflict
  changed;
- conclusions/recommendations/gaps: added/removed/changed theo stable semantic
  key và basis IDs;
- figures/renderers: content hash hoặc renderer version changed;
- prose: optional bounded text diff để đọc, không dùng làm source cho scientific
  change classification.

Diff response có `schema_version`, `algorithm_version` và `content_sha256` riêng.
Adapter normalize v1/v2 sang comparison model; field không có là `unknown`, không
được coi là `unchanged`.

**UI:** history drawer + chọn hai version + nhóm thay đổi; numeric delta không
dùng màu đỏ/xanh như verdict và luôn hiện old/new.  
**Acceptance/tests:** no-change rebuild, model-only, threshold-only,
evidence/conflict-only, renderer-only, v2→v3, large report bound, reversed base,
cross-session denial.  
**Estimate:** 6–7 person-days. Phụ thuộc P2-GOV-01 và artifact v3.

### P2-GOV-03 — Export manifest

Thêm manifest hai lớp để tránh dependency vòng:

- `ArtifactManifest v1` là deterministic, được nhúng vào export và chứa report,
  analysis, reference và figure integrity;
- `DeliveryManifest v1` do endpoint trả sau khi render, bao `ArtifactManifest`
  và thêm metadata/checksum của các rendering đã persist.

`report-manifest-v1.json` nhúng trong export gồm:

- report id/version/schema/content hash, lineage parent;
- analysis id/hash;
- compiler và renderer versions;
- từng figure: id, media type, byte size, SHA-256, renderer version,
  endpoint/task;
- từng reference snapshot: evidence id/number và deterministic metadata hash;

Quy tắc tránh self-reference:

- manifest bên trong một bundle **không** chứa checksum của chính archive chứa
  nó;
- checksum rendering được trả ở API metadata/`ETag`/`Digest` và trong
  `DeliveryManifest` từ manifest endpoint, không nhét ngược vào file rồi hash
  lại;
- `markdown_bundle` chứa `report.md`, `figures/` và
  `report-manifest-v1.json`;
- HTML embed manifest JSON inert (`application/json`); PDF ghi report hash ngắn
  ở metadata/footer và tải full manifest qua endpoint;
- manifest generation là deterministic từ artifact + persisted metadata và
  không fetch external source.

```text
GET /v1/sessions/{session_id}/reports/{report_id}/manifest
```

**Acceptance/tests:** verify script kiểm toàn bộ figure bytes, tamper detection,
deterministic JSON order, unsafe filename/path traversal, missing/expired object,
bundle golden.  
**Estimate:** 3–4 person-days. Phụ thuộc P1 artifact ổn định.

## 8. P2 — Telemetry và retention

### P2-OPS-01 — Telemetry contract

Ưu tiên OpenTelemetry-compatible instruments sau đây; tên cuối được namespace
theo convention hiện tại của deployment:

| Metric | Type | Attributes hữu hạn |
|---|---|---|
| `report.build.duration` | histogram | outcome, stage |
| `report.build.completed` | counter | status, has_gaps |
| `report.validation.failure` | counter | violation_code, attempt |
| `report.explanation.cache` | counter | endpoint, outcome=hit/miss |
| `report.explanation.duration` | histogram | endpoint, status, reason_code |
| `report.figure.render.failure` | counter | endpoint, reason_code |
| `report.citation.unresolved` | counter | provider, reason_code |
| `report.export.duration` | histogram | format, outcome |
| `report.export.failure` | counter | format, reason_code |
| `report.diff.duration` | histogram | outcome, schema_pair |
| `report.retention.action` | counter | entity_kind, action, outcome |

Tracing span boundaries: build, explanation resolve/cache, evidence resolve,
validation, compile, each render, persistence, diff. Trace/log correlation dùng
opaque run/report/build id trong logs có access control; metrics không dùng các
ID này làm labels.

**Privacy/cardinality gate**

- Không record SMILES, substance name, prose, URL, evidence title/excerpt,
  contributor list, owner id hoặc session/report id trong metric attributes.
- Endpoint/task chỉ dùng allowlist; free-form reason map về closed reason code.
- Sampling/export failure không được làm fail report build.
- Event/outbox là audit/product event, không thay cho metrics; metrics không là
  source of truth cho history.

**Dashboard/alerts**

- Dashboard: build outcome/duration, explanation hit ratio/latency, figure and
  export failure, unresolved citations, gap reasons.
- Thu tối thiểu một tuần alpha rồi mới khóa SLO/p95 và alert thresholds.
- Trước alpha chỉ alert invariant breach: artifact/hash corruption,
  cross-scope denial anomaly và sustained zero-success với có traffic.

**Tests:** in-memory metric exporter, exact once/attempt semantics, redaction
test, bounded-cardinality test, failure injection từng stage.  
**Estimate:** 5–6 person-days.

### P2-OPS-02 — Retention/deletion policy

#### Default đề xuất cần security/product xác nhận

| Data class | Retention class | Default đề xuất | Khi hết hạn |
|---|---|---|---|
| Canonical report JSON + reference snapshot | `audit` | Theo lifecycle session, đề xuất 365 ngày sau archive | Tombstone metadata tối thiểu, report không còn đọc được |
| Explanation/structure figure được report dùng | `audit` | Ít nhất bằng report dài nhất tham chiếu nó | Xóa chỉ khi không còn report sống tham chiếu |
| Download rendering | `session` hoặc derived | 90 ngày; có thể regenerate khi canonical artifact + figure còn | Xóa object/row hoặc đánh expired; regenerate có audit event |
| Accepted normalized evidence metadata/excerpt | `audit` | Ít nhất bằng report đang snapshot/cite | Report giữ snapshot; raw provider payload theo dòng dưới |
| Raw provider payload | `transient` | Không lưu nếu không cần; nếu lưu, đề xuất 7 ngày | Hard-delete object, giữ hash/reason nếu policy cho phép |
| Telemetry operational | tách khỏi attachment class | Đề xuất 30 ngày | Aggregate hoặc delete theo backend telemetry |
| Deletion audit/tombstone | audit tối thiểu | Đề xuất 365 ngày, không chứa nội dung đã xóa | Giữ entity kind/hash/action/time/policy version |

Con số trên là operational default, không phải tư vấn pháp lý. PR retention
không được merge production trước khi có owner phê duyệt duration, legal hold,
user delete semantics và backup expiry.

#### Implementation

1. Viết versioned `RetentionPolicy` mapping entity/media/purpose → class,
   duration, legal-hold behavior và deletion order.
2. Gán `expires_at` khi tạo attachment/rendering; hiện figure mặc định `AUDIT`
   nhưng duration chưa được enforce.
3. Xây reference graph report → figure attachment/rendering/evidence snapshot để
   tránh xóa object còn được artifact sống tham chiếu.
4. Worker claim theo batch/idempotent: mark expired → delete object → delete hoặc
   tombstone row → emit audit event. Retry an toàn khi object đã không còn.
5. API trả typed `artifact_expired`, không dùng 404 để ngầm tiết lộ existence;
   authorization vẫn chạy trước retention state.
6. Session delete/archive gọi cùng lifecycle service, không tự cascade object
   store mà bỏ orphan.
7. Backup policy phải bảo đảm deletion propagation và có restore drill chứng
   minh object hết hạn không sống lại ngoài policy.

**Acceptance/tests**

- Clock-controlled expiry cho từng class.
- Shared figure không bị xóa khi còn một report sống tham chiếu.
- Idempotent retry sau DB/object-store failure ở từng bước.
- Legal hold ngăn deletion nhưng vẫn audit attempted action.
- Cross-session caller không phân biệt expired với nonexistent.
- Restore/deletion drill có biên bản; orphan scanner về 0 ở fixture.

**Estimate:** 8–10 person-days. Phụ thuộc quyết định policy, P2-GOV-01 và
P2-OPS-01.

## 9. P2-QA-01 — Frozen regression fixture

Tạo fixture hoàn toàn offline, không phụ thuộc provider/model live:

- một canonical molecule có attribution dương và âm;
- ít nhất hai endpoint, trong đó một target có `mixed` evidence;
- external evidence gồm primary + database/regulatory metadata, một URL HTTPS
  và một source không có link an toàn;
- organism/assay/dose mismatch tạo conflict differentiator;
- một explanation/figure thành công và một gap có reason code cụ thể;
- một threshold dùng artifact, một threshold override;
- recommendations có basis;
- hai report versions để test semantic diff;
- figure SVG cố định và hashes được pin.

Artifacts cần commit:

```text
backend/control/evals/fixtures/report-v3/
  source-observations.json
  evidence-records.json
  report-draft.json
  report-artifact.golden.json
  report-artifact-v2.golden.json
  report-diff.golden.json
  figures/*.svg
  expected-manifest.json
  expected-report.md
frontend/src/components/transcript/__fixtures__/report-v3.json
```

Không pin raw PDF bytes nếu renderer có metadata/timestamp không deterministic;
thay bằng text/layout assertions, page count bound và visual snapshots ở pinned
container. Markdown/HTML/manifest phải golden deterministic.

**CI lanes**

- Fast PR: compiler/validator/unit/React, Markdown+HTML+manifest golden.
- Integration PR: PostgreSQL/object store, auth scope, history/diff, retention
  clock/failure injection.
- Pinned rendering: PDF + SVG visual snapshots.
- Nightly/alpha: Playwright desktop/tablet/mobile và optional live provider;
  live result không thay golden fixture.

**Estimate:** 4–5 person-days, làm dần cùng các PR chứ không để cuối.

## 10. Trình tự PR và dependency

| PR | Nội dung | Phụ thuộc | Exit gate | Estimate |
|---|---|---|---|---:|
| PR-RP0 | ADR + v3 wire/domain skeleton + compatibility readers | P0 hiện có | Contract/golden skeleton được review | 3d |
| PR-RP1 | Deterministic endpoint assessments + threshold context | RP0 | Numeric/classification fidelity 100% | 5–6d |
| PR-RP2 | Applicability panel | RP1 | Không có learned-OOD claim sai | 3d |
| PR-RP3 | Concordance + relation truth table | RP0, RP1 | 5 relation outcomes pass | 5–6d |
| PR-RP4 | Conflict objects/validator/renderers | RP3 | Mixed luôn có visible conflict | 4–5d |
| PR-RP5 | Method metadata + next experiments | RP1, RP3 | Basis/linkage policy pass | 6–8d |
| PR-RP6 | Sticky TOC + semantic summary/full + backlinks | RP2–RP5 | Responsive/a11y component tests pass | 7–8d |
| PR-RP7 | Atomic lineage/history/rebuild | RP0 | Concurrent lineage test pass | 5–6d |
| PR-RP8 | Semantic diff service/API/UI | RP7, RP1–RP5 | v2→v3 and no-change diff pass | 6–7d |
| PR-RP9 | Export manifest + verification CLI/test | RP0, RP1 | Tamper test pass | 3–4d |
| PR-RP10 | Telemetry contract/export/dashboard seed | RP1, RP7 | Redaction/cardinality gate pass | 5–6d |
| PR-RP11 | Retention worker/policy/delete/restore drill | RP7, RP10 | Lifecycle failure-injection suite pass | 8–10d |
| PR-RP12 | Frozen fixture + CI lane consolidation | RP1–RP11 tăng dần | Toàn bộ golden/integration/visual lanes pass | 4–5d |

RP1/RP2 và RP3 có thể chạy song song sau RP0. RP7/RP9 có thể bắt đầu khi v3
contract ổn định; RP10 instrument dần các service khi chúng merge. RP11 đứng sau
lineage/reference graph và quyết định retention.

### Milestone đề xuất

1. **M0 — Contract freeze (tuần 1):** RP0, product khóa D-P12-01..08.
2. **M1 — Scientific vertical slice (tuần 2–3):** RP1–RP4 trên frozen fixture.
3. **M2 — P1 complete (tuần 4):** RP5–RP6, parity bốn renderer, SME review.
4. **M3 — Governance core (tuần 4–5):** RP7–RP9.
5. **M4 — Operations hardening (tuần 6–7):** RP10–RP11, staging drills.
6. **M5 — Alpha observation (ít nhất 1 tuần):** đo baseline, sau đó mới chốt
   SLO/alert và retention production.

## 11. File/module dự kiến tác động

Backend/control:

- `domain/report.py`: v3 types, enums, content hash.
- `validation/report_wire.py`, `validation/report_validator.py`: candidates và
  deterministic gates.
- `report/compiler.py`: projector assessment/applicability/concordance.
- `report/renderers.py`: matrix, panel, conflicts, method note, manifest.
- `application/submit_report_draft.py`: compile/render instrumentation và
  lineage allocation.
- module mới `report/diff.py`, `report/manifest.py`, `retention/` và
  `telemetry/report_metrics.py` (tên cuối theo package convention).
- `persistence/schema.py`, migration, interfaces/repositories: lineage,
  rendering expiry/tombstone nếu cần.
- `api/routes.py`, `api/schemas.py`: history, rebuild, diff, manifest.
- `agent_profiles/report_build/`: compose/research/preflight instructions và
  profile hash bump.

Frontend:

- `lib/api/types.ts`, `lib/api/endpoints.ts`.
- `components/transcript/ReportBlock.tsx` được tách nhỏ hơn.
- component mới/riêng cho EndpointMatrix, ApplicabilityPanel,
  ConcordanceTable, ConflictSection, ReportNavigation, ReportHistory và
  ReportDiff.

Tests/evals/docs:

- report v3 frozen fixture và golden renderers.
- integration tests cho persistence/API/auth/object-store/retention.
- Playwright viewport matrix và visual snapshots.
- ADR report v3, ADR retention, runbook expiry/restore và metric catalog.

## 12. Release, migration và rollback

### Deploy order

1. Deploy backend reader hỗ trợ v3 nhưng writer vẫn v2.
2. Chạy migration additive và verify indexes/constraints.
3. Deploy frontend hiểu v1/v2/v3.
4. Bật v3 writer bằng feature flag cho internal/staging.
5. Chạy frozen fixture, restart/rehydration, export verification và SME review.
6. Bật history/diff/manifest.
7. Bật telemetry; quan sát trước khi alert.
8. Chạy retention dry-run chỉ report count/bytes; review sample; sau đó mới bật
   delete worker.

### Rollback

- Tắt v3 writer không làm mất report v3 đã tạo; backend reader vẫn phải giữ v3.
- Tắt UI components mới thì fallback về structured sections/tables hiện tại.
- Diff/manifest là derived, có thể disable độc lập.
- Retention worker có kill switch và dry-run; không rollback bằng cách phục hồi
  object đã xóa nếu policy yêu cầu deletion.
- Không downgrade DB bằng migration phá dữ liệu trong incident; migration P1/P2
  phải additive trước giai đoạn cleanup riêng.

## 13. Definition of Done

### P1 complete

- Artifact v3 có endpoint assessments, applicability và concordance typed,
  hashed, persisted và rehydrated đúng.
- Mỗi selected target có matrix/concordance outcome hoặc gap.
- Numeric/threshold/label fidelity và citation/read-before-cite đạt 100%.
- Mixed evidence luôn có conflict section với assay/dose/organism context.
- Applicability wording không biến element whitelist thành learned OOD claim.
- Recommendations có enum, target và valid basis claims.
- Summary/Full dùng cùng artifact; sticky TOC/backlinks dùng được bằng keyboard.
- UI, Markdown, bundle, HTML và PDF cùng nội dung khoa học cốt lõi.
- Không có aggregate safety verdict.

### P2 complete

- Rebuild tạo lineage atomic, old report không bị mutate.
- History và v1 diff trả kết quả scoped/authenticated, hỗ trợ v2→v3.
- Manifest xác minh được report hash và mọi figure checksum; tamper bị phát hiện.
- Metrics có redaction/cardinality tests và dashboard alpha.
- Retention policy được owner phê duyệt, worker qua dry-run/failure injection,
  legal hold/delete/restore drill.
- Frozen fixture chạy ở fast, integration, rendering và Playwright lanes.
- Runbook release/rollback/expiry và ADR được cập nhật.

## 14. Các gate bắt buộc trước production

1. Product xác nhận D-P12-01..07; security/privacy xác nhận D-P12-08.
2. Scientific reviewer duyệt wording applicability, concordance/conflict và
   recommendation policy trên ít nhất các scenario 4, 8, 9 của evaluation
   matrix hiện có.
3. Migration/concurrent rebuild test trên PostgreSQL thật.
4. Export manifest verify qua object store thật; PDF chạy bằng renderer pinned.
5. Playwright desktop/tablet/mobile chạy với server thật, không chỉ component
   test; có baseline visual được review.
6. Retention dry-run và restore/delete drill hoàn tất trước khi bật deletion.
7. Có ít nhất một tuần alpha telemetry trước khi khóa SLO và alert threshold.
