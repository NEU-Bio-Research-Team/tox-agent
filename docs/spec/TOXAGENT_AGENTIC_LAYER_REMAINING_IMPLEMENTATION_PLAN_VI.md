# ToxAgent Agentic Layer — Kế hoạch triển khai toàn bộ phần còn lại

> **Ngày lập:** 2026-09-05  
> **Trạng thái:** kế hoạch thực thi, chưa phải báo cáo tiến độ  
> **Nguồn contract:** [TOXAGENT_AGENTIC_LAYER_REBUILD_PLAN_VI.md](TOXAGENT_AGENTIC_LAYER_REBUILD_PLAN_VI.md)  
> **Nguồn trạng thái:** [TOXAGENT_AGENTIC_LAYER_PROGRESS_VI.md](TOXAGENT_AGENTIC_LAYER_PROGRESS_VI.md)

## 1. Mục đích và cách dùng tài liệu

Tài liệu này chuyển toàn bộ khoảng trống còn lại của agentic layer thành một
backlog có thứ tự phụ thuộc, checklist, đầu ra kiểm chứng được và exit gate. Nó
không thay đổi các contract khoa học hay kiến trúc đã chốt trong rebuild plan.

Ba tài liệu có vai trò khác nhau:

- `REBUILD_PLAN`: contract và Definition of Done dài hạn.
- `PROGRESS`: bằng chứng đã làm và kết quả chạy thực tế.
- File này: thứ tự triển khai từ trạng thái hiện tại tới internal alpha và
  production candidate.

Khi một checkbox hoàn tất, phải ghi bằng chứng vào `PROGRESS` (commit, test,
manifest, run ID hoặc runbook). Không đánh dấu xong chỉ vì code đã tồn tại; exit
gate tương ứng phải được chạy.

## 2. Baseline đã xác minh khi lập kế hoạch

### 2.1 Đã có và không xây lại

- Control plane sở hữu session/message/run/analysis/observation/evidence/answer,
  SQL store và transactional outbox.
- Lane D phân tích deterministic qua ToxPred; Lane A/Mixed chạy qua
  `AgentRuntimeGateway`.
- Tool registry, MCP capability token, deny-all OpenCode profile, validator,
  correction loop và deterministic fallback.
- OpenCode `1.17.11` chạy live; report Q&A, runtime-loss recovery và evidence
  EuropePMC đã được xác nhận thật.
- Bộ eval 50 task, frozen fixtures, graders, scripted driver và remote OpenCode
  driver.
- Workspace frontend ba vùng, REST + SSE, answer renderer, analysis/run/audit
  artifact và ba cách nhập cấu trúc: SMILES, ảnh, vẽ 2D.
- `toxocr` dùng MolScribe như deployable riêng; ảnh → SMILES → ToxPred đã chạy
  live.

### 2.2 Khoảng trống quan sát trực tiếp trong source

- `potentially_billed` có trong domain/schema/API nhưng không có đường code nào
  đặt thành `true`.
- `runtime.usage.reported` được adapter nhận diện nhưng gateway chưa persist hay
  tổng hợp token/cost; package `telemetry/` chưa có implementation.
- Attribution có implementation và integration test nhưng chưa có live gate.
- SSE reconnect, abort/cancel và một số contract OpenCode mới chỉ được xác minh
  bằng mock hoặc test cục bộ, chưa qua failure injection live đầy đủ.
- Object-store interface/implementation chưa tồn tại; image bytes của luồng OCR
  chỉ nằm trong memory của worker; `EvidenceRecord.raw_payload_ref` luôn `None`.
- Backend có SQLAlchemy/Alembic và hỗ trợ dialect PostgreSQL, nhưng chưa có
  deployment/test matrix PostgreSQL thật.
- Auth alpha là static bearer token. Backend có JWT HMAC cơ bản nhưng chưa có
  OIDC/JWKS production flow; frontend vẫn yêu cầu người dùng dán token.
- CI hiện tại chỉ phủ ToxPred; chưa có job bắt buộc cho control plane, frontend,
  toxocr, eval và multi-service smoke.
- Frontend chưa có automated test suite. Build hiện đạt, nhưng chunk
  `WorkbenchPage` khoảng 1,65 MB trước gzip và cần budget/code splitting.
- Test control plane không chạy trong shell Python mặc định vì thiếu
  `sqlalchemy`; cần chuẩn hoá môi trường chạy thay vì coi đó là lỗi sản phẩm.
- Lần chạy `toxocr/tests` trong shell hiện tại không trả output sau khoảng 90
  giây và đã được dừng; cần tách contract test nhẹ khỏi model smoke có checkpoint.

### 2.3 Đính chính trạng thái DSH

Mục “Phase 4 bị chặn vì chưa có carrier chính chủ” trong progress doc đã lỗi
thời tại ngày lập file này. DeepSeek hiện phát hành:

- [`deepseek-harness-sdk`](https://pypi.org/project/deepseek-harness-sdk/) cho
  Python; package kéo đúng phiên bản `deepseek-harness-runtime-bin`.
- Runtime wheel chính chủ, mang binary và profile SDK, giao tiếp JSON-RPC theo
  dòng qua stdio. Nguồn chính thức mô tả tại
  [DeepSeek Harness Python SDK](https://github.com/deepseek-ai/deepseek-harness/blob/master/python/README.md)
  và [SDK protocol](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/sdk/protocol/README.md).

Bản phân phối vẫn là pre-release, protocol chưa có prompt cancel/session close,
và SDK profile mặc định có tool coding không phù hợp ToxAgent. Vì vậy Phase 4
không còn bị chặn ở bước tìm carrier, nhưng chỉ được mở sau một spike cô lập và
phải dùng custom profile deny-all.

## 3. Nguyên tắc triển khai bắt buộc

- Không thay đổi nghĩa khoa học của ToxPred và không thêm aggregate
  toxicity/safety verdict ở bất kỳ tầng nào.
- Mọi số/classification được chấp nhận phải truy ngược tới observation chính
  xác; evidence không được dùng thay số của predictor.
- Model text chỉ trở thành product answer sau `submit_grounded_answer` và
  validator.
- Runtime không sở hữu product state, user auth hoặc tool authorization.
- Mọi tool authority được kiểm tra hai lần: model-visible surface và MCP
  transport.
- Fix eval phải được phân loại rõ là product, prompt/policy, grader, task spec,
  provider variance hay hạ tầng. Không nới gate chỉ để tăng pass rate.
- Live test có model/provider phải ghi manifest, chi phí, thời gian và dependency
  version; không chạy ẩn trong CI của mọi commit.
- Mỗi migration phải có upgrade test; không dùng `create_all` cho deployment.
- Internal alpha dùng OpenCode làm primary. DSH chỉ trở thành runtime được hỗ
  trợ sau khi qua cùng contract/eval gate.

## 4. Bản đồ workstream và thứ tự phụ thuộc

| ID | Workstream | Ưu tiên | Phụ thuộc | Kết quả chính |
|---|---|---:|---|---|
| W0 | Khoá baseline và đồng bộ trạng thái | P0 | — | Một commit/test baseline tái lập được |
| W1 | Đóng eval và chất lượng agent | P0 | W0 | Pass@1 ổn định, critical `pass^3=100%` |
| W2 | Runtime, recovery, cancel và streaming | P0 | W0 | Failure injection live đạt |
| W3 | Attribution và evidence quality | P0 | W0, một phần W1 | Citation/support gate đạt alpha |
| W4 | Persistence, object store và data lifecycle | P0 | W0 | PostgreSQL + blob lifecycle chạy thật |
| W5 | Hoàn thiện product UI | P0 | W2–W4 contracts ổn định | UI qua reload/reconnect/recovery |
| W6 | Telemetry, CI/CD và deploy topology | P0 | W0; song song W1–W5 | Stack build/deploy/quan sát được |
| W7 | DSH conformance runtime | P1 | W0, W1 harness ổn định | DSH pass hoặc được ghi unsupported có bằng chứng |
| W8 | Internal alpha và SME loop | P0 | W1–W6 | Alpha gate + feedback thành eval |
| W9 | Production hardening | P0 sau alpha | W8; W7 theo quyết định runtime | Production candidate + rollback |
| W10 | Product backlog sau alpha | P2 | W8 | Session management và UX bổ sung |

Critical path dự kiến:

```text
W0 → {W1, W2, W3, W4, W6} → W5 → W8 → W9
          └──────── W7 chạy song song sau khi eval contract ổn định ───────┘
```

## 5. W0 — Khoá baseline và đồng bộ trạng thái

### 5.1 Workspace và commit hiện tại

- [ ] W0-01 Chụp `git status`, commit SHA, dependency lock và danh sách thay đổi
  đang dở; không trộn thay đổi ngoài agentic-layer vào PR.
- [ ] W0-02 Hoàn tất lát UI/OCR đang uncommitted: review diff, chạy contract
  tests, frontend build/policy lint và một live smoke cho cả ba input.
- [ ] W0-03 Commit lát UI/OCR riêng với ADR 0006 và cập nhật progress; chỉ push
  khi quy trình repository yêu cầu.
- [ ] W0-04 Chuẩn hoá lệnh test theo từng environment (`toxagent-control`,
  `toxocr`, ToxPred, frontend); thêm script không phụ thuộc shell đang active.
- [ ] W0-05 Ghi baseline mới: số test, build sizes, 35-task pass@1 gần nhất,
  critical failures, latency và fallback rate.

### 5.2 Đồng bộ tài liệu

- [ ] W0-06 Sửa bảng trạng thái đầu `PROGRESS`: Phase 3 đã commit, eval không
  rỗng, Phase 5 core đã live, Phase 6/UI đã triển khai một phần đáng kể.
- [ ] W0-07 Sửa mục DSH: carrier chính chủ đã xuất hiện; link SDK/runtime/protocol
  và chuyển blocker thành quyết định pin pre-release.
- [ ] W0-08 Lập decision table hiện hành cho DEC-01…DEC-10; mỗi dòng phải là
  `accepted`, `pending` hoặc `superseded`, kèm ADR/source.
- [ ] W0-09 Version bộ eval hiện tại trước khi thêm task OCR/production, tránh
  thay đổi mẫu số của baseline mà không ghi nhận.

**Exit gate W0**

- [ ] Working tree của lát hiện tại đã được review và baseline có thể chạy bằng
  một nhóm lệnh được ghi trong runbook.
- [ ] Không còn mâu thuẫn lớn giữa bảng trạng thái và nhật ký mới nhất.

## 6. W1 — Đóng eval và chất lượng agent

### 6.1 Tách ba chế độ fixture đúng contract

- [ ] W1-01 Thêm `fixture_mode` rõ trong task/manifest:
  `frozen`, `predictor_integration`, `live_evidence`; không suy chế độ chỉ từ
  nội dung expectation.
- [ ] W1-02 Xây đường chạy **agentic + frozen predictor** để 12 task
  `numeric_fidelity` dùng model thật nhưng số nguồn vẫn cố định. Có thể chạy
  control plane local với frozen ToxPred adapter và OpenCode thật; không đổi
  expected number theo predictor live.
- [ ] W1-03 Giữ `predictor_integration` cho semantic/wording gates trên ToxPred
  thật, đồng thời ghi artifact hashes từ `/v1/models` vào manifest.
- [ ] W1-04 Giữ `live_evidence` riêng vì kết quả nguồn thay đổi theo thời gian;
  lưu query, provider, retrieval time và content hashes để audit.
- [ ] W1-05 Mỗi task không chạy được phải có `skipped_reason` typed; task bị skip
  không được tính pass hay fail.

### 6.2 Đóng sáu failure còn lại

- [ ] W1-06 Chạy lại đầy đủ 35 task live-compatible sau toàn bộ tám fix ở
  progress §3.13, lưu manifest/result nguyên vẹn.
- [ ] W1-07 `adv-05-ignore-the-limitations`: xác định vì sao thiếu
  `uncalibrated_probability`; fix prompt/limitation derivation nếu product sai,
  giữ nguyên hard gate vì đây là task critical.
- [ ] W1-08 `qa-06-attribution-request`: xác định limitation bị mất ở projection,
  prompt, candidate hay validator; chạy lại qua attribution thật.
- [ ] W1-09 `evsyn-03-conflicting-evidence`: thay exact-word grader bằng semantic
  condition có tính phủ định/xung đột nhưng vẫn deterministic nếu có thể; thêm
  positive và negative counterexamples trước khi đổi task.
- [ ] W1-10 `evsyn-05-no-evidence-found`: grade hành vi “không tìm thấy” theo
  nghĩa và citation count, không khoá vào một cụm tiếng Anh duy nhất.
- [ ] W1-11 `numeric-07` và `qa-02`: quyết định `kind=comparison` có thật sự là
  contract bắt buộc. Nếu có, làm rõ tool description/prompt và validator; nếu
  không, grade graph nguồn và phép so sánh thay vì enum do model chọn.
- [ ] W1-12 Với mỗi fix, thêm regression task/test nhỏ nhất tái hiện nguyên nhân;
  không thêm regex rộng thiếu negative cases.

### 6.3 Gate lặp lại và báo cáo chất lượng

- [ ] W1-13 Khi pass@1 ổn định, chạy toàn bộ critical set ba trial độc lập trên
  cùng manifest family; yêu cầu `pass^3=100%`.
- [ ] W1-14 Chạy numeric fidelity bằng frozen-agentic mode; yêu cầu 100% exact
  source/rounding/rendered-value gate.
- [ ] W1-15 Báo cáo theo category: pass@1, pass^3, first-candidate acceptance,
  fallback rate, tool calls, deadline failures, latency, token và cost.
- [ ] W1-16 Thêm regression comparison với baseline; CI fail khi hard gate giảm,
  còn live quality regression được đưa vào release review thay vì chạy mỗi PR.
- [ ] W1-17 Thêm suite OCR/structure-recognition version mới: ảnh hợp lệ, MIME
  giả, base64 lỗi, quá kích thước, OCR unavailable, OCR không nhận diện, SMILES
  OCR không hợp lệ và success tạo analysis đúng.

**Exit gate W1**

- [ ] 35 task live-compatible có một baseline sạch sau cùng một code revision.
- [ ] Critical `pass^3=100%`; numeric claim fidelity 100%; unsupported critical
  claims bằng 0.
- [ ] Mọi failure còn lại có owner/category và quyết định rõ, không còn “chưa
  biết là bug hay variance”.

## 7. W2 — Runtime, recovery, cancel và streaming

### 7.1 Hoàn tất OpenCode contract suite

- [ ] W2-01 Live test direct denied call: shell/edit/subagent/direct web vừa
  không hiện trên surface vừa bị transport từ chối.
- [ ] W2-02 Live test abort một turn đang chạy; chỉ render “đã huỷ” sau khi run
  thật sự thành `cancelled`.
- [ ] W2-03 Live test event stream disconnect/reconnect với
  `after_sequence`, duplicate delivery và sequence gap.
- [ ] W2-04 Live test runtime restart làm binding cũ thành `lost`, tạo đúng một
  recovery run và không nối transcript runtime cũ.
- [ ] W2-05 Snapshot/diff OpenCode OpenAPI trong CI. Mọi version bump phải tạo
  diff được review và chạy lại surface/cancel/recovery contract.

### 7.2 Startup reconciliation và failure injection

- [ ] W2-06 Thêm startup reconciler cho run `queued/running/validating` còn lại
  sau control-plane crash. Mỗi run phải đi tới terminal/recovery có audit event,
  không nằm treo vô hạn.
- [ ] W2-07 Persist đủ recovery input hoặc recovery plan để restart control
  plane không phụ thuộc `RunContext` chỉ tồn tại trong memory.
- [ ] W2-08 Đảm bảo deterministic observation đã commit được reuse; không gọi
  lại predictor/provider khi retry nếu idempotency key/source graph đã tồn tại.
- [ ] W2-09 Viết failure-injection orchestrator điều khiển các service độc lập:
  kill runtime trước request, sau tool call, sau accepted candidate; kill control
  plane; treo event stream; provider timeout; DB reconnect.
- [ ] W2-10 Chạy lại `fail-04/05/06` và `adv-04` bằng orchestrator thay vì loại
  khỏi live-compatible rate.
- [ ] W2-11 Quét/reap runtime workspace và orphan process cả khi shutdown sạch
  lẫn khi process cha chết; có soak test khẳng định orphan count bằng 0.

### 7.3 Billing/usage semantics

- [ ] W2-12 Chốt ngữ nghĩa `potentially_billed`: chỉ bật cho run thất bại/hủy
  sau khi runtime đã nhận provider turn mà charge outcome không xác định.
- [x] W2-13 Persist normalized usage events theo run/provider/model, gồm token
  fields runtime thật cung cấp; không bịa số còn thiếu.
- [x] W2-14 Khi usage/cost không có, lưu `unknown` thay vì `0`; API/UI phân biệt
  “không tốn” và “không biết”.
- [ ] W2-15 Test các ranh giới: fail trước send = không potentially billed; receipt
  accepted rồi mất runtime = potentially billed; completed có usage = usage
  audit; recovery có usage riêng, không cộng trùng.

**Exit gate W2**

- [ ] Toàn bộ OpenCode contract tests có bằng chứng live ở một pinned version.
- [ ] Restart runtime/control plane, cancel và SSE reconnect đều tái dựng đúng
  state; không duplicate answer/provider call ngoài policy.
- [ ] Orphan process/workspace bằng 0 trong soak; billing uncertainty hiển thị
  trung thực.

## 8. W3 — Attribution và evidence quality

### 8.1 Attribution end-to-end

- [ ] W3-01 Chạy live `request_attribution` cho hERG và một assay Tox21; ghi
  latency, model artifact hash và observation projection.
- [ ] W3-02 Xác minh attribution chỉ cho một endpoint/task, không biến token
  importance thành causal mechanism hay aggregate toxicity.
- [ ] W3-03 Đảm bảo answer bắt buộc có `attribution_not_causality`, claim source
  trỏ đúng attribution observation và numeric value vẫn trỏ predictor source.
- [ ] W3-04 Thêm live/contract cases: assay thiếu, endpoint unavailable, timeout,
  partial attribution và cache hit.

### 8.2 Evidence provider robustness

- [ ] W3-05 Contract test EuropePMC search/detail normalization bằng captured,
  redacted fixtures; kiểm stable identifier, canonical URL, dedupe và hash.
- [ ] W3-06 Failure cases: 429, timeout, malformed payload, empty result, duplicate,
  disallowed host và provider circuit breaker/backoff.
- [ ] W3-07 Citation validator yêu cầu model đã gọi detail/read trước khi cite;
  mọi citation phải là accepted record cùng session/share scope.
- [ ] W3-08 Giữ evidence text trong untrusted projection có delimiter/type;
  adversarial suite phải chứng minh instruction trong title/abstract không tăng
  tool authority hay tạo model-authored URL.
- [ ] W3-09 Chốt DEC-10: metadata + accepted excerpt mặc định; raw payload chỉ
  lưu khi policy yêu cầu và qua object store có TTL/ACL.

### 8.3 Scientific support và SME grading

- [ ] W3-10 Viết rubric citation support tách “URL tồn tại”, “nguồn nói về đúng
  chủ đề”, “nguồn hỗ trợ đúng claim” và “chất lượng nguồn”.
- [ ] W3-11 Hai SME chấm mù tối thiểu 20% capability set và mọi critical
  evidence failure; lưu disagreement/adjudication.
- [ ] W3-12 Chuyển disagreement lặp lại thành task/rubric version mới; không sửa
  kết quả cũ.
- [ ] W3-13 Alpha gate: citation validity 100%, scientific citation support
  >=95%, major SME correction <=15%.

**Exit gate W3**

- [ ] Attribution và evidence đều có live runs accepted qua cùng GroundedAnswer
  contract.
- [ ] Prompt injection không mở rộng authority; citation/support đạt alpha gate.

## 9. W4 — Persistence, object store và data lifecycle

### 9.1 PostgreSQL và multi-instance correctness

- [x] W4-01 Dựng PostgreSQL ephemeral trong integration CI; chạy Alembic từ DB
  rỗng, toàn bộ repository/integration tests và schema constraint checks.
- [x] W4-02 Test transaction giữa domain mutation và outbox, monotonic sequence,
  idempotency keys, unique accepted answer và claim-source foreign keys trên
  PostgreSQL thật.
- [x] W4-03 Thay admission/concurrent-run guard chỉ trong memory bằng cơ chế
  đúng khi có nhiều control-plane instance (DB constraint/lock có bounded
  retry); test hai instance cùng nhận request.
- [x] W4-04 Test cross-instance: instance A ghi event, instance B phục vụ REST/SSE
  reconcile mà không mất state.
- [x] W4-05 Viết [migration policy/runbook](../runbooks/TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md):
  forward-only production, pre-deploy migrate, backup trước migration,
  compatibility window khi rolling deploy.

### 9.2 Object store và attachment

- [x] W4-06 Tạo `ObjectStore` interface (`put/get/delete/signed_read_ref`) và fake
  filesystem/in-memory cho test; production adapter ưu tiên GCS vì deployment
  hiện tại ở GCP, nhưng application không import SDK GCS trực tiếp.
- [x] W4-07 Upload ảnh: persist bytes trước khi queue OCR run; message dùng
  `attachment_id/image_ref`, worker/recovery đọc qua owner/session ACL.
- [x] W4-08 Verify MIME bằng magic bytes, giới hạn kích thước sau base64 decode,
  hash nội dung, chặn SVG/HTML/polyglot không được hỗ trợ và không auto-serve
  user payload inline.
- [ ] W4-09 Lưu raw evidence/provider payload theo DEC-10 khi bật; trả
  `raw_payload_ref` opaque, không trả object URI/credential cho model hoặc user
  không có role auditor.
- [ ] W4-10 TTL cleanup idempotent cho transient upload/raw payload; DB row và
  object không bị orphan khi một phía delete lỗi.

### 9.3 Retention, deletion và restore

- [ ] W4-11 Chốt DEC-04 theo class `transient/session/audit`; policy là config
  versioned, không hard-code trong handler.
- [ ] W4-12 Implement session deletion workflow có tombstone/audit, cascade theo
  policy và object cleanup; API không leak session đã xoá của owner khác.
- [ ] W4-13 Backup/restore PostgreSQL và object store; diễn tập restore vào môi
  trường cô lập rồi kiểm source graph, event sequence và hashes.

**Exit gate W4**

- [ ] PostgreSQL migration/integration và cross-instance tests đạt.
- [ ] Ảnh/raw payload sống qua worker restart, obey ACL/TTL, và restore được.

## 10. W5 — Hoàn thiện product UI

### 10.1 State/reconnect/reload

- [x] W5-01 Biến event handling thành reducer có test: cursor chỉ tăng, dedupe
  `event_id`, phát hiện gap và chạy REST reconcile trước khi nối lại SSE.
- [x] W5-02 Bootstrap từ `GET session/messages/events:list` đủ trang để tái dựng
  analysis-by-run, recovery banners và validation history sau reload; không phụ
  thuộc map chỉ thu được khi browser đang mở.
- [ ] W5-03 Test offline → reconnect, tab sleep/wake, duplicate event, missed
  event, expired token và session switch; chỉ draft/UI preference vào
  `localStorage`.
- [x] W5-04 Thêm pending user-send theo `client_message_id`; không optimistic
  assistant answer hoặc analysis.

### 10.2 Artifact và scientific UX

- [ ] W5-05 Nối evidence artifact/list/detail; claim citation mở đúng record,
  hiện title/authors/source/retrieved-at/excerpt/status và external link đã
  normalize.
- [ ] W5-06 Nối attribution viewer cho một endpoint/assay; hiển thị token/atom
  contribution trung tính và limitation không-causality ngang hàng nội dung.
- [ ] W5-07 Deep-link claim → observation/evidence → field path; nếu artifact
  không còn do retention, hiện trạng thái “đã hết hạn” thay vì 404 thô.
- [x] W5-08 Fallback badge cạnh tiêu đề, violations sau nút Chi tiết, limitations
  không collapse mặc định và không dùng màu để suy mức độc.
- [ ] W5-09 Hiển thị cancel/recovery/deadline/predictor unavailable đúng state.
  `requested=true` không đồng nghĩa run đã cancelled; hiện cảnh báo
  `potentially_billed` khi backend cung cấp.
- [ ] W5-10 OCR UI hiển thị file/preview an toàn, progress, recognized SMILES,
  confidence nếu contract cho phép, khả năng sửa SMILES trước một phân tích mới
  và lỗi capability unavailable rõ ràng.

### 10.3 Quality frontend

- [x] W5-11 Thêm Vitest + React Testing Library cho reducer, markdown sanitizer,
  exact rendered-value linking và critical state components.
- [ ] W5-12 Thêm Playwright E2E cho SMILES/ảnh/vẽ, report Q&A, evidence,
  attribution, cancel, recovery, reload/reconnect và permission boundaries.
- [ ] W5-13 Accessibility: keyboard/focus cho dialogs/panel, semantic labels,
  screen-reader announcements cho run status; kiểm desktop/tablet/mobile.
- [x] W5-14 Code-split molecule editor, artifact viewers và route chunks; đặt
  bundle budget, không tải `openchemlib` trước khi người dùng mở editor.
- [ ] W5-15 Không hardcode backend enums có thể mở rộng như violation code; sinh
  hoặc kiểm API types từ OpenAPI trong CI.

**Exit gate W5**

- [ ] Reload/cross-instance/reconnect không làm mất hoặc bịa state.
- [ ] Toàn bộ luồng alpha có browser E2E; policy lint, typecheck, build và bundle
  budget đạt.

## 11. W6 — Telemetry, CI/CD và deployment

### 11.1 Observability

- [ ] W6-01 Structured logs có `request_id/session_id/run_id/binding_id`, nhưng
  redact Authorization, prompt/evidence raw và capability token.
- [ ] W6-02 OpenTelemetry spans qua API → scheduler → runtime → MCP tool →
  predictor/evidence/OCR; trace IDs đi qua service boundary.
- [ ] W6-03 Metrics theo plan §15: product outcome, runtime health/loss/cancel,
  model token/cost, tool latency/error, validator/fallback, evidence yield,
  outbox lag/SSE reconnect và restore failure.
- [ ] W6-04 Dashboard alpha: success/failure theo intent, first-pass acceptance,
  fallback, p50/p95 latency, token/cost, citation support sample và dependency
  readiness.
- [ ] W6-05 Alert ban đầu chỉ cho invariant nghiêm trọng: cross-session leak,
  incomplete claim-source graph, stuck runs, readiness outage, outbox lag và
  orphan process; tinh chỉnh ngưỡng sau dữ liệu alpha.

### 11.2 CI

- [ ] W6-06 Thêm control-plane job: install locked dev deps; unit, contract,
  integration, eval schema/graders và scripted eval.
- [x] W6-07 Thêm frontend job: typecheck, policy lint, unit/component tests,
  production build và bundle budget.
- [x] W6-08 Thêm toxocr contract job không tải checkpoint; model/checkpoint smoke
  chạy riêng theo schedule/manual hoặc runner có artifact cache và timeout.
- [x] W6-09 Thêm PostgreSQL service job + Alembic migration test.
- [ ] W6-10 Thêm container build/smoke cho control plane, frontend và toxocr;
  multi-service smoke dùng predictor stub cho PR và model thật ở protected job.
- [ ] W6-11 Pin dependency/runtime hashes và cache hợp lý; live provider secrets
  chỉ có trong protected manual/release workflow.
- [x] W6-12 Lưu eval manifest/results, test reports và SBOM làm CI artifacts;
  không upload raw secrets/evidence ngoài retention policy.

### 11.3 Deploy topology và runbooks

- [ ] W6-13 Tạo deploy artifacts cho bốn boundary: frontend, control plane,
  ToxPred, toxocr; OpenCode/DSH chạy private runtime host, không expose management
  port public.
- [ ] W6-14 Cấu hình health/live và health/ready đúng dependency semantics;
  startup không báo ready trước predictor/runtime/OCR bắt buộc.
- [ ] W6-15 Cấu hình private MCP URL, egress allowlist cho EuropePMC/provider,
  CORS origin chính xác, secret manager và least-privilege service accounts.
- [ ] W6-16 Viết runbook deploy, migration, smoke, rollback, rotate secret,
  dependency outage, stuck run, orphan cleanup, backup và restore.
- [x] W6-17 Xoá hoặc viết lại runbook Docker legacy đang trỏ
  `model_server/main.py`, `/analyze` và aggregate verdict đã bị loại bỏ.

**Exit gate W6**

- [ ] Mọi package/deployable có CI bắt buộc và image provenance.
- [ ] Staging stack dựng từ đầu bằng runbook, có dashboard/log/trace và smoke
  qua cả bốn boundary.

## 12. W7 — DSH conformance runtime

### 12.1 Spike và DEC-06

- [ ] W7-01 Tạo environment cô lập; cài exact official SDK + matching runtime
  wheel. Tại ngày lập plan, candidate mới nhất quan sát được là pre-release
  `0.1.2rc1`; phải kiểm lại và pin version/hash tại lúc implement.
- [ ] W7-02 Smoke `initialize → session/prompt → session.event/status → shutdown`
  qua Python SDK; capture protocol fixture và binary/package hashes.
- [ ] W7-03 Xác minh platform wheel, startup latency, stderr/stdout purity,
  process reaping, session root và behavior khi provider credential thiếu.
- [ ] W7-04 Chốt ADR DEC-06: version, carrier, supported platforms, upgrade
  policy và developer-preview risk. Không dùng package trùng tên
  `deepseek-harness` của tác giả khác.

### 12.2 Profile và adapter

- [ ] W7-05 Tạo custom `cordis.yml` từ profile SDK tối thiểu: loại bash/file
  edit/subagent/direct web; chỉ nạp MCP client và ToxAgent instructions cần thiết.
- [ ] W7-06 Cô lập `DSH_HOME`, workspace và session root theo run; không discovery
  `~/.dsh`; env allowlist chỉ mang credential/provider cần thiết.
- [ ] W7-07 Capture model-visible surface và chứng minh exact ToxAgent MCP
  allowlist. Direct denied execution phải fail ở transport.
- [ ] W7-08 Implement adapter mỏng theo `AgentRuntimeProvider`: health,
  capabilities, create, send, normalize event/status, close; không đưa DSH type
  vào application/domain.
- [ ] W7-09 Map protocol limitation trung thực: hiện không có mid-turn cancel hay
  session close; `runtime_cancel_supported=false`. Nếu policy chọn kill owned
  process để dừng, action phải nói rõ process termination, không giả là prompt
  cancel.
- [ ] W7-10 Shutdown/timeout luôn reap process và child; stdout chỉ có JSON-RPC,
  diagnostics đọc stderr có bound/redaction.
- [ ] W7-11 Contract snapshot và version diff gate tương tự OpenCode.

### 12.3 Conformance/eval

- [ ] W7-12 Chạy cùng tool contract và deterministic fixtures qua DSH.
- [ ] W7-13 Paired OpenCode/DSH Track A: cùng provider/model, prompt, tools,
  fixtures và budgets; tối thiểu ba trial/release candidate.
- [ ] W7-14 Track B: deployment thực tế, so reliability/latency/cost/ops burden;
  không gán mọi chênh lệch cho harness.
- [ ] W7-15 Scientific observations và accepted source graph phải byte/semantic
  equivalent bất kể runtime; denied tool/cross-session leak bằng 0.

**Exit gate W7**

- [ ] DSH qua same-tool/eval contracts, unsupported cancel được report đúng và
  orphan process bằng 0; hoặc ADR ghi rõ runtime unsupported với lỗi smoke cụ
  thể. Không để adapter giả/mock-only được tính là hoàn thành.

## 13. W8 — Internal alpha và vòng phản hồi SME

### 13.1 Chuẩn bị alpha

- [ ] W8-01 Deploy staging/alpha dùng PostgreSQL, object store, OpenCode pinned,
  EuropePMC và toxocr; static token chỉ được phép vì environment là alpha.
- [ ] W8-02 Chọn tập use case: analysis, report Q&A, attribution, evidence,
  conflicting/no evidence, OCR và recovery; cung cấp sample molecules không có
  dữ liệu nhạy cảm.
- [ ] W8-03 Viết hướng dẫn phạm vi: screening/decision support, không phải chẩn
  đoán, safety assessment hay regulatory decision.
- [ ] W8-04 Bật telemetry/redaction, cost budget theo user/run, feedback form gắn
  session/run/answer ID và cơ chế báo scientific concern.
- [ ] W8-05 Diễn tập dependency outage, cancel, runtime recovery, DB restore và
  rollback trước khi mời reviewer.

### 13.2 Chạy và đóng feedback loop

- [ ] W8-06 Chạy tối thiểu một tuần hoặc đủ sample đã chốt; không chốt SLO từ
  vài smoke run.
- [ ] W8-07 Hai SME review mù tối thiểu 20% capability answers và tất cả critical
  failures/fallbacks.
- [ ] W8-08 Triage feedback trong 48 giờ làm việc thành: bug, eval task, rubric
  clarification, UX issue, provider issue hoặc out-of-scope request.
- [ ] W8-09 Mỗi scientific regression có reproduction fixture/task trước fix;
  rerun affected category và critical set.
- [ ] W8-10 Chốt measured baseline về success, latency, cost, fallback, citation
  support, correction rate, reconnect và restore.

**Exit gate W8**

- [ ] Critical `pass^3=100%`, capability pass@1 >=80%, numeric fidelity 100%,
  citation support >=95%, unsupported critical claim bằng 0, major SME
  correction <=15%.
- [ ] Không mất state qua reload/cross-instance; feedback quan trọng đã thành
  task có owner và priority.

## 14. W9 — Production hardening

### 14.1 Identity, security và credentials

- [ ] W9-01 Chốt DEC-07 và provider terms: server-workload credential, không dùng
  shared personal OAuth/subscription cho production.
- [ ] W9-02 Backend OIDC/JWKS verifier: issuer/audience/expiry/algorithm allowlist,
  key rotation/cache, role mapping và fail-closed; không dùng cùng secret với MCP
  capability token.
- [ ] W9-03 Frontend Authorization Code + PKCE/session flow; token không nhập tay,
  logout/expiry/refresh rõ ràng và không log token.
- [ ] W9-04 Security review: owner/share scope, IDOR, MCP replay/revocation,
  prompt injection, SSRF/egress, file upload, CORS/CSRF assumptions, secret
  redaction và dependency/container scan.
- [ ] W9-05 Abuse controls phân tán: per-user/session concurrency, size/batch/run
  budgets, provider rate limit/circuit breaker và duplicate/cyclic tool detector.

### 14.2 Reliability, SLO và release

- [ ] W9-06 Load test deterministic analysis và mixed workload; đo DB pool,
  outbox lag, SSE connections, runtime host capacity, OCR queue và provider
  throttling.
- [ ] W9-07 Soak/failure injection: predictor 503/malformed/slow, evidence
  429/timeout, OCR hang, runtime disconnect/hung, DB conflict, outbox duplicate,
  object store unavailable và node restart.
- [ ] W9-08 Chốt SLO từ alpha data; alert có owner, severity và runbook. Không
  dùng candidate numbers trong rebuild plan như cam kết trước khi đo.
- [ ] W9-09 Production eval: critical `pass^5=100%`, capability pass@1 >=85%
  và không category <80%, citation support >=98%, numeric source 100%, major SME
  correction <=10%.
- [ ] W9-10 Canary runtime/model/provider/profile/tool/prompt upgrade; manifest
  diff, eval non-regression, rollback tự động hoặc một lệnh đã diễn tập.
- [ ] W9-11 Restore drill từ backup, deletion/retention audit và disaster recovery
  runbook được người không viết runbook thực hiện thành công.
- [ ] W9-12 Production go/no-go review có sign-off từ engineering, security,
  product/SME và owner của provider/credential terms.

**Exit gate W9**

- [ ] Toàn bộ Definition of Done §23 trong rebuild plan có bằng chứng.
- [ ] Security không còn critical finding; SLO/alerts/rollback/retention/delete/
  restore đã diễn tập; primary runtime/provider được chọn bằng dữ liệu.

## 15. W10 — Product backlog sau alpha

Nhóm này không chặn alpha nhưng nằm trong “toàn bộ phần còn lại” của product
backlog hiện có.

- [ ] W10-01 Session search toàn lịch sử bằng API, không chỉ filter 25/50 rows đã
  tải ở frontend.
- [ ] W10-02 Rename/pin/archive/delete session với audit và retention semantics.
- [ ] W10-03 Pagination/cursor ổn định cho session history và artifact lists.
- [ ] W10-04 Export report/audit bundle đã sanitize, gồm manifest và source graph,
  không kèm raw evidence trái policy.
- [ ] W10-05 Expert-only threshold override UI; server role check là authority,
  snapshot hiện `threshold_source=request_override` rõ ràng.
- [ ] W10-06 Đánh giá provider evidence thứ hai chỉ khi alpha cho thấy EuropePMC
  thiếu coverage; phải có ADR và paired normalization/citation tests.

## 16. Ma trận kiểm thử bắt buộc

| Lớp | Mỗi PR | Merge/release | Manual/protected |
|---|---|---|---|
| Unit/domain/validator | Có | Có | — |
| API/MCP/runtime contract mock | Có | Có | — |
| PostgreSQL integration + migration | Có | Có | Restore drill |
| Frontend unit/policy/build | Có | Có | Accessibility review |
| Browser E2E với stub/frozen deps | Có, nhóm critical | Toàn suite | — |
| Scripted eval | Có | Có | — |
| OpenCode frozen-agentic eval | — | Critical/full theo gate | Có credential |
| ToxPred real integration | — | Release | Có artifact runner |
| EuropePMC live evidence | — | Release/sample | Có network/provider budget |
| toxocr checkpoint smoke | — | Release/scheduled | Có checkpoint/CPU-GPU runner |
| Failure injection/soak | — | Trước release | Staging cô lập |
| DSH paired eval | — | Khi runtime candidate đổi | Có SDK/provider credential |
| SME review | — | Alpha/production gate | Hai reviewer mù |

Quy tắc: test live thất bại vì dependency phải mang error category rõ. Không
chuyển nó thành pass, và cũng không gộp vào quality failure của model nếu chưa
phân biệt được.

## 17. Chuỗi PR đề xuất

Mỗi PR dưới đây phải review được độc lập và cập nhật progress ngay sau khi merge:

1. **PR-R0 — Baseline/docs/current UI-OCR slice:** đóng working tree hiện tại,
   run commands và đồng bộ trạng thái.
2. **PR-R1 — Eval modes và reproducibility:** fixture modes,
   frozen-agentic path, manifest/result taxonomy.
3. **PR-R2 — Sáu eval failures:** chia tối đa 2–3 PR theo product/prompt và
   grader/task; không gom một diff lớn khó quy nguyên nhân.
4. **PR-R3 — Runtime reliability:** live contract harness, startup reconciler,
   persisted recovery context, potentially-billed/usage.
5. **PR-R4 — Evidence/attribution closure:** live attribution, provider failure
   policy, citation support rubric.
6. **PR-R5 — PostgreSQL/multi-instance:** migration CI, locks/idempotency,
   cross-instance tests.
7. **PR-R6 — Object store/retention:** attachment persistence, raw evidence,
   cleanup/deletion.
8. **PR-R7 — Frontend state and artifacts:** reconcile reducer,
   evidence/attribution/billing UI, critical tests.
9. **PR-R8 — Frontend performance/E2E:** lazy chunks, bundle budget, Playwright,
   accessibility fixes.
10. **PR-R9 — Observability/CI/deploy:** telemetry, all-package CI, images,
    staging topology và runbooks.
11. **PR-R10 — DSH spike/ADR:** package pin + captured contract; không kèm adapter
    nếu smoke chưa đạt.
12. **PR-R11 — DSH adapter/conformance:** chỉ mở sau PR-R10 đạt.
13. **PR-R12 — Alpha fixes:** mỗi nhóm regression tách theo nguyên nhân.
14. **PR-R13 — Production auth/hardening:** OIDC, distributed limits, security,
    SLO/release gates; có thể tách nhỏ theo owner hạ tầng.

## 18. Decision checklist còn phải chốt

- [ ] D-REM-01 Exact DSH SDK/runtime version và supported platform (DEC-06).
- [ ] D-REM-02 Retention duration cho transient/session/audit (DEC-04).
- [ ] D-REM-03 Raw evidence nào được lưu, trong bao lâu, ai được đọc (DEC-10).
- [ ] D-REM-04 Production object store và encryption/key ownership.
- [ ] D-REM-05 OIDC issuer, audience, role/group claim và admin/expert mapping.
- [ ] D-REM-06 Provider credential/legal topology (DEC-07).
- [ ] D-REM-07 Alpha participant set, SME reviewers và dữ liệu được phép nhập.
- [ ] D-REM-08 SLO/alert thresholds sau khi có một tuần alpha telemetry.
- [ ] D-REM-09 Primary runtime cho production dựa trên paired data; OpenCode giữ
  mặc định cho alpha.
- [ ] D-REM-10 Có cần evidence provider thứ hai hay EuropePMC đủ cho scope alpha.

Các quyết định D-REM-02/03/05/06/07 cần product/security/SME owner; engineering
có thể chuẩn bị interface, test và phương án đề xuất trước khi xin sign-off.

## 19. Ước lượng và khả năng chạy song song

Ước lượng là effort để lập lịch, không thay exit gate:

| Workstream | Ước lượng |
|---|---:|
| W0 | 1–2 engineer-days |
| W1 | 4–7 engineer-days + chi phí live model |
| W2 | 4–7 engineer-days |
| W3 | 4–7 engineer-days + thời gian SME |
| W4 | 6–10 engineer-days |
| W5 | 5–9 engineer-days |
| W6 | 5–9 engineer-days |
| W7 | 4–8 engineer-days nếu official smoke đạt |
| W8 | 2–4 engineer-days + tối thiểu 1 tuần lịch quan sát |
| W9 | 7–12 engineer-days |
| W10 | 4–8 engineer-days, sau alpha |

Một engineer làm tuần tự: khoảng **42–68 engineer-days**, chưa tính thời gian chờ
SME/security/legal và hạ tầng. Hai engineer có thể chạy song song:

- Track A: W1 → W3 → W7.
- Track B: W2 → W4 → W6.
- W5 bắt đầu sau khi event/data contracts của W2–W4 ổn định.
- W8 và W9 hội tụ cả hai track.

## 20. Definition of Done cho toàn bộ kế hoạch

- [ ] Mọi invariant SCI/PROD trong rebuild plan còn nguyên và có regression gate.
- [ ] OpenCode đạt full live contract và critical release eval; DSH đạt conformance
  hoặc được ghi rõ unsupported bằng ADR/bằng chứng thực.
- [ ] Attribution, evidence và OCR đều có live end-to-end runs, typed failures và
  UI tương ứng.
- [ ] Session/run/source graph tái dựng qua runtime loss, control-plane restart,
  SSE gap, cross-instance và restore.
- [ ] PostgreSQL, object store, retention/deletion, OIDC, distributed limits và
  secret boundaries chạy trong production topology.
- [ ] Frontend không làm source of truth, không render nội dung ngoài thiếu
  sanitize, không dùng màu/aggregate để diễn giải độc tính.
- [ ] CI phủ mọi deployable; release workflow lưu manifest/SBOM/eval evidence và
  có rollback đã diễn tập.
- [ ] Alpha và production gates ở rebuild plan §16.10 đạt, có SME/security/product
  sign-off.
- [ ] `PROGRESS` phản ánh trạng thái cuối, không còn checklist mở không có owner
  hoặc quyết định trì hoãn rõ ràng.

## 21. Checklist bắt đầu ngay

Đây là lát đầu tiên nên triển khai, theo đúng critical path:

- [ ] 1. Đóng và commit lát UI/OCR hiện tại; chạy đúng environment cho 563 + 6
  tests, frontend build/policy lint và live three-input smoke.
- [ ] 2. Đồng bộ bảng progress và đính chính DSH carrier.
- [ ] 3. Chạy full 35-task pass@1 trên một revision duy nhất.
- [ ] 4. Root-cause `adv-05` trước vì critical; sau đó năm failure còn lại.
- [ ] 5. Thêm frozen-agentic eval mode và chạy numeric suite với model thật.
- [ ] 6. Dựng failure-injection harness cho cancel/SSE/runtime/control restart.
- [ ] 7. Song song, mở DSH spike chỉ tới contract capture + ADR; chưa viết adapter
  trước khi smoke official carrier đạt.
