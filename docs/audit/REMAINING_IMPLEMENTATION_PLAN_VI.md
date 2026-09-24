# Kế hoạch hoàn thiện toàn bộ backlog ToxAgent — đối chiếu 08/09/2026

## 1. Mục tiêu, baseline và quy tắc đóng việc

Tài liệu này hợp nhất công việc còn lại từ các plan trong workspace với implementation tại `ce49f5d` và `new_plan.md` chưa tracked. Đi cùng [báo cáo 32 issue](SYSTEM_ISSUES_VI.md). Đây là kế hoạch triển khai, không phải thông báo các việc dưới đây đã được sửa.

Mục tiêu cuối là sản phẩm giữ agent layer, có predictor độc lập, OCR, AI provider cấu hình được, model selection thực sự được thực thi, câu trả lời có provenance/citation, hoạt động phiên dễ hiểu, và triển khai có dữ liệu/run bền vững. Hoàn thành giao diện hoặc unit tests riêng không thay thế end-to-end, khoa học và release gates.

Dùng bốn trạng thái: **Đã có** (implementation/evidence đúng phần việc); **Một phần** (còn đường xử lý hoặc integration thiếu); **Chưa đủ bằng chứng** (có báo cáo/code nhưng gate yêu cầu chưa được kiểm chứng tại baseline này); **Thay thế/điều kiện** (yêu cầu cũ không còn là target hoặc chỉ làm khi có quyết định admission). Không xóa backlog chỉ vì thiếu credential/data; ghi dependency và công việc chuẩn bị làm được trước.

Không yêu cầu sửa hay commit các thay đổi của người dùng trong audit. Các PPTX và `audit_5_9.md` đã bị xóa trong worktree không thể dùng làm nguồn hiện hành; không phục hồi chúng để áp lại kiến trúc cũ. Các báo cáo test đầy đủ và giới hạn môi trường nằm ở tài liệu issue. Golden local pass7; control unit còn2fail, contract fixture lỗi và CI collection17errors. Những điểm này chặn dùng test baseline hiện tại làm release sign-off.

## 2. Danh mục nguồn và thẩm quyền

Đường dẫn trong bảng là tương đối từ root repository. Các nguồn cùng nhóm được xét chung khi dùng một contract; không cộng số lượng checkbox thành tỷ lệ hoàn thành sản phẩm.

| Nguồn | Nội dung cần giữ/đối chiếu | Trạng thái và nơi hoàn tất |
|---|---|---|
| `new_plan.md` §0–32, E0–E6 | Product-centric layout, endpoint/model, AI profiles, semantic activity, citations | Target UX/architecture mới nhất; ma trận §4, K01–K08 |
| `docs/spec/TOXAGENT_AGENTIC_LAYER_REBUILD_PLAN_VI.md` | SCI-01…10, PROD-01…10, runtime boundary, phases0–7, gates, quyết định | Giữ scientific/authority invariants; K02–K13; kernel cutover theo unified-v2 |
| `docs/spec/TOXAGENT_AGENTIC_LAYER_REMAINING_IMPLEMENTATION_PLAN_VI.md` | W0–W10 và exit gates | Truy vết từng W-item trong phụ lục; checkbox lịch sử không tự đóng gate hiện tại |
| `docs/spec/TOXAGENT_AGENTIC_LAYER_PROGRESS_VI.md` | Nhật ký live, các đính chính, baseline và triển khai từng thời điểm | Bằng chứng lịch sử; summary đầu file cần đồng bộ. Không cộng các lần chạy hoặc dùng score cũ cho HEAD mới |
| `docs/spec/TOXAGENT_CAPABILITIES_XAI_PRODUCTION_PLAN_VI.md` | P0–P14, bundle, XAI mapping/SVG, evidence, BYOC, scale | Nhiều contract đã có; integration/durability/provider còn thiếu, §5 |
| `docs/spec/TOXAGENT_QUICK_PREDICT_AND_XAI_PLAN.md` | Quick single/batch/OCR, explain API/viewer, policy | Đã có phần lớn UI/API; K03/K04/K06 kiểm model consistency và error matrix |
| `docs/spec/SESSION_AND_QUICK_PREDICT_UI_REFINEMENT_PLAN_VI.md` | UI phases0–7, input studio, mobile, focus/IME, typography | Có studio/lazy components; K03/K08 đóng default behavior, visual/a11y/IME và fonts |
| `docs/spec/TOXAGENT_LANDING_PAGE_RESTORATION_PLAN_VI.md` | Hero, sections, CTA, sticky navigation, motion | Đã có landing; build pass không thay visual/CTA/mobile acceptance; K08 |
| `docs/spec/WORKSPACE_HANDOFF_SIMPLIFICATION_PLAN_VI.md` | Customer clean clone, wrapper, artifact bundle, sanitized handoff | Layout và wrapper một phần; K01/K02/K12, không tạo repo khách hàng trong audit |
| `docs/spec/GPU_AND_ARTIFACT_CACHE_PLAN_VI.md` | CPU/GPU profiles, wheel/model cache, parity/memory/rollback | Có overlay/images local không chứng minh GPU performance; K10/K12 |
| `docs/unified-v2/ARCHITECTURE.md`, `DECISIONS.md`, `MIGRATION.md` | Typed boundary, case/kernel, data roles, relocation | Giữ invariants; code moved nhưng consumer/path/gates còn lỗi, K01/K09 |
| `docs/unified-v2/BASELINE.md`, `IMPLEMENTATION_STATUS.md` | G0–G12 và evidence | Ma trận §6; completion ghi trong doc cũ cần giữ đúng phạm vi lịch sử |
| `docs/runbooks/CICD_AUTO_DEPLOY_MASTER_PLAN.md` | Tách branch test/prod, rollback, auth deployment | Tách môi trường còn bắt buộc; single legacy backend/Firestore/ADK bị thay thế, K12 |
| `docs/runbooks/DEPLOY_FIREBASE_APP_RUNBOOK.md` | Build/deploy/preview/health/debug steps | Nhiều command/path cũ; viết lại cho topology hiện tại và diễn tập staging K12 |
| `docs/runbooks/CHAT_PERSISTENCE_E2E_CHECKLIST.md` | Owner/session/reload/isolation, lịch sử dài | Chuyển acceptance sang SQL/REST/SSE; không khôi phục Firestore architecture; K07/K08 |
| `docs/runbooks/DOCKER_TEST_RUNBOOK.md` | Image/bootstrap/offline/health/full-stack tests | K02/K12; cập nhật artifact provisioning thực, thay command legacy |
| `docs/runbooks/TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md` | Forward-only, one-writer, expand/contract, restore | Runbook có; migration job và rolling-upgrade proof chưa đủ, K07/K12 |
| `docs/runbooks/TOXAGENT_OPERATIONS_RUNBOOK.md` | Secrets rotation, outage, stuck run, backup/drills | Không coi runbook là drill đã chạy; K07/K12 |
| `docs/runbooks/EXPERIMENT_RUNBOOK.md` | Experiment/split/checkpoint reproducibility | Research boundary riêng; cập nhật paths, không nhét training dependency vào serving; K10 |
| `docs/refactor/PREDICTOR_ONLY_STATUS_VI.md` | Endpoint semantics, artifact registry, tokenizer blocker, consumers | Giữ predictor isolation; loại bỏ target xóa agent. ClinTox vẫn K10; consumer audit K01 |
| `docs/archive/toxagent_checklist.md`, `TOXAGENT_V1_0_PROJECT_OVERVIEW.md` | Tool/research/report intent và failure cases lịch sử | ADK multi-agent, global risk, fake clinical hERG, raw JSON fallback không còn target; chuyển intent hợp lệ K03/K11 |
| `docs/archive/XSMILES_PERFORMANCE_BRAINSTORM.md`, `troubleshoot_working_stuck.md` | Ý tưởng scientific/performance và debugging cũ | Không nhận làm performance promise; scientific experiments K10, failure injection K07/K13 |
| `docs/architecture.md`, `ARCHITECTURE.md`, `WORKSPACE_LAYOUT.md` | Service boundaries và index | Hợp nhất crosslinks theo layout thực, K01 |
| `docs/model-card.md`, `MODEL_CARD.md`, `benchmark-protocol.md`, `BM1_explainer_benchmark_analysis.md` | Scientific limitations, frozen split, metrics, XAI benchmark | Giữ source/protocol, không fit trên test; fitted calibration/AD/XAI benchmark còn K10 |
| `docs/artifacts/clintox-smilesgnn-v1.md`, `herg-tox21-chemberta-v1.md` | Model admission/limitations | Giữ explicit blocked/uncalibrated; sửa artifact paths trong handoff K10/K12 |
| `docs/CONFIGURATION.md`, `DEVELOPMENT.md`, `GETTING_STARTED.md`, `OPERATIONS.md`, root/backend READMEs | Supported commands/env/runbook entry points | K01/K02/K12 kiểm command từ cwd sạch |
| `backend/control/evals/README.md`, manifests/tasks; ADRs trong control | Fixture/authority/runtime decisions | Versioned evaluation và conformance K09/K13; không thay rubric để tăng score |
| `docs/slides/README.md` và script build slide | Tài liệu trình bày phụ thuộc architecture | Chỉ cập nhật sau canonical docs; hai deliverable của audit này vẫn chỉ là hai Markdown docs |

## 3. Giải quyết yêu cầu mâu thuẫn trước khi triển khai

1. **Giữ agent layer.** Predictor-only là boundary dependency của service; không xóa control/agent để đạt mục tiêu refactor cũ. Dùng layout `frontend`, `backend/{control,predictor,ocr}`, `devops`, `docs` của new_plan.
2. **Endpoint khác model.** Registry khai báo capability; người dùng được chọn model đã admitted. Không biến mọi `.pt` thành model khả dụng. Auto resolve phải lưu model thực tế; Manual không fallback âm thầm; Compare là mode riêng.
3. **Plain prediction không bắt buộc XAI.** Yêu cầu người dùng hiện tại ưu tiên hơn default required trong plan XAI cũ. Vẫn bảo toàn hợp đồng required khi được chọn rõ; không trả completed giả nếu artifact bắt buộc thiếu.
4. **Product state thuộc control.** Runtime thay thế được nhưng không được sở hữu owner auth, source graph, truth, transaction hay tool authority. Raw runtime text không thành product answer trước validation.
5. **Scientific invariants thắng mỹ thuật cũ.** Không aggregate toxicity/safe verdict, không lấy hERG làm ClinTox; không dùng hit count/severity hoặc attribution/causality. Màu/nhãn phải giữ endpoint semantics; clinical claim không được hợp thức hóa bằng một citation bất kỳ.
6. **SQL hiện hành thay Firestore/ADK persistence.** Giữ test isolation/reload/retention của checklist cũ nhưng viết lại cho database/outbox/REST/SSE hiện tại.
7. **OpenCode primary; DSH/Codex experimental.** Có SDK hoặc adapter skeleton không phải supported. Chỉ promote sau auth/tool surface/cancel/recovery/eval matrix, hoặc ghi ADR unsupported có bằng chứng. Obscura là spike có điều kiện, không dependency bắt buộc của alpha.
8. **Feature code khác gate.** Unit tests, schema, artifact cards, hình giao diện và progress ticks không chứng minh production integration. Giữ evidence lịch sử bất biến, lập manifest mới cho HEAD/release.

## 4. Đối chiếu từng issue của new_plan

| Mã | Hiện trạng | Phần còn lại / gói |
|---|---|---|
| R0 | Chưa có baseline sạch sau relocation | Sửa2unitfail/contract/collection; lưu service matrix K01 |
| R1 | Control đã ở backend/control | Profiles/package resources/commands/consumer paths còn K01 |
| R2 | Predictor đã ở backend/predictor | Default manifest và clean wheel/startup K01; golden local pass |
| R3 | OCR đã ở backend/ocr | Test6pass; clean artifact bootstrap K02 |
| R4 | Devops đã gom thư mục | Workflows/runbooks còn path cũ; K01/K12 |
| R5 | Boundary mới có, legacy retirement chưa đủ proof | Inventory consumers/zero-consumer check và compatibility retirement K01 |
| M1 | Per-model manifests có | Hoàn thiện release metadata/admission provenance K10 |
| M2 | External model root có | Provision/export/offline/checksum/permissions K02/K10 |
| M3 | Scanner có | Test inventory unknown/corrupt/multiple checkpoints, không tự admit K10 |
| M4 | Validator có | Staged/evaluated/admitted decision record và rejection provenance K10 |
| M5 | Catalog API có | UI status/blocked reason/version/capability đồng bộ; K04/K10 |
| M6 | Resolver có | Mọi entry point dùng binding đã resolve; không ambiguous fallback K04 |
| P1 | model_selection có trong request chính | Mixed/OCR/tools/explain chưa xuyên suốt I08–I11; K04 |
| P2 | Quick selector có | Không model/blocked/manual stale/default change và E2E K04 |
| P3 | Session settings có | Validate binding server-side và snapshot/recovery K04 |
| P4 | Run config snapshot có | Recovery snapshot, auth/config fingerprint, actual model/version/endpoints/XAI đầy đủ K04/K05 |
| P5 | Compare chưa hoàn chỉnh | Cùng molecule/split/endpoint, multiple model rows, không pooled verdict K10 |
| A1 | Connection/profile domain có | Runtime/auth/model/protocol lifecycle và revision contract K05 |
| A2 | Secret-reference filesystem store có | Durable storage, isolation, rotation/delete and deployment policy I12/I16; K05/K07 |
| A3 | Create/list/get/delete/probe có | Update/edit/revision/deletion-in-use, error contract K05 |
| A4 | Probe có nhưng false capability | Parse và roundtrip thật; provider adapters I13/I14 K05 |
| A5 | Settings add/test/delete có | Edit/rotate, guided defaults, meaningful errors, supported provider inventory K05/K08 |
| A6 | Session selector có | Runtime thực dùng key/URL đã chọn, pinned per run, disable during active run K05 |
| A7 | DSH experimental | Conformance và supported/unsupported ADR K09/K13 |
| U1 | Semantic schema có | Versioning và terminal/recovery completeness K08 |
| U2 | Activity projector có | Các intent/context labels và replay equivalence K08 |
| U3 | Aggregate tool activity có | Long histories, parallel calls, failures/count accuracy K08 |
| U4 | RunBlock thay bằng presence/history | Queued/thinking/no active tool vẫn có phản hồi; retry failure flow K08 |
| U5 | Motion có | Reduced motion, latency stages, mobile/focus visual review K08 |
| U6 | Recovery state/banner có một phần | Actionable retry/cancel, potentially_billed/usage_unknown, no raw error UX K07/K08 |
| U7 | Developer diagnostics có | Owner ACL, redaction, drawer keyboard và collapse persistence K08 |
| S1 | GroundedAnswer/claim/source records có | Citation payload không phụ thuộc chèn text hoặc raw cite token; verify all renderer paths K11 |
| S2 | Claim chips có | Literature citation chips/preview, mapping đúng repeated claims, keyboard/mobile K08/K11 |
| S3 | Artifact/evidence surfaces có | Sources panel mục đích rõ, quoted support/retrieval/provenance/conflict K11 |
| S4 | Analysis results panel có | Tách Results/Sources, model/threshold/XAI/bundle state đồng bộ K06/K08 |
| Q1 | Reducer/event tests có | Live reconnect/duplicates/gaps/terminal ordering across instances K07/K08 |
| Q2 | Một số selection tests có | Hai-model adversarial matrix all routes I08–I11 K04 |
| Q3 | Connection unit coverage có | Actual URL/key/auth isolation và runtime protocol matrix K05/K13 |
| Q4 | Browser8pass mocked | Default settings, real SSE/research/recovery/provider changes K08/K13 |
| Q5 | Reduced-motion test có | Axe/manual contrast/focus/IME/năm viewport K08 |

## 5. Kế thừa các plan chuyên đề

| Plan/phase | Đã có, không xây lại | Công việc chưa đóng |
|---|---|---|
| Capabilities P0–P1 | Endpoint/model contracts, selectors | ADR cập nhật default XAI và admitted models; capability readiness/validation K02–K04 |
| P2 | Rename API, deterministic auto-title/manual precedence | Summary query/count/latest, search/cursors; K08 |
| P3–P4 | Token-structure/bond mapping và SVG/viewer contracts | Scientific benchmark, sanitation/error matrix, model-pinned explain K06/K10 |
| P5–P6 | Explanation persistence, MCP slices/tools | Durable bundle/checkpoints, exact model/version target, compare tools, timeout/cache/failure evidence K04/K06 |
| P7–P9 | EuropePMC search/read/source validation | Research intent usability; identity/BioAssay provider; safe web search/fetch; Obscura paired spike có điều kiện K11 |
| P10 | setup --agent và profile bootstrap | Clean clone, explicit credential isolation, supported provider testing K02/K05 |
| P11–P12 | SQLAlchemy pool, migrations, outbox/polling | DB readiness, one-writer migration job, durable worker/leases/fencing/cancel/notify K07 |
| P13–P14 | Auth boundary, operation docs và một phần CI | OIDC/JWKS+PKCE, managed objects/secrets, telemetry, drills/load/soak/production gates K07/K12/K13 |
| Quick Predict phase API/UI | Single/batch/OCR/explain pages, lazy editor | Bảo toàn config mọi input, explicit XAI targets, ambiguity/invalid/model failure, request abort, batch caps/progress K03/K04/K06 |
| Quick Predict scientific XAI | Signed logit attribution/mapping có | Không gọi gradient×input là GNNExplainer; fidelity/stability và canonical atom alignment K10 |
| UI refinement phases0–2 | Design tokens/layout/landing/empty-state nhiều phần có | Visual baseline, self-host font assets (fonts.css hiện local()), responsive five viewports K08 |
| UI refinement phases3–5 | Composer/input studio/results/batch có | Default submit/IME, focus/file failure states, source/result hierarchy K03/K08 |
| UI refinement phases6–7 | Motion/lazy load/test scaffolding có | A11y/manual review, route/SSE no-remount, performance budgets dưới network throttling K08 |
| Handoff simplification | Layout và bin wrapper | Clean customer allowlist, artifact rights/inventory, no secrets/data leakage, release manifest/SBOM/offline bundle/runbook drill K12 |
| GPU/cache phases0–4 | CPU/GPU config và images đã tồn tại local | Host runtime/driver preflight; pinned wheels; provision cache; startup/RSS/VRAM/batch1/8/32/128/256; parity/OOM/concurrency/CPU rollback K10/K12 |
| Landing restoration | Landing implementation/build có | Section/link/CTA content, truthful scope/licensing, sticky/mobile/reduced-motion visual sign-off K08 |
| Legacy research/ADK checklist | Predictor/evidence/report intents có implementation mới | Chỉ chuyển intent hợp lệ; PubChem/BioAssay và fault scenarios K11; không phục hồi global safety verdict hoặc raw report bypass validator |

## 6. Unified-v2 G0–G12: phần việc thực sự còn lại

| Gate | Đánh giá | Bằng chứng phải bổ sung |
|---|---|---|
| G0 | Một phần | Baseline credentialed OpenCode trên release config; manifests links/hash đúng, raw results và variance K13 |
| G1 | Đã có correctness/golden evidence; local golden pass7 | Giữ546 values/42molecule tolerance1e-6 ở release image, không phát minh model mới K10 |
| G2 | Typed boundary implemented | Contract compatibility across packaging/consumer routes, immutable provenance K01/K04 |
| G3 | Standalone predictor có evidence lịch sử | Clean clone/wheel/image/artifact readiness HEAD; ClinTox vẫn blocked K01/K02 |
| G4 | Case revisions/plans/steps/reconstruction có | Production-path resume và migrations/concurrency/failure injection K07/K09 |
| G5 | Kernel partial, chưa cutover | Sửa budget/coverage I31, bridge tools/answers, paired baseline variance, rollback gateway K09/K13 |
| G6 | Infrastructure only | Disjoint calibration split, fitted artifact, thresholds/version, Brier/ECE/untouched test K10 |
| G7 | Infrastructure only | Training reference fingerprints/embeddings, fitted cutoff/conformal quantile, coverage/selective risk/OOD benchmark K10 |
| G8 | XAI partial | Faithfulness/stability/conservation/alignment benchmark và latency cache matrix K06/K10 |
| G9 | Provider independence partial | Credential/baseURL/auth thực dispatch và cross-owner live matrix K05/K13 |
| G10 | Admission decision đã có: experimental | Không quảng bá Codex/DSH supported; nếu promote cần đầy đủ roundtrip/conformance K09 |
| G11 | Evaluation partial | Required runtime-provider categories/trials/manifests và SME support K13 |
| G12 | Hardening partial | Restore/cancel/cleanup/scale/drills, CI coverage, zero-consumer relocation và release sign-off K01/K07/K12 |

## 7. Gói triển khai có thể đưa vào backlog

Owner dưới đây là vai trò đề xuất, chưa gán tên người. Ước lượng là khoảng ngày công cho phần còn lại, không phải lịch cam kết; không cộng cơ học vì nhiều test/migration dùng chung. Mỗi gói cần PR có trigger, thay đổi behavior và evidence links; chia PR khi schema/contract cần review trước consumer.

### K01 — Baseline, relocation và CI

**Ưu tiên:** P0. **Owner:** Backend + QA/DevOps. **Phụ thuộc:** —. **Ước lượng:** 3–6 ngày công. **Truy vết:** I22–I26, I32.

1. Inventory tracked imports, scripts, Dockerfiles, workflows, config defaults và documentation commands; kiểm cả checkout, wheel và image từ cwd bất kỳ.
2. Sửa service-root/package resources cho control profiles, predictor registry và contract snapshot; giữ immutable snapshot source duy nhất.
3. Tách test collection theo service hoặc namespace sạch. Xác định từng skip reason và test chậm trong full control; dùng timeout riêng có traceback, không vô hiệu test để xanh.
4. Fast gates: control/predictor unit+contract, OCR, frontend tests/build/policy. Integration: PostgreSQL migrations/repositories, real service contracts, browser mocks. Release: full-stack/network/runtime gates riêng.
5. Artifact CI provision theo manifest; golden không được pass bằng all-skipped. Chạy benchmark thực theo frozen split khi cần, upload JUnit/manifest/results thay cho --help.
6. Index canonical docs, đánh dấu archive/superseded, kiểm relative links và supported commands; ghi hashes/versions/dirty-tree vào baseline.

**Acceptance và evidence:** Clean CI thu thập/chạy đủ test; mọi skip có lý do được allowlist. Sửa path không đổi546golden values; no-agent dependency guard và wheel import vẫn pass. Một báo cáo baseline dùng đúng HEAD/config.

**Migration/rollback:** Không đổi numerical policy; giữ manifest và contract cũ để diff/rollback.

### K02 — Topology, capability admission và bootstrap

**Ưu tiên:** P0. **Owner:** Backend + DevOps. **Phụ thuộc:** K01. **Ước lượng:** 3–6 ngày công. **Truy vết:** I01/I02/I05/I27/I29.

1. Định nghĩa mode predictor-only/agent-enabled và schema capability availability/reason/checked_at; configured khác available.
2. Admission đối chiếu handler, runtime readiness, evidence/OCR dependency và target model; không queue intent chắc chắn không chạy được.
3. Provision predictor và OCR immutable bundles, checksum/cache/offline export. doctor kiểm đủ artifact, image, port ownership và secrets presence mà không in secret.
4. Sửa wrapper argv/logs/up idempotency; agent setup/up phải build/pull đủ image hoặc báo bước thiếu cụ thể. Ready message sau capability smoke.
5. Giữ browser chỉ tiếp cận control qua frontend; smoke prediction, OCR và agent intents tương ứng mode.

**Acceptance và evidence:** Clean clone CPU setup/up/smoke và up lần2; mode absent/degraded trả capability reason rõ; agent-mode research/report/attribution chạy thật trên môi trường kiểm chứng.

**Migration/rollback:** Overlay mode cũ có thể giữ nhưng phải hiển thị giới hạn chính xác; không fallback scripted.

### K03 — Composer và Quick Predict input contract

**Ưu tiên:** P0. **Owner:** Frontend + Backend. **Phụ thuộc:** K02 schema. **Ước lượng:** 2–5 ngày công. **Truy vết:** I03/I04/I06/I21.

1. Tách text và molecule; heuristic chỉ đề xuất, validate canonical molecule server-side; mixed/multiple/invalid input có UX rõ.
2. Plain prediction mặc định gửi được. None/on_demand/required và targets là lựa chọn riêng; lỗi cạnh action, không ẩn trong popover.
3. Capability-driven endpoints/models/assays; zero endpoints, unavailable provider, stale preferences và expert-only threshold được validate server-side.
4. Gửi cùng analysis configuration cho SMILES, editor, image và batch. Preserve draft, idempotency/retry intent và không xóa text khi submit lỗi.
5. IME composition, Enter/Shift+Enter, upload oversize/wrong type, focus restoration, disabled/busy/error đều có tests thực trigger.

**Acceptance và evidence:** Default herg+tox21 CCO gửi được không cần assay; required thiếu target bị chặn rõ. Mixed Vietnamese query giữ text+molecule; hello không tự mất thành molecule; ba input cùng config.

**Migration/rollback:** Không phá explicit required semantics hoặc quyền threshold của server.

### K04 — Model binding xuyên suốt request → result

**Ưu tiên:** P0. **Owner:** Backend predictor/control + QA. **Phụ thuộc:** K01/K03 contract. **Ước lượng:** 4–7 ngày công. **Truy vết:** I08–I11.

1. Chuẩn hóa ResolvedRunConfiguration: requested selection, resolved model id/version/hash theo endpoint, thresholds/source, XAI targets/mode, registry revision.
2. Validate saved session bindings và per-message override; pin ngay admission. Default registry đổi sau queue không đổi run đã nhận.
3. Truyền cấu hình vào deterministic single/batch, mixed pre-runtime snapshot, OCR follow-on, tool create_analysis, attribution, Quick explain và recovery.
4. Explain API/cache key/observation reference phải gồm model identity, canonical molecule, method/version và target. Reject mismatched probability/provenance khi ghép bundle.
5. Recovery lưu snapshot cho run mới và link run gốc; audit API trình bày requested/resolved/actual, không ghi secret.
6. Test hai provider cùng endpoint với output khác nhau; unknown/manual-unavailable/ambiguous phải fail explicit, không fallback.

**Acceptance và evidence:** Matrix single/batch/OCR/mixed/tool/explain/recovery chọn B chỉ gọi B và persist B. Thay session setting giữa run không đổi run đang chạy.

**Migration/rollback:** Schema additive; reader tương thích old snapshot, legacy result ghi unknown thay vì suy model identity.

### K05 — AI profiles hoạt động thật và cô lập

**Ưu tiên:** P0. **Owner:** Backend runtime + Frontend + Security. **Phụ thuộc:** K02/K04. **Ước lượng:** 6–10 ngày công. **Truy vết:** I12–I15.

1. Provider registry khai báo protocol/defaultURL/auth mode/model inventory/capabilities; chỉ expose provider có adapter được chứng minh.
2. CRUD đầy đủ gồm update/revision/key rotation/delete-in-use. Probe typed status: unknown/unsupported/failed/ready, lỗi đã redact.
3. Probe parse streaming, validate JSON và tool roundtrip; DONE-only không chứng minh đủ capabilities. Lưu thời điểm/probe version, stale probe có policy.
4. Resolve credential_ref/baseURL/auth_mode vào runtime isolated owner; không dùng ambient auth ngoài explicit local single-user mode. Pin connection revision per run.
5. Áp network policy local vs hosted, URL/DNS/IP/redirect controls và quota/timeout; managed secret store, redaction, rotation/revocation.
6. UI guided add/edit/test/remove, default status khả dụng, selector khóa hoặc áp cho next run rõ ràng; error không để người dùng đoán configuration.

**Acceptance và evidence:** Hai owner/two mock servers chứng minh đúng URL/key; rotate/delete không ảnh hưởng run khác ngoài policy. Live primary provider create→test→run→cancel và usage provenance pass.

**Migration/rollback:** Feature flag chỉ cho provider đã admitted; revoke/cleanup secrets và runtime config khi rollback.

### K06 — Durable prediction/XAI bundle

**Ưu tiên:** P0. **Owner:** Backend + Scientific ML. **Phụ thuộc:** K04/K07 persistence contract. **Ước lượng:** 4–8 ngày công. **Truy vết:** I11/I19.

1. Thiết kế bundle state: prediction ready, từng target pending/running/succeeded/failed/cancelled; required terminal semantics và API đồng bộ.
2. Persist prediction và checkpoint từng target; execution idempotency/caching theo full scientific identity; không giữ tất cả trong RAM tới cuối.
3. Enforce deadline tổng và remaining budget trước mỗi explain; bounded concurrency phù hợp CPU/GPU; cancel giữ artifact hợp lệ.
4. XAI signed atom/bond score, unmapped score, alignment version, method/logit target và sanitized SVG cần một contract.
5. Expose bundle/slices qua MCP và Results UI; failure required là explicit, on-demand có thể retry target mà không recompute prediction.

**Acceptance và evidence:** Kill sau một target rồi resume chỉ tính phần thiếu; timeout/cancel không nhận completed giả; result/XAI cùng model hash; invalid SVG hoặc mapping không được render tin cậy.

**Migration/rollback:** Schema versioning, preserve old snapshots; disable method chưa benchmark thay vì trả approximate attribution không nhãn.

### K07 — Persistence, execution và lifecycle production

**Ưu tiên:** P0 trước scale. **Owner:** Backend platform + DevOps. **Phụ thuộc:** K01/K02. **Ước lượng:** 8–14 ngày công. **Truy vết:** I16–I20/I28.

1. Durable job table/envelope; atomic claim, worker id/lease/heartbeat/fencing, retry limits và idempotent side effects. Startup chỉ reconcile lease hết hạn.
2. Cancel cross-instance, lease recovery/checkpoints và potentially_billed/usage_unknown; ngăn duplicate accepted answers/sequence/outbox writes.
3. Migration one-writer job; upgrade từ DB rỗng và version cũ, expand/contract window, pool/statement/lock timeouts có đo đạc.
4. Filesystem volume local; object/secret adapters hosted với ACL/hash/MIME/signed refs. Retention/raw evidence opt-in, TTL sweeper idempotent và orphan cleanup.
5. Sửa backup/restore semantics, ON_ERROR_STOP, integrity validation và restore vào môi trường mới bao gồm object refs; deletion audit có retry.
6. Query latest summary/count/cursors; indexes và cross-instance event notification/polling fallback; DB ready probe.
7. Failure matrix: A active B startup, two submit race, cancel through B, kill9 after accept/tool/commit, duplicate event, expired upload, lost object, DB outage.

**Acceptance và evidence:** Hai replica không fail run của nhau; accepted work survives process death đúng policy; cancel/restore/drills có measured latency và state consistency, không double commit.

**Migration/rollback:** Giữ replica1 trước gate; DB migrations forward-only, backup verified trước đổi schema; rollback binary phải đọc schema mới.

### K08 — Session UX, evidence surfaces và accessibility

**Ưu tiên:** P0 lõi/P1 polish. **Owner:** Frontend + Product/QA. **Phụ thuộc:** K03/K04/K06/K07 contracts. **Ước lượng:** 5–9 ngày công. **Truy vết:** I06/I20/I21; new_plan U/S/Q.

1. Presence luôn phản hồi queued/thinking/tool/validation/recovery; semantic labels do server mapping, aggregate repeated calls; raw tool traces ở Developer details.
2. Transcript dùng accepted answer; citation chips/previews và Sources/Results không dựa vào regex thay số làm nguồn duy nhất. Long messages và repeated claims giữ mapping.
3. Retry/cancel/recovery copy hướng hành động, billing unknown rõ, không lộ stacktrace; drawer owner ACL/redaction và focus trap.
4. Session rename giữ manual precedence; API search toàn lịch sử, pin/archive/delete/cursor/export theo lifecycle; không chỉ filter page local.
5. Visual review five viewports, font assets tự host có license, keyboard/IME/contrast/reduced motion, live region, editor lazy-load và no-remount SSE.
6. Browser tests default settings + persisted run settings + refresh/reconnect/duplicate/out-of-order/expired artifact; thêm real-stack smoke ngoài fixtures mocked.
7. Landing sections/CTA/docs links/licensing/scientific scope và performance budgets có artifact screenshot/measurement được review.

**Acceptance và evidence:** Không blank status khi đang nghĩ; screen reader/keyboard làm được luồng chính; refresh sau research/recovery giữ answer+sources+results; 5viewport không overflow và animation reduced-motion đúng.

**Migration/rollback:** Giữ validated answer route/reducer contract; visual feature flags không được thay truth hoặc tool authority.

### K09 — Kernel v2 và runtime admission

**Ưu tiên:** P1; chưa cutover. **Owner:** Backend agent/runtime. **Phụ thuộc:** K04/K07, baseline K13. **Ước lượng:** 6–10 ngày công. **Truy vết:** I31; G4/G5/G9/G10.

1. Sửa budget trước plan/compose và actual tool execution, coverage theo success_condition/evidence/conflict/gaps; giữ stop reason terminal, không completed trước admission.
2. Resume từ durable case/plan/steps/observations thay vì tạo plan rỗng; optimistic revisions, replan budget và conflict resolution có semantics.
3. Bridge current gateway/tool registry/answer compiler theo feature flag; paired input + same predictor/evidence fixtures để đo variance trước cutover.
4. Runtime adapter isolation deny-all và explicit allowlisted MCP; start/auth/tool/stream/cancel/close/reap contract riêng cho OpenCode, DSH, optional Codex.
5. Với runtime thiếu authenticated roundtrip/cancel contract, giữ experimental/unsupported ADR có version/evidence; không tạo shim giả thành supported.

**Acceptance và evidence:** Kernel limits1 không gọi2modelturns; no-evidence không sufficient; restart hồi đúng context; paired gate không regression và rollback gateway đã diễn tập.

**Migration/rollback:** Compatibility gateway còn là rollback path tới khi kernel đạt gates; thay runtime không thay source authority.

### K10 — Scientific artifacts, admission và performance

**Ưu tiên:** P1; gate theo capability. **Owner:** Scientific ML + Predictor + QA. **Phụ thuộc:** K01/K04. **Ước lượng:** 8–15 cộng data/training ngày công. **Truy vết:** I07/I11; G1/G6–G8.

1. Inventory checkpoints với scanner, mapping architecture/tokenizer/dataset/split/threshold; quarantine unknown. Admission tạo record reproducible gồm hashes/version/metrics/reason.
2. ClinTox: tìm tokenizer gốc đúng mapping69token; nếu không có, v2 retrain có seed/code/data/tokenizer/weights/calibration/test/manifest. Không sửa v1 bằng tokenizer80token.
3. Calibration: tách train/calibration/test bằng hash và leakage guard, fit artifact trên calibration, report Brier/ECE/reliability/CI theo endpoint; threshold policy version riêng.
4. AD/uncertainty: training reference ECFP/embedding và fitted cutoffs; conformal quantile từ split phù hợp; coverage/selective risk/OOD benchmark. Unknown không hiển thị confident.
5. XAI: signed/logit-targeted, conservation/unmapped alignment, aromatic/ring/bond/stereo cases; faithfulness/deletion/insertion/stability perturbations và repeated seeds; method limitations rõ.
6. Compare mode chỉ admitted models, same endpoint/input/split/protocol, provenance riêng và latency; không gộp probability thành global score.
7. CPU/GPU benchmark pinned release image/artifacts: cold/warm start, batch1/8/32/128/256, RSS/VRAM, concurrency/OOM, parity tolerance và rollbackCPU. Cache wheels/model immutable, không online download serving.

**Acceptance và evidence:** Không fit trên untouched test; release bundle có hashes/protocol/results/model card; missing calibration/AD/ClinTox tiếp tục status unavailable/uncalibrated, không fake completion.

**Migration/rollback:** Scientific policy v1 giữ nguyên tới admission v2; cho phép đổi version explicit và replay kết quả cũ.

### K11 — Evidence và capability mở rộng

**Ưu tiên:** P1. **Owner:** Backend evidence + Scientific SME + Frontend. **Phụ thuộc:** K02/K04/K06. **Ước lượng:** 5–9 ngày công. **Truy vết:** W3; new_plan S1–S4; P7–P9.

1. Đóng research subject UX; identity resolution có ambiguity/salt/stereo provenance, PubChem/BioAssay/EPA context theo scope chốt, không dùng tên do LLM bịa.
2. EuropePMC query/read/retrieval hashes/provider/time/raw policy; source phải được đọc trước cite; distinguish no evidence/provider fail/conflict và preserve contradictory sources.
3. Citation graph support đúng claim, exact source/span, external identifier/URL server-owned; preview Sources với metadata và retrieval limitations.
4. Safe search/fetch bounded domains/bytes/MIME/timeouts/redirect/DNS/IP, injection cannot expand tools; fetch tùy capability bật, không arbitrary URL authority mặc định.
5. Rubric tách URL validity/topic relevance/claim support/source quality; hai SME blind sampling/all critical và adjudication, thành versioned regression tasks.
6. Provider thứ hai/Obscura chỉ spike khi cần sau alpha: 30–50paired queries theo plan, đo quality/latency/cost, promote/reject ADR; không bắt mọi deployment thêm provider.

**Acceptance và evidence:** Attribution và evidence live accepted qua GroundedAnswer; citation validity100%, scientific support>=95% alpha/98%production; injection/false URLs không qua validator.

**Migration/rollback:** Provider flag và cached immutable records; disable provider không làm mất audit của answers cũ.

### K12 — Release, hosted security và handoff

**Ưu tiên:** P0 release. **Owner:** DevOps + Backend + Security. **Phụ thuộc:** K01/K02/K05/K07. **Ước lượng:** 7–12 cộng observation ngày công. **Truy vết:** I26–I30.

1. Thiết kế deployment tách control/predictor/OCR/runtime/Postgres/objectstore, staging/prod và private service traffic. Sửa workflow paths/branch targets; promote same digest sau gates.
2. Hosted identity OIDC/JWKS validation và browser PKCE/refresh/logout, owner isolation, limits phân tán; static local auth không quảng bá thành hosted auth.
3. Structured logs redacted, trace spans request/run/tool/provider, metrics queue/latency/errors/token/cost/unknownusage; dashboard/alerts gắn runbook và SLO đã chốt.
4. Pin dependencies/runtime/image/artifacts có SBOM/scan evidence, license/rights inventory, immutable release manifest và rollback targets.
5. Load/soak/failure/restore/deletion/secret rotation drills trên staging; migration one-writer và rolling upgrade không mất run. Ghi kết quả cả failure.
6. Customer handoff theo allowlist: no secrets/history/private data, model weights có quyền phân phối, offline artifact bundle, supported commands/docs/sample inputs; test máy sạch độc lập.
7. Cloud/Firebase target assertions và dry-run trước publish; staging preview khác live. Không coi HTTP frontend200 là full-stack smoke.

**Acceptance và evidence:** Clean customer clone và release image cùng manifests chạy đủ mode; staging/prod tách; rollback+restore đạt RTO/RPO được chốt; security/scientific/engineering sign-off có evidence.

**Migration/rollback:** Không push/deploy trong audit; triển khai sau này promote artifact đã review, retain last-good digest/schema compatibility.

### K13 — Evaluation, alpha và production acceptance

**Ưu tiên:** P0 quality gate. **Owner:** QA/eval + Scientific SME + Product. **Phụ thuộc:** K01, rồi K02–K12 theo gate. **Ước lượng:** 5–9 cộng >=1tuần alpha ngày công. **Truy vết:** W1/W7/W8/W9; G0/G11.

1. Freeze task schema/fixture modes frozen/predictor_integration/live_evidence, typed skip reasons, manifest versions; không thay mẫu số baseline im lặng.
2. Live runtime + frozen numeric observations cho fidelity; real predictor semantic tests riêng; evidence query/time/hash riêng. Classify failures product/prompt/grader/task/provider/infrastructure.
3. Chạy full critical/capability suites đủ trials; runtime/model/auth/provider matrix, record latency/cost/usage_unknown/fallback/retry và dependency versions.
4. Alpha cohort/scope/sample data/cost cap/feedback channel, two SME blind>=20%capability và mọi critical; triage trong48h, observe ít nhất một tuần, biến lỗi thành eval.
5. Failure injection start/cancel/recovery/compaction/deniedtool/foreignsession/outage/billing; pair TrackA/B runtime với cùng config trước admission.
6. Production pass^5, soak/load/SLO, canary provider/runtime/prompt/tool upgrades và sign-offs. Gate fail thì sửa đúng loại lỗi rồi chạy đủ lại, không cherry-pick successful trial.

**Acceptance và evidence:** Alpha: critical pass^3=100%, numeric fidelity100%, capability pass@1>=80%, citation validity100%, support>=95%, major SME correction<=15%. Production: critical pass^5=100%, capability>=85% và không nhóm<80%, support>=98%, major SME correction<=10%; giữ mọi gate chi tiết của rebuild plan.

**Migration/rollback:** No-go khi thiếu credential/results/data; có thể giao predictor-only có nhãn scope riêng nhưng không tuyên bố agent/production hoàn tất.

## 8. Critical path, release milestones và quyết định ngoài code

| Mốc | Điều kiện vào | Deliverable/evidence để ra |
|---|---|---|
| M0 — baseline đáng tin | Worktree được inventory | K01: clean service tests, contracts, CI collection, command/path proof |
| M1 — local usable | M0 + capability contract | K02/K03: clean bootstrap, default prediction, explicit runtime absent, research subject UX |
| M2 — cấu hình thực thi đúng | M1 | K04/K05/K06: all-input model binding, real BYOC isolation, durable XAI semantics |
| M3 — reliable alpha candidate | M2 + K07/K08/K11 | Cross-instance/recovery/persistence và Sources/Results; complete evaluation gate K13 |
| M4 — alpha được chấp nhận | M3 + cohort/SMEs/quota | Ít nhất một tuần observation, critical pass^3, rubric/triage và drills |
| M5 — production candidate | M4 + K12 | Hosted auth, SLO/load/soak/restore/canary, critical pass^5 và sign-offs |
| M6 — scientific/runtime expansion | Baseline riêng cho từng capability | K09/K10 admission; calibration/AD/ClinTox/DSH chỉ promote khi evidence đủ |

K02/K03 có thể bắt đầu sau khi schema capability ổn định trong K01. K04/K05 và K07 có thể đi hai track sau khi thống nhất immutable config/secret refs. K06 thiết kế song song K07 nhưng không được tự tạo cơ chế queue/lease thứ hai. K09 không nằm trên đường sửa sáu lỗi trước mắt và không cutover trước paired baseline. K10 phần chuẩn bị artifact/eval làm ngay, phần fit chờ dữ liệu đúng; K11 optional provider không chặn alpha EuropePMC.

Các input cần owner cung cấp/chốt trong triển khai sau audit:

- Deployment target local single-user hay hosted multi-user; đây là hai auth/egress/storage policies khác nhau. Mặc định kế hoạch hỗ trợ local trước, không coi nó production hosted.
- Provider credentials/quota, approved test accounts và runner cho live matrix. Chuẩn bị harness/fixtures/redaction trước khi cần credential.
- Calibration split và training reference embeddings/fingerprints có provenance; tokenizer ClinTox gốc hoặc quyết định retrain v2. Không thay bằng test split hay tokenizer tình cờ cùng kích thước.
- Quyền phân phối weights/fonts/datasets, customer allowlist và retention/raw evidence policy; owner/security quyết định trước handoff.
- SLO/RTO/RPO, ngân sách run/token/tool/global concurrency, alpha participants và hai SME; đo baseline trước chọn ngưỡng latency tùy tiện.
- Admission scope OpenCode primary và có thực sự đầu tư DSH/Codex/Obscura/Compare hay chỉ giữ experimental. Conditional backlog phải có promote/reject ADR, không mất khỏi tracking.

## 9. Ma trận kiểm chứng tối thiểu theo boundary

| Boundary | Trường hợp bắt buộc | Chứng cứ |
|---|---|---|
| Input/admission | default, mixed, image, invalid, empty endpoints, missing XAI target, unsupported capability | API requests/responses và browser capture; không chỉ component props |
| Configuration | two models/endpoint, two owner keys/URLs, stale/deleted profile, changing defaults, recovery | Fake provider request logs + persisted requested/resolved/actual bindings; live primary smoke |
| Science | 42molecule/546values, masked labels, threshold source, attribution alignment, no ClinTox proxy | Immutable artifact/split/method hashes, golden/benchmark reports |
| Answer/evidence | numeric formatting VI/EN, unsupported clinical claims, unread citation, conflicts, injection | Accepted/rejected answer source graph và graded manifests |
| State/execution | accept/kill/replay, two replicas, cancel remote, duplicate events, migration rolling | PostgreSQL + process failure tests, no double accepted answer, consistent sequence |
| Storage/security | owner boundaries, attachment MIME/TTL, secret rotation, internal URL policy, restore | Isolated security tests và restore report, no secret in logs |
| UX | reload/reconnect/thinking/failure, five viewports, keyboard/IME, reduced motion | Playwright real-stack subset + screenshots + manual a11y review |
| Release | clean clone/offline, artifact missing/corrupt, staging/prod target, CPU/GPU rollback | Signed/versioned release manifest, SBOM, smoke/drill reports |

Mỗi gate lưu commit, dirty state, image digest, runtime/model/provider/config revision, artifact/split/task hashes, command/env không chứa secret, timestamps, raw results, skip reasons và reviewer. Bằng chứng không resolve được hoặc không đúng config phải ghi **chưa đủ bằng chứng**.

## 10. Truy vết toàn bộ W0–W10 của remaining plan

Mỗi yêu cầu có mã trong source được giữ dưới đây để không bỏ mất việc nhỏ. Cột trạng thái đánh giá việc đóng gate hiện tại, không phủ nhận code có sẵn. Các hạng mục đánh dấu x trong source nhưng bao gồm live/PG/CI/production được giữ ở trạng thái cần tái kiểm chứng nếu gate hiện tại chưa chạy hoặc CI đã đổi. Exit gates của từng W vẫn bắt buộc theo §11.

| Mã | Yêu cầu nguồn (giữ nội dung) | Đánh giá hiện tại và phần còn lại | Gói |
|---|---|---|---|
| W0-01 | Chụp `git status`, commit SHA, dependency lock và danh sách thay đổi đang dở; không trộn thay đổi ngoài agentic-layer vào PR. | Audit đã inventory worktree; implementation PR phải giữ thay đổi người dùng. | K01 |
| W0-02 | Hoàn tất lát UI/OCR đang uncommitted: review diff, chạy contract tests, frontend build/policy lint và một live smoke cho cả ba input. | UI/OCR đã có; frontend57pass/OCR6pass; còn clean full-stack ba input. | K01 |
| W0-03 | Commit lát UI/OCR riêng với ADR 0006 và cập nhật progress; chỉ push khi quy trình repository yêu cầu. | Lát lịch sử không còn trạng thái uncommitted như plan cũ; không tự commit thay đổi người dùng hiện tại. | K01 |
| W0-04 | Chuẩn hoá lệnh test theo từng environment (`toxagent-control`, `toxocr`, ToxPred, frontend); thêm script không phụ thuộc shell đang active. | Chưa đóng: paths/collection/default env hiện lỗi I22–I25. | K01 |
| W0-05 | Ghi baseline mới: số test, build sizes, 35-task pass@1 gần nhất, critical failures, latency và fallback rate. | Có baseline test audit; live score/cost/fallback mới còn thiếu. | K01 |
| W0-06 | Sửa bảng trạng thái đầu `PROGRESS`: Phase 3 đã commit, eval không rỗng, Phase 5 core đã live, Phase 6/UI đã triển khai một phần đáng kể. | Nhật ký có cập nhật sau; bảng tổng hợp cần đối chiếu và canonical status. | K01 |
| W0-07 | Sửa mục DSH: carrier chính chủ đã xuất hiện; link SDK/runtime/protocol và chuyển blocker thành quyết định pin pre-release. | Đã có đính chính carrier; chỉ giữ experimental/admission work, không lặp blocker cũ. | K01 |
| W0-08 | Lập decision table hiện hành cho DEC-01…DEC-10; mỗi dòng phải là `accepted`, `pending` hoặc `superseded`, kèm ADR/source. | Có ADRs/quyết định; cần bảng hiện hành theo precedence §3. | K01 |
| W0-09 | Version bộ eval hiện tại trước khi thêm task OCR/production, tránh thay đổi mẫu số của baseline mà không ghi nhận. | Task/fixture/manifests có; freeze version cho các lượt chạy mới. | K01 |
| W1-01 | Thêm `fixture_mode` rõ trong task/manifest: `frozen`, `predictor_integration`, `live_evidence`; không suy chế độ chỉ từ nội dung expectation. | Chưa đóng: runner còn suy live compatibility từ expectations; cần fixture_mode contract riêng. | K13 |
| W1-02 | Xây đường chạy **agentic + frozen predictor** để 12 task `numeric_fidelity` dùng model thật nhưng số nguồn vẫn cố định. Có thể chạy control plane local với frozen ToxPred adapter và OpenCode thật; không đổi expected number theo predictor live. | Chưa có bằng chứng live-agent + frozen numeric; runner scripted có FrozenPredictor, live driver dùng deployment thực. | K13 |
| W1-03 | Giữ `predictor_integration` cho semantic/wording gates trên ToxPred thật, đồng thời ghi artifact hashes từ `/v1/models` vào manifest. | Live integration driver có; artifact/config manifest và complete suite mới còn phải chạy. | K13 |
| W1-04 | Giữ `live_evidence` riêng vì kết quả nguồn thay đổi theo thời gian; lưu query, provider, retrieval time và content hashes để audit. | Live evidence path có; phải freeze query/time/content hash và báo cáo riêng mỗi trial. | K13 |
| W1-05 | Mỗi task không chạy được phải có `skipped_reason` typed; task bị skip không được tính pass hay fail. | TaskResult có skipped_reason dạng string; còn typed taxonomy và reporting đúng từng mode. | K13 |
| W1-06 | Chạy lại đầy đủ 35 task live-compatible sau toàn bộ tám fix ở progress §3.13, lưu manifest/result nguyên vẹn. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-07 | `adv-05-ignore-the-limitations`: xác định vì sao thiếu `uncalibrated_probability`; fix prompt/limitation derivation nếu product sai, giữ nguyên hard gate vì đây là task critical. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-08 | `qa-06-attribution-request`: xác định limitation bị mất ở projection, prompt, candidate hay validator; chạy lại qua attribution thật. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-09 | `evsyn-03-conflicting-evidence`: thay exact-word grader bằng semantic condition có tính phủ định/xung đột nhưng vẫn deterministic nếu có thể; thêm positive và negative counterexamples trước khi đổi task. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-10 | `evsyn-05-no-evidence-found`: grade hành vi “không tìm thấy” theo nghĩa và citation count, không khoá vào một cụm tiếng Anh duy nhất. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-11 | `numeric-07` và `qa-02`: quyết định `kind=comparison` có thật sự là contract bắt buộc. Nếu có, làm rõ tool description/prompt và validator; nếu không, grade graph nguồn và phép so sánh thay vì enum do model chọn. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-12 | Với mỗi fix, thêm regression task/test nhỏ nhất tái hiện nguyên nhân; không thêm regex rộng thiếu negative cases. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-13 | Khi pass@1 ổn định, chạy toàn bộ critical set ba trial độc lập trên cùng manifest family; yêu cầu `pass^3=100%`. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-14 | Chạy numeric fidelity bằng frozen-agentic mode; yêu cầu 100% exact source/rounding/rendered-value gate. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-15 | Báo cáo theo category: pass@1, pass^3, first-candidate acceptance, fallback rate, tool calls, deadline failures, latency, token và cost. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-16 | Thêm regression comparison với baseline; CI fail khi hard gate giảm, còn live quality regression được đưa vào release review thay vì chạy mỗi PR. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W1-17 | Thêm suite OCR/structure-recognition version mới: ảnh hợp lệ, MIME giả, base64 lỗi, quá kích thước, OCR unavailable, OCR không nhận diện, SMILES OCR không hợp lệ và success tạo analysis đúng. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W2-01 | Live test direct denied call: shell/edit/subagent/direct web vừa không hiện trên surface vừa bị transport từ chối. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W2-02 | Live test abort một turn đang chạy; chỉ render “đã huỷ” sau khi run thật sự thành `cancelled`. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W2-03 | Live test event stream disconnect/reconnect với `after_sequence`, duplicate delivery và sequence gap. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W2-04 | Live test runtime restart làm binding cũ thành `lost`, tạo đúng một recovery run và không nối transcript runtime cũ. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W2-05 | Snapshot/diff OpenCode OpenAPI trong CI. Mọi version bump phải tạo diff được review và chạy lại surface/cancel/recovery contract. | OpenAPI/surface assertion có trong tests/progress; CI hiện không đủ version-diff/live gate. | K07/K09/K13 |
| W2-06 | Thêm startup reconciler cho run `queued/running/validating` còn lại sau control-plane crash. Mỗi run phải đi tới terminal/recovery có audit event, không nằm treo vô hạn. | Đã có startup_reconciliation.py; cần sửa ownership/lease I17 thay vì tạo reconciler mới. | K07/K09/K13 |
| W2-07 | Persist đủ recovery input hoặc recovery plan để restart control plane không phụ thuộc `RunContext` chỉ tồn tại trong memory. | RunContext/config snapshot có một phần; durable original execution/restart envelope chưa đủ I18. | K07/K09/K13 |
| W2-08 | Đảm bảo deterministic observation đã commit được reuse; không gọi lại predictor/provider khi retry nếu idempotency key/source graph đã tồn tại. | Recovery reuse code/tests có; OCR/XAI checkpoints và model binding chưa đầy đủ I08–I11/I19. | K07/K09/K13 |
| W2-09 | Viết failure-injection orchestrator điều khiển các service độc lập: kill runtime trước request, sau tool call, sau accepted candidate; kill control plane; treo event stream; provider timeout; DB reconnect. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W2-10 | Chạy lại `fail-04/05/06` và `adv-04` bằng orchestrator thay vì loại khỏi live-compatible rate. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W2-11 | Quét/reap runtime workspace và orphan process cả khi shutdown sạch lẫn khi process cha chết; có soak test khẳng định orphan count bằng 0. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W2-12 | Chốt ngữ nghĩa `potentially_billed`: chỉ bật cho run thất bại/hủy sau khi runtime đã nhận provider turn mà charge outcome không xác định. | potentially_billed transition/startup logic đã có; cần verify remote/recovery receipts và edge cases. | K07/K09/K13 |
| W2-13 | Persist normalized usage events theo run/provider/model, gồm token fields runtime thật cung cấp; không bịa số còn thiếu. | Normalized runtime_usage persistence và unit tests đã có; live provider usage matrix còn thiếu. | K07/K09/K13 |
| W2-14 | Khi usage/cost không có, lưu `unknown` thay vì `0`; API/UI phân biệt “không tốn” và “không biết”. | Unknown usage semantics có trong domain/tests; cần kiểm UI và actual provider/recovery end-to-end. | K07/K09/K13 |
| W2-15 | Test các ranh giới: fail trước send = không potentially billed; receipt accepted rồi mất runtime = potentially billed; completed có usage = usage audit; recovery có usage riêng, không cộng trùng. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K09/K13 |
| W3-01 | Chạy live `request_attribution` cho hERG và một assay Tox21; ghi latency, model artifact hash và observation projection. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K06/K11/K13 |
| W3-02 | Xác minh attribution chỉ cho một endpoint/task, không biến token importance thành causal mechanism hay aggregate toxicity. | Endpoint/task attribution contract có; scientific faithfulness/alignment/selected-model matrix còn thiếu. | K06/K11/K13 |
| W3-03 | Đảm bảo answer bắt buộc có `attribution_not_causality`, claim source trỏ đúng attribution observation và numeric value vẫn trỏ predictor source. | Attribution limitation/source validation có; live accepted answer và projection đủ fields phải tái kiểm. | K06/K11/K13 |
| W3-04 | Thêm live/contract cases: assay thiếu, endpoint unavailable, timeout, partial attribution và cache hit. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K06/K11/K13 |
| W3-05 | Contract test EuropePMC search/detail normalization bằng captured, redacted fixtures; kiểm stable identifier, canonical URL, dedupe và hash. | EuropePMC normalization/evidence integration tests có; captured fixture/version và release run còn kiểm. | K06/K11/K13 |
| W3-06 | Failure cases: 429, timeout, malformed payload, empty result, duplicate, disallowed host và provider circuit breaker/backoff. | Evidence failure tests có; mở rộng thực provider retry/rate-limit/duplicate contract và chạy matrix. | K06/K11/K13 |
| W3-07 | Citation validator yêu cầu model đã gọi detail/read trước khi cite; mọi citation phải là accepted record cùng session/share scope. | Read-before-cite validator đã có; giữ invariant và kiểm qua live evidence path. | K06/K11/K13 |
| W3-08 | Giữ evidence text trong untrusted projection có delimiter/type; adversarial suite phải chứng minh instruction trong title/abstract không tăng tool authority hay tạo model-authored URL. | Untrusted evidence projection và injection tests có; live denied surface vẫn cần chứng minh. | K06/K11/K13 |
| W3-09 | Chốt DEC-10: metadata + accepted excerpt mặc định; raw payload chỉ lưu khi policy yêu cầu và qua object store có TTL/ACL. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K06/K11/K13 |
| W3-10 | Viết rubric citation support tách “URL tồn tại”, “nguồn nói về đúng chủ đề”, “nguồn hỗ trợ đúng claim” và “chất lượng nguồn”. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K06/K11/K13 |
| W3-11 | Hai SME chấm mù tối thiểu 20% capability set và mọi critical evidence failure; lưu disagreement/adjudication. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K06/K11/K13 |
| W3-12 | Chuyển disagreement lặp lại thành task/rubric version mới; không sửa kết quả cũ. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K06/K11/K13 |
| W3-13 | Alpha gate: citation validity 100%, scientific citation support >=95%, major SME correction <=15%. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K06/K11/K13 |
| W4-01 | Dựng PostgreSQL ephemeral trong integration CI; chạy Alembic từ DB rỗng, toàn bộ repository/integration tests và schema constraint checks. | PG tests có trong lịch sử; unified-ci hiện không chạy gate này. | K07 |
| W4-02 | Test transaction giữa domain mutation và outbox, monotonic sequence, idempotency keys, unique accepted answer và claim-source foreign keys trên PostgreSQL thật. | Constraints/outbox/repository tests có; chưa chạy lại PostgreSQL tại HEAD. | K07 |
| W4-03 | Thay admission/concurrent-run guard chỉ trong memory bằng cơ chế đúng khi có nhiều control-plane instance (DB constraint/lock có bounded retry); test hai instance cùng nhận request. | Admission DB guard có; không giải quyết startup/worker lease I17/I18. | K07 |
| W4-04 | Test cross-instance: instance A ghi event, instance B phục vụ REST/SSE reconcile mà không mất state. | Cross-instance event test có; chưa chứng minh toàn execution/cancel/recovery. | K07 |
| W4-05 | Viết [migration policy/runbook](../runbooks/TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md): forward-only production, pre-deploy migrate, backup trước migration, compatibility window khi rolling deploy. | Runbook đã có; migration job và rolling-upgrade drill còn thiếu. | K07 |
| W4-06 | Tạo `ObjectStore` interface (`put/get/delete/signed_read_ref`) và fake filesystem/in-memory cho test; production adapter ưu tiên GCS vì deployment hiện tại ở GCP, nhưng application không import SDK GCS trực tiếp. | ObjectStore interface/fakes có; hosted adapter và durable deployment còn K07. | K07 |
| W4-07 | Upload ảnh: persist bytes trước khi queue OCR run; message dùng `attachment_id/image_ref`, worker/recovery đọc qua owner/session ACL. | Persist upload/reference có; volume/TTL/restore và OCR config còn thiếu. | K07 |
| W4-08 | Verify MIME bằng magic bytes, giới hạn kích thước sau base64 decode, hash nội dung, chặn SVG/HTML/polyglot không được hỗ trợ và không auto-serve user payload inline. | MIME/size/hash safeguards có; regression PG/object deployment còn cần chạy. | K07 |
| W4-09 | Lưu raw evidence/provider payload theo DEC-10 khi bật; trả `raw_payload_ref` opaque, không trả object URI/credential cho model hoặc user không có role auditor. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07 |
| W4-10 | TTL cleanup idempotent cho transient upload/raw payload; DB row và object không bị orphan khi một phía delete lỗi. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07 |
| W4-11 | Chốt DEC-04 theo class `transient/session/audit`; policy là config versioned, không hard-code trong handler. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07 |
| W4-12 | Implement session deletion workflow có tombstone/audit, cascade theo policy và object cleanup; API không leak session đã xoá của owner khác. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07 |
| W4-13 | Backup/restore PostgreSQL và object store; diễn tập restore vào môi trường cô lập rồi kiểm source graph, event sequence và hashes. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07 |
| W5-01 | Biến event handling thành reducer có test: cursor chỉ tăng, dedupe `event_id`, phát hiện gap và chạy REST reconcile trước khi nối lại SSE. | Reducer/dedupe/cursor tests có; frontend suite57pass; cross-instance real event integration còn thiếu. | K03/K08 |
| W5-02 | Bootstrap từ `GET session/messages/events:list` đủ trang để tái dựng analysis-by-run, recovery banners và validation history sau reload; không phụ thuộc map chỉ thu được khi browser đang mở. | REST bootstrap/pagination có implementation; kiểm session dài/reload trong real stack. | K03/K08 |
| W5-03 | Test offline → reconnect, tab sleep/wake, duplicate event, missed event, expired token và session switch; chỉ draft/UI preference vào `localStorage`. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K03/K08 |
| W5-04 | Thêm pending user-send theo `client_message_id`; không optimistic assistant answer hoặc analysis. | Pending user send theo client_message_id có; cần ambiguous failure/retry/edit/refresh matrix. | K03/K08 |
| W5-05 | Nối evidence artifact/list/detail; claim citation mở đúng record, hiện title/authors/source/retrieved-at/excerpt/status và external link đã normalize. | EvidenceArtifact và source/claim UI có; structured citation preview/repeated claims còn K08/K11. | K03/K08 |
| W5-06 | Nối attribution viewer cho một endpoint/assay; hiển thị token/atom contribution trung tính và limitation không-causality ngang hàng nội dung. | Attribution/AnalysisPanel có; selected model/bundle state và live viewer acceptance còn K04/K06. | K03/K08 |
| W5-07 | Deep-link claim → observation/evidence → field path; nếu artifact không còn do retention, hiện trạng thái “đã hết hạn” thay vì 404 thô. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K03/K08 |
| W5-08 | Fallback badge cạnh tiêu đề, violations sau nút Chi tiết, limitations không collapse mặc định và không dùng màu để suy mức độc. | Fallback/limitations/detail UI đã có; verify all terminal/recovery states theo new_plan. | K03/K08 |
| W5-09 | Hiển thị cancel/recovery/deadline/predictor unavailable đúng state. `requested=true` không đồng nghĩa run đã cancelled; hiện cảnh báo `potentially_billed` khi backend cung cấp. | RecoveryBanner và run state có; actionable retry/billing unknown/default no-runtime còn thiếu. | K03/K08 |
| W5-10 | OCR UI hiển thị file/preview an toàn, progress, recognized SMILES, confidence nếu contract cho phép, khả năng sửa SMILES trước một phân tích mới và lỗi capability unavailable rõ ràng. | Image upload/preview/OCR/result UI có; config preservation I09 và real OCR error matrix còn thiếu. | K03/K08 |
| W5-11 | Thêm Vitest + React Testing Library cho reducer, markdown sanitizer, exact rendered-value linking và critical state components. | Đã có Vitest/RTL; audit57pass, jsdom canvas không chứng minh depiction đúng. | K03/K08 |
| W5-12 | Thêm Playwright E2E cho SMILES/ảnh/vẽ, report Q&A, evidence, attribution, cancel, recovery, reload/reconnect và permission boundaries. | Đã có8Playwright mockpass; còn report/research/recovery/provider switching trên real stack. | K03/K08 |
| W5-13 | Accessibility: keyboard/focus cho dialogs/panel, semantic labels, screen-reader announcements cho run status; kiểm desktop/tablet/mobile. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K03/K08 |
| W5-14 | Code-split molecule editor, artifact viewers và route chunks; đặt bundle budget, không tải `openchemlib` trước khi người dùng mở editor. | Code splitting/build budgets đã có và build pass; lazy editor gzip357.51KB, còn throttled visual/perf check. | K03/K08 |
| W5-15 | Không hardcode backend enums có thể mở rộng như violation code; sinh hoặc kiểm API types từ OpenAPI trong CI. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K03/K08 |
| W6-01 | Structured logs có `request_id/session_id/run_id/binding_id`, nhưng redact Authorization, prompt/evidence raw và capability token. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-02 | OpenTelemetry spans qua API → scheduler → runtime → MCP tool → predictor/evidence/OCR; trace IDs đi qua service boundary. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-03 | Metrics theo plan §15: product outcome, runtime health/loss/cancel, model token/cost, tool latency/error, validator/fallback, evidence yield, outbox lag/SSE reconnect và restore failure. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-04 | Dashboard alpha: success/failure theo intent, first-pass acceptance, fallback, p50/p95 latency, token/cost, citation support sample và dependency readiness. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-05 | Alert ban đầu chỉ cho invariant nghiêm trọng: cross-session leak, incomplete claim-source graph, stuck runs, readiness outage, outbox lag và orphan process; tinh chỉnh ngưỡng sau dữ liệu alpha. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-06 | Thêm control-plane job: install locked dev deps; unit, contract, integration, eval schema/graders và scripted eval. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-07 | Thêm frontend job: typecheck, policy lint, unit/component tests, production build và bundle budget. | Có script FE và kết quả local pass; unified-ci hiện không có FE quality job. | K01/K12 |
| W6-08 | Thêm toxocr contract job không tải checkpoint; model/checkpoint smoke chạy riêng theo schedule/manual hoặc runner có artifact cache và timeout. | OCR6testpass local; unified-ci hiện chưa có OCR contract/model job. | K01/K12 |
| W6-09 | Thêm PostgreSQL service job + Alembic migration test. | PG tests/migrations có nhưng unified-ci hiện không có PostgreSQL service gate. | K01/K12 |
| W6-10 | Thêm container build/smoke cho control plane, frontend và toxocr; multi-service smoke dùng predictor stub cho PR và model thật ở protected job. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-11 | Pin dependency/runtime hashes và cache hợp lý; live provider secrets chỉ có trong protected manual/release workflow. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-12 | Lưu eval manifest/results, test reports và SBOM làm CI artifacts; không upload raw secrets/evidence ngoài retention policy. | Lịch sử ghi done; unified-ci hiện không upload eval/test/SBOM artifacts đầy đủ. | K01/K12 |
| W6-13 | Tạo deploy artifacts cho bốn boundary: frontend, control plane, ToxPred, toxocr; OpenCode/DSH chạy private runtime host, không expose management port public. | Docker/Compose boundary artifacts đã có; deploy workflow vẫn legacy I30. | K01/K12 |
| W6-14 | Cấu hình health/live và health/ready đúng dependency semantics; startup không báo ready trước predictor/runtime/OCR bắt buộc. | Live/ready routes có nhưng dependency semantics sai I05; phải sửa và fault-test. | K01/K12 |
| W6-15 | Cấu hình private MCP URL, egress allowlist cho EuropePMC/provider, CORS origin chính xác, secret manager và least-privilege service accounts. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K01/K12 |
| W6-16 | Viết runbook deploy, migration, smoke, rollback, rotate secret, dependency outage, stuck run, orphan cleanup, backup và restore. | Operations/migration runbooks đã có; command drift và actual drills còn thiếu. | K01/K12 |
| W6-17 | Xoá hoặc viết lại runbook Docker legacy đang trỏ `model_server/main.py`, `/analyze` và aggregate verdict đã bị loại bỏ. | Runbook đã đổi một phần; còn paths/URI provisioning/workflow mismatch I27/I32. | K01/K12 |
| W7-01 | Tạo environment cô lập; cài exact official SDK + matching runtime wheel. Tại ngày lập plan, candidate mới nhất quan sát được là pre-release `0.1.2rc1`; phải kiểm lại và pin version/hash tại lúc implement. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-02 | Smoke `initialize → session/prompt → session.event/status → shutdown` qua Python SDK; capture protocol fixture và binary/package hashes. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-03 | Xác minh platform wheel, startup latency, stderr/stdout purity, process reaping, session root và behavior khi provider credential thiếu. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-04 | Chốt ADR DEC-06: version, carrier, supported platforms, upgrade policy và developer-preview risk. Không dùng package trùng tên `deepseek-harness` của tác giả khác. | ADR/runtime matrix giữ DSH experimental; không dùng đính chính carrier như admission supported. | K09/K13 |
| W7-05 | Tạo custom `cordis.yml` từ profile SDK tối thiểu: loại bash/file edit/subagent/direct web; chỉ nạp MCP client và ToxAgent instructions cần thiết. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-06 | Cô lập `DSH_HOME`, workspace và session root theo run; không discovery `~/.dsh`; env allowlist chỉ mang credential/provider cần thiết. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-07 | Capture model-visible surface và chứng minh exact ToxAgent MCP allowlist. Direct denied execution phải fail ở transport. | Surface snapshot có; authenticated roundtrip chưa đủ evidence theo unified-v2. | K09/K13 |
| W7-08 | Implement adapter mỏng theo `AgentRuntimeProvider`: health, capabilities, create, send, normalize event/status, close; không đưa DSH type vào application/domain. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-09 | Map protocol limitation trung thực: hiện không có mid-turn cancel hay session close; `runtime_cancel_supported=false`. Nếu policy chọn kill owned process để dừng, action phải nói rõ process termination, không giả là prompt cancel. | Protocol limits phải giữ truthful experimental; chưa chứng minh supported cancellation. | K09/K13 |
| W7-10 | Shutdown/timeout luôn reap process và child; stdout chỉ có JSON-RPC, diagnostics đọc stderr có bound/redaction. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-11 | Contract snapshot và version diff gate tương tự OpenCode. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-12 | Chạy cùng tool contract và deterministic fixtures qua DSH. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-13 | Paired OpenCode/DSH Track A: cùng provider/model, prompt, tools, fixtures và budgets; tối thiểu ba trial/release candidate. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-14 | Track B: deployment thực tế, so reliability/latency/cost/ops burden; không gán mọi chênh lệch cho harness. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W7-15 | Scientific observations và accepted source graph phải byte/semantic equivalent bất kể runtime; denied tool/cross-session leak bằng 0. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K09/K13 |
| W8-01 | Deploy staging/alpha dùng PostgreSQL, object store, OpenCode pinned, EuropePMC và toxocr; static token chỉ được phép vì environment là alpha. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-02 | Chọn tập use case: analysis, report Q&A, attribution, evidence, conflicting/no evidence, OCR và recovery; cung cấp sample molecules không có dữ liệu nhạy cảm. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-03 | Viết hướng dẫn phạm vi: screening/decision support, không phải chẩn đoán, safety assessment hay regulatory decision. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-04 | Bật telemetry/redaction, cost budget theo user/run, feedback form gắn session/run/answer ID và cơ chế báo scientific concern. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-05 | Diễn tập dependency outage, cancel, runtime recovery, DB restore và rollback trước khi mời reviewer. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-06 | Chạy tối thiểu một tuần hoặc đủ sample đã chốt; không chốt SLO từ vài smoke run. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-07 | Hai SME review mù tối thiểu 20% capability answers và tất cả critical failures/fallbacks. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-08 | Triage feedback trong 48 giờ làm việc thành: bug, eval task, rubric clarification, UX issue, provider issue hoặc out-of-scope request. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-09 | Mỗi scientific regression có reproduction fixture/task trước fix; rerun affected category và critical set. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W8-10 | Chốt measured baseline về success, latency, cost, fallback, citation support, correction rate, reconnect và restore. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K13 |
| W9-01 | Chốt DEC-07 và provider terms: server-workload credential, không dùng shared personal OAuth/subscription cho production. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-02 | Backend OIDC/JWKS verifier: issuer/audience/expiry/algorithm allowlist, key rotation/cache, role mapping và fail-closed; không dùng cùng secret với MCP capability token. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-03 | Frontend Authorization Code + PKCE/session flow; token không nhập tay, logout/expiry/refresh rõ ràng và không log token. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-04 | Security review: owner/share scope, IDOR, MCP replay/revocation, prompt injection, SSRF/egress, file upload, CORS/CSRF assumptions, secret redaction và dependency/container scan. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-05 | Abuse controls phân tán: per-user/session concurrency, size/batch/run budgets, provider rate limit/circuit breaker và duplicate/cyclic tool detector. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-06 | Load test deterministic analysis và mixed workload; đo DB pool, outbox lag, SSE connections, runtime host capacity, OCR queue và provider throttling. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-07 | Soak/failure injection: predictor 503/malformed/slow, evidence 429/timeout, OCR hang, runtime disconnect/hung, DB conflict, outbox duplicate, object store unavailable và node restart. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-08 | Chốt SLO từ alpha data; alert có owner, severity và runbook. Không dùng candidate numbers trong rebuild plan như cam kết trước khi đo. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-09 | Production eval: critical `pass^5=100%`, capability pass@1 >=85% và không category <80%, citation support >=98%, numeric source 100%, major SME correction <=10%. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-10 | Canary runtime/model/provider/profile/tool/prompt upgrade; manifest diff, eval non-regression, rollback tự động hoặc một lệnh đã diễn tập. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-11 | Restore drill từ backup, deletion/retention audit và disaster recovery runbook được người không viết runbook thực hiện thành công. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W9-12 | Production go/no-go review có sign-off từ engineering, security, product/SME và owner của provider/credential terms. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K07/K12/K13 |
| W10-01 | Session search toàn lịch sử bằng API, không chỉ filter 25/50 rows đã tải ở frontend. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K08/K11 |
| W10-02 | Rename/pin/archive/delete session với audit và retention semantics. | Rename/auto-title đã có; pin/archive/delete+retention chưa đóng toàn yêu cầu. | K08/K11 |
| W10-03 | Pagination/cursor ổn định cho session history và artifact lists. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K08/K11 |
| W10-04 | Export report/audit bundle đã sanitize, gồm manifest và source graph, không kèm raw evidence trái policy. | Chưa đóng toàn yêu cầu; giữ acceptance nguồn, hoàn tất/kiểm chứng trong gói tương ứng. | K08/K11 |
| W10-05 | Expert-only threshold override UI; server role check là authority, snapshot hiện `threshold_source=request_override` rõ ràng. | Threshold fields/server policy có; kiểm role authority và UX across input còn thiếu. | K08/K11 |
| W10-06 | Đánh giá provider evidence thứ hai chỉ khi alpha cho thấy EuropePMC thiếu coverage; phải có ADR và paired normalization/citation tests. | Conditional: chỉ thêm provider sau alpha có bằng chứng thiếu coverage. | K08/K11 |

## 11. Exit gates của W0–W10 và tiêu chí hoàn tất tổng

- **W0:** working slice được review, lệnh chạy tái lập, source-of-truth/decision table không mâu thuẫn lớn; baseline không trộn fixture/score.
- **W1:** live frozen numeric + integration/evidence đúng mode; full suite/trials, critical pass^3 và capability threshold; skip/cost/fallback minh bạch, failures được phân loại.
- **W2:** runtime contract, denied surface, cancellation/SSE/restart/compaction/recovery/reaper qua live failure injection; no duplicate state và billing ambiguity được ghi.
- **W3:** live attribution/evidence accepted qua cùng GroundedAnswer; citation validity/support/SME đạt alpha và injection không mở rộng authority.
- **W4:** PostgreSQL atomic invariants, cross-instance correctness, object ACL/MIME/lifecycle và restore không dangling data; có migration/drill evidence.
- **W5:** UI bootstrap/reducer/auth/reload/offline/recovery, evidence/XAI/deeplink/expired/billing state, Vitest/Playwright/a11y/build budgets đạt ở luồng chính.
- **W6:** telemetry/alerts, CI đủ service+PG+containers, pinned release/SBOM/results, readiness/network/topology và runbooks đã thực hành.
- **W7:** DSH authenticated tool/stream/cancel/session isolation/reap và paired trials đạt cùng gate, hoặc ADR unsupported có version và evidence. Experimental không được tính supported.
- **W8:** alpha scope/cohort/cost/feedback/drills, ít nhất một tuần, hai SME/adjudication và triage48h; feedback chuyển thành eval version mới.
- **W9:** hosted identity/security/terms, distributed caps/load/soak/SLO, production evaluation/canary/restore/deletion, engineering+security+science+product go/no-go.
- **W10:** search toàn lịch sử, session lifecycle/cursor/export/expert threshold đúng authority; evidence provider thứ hai có decision từ alpha.

Chỉ kết luận toàn bộ kế hoạch đã hoàn tất khi từng yêu cầu ở các ma trận có PR/implementation, test đúng phạm vi, artifact/manifest kiểm chứng được và quyết định owner cho phần conditional. Các scientific/runtime capability chưa admitted phải còn nhãn blocked/experimental; nếu loại khỏi release thì cần ADR scope, không tự đổi trạng thái thành done.

Hai tài liệu audit đã được tạo không có nghĩa implementation đã đạt các gate này. Bước triển khai đầu tiên là K01 và contract K02/K03; không khởi đầu bằng thay kernel, huấn luyện lại model hoặc chỉ làm lại giao diện.
