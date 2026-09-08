# Kiểm toán issue hệ thống ToxAgent — 08/09/2026

## Phạm vi và kết luận

Đối chiếu worktree tại HEAD `ce49f5d` cùng `new_plan.md` chưa tracked, source, cấu hình, artifact cards và các kiểm thử dưới đây. Có **32 issue**, gồm sáu vấn đề người dùng cung cấp và các lỗi/gap vận hành bổ sung. Các issue có chung nguyên nhân vẫn tách khi đường sửa và acceptance khác nhau. P1 là chặn luồng chính hoặc release phù hợp phạm vi; P2 là correctness/UX hoặc rủi ro có điều kiện. Không đồng nhất mọi P1 với sự cố production đang xảy ra.

Hệ thống đã có predictor, OCR, SQL persistence, runtime gateway, evidence validation, model catalog/selection và semantic activity. Các boundary này chưa được nối nhất quán: cấu hình đã lưu không luôn đi tới nơi thực thi, capability đã cấu hình chưa chắc khả dụng, và đường triển khai sạch chưa được CI kiểm chứng đầy đủ.

Audit không thay đổi implementation, không khởi động lại stack, không gọi model trả phí hay deploy cloud. Khi kiểm tra Docker, các container ToxAgent đã dừng; vì vậy thông tin run/session live trong yêu cầu được coi là bằng chứng người dùng cung cấp, không phải run vừa tái hiện. Không tuyên bố đã chứng minh hệ thống không còn lỗi ngoài danh sách.

Các xóa sẵn có `ToxAgent_02_Harness_Architecture.pptx`, `ToxAgent_03_Harness_Master_Plan.pptx`, `audit_5_9.md` và file mới `new_plan.md` được giữ nguyên. Thư mục ignored/legacy trên máy không tự động được coi là code đang serve.

## Bằng chứng kiểm thử trong phiên audit

| Phép kiểm | Kết quả thực tế | Phạm vi chứng minh |
|---|---|---|
| Frontend `npm test -- --run` | 18 files, 57 passed | Unit/component; jsdom cảnh báo canvas chưa implement, không chứng minh hình vẽ đúng |
| Frontend `npm run build`, `npm run lint:policy` | Pass | Type/build và policy; không phải visual approval |
| Frontend Playwright, 1 worker | 8 passed | API mock; không phải runtime/model/evidence live |
| Predictor unit + contract | 151 passed, 26 skipped | Các test được chạy; skip không tính pass |
| Predictor golden `PYTHONPATH=src …python -m pytest tests/golden -q -rs` | 7 passed, 7.47s | Artifact local, numerical parity và assertions golden trong bộ test; không phải clean-image release toàn stack |
| OCR test suite | 6 passed | Unit/contract OCR, không phải benchmark OCR accuracy |
| Control unit | 302 passed, 4 skipped, 2 failed | Hai lỗi profile path; 12.06s |
| Control `tests -x`, loại live/postgres | 41 passed, 12 skipped, 5 deselected, 1 error | Dừng tại snapshot path, 8.75s; không phải tổng kết full suite |
| Combined CI-style collection | 382 collected, 17 errors | Collection lỗi; chưa chạy assertions |
| Control full non-live suite, timeout180 | Exit124, log chưa hoàn tất | Không được ghi là pass/fail tổng; cần điều tra test chậm sau khi sửa fixture |
| `./bin/toxagent restore <file-tồn-tại>` | Exit1, chỉ in usage | Lỗi argv được tái hiện; không có restore DB thật |

Logs cục bộ ở `/tmp/toxagent-audit-20260908/`: `frontend-build.log`, `browser.log`, `predictor.log`, `golden.log`, `ocr.log`, `control-unit.log`, `control-first-error.log`, `ci-collection.log`, `control.log`. `/tmp` không phải kho release bền vững; số liệu và phạm vi đã được ghi lại trong tài liệu này. Python test env là `drug-tox-env` 3.10, riêng OCR dùng `toxocr-env`; không cài thêm dependency vào môi trường người dùng.

**Đính chính qua kiểm chứng:** đường dẫn golden dùng `parents[4]` là đúng và bộ golden thực tế pass. Không đưa nghi ngờ đó thành bug. CI vẫn cần provision artifact và fail khi unexpected skip; đây là vấn đề khác. Các con số benchmark/gate lịch sử trong progress không được cộng với kết quả phiên này.

## Danh mục issue

| ID | Mức | Căn cứ | Vấn đề | Gói sửa |
|---|---|---|---|---|
| I01 | P1 | Nguồn | Stack mặc định nhận diện một runtime không được khởi tạo | K02 |
| I02 | P1 | Nguồn | Admission research không kiểm tra đủ handler/runtime | K02 |
| I03 | P1 | Nguồn | Composer mặc định khóa prediction bởi target XAI Tox21 | K03 |
| I04 | P1 | Nguồn | Heuristic molecule bỏ sót câu hỗn hợp và nhận nhầm từ thường | K03 |
| I05 | P1 | Nguồn | Health ready chưa chứng minh toàn bộ khả năng sản phẩm | K02 |
| I06 | P2 | Nguồn | Session Config ghi Runtime mặc định khi không có runtime | K03 |
| I07 | P1 | Artifact/nguồn | ClinTox v1 vẫn bị chặn bởi tokenizer chính xác | K10 |
| I08 | P1 | Nguồn | Lựa chọn predictor bị mất ở mixed run | K04 |
| I09 | P1 | Nguồn | OCR không bảo toàn tùy chọn prediction/XAI/model | K04 |
| I10 | P1 | Nguồn | Tool tạo analysis/attribution chưa kế thừa binding model của run | K04 |
| I11 | P1 | Nguồn | Prediction model được chọn nhưng explanation không chọn cùng model | K04 |
| I12 | P1 | Nguồn | Profile AI lưu credential/base URL nhưng dispatch không sử dụng | K05 |
| I13 | P1 | Nguồn | Provider dựng sẵn không có base URL nhưng probe bắt buộc URL | K05 |
| I14 | P1 | Nguồn | Capability probe có thể chứng nhận tính năng chưa được kiểm tra | K05 |
| I15 | P1 | Nguồn; cần security test cô lập | Probe URL do người dùng nhập chưa có policy truy cập mạng | K05 |
| I16 | P1 | Nguồn | Secret và attachment không có storage bền vững trong Compose | K07 |
| I17 | P1 | Nguồn | Khởi động replica mới có thể fail run đang chạy trên replica cũ | K07 |
| I18 | P1 | Nguồn | Queue/cancel/recovery chưa đủ cho nhiều process | K07 |
| I19 | P1 | Nguồn | XAI bundle chưa checkpoint phần việc đã hoàn thành | K06 |
| I20 | P2 | Nguồn | Session summary trả preview cũ và đếm run bị giới hạn 10 | K08 |
| I21 | P2 | Nguồn | Enter chưa bảo vệ IME composition | K03 |
| I22 | P1 | Test | Đường dẫn profile mặc định sai sau chuyển sang src layout | K01 |
| I23 | P1 | Nguồn | Predictor manifest mặc định trỏ thư mục không tồn tại | K01 |
| I24 | P1 | Test | Control contract test đọc snapshot tại đường dẫn cũ | K01 |
| I25 | P1 | Test | CI gom hai package tests gây lỗi collection | K01 |
| I26 | P1 | Nguồn + test giới hạn | CI thiếu các gate đã từng được báo là hoàn tất | K01 |
| I27 | P1 | Nguồn | Fresh clone setup chưa provision predictor artifacts | K02 |
| I28 | P1 | Test + nguồn | Wrapper restore không nhận cú pháp được hướng dẫn | K07 |
| I29 | P2 | Nguồn | Wrapper logs/up/agent bootstrap chưa đúng contract thao tác | K02 |
| I30 | P1 | Nguồn | Workflow deploy vẫn trỏ topology cũ và đích live từ agent_test | K12 |
| I31 | P2 | Nguồn; đường thử nghiệm | Kernel budget/coverage chưa đủ điều kiện cutover | K09 |
| I32 | P2 | Nguồn | Tài liệu vận hành và bằng chứng lịch sử chưa theo layout hiện tại | K01 |

## I01 — Stack mặc định nhận diện một runtime không được khởi tạo

**P1 · Nguồn · K02.** Nguồn: [devops/compose/compose.yaml](../../devops/compose/compose.yaml), [backend/control/src/toxagent/api/app.py](../../backend/control/src/toxagent/api/app.py).

**Bằng chứng và điều kiện:** Compose đặt runtime kind scripted, nhưng app chỉ tự dựng adapter OpenCode. Handler agentic chỉ được đăng ký khi có runtime provider.

**Tác động:** Người dùng mở Session trong stack mặc định rồi yêu cầu research/report/attribution sẽ không có agent thực thi. Đây là cấu hình có chủ ý cho prediction/OCR nhưng không đáp ứng trải nghiệm Session được quảng bá.

**Hướng xử lý:** Khai báo deployment mode và runtime configured/available riêng; luồng setup agent phải dựng và kiểm tra runtime thật. Không inject scripted để giả lập production.

**Tiêu chí đóng:** Stack predictor-only báo rõ giới hạn; stack agent-enabled chạy đủ ba intent với provider thật.

## I02 — Admission research không kiểm tra đủ handler/runtime

**P1 · Nguồn · K02.** Nguồn: [backend/control/src/toxagent/api/app.py](../../backend/control/src/toxagent/api/app.py), [backend/control/src/toxagent/application/submit_message.py](../../backend/control/src/toxagent/application/submit_message.py), [backend/control/src/toxagent/application/run_scheduler.py](../../backend/control/src/toxagent/application/run_scheduler.py).

**Bằng chứng và điều kiện:** evidence_research_available lấy từ research_provider is not None; điều này không chứng minh scheduler có handler. Scheduler thất bại tại nhánh no handler is registered.

**Tác động:** Request được nhận, tạo message/run rồi fail ngay dù có thể từ chối trước. Có EuropePMC provider không đồng nghĩa có agent runtime.

**Hướng xử lý:** Một capability resolver dùng chung cho UI, API admission và scheduler; trả lỗi cấu hình có thể hành động trước khi queue.

**Tiêu chí đóng:** Provider evidence tồn tại nhưng runtime absent: không tạo run mồ côi; mã lỗi và UI thống nhất.

## I03 — Composer mặc định khóa prediction bởi target XAI Tox21

**P1 · Nguồn · K03.** Nguồn: [frontend/src/components/workbench/MessageComposer.tsx](../../frontend/src/components/workbench/MessageComposer.tsx).

**Bằng chứng và điều kiện:** Default endpoints herg/tox21, tox21Tasks rỗng; needsTox21Target tham gia canSend. Payload SMILES luôn đặt explanation_mode required.

**Tác động:** Nhập CCO trên cấu hình mặc định bị khóa dù chỉ muốn prediction; hướng dẫn nằm trong tùy chọn nâng cao.

**Hướng xử lý:** Tách prediction và explanation mode none/on_demand/required. Chỉ bắt assay khi người dùng thực sự chọn required; hiện lý do cạnh nút gửi.

**Tiêu chí đóng:** CCO gửi được ngay với defaults; required Tox21 thiếu assay bị chặn rõ; không âm thầm thay đổi lựa chọn required.

## I04 — Heuristic molecule bỏ sót câu hỗn hợp và nhận nhầm từ thường

**P1 · Nguồn · K03.** Nguồn: [frontend/src/components/workbench/MessageComposer.tsx](../../frontend/src/components/workbench/MessageComposer.tsx), [backend/control/src/toxagent/application/router.py](../../backend/control/src/toxagent/application/router.py).

**Bằng chứng và điều kiện:** looksLikeSmiles chỉ kiểm tra chuỗi không khoảng trắng thuộc tập ký tự rộng. Từ hello/aspirin cũng khớp; câu có SMILES lại không khớp. Backend yêu cầu subject rõ ràng.

**Tác động:** Câu hỏi kèm molecule có thể research_subject_missing; một từ thường có thể bị chuyển thành SMILES và mất text gốc. Ví dụ c1ccc1 của báo cáo cũ còn cần validation hóa học; dùng CCO khi kiểm tra luồng nhập hợp lệ.

**Hướng xử lý:** Có ô molecule rõ ràng hoặc parser đề xuất ứng viên có xác nhận và RDKit validation phía server; luôn giữ câu hỏi gốc. Không dùng LLM đoán danh tính không kiểm chứng.

**Tiêu chí đóng:** CCO, câu tiếng Việt + CCO, nhiều ứng viên, tên chất, chuỗi sai và câu hỏi về analysis hiện tại đều có hành vi xác định.

## I05 — Health ready chưa chứng minh toàn bộ khả năng sản phẩm

**P1 · Nguồn · K02.** Nguồn: [backend/control/src/toxagent/api/routes.py](../../backend/control/src/toxagent/api/routes.py).

**Bằng chứng và điều kiện:** Ready kiểm tra predictor và gateway nếu có; DB không được probe. OCR capability dựa vào object được cấu hình, không phải model OCR đã sẵn sàng. Kind vẫn là cấu hình.

**Tác động:** Có thể ready true trong deployment không có agent, DB lỗi hoặc OCR không dùng được; monitoring và UI dễ hiểu quá phạm vi.

**Hướng xử lý:** Tách liveness, readiness lõi và capability readiness với reason/checked_at; probe phụ thuộc bắt buộc có timeout. UI cần tiêu thụ capability thực.

**Tiêu chí đóng:** Fault injection DB/OCR/runtime; HTTP và capabilities phản ánh mode triển khai mà không làm predictor-only thất bại vì feature cố ý tắt.

## I06 — Session Config ghi Runtime mặc định khi không có runtime

**P2 · Nguồn · K03.** Nguồn: [frontend/src/components/workbench/SessionConfigPopover.tsx](../../frontend/src/components/workbench/SessionConfigPopover.tsx).

**Bằng chứng và điều kiện:** ai_profile_id null được hiển thị Runtime mặc định; popover không xác thực khả dụng runtime từ health.

**Tác động:** Không phân biệt chưa cấu hình với một cấu hình mặc định sử dụng được.

**Hướng xử lý:** Hiện Chưa cấu hình AI cùng đường dẫn Settings; chỉ dùng nhãn mặc định nếu server xác nhận có default khả dụng.

**Tiêu chí đóng:** Null profile + danh sách rỗng + gateway absent không mang thông điệp sẵn sàng.

## I07 — ClinTox v1 vẫn bị chặn bởi tokenizer chính xác

**P1 · Artifact/nguồn · K10.** Nguồn: [docs/artifacts/clintox-smilesgnn-v1.md](../../docs/artifacts/clintox-smilesgnn-v1.md), [backend/predictor/registry/predictor-manifest.yaml](../../backend/predictor/registry/predictor-manifest.yaml).

**Bằng chứng và điều kiện:** Checkpoint yêu cầu mapping 69 token; tokenizer khác trên máy không chứng minh đúng mapping huấn luyện. Manifest giữ trạng thái không serve.

**Tác động:** Không thể cung cấp endpoint ClinTox từ artifact v1 hiện có; hERG/Tox21 không phải substitute. Đây là thiếu capability, không phải nguyên nhân lỗi runtime.

**Hướng xử lý:** Khôi phục tokenizer gốc có hash/mapping/special tokens hoặc retrain v2 tái lập đầy đủ. Tiếp tục fail closed cho v1.

**Tiêu chí đóng:** Admission kiểm tra vocab và mapping, golden parity hoặc bộ benchmark v2 độc lập, threshold provenance.

## I08 — Lựa chọn predictor bị mất ở mixed run

**P1 · Nguồn · K04.** Nguồn: [backend/control/src/toxagent/harness/gateway.py](../../backend/control/src/toxagent/harness/gateway.py).

**Bằng chứng và điều kiện:** _snapshot_before_runtime truyền smiles/endpoints/threshold_overrides nhưng bỏ model_selection và cấu hình explanation đang có trong RunContext.

**Tác động:** Session chọn model B nhưng câu hỏi kèm molecule có thể tạo snapshot bằng default A; run config và kết quả không nhất quán.

**Hướng xử lý:** Một cấu hình run đã resolve bất biến được truyền xuyên suốt mọi entry point.

**Tiêu chí đóng:** Spy hai provider cùng endpoint: mixed run chỉ gọi B; snapshot và observation ghi B, kể cả sau recovery.

## I09 — OCR không bảo toàn tùy chọn prediction/XAI/model

**P1 · Nguồn · K04.** Nguồn: [frontend/src/components/workbench/MessageComposer.tsx](../../frontend/src/components/workbench/MessageComposer.tsx), [backend/control/src/toxagent/api/app.py](../../backend/control/src/toxagent/api/app.py), [backend/control/src/toxagent/application/recognize_structure.py](../../backend/control/src/toxagent/application/recognize_structure.py).

**Bằng chứng và điều kiện:** Composer chỉ gửi analysis_options khi effectiveSmiles có giá trị; handler OCR/CreateAnalysis không truyền toàn bộ model_selection và explanation targets.

**Tác động:** Ảnh là đầu vào duy nhất có thể dùng endpoints/model/XAI mặc định thay vì lựa chọn vừa lưu hoặc vừa chọn.

**Hướng xử lý:** Đưa cấu hình analysis độc lập loại input; OCR chỉ chuyển ảnh thành molecule, không reset cấu hình run.

**Tiêu chí đóng:** Ảnh + chọn hERG/model B/threshold riêng giữ nguyên cả ba; required XAI có artifact hoặc failure minh bạch.

## I10 — Tool tạo analysis/attribution chưa kế thừa binding model của run

**P1 · Nguồn · K04.** Nguồn: [backend/control/src/toxagent/tools/definitions/analysis.py](../../backend/control/src/toxagent/tools/definitions/analysis.py).

**Bằng chứng và điều kiện:** Lời gọi create_analysis.execute và predictor.attribution không nhận selected model từ run context.

**Tác động:** Agent tool có thể dùng default khác với lựa chọn session. Không nên trao quyền tùy ý chọn provider cho model ngôn ngữ để vá lỗi này.

**Hướng xử lý:** Server inject immutable model binding vào ToolContext và mọi tool invocation.

**Tiêu chí đóng:** Tool không có đối số tùy ý để vượt binding; cross-endpoint/cross-owner bị chặn và provenance đúng.

## I11 — Prediction model được chọn nhưng explanation không chọn cùng model

**P1 · Nguồn · K04.** Nguồn: [backend/control/src/toxagent/application/create_analysis.py](../../backend/control/src/toxagent/application/create_analysis.py).

**Bằng chứng và điều kiện:** predict nhận model_selection tại dòng 98; explain tại dòng 105 chỉ nhận canonical_smiles, endpoint, task.

**Tác động:** Khi có nhiều model một endpoint, XAI có thể giải thích default model hoặc gặp ambiguity thay vì model đã prediction. Với chỉ một model hiện tại lỗi chưa nhất thiết lộ ra.

**Hướng xử lý:** Pin model/version/hash/target cho explain API và cache key; kiểm tra consistency trước khi ghép bundle.

**Tiêu chí đóng:** Hai provider trả probability và attribution khác nhau; yêu cầu B không bao giờ nhận attribution A.

## I12 — Profile AI lưu credential/base URL nhưng dispatch không sử dụng

**P1 · Nguồn + mock reproduction · K05.** Nguồn: [backend/control/src/toxagent/connections/service.py](../../backend/control/src/toxagent/connections/service.py), [backend/control/src/toxagent/harness/gateway.py](../../backend/control/src/toxagent/harness/gateway.py).

**Bằng chứng và điều kiện:** _resolve_ai_profile chỉ trả provider_id/model_id/connection.id; gateway không resolve credential_ref/base_url/auth_mode vào cấu hình runtime.

**Tác động:** Profile test thành công không chứng minh run dùng credential hoặc server đó. Runtime có thể dùng auth sẵn trên host, sai account/cost boundary hoặc không kết nối được.

**Hướng xử lý:** Resolve profile đầy đủ tại trust boundary; secret được đưa vào runtime cô lập của owner và thu hồi sau dùng; binding lưu fingerprint/config không lưu secret.

**Tiêu chí đóng:** Hai owner cùng provider/model nhưng URL/key khác: stub server của từng owner nhận đúng request; không dùng ambient auth.

## I13 — Provider dựng sẵn không có base URL nhưng probe bắt buộc URL

**P1 · Nguồn · K05.** Nguồn: [frontend/src/pages/SettingsPage.tsx](../../frontend/src/pages/SettingsPage.tsx), [backend/control/src/toxagent/connections/service.py](../../backend/control/src/toxagent/connections/service.py).

**Bằng chứng và điều kiện:** OpenAI/Anthropic/Gemini mặc định baseUrl rỗng; UI ghi optional, probe ném ValueError nếu absent. Tất cả dùng một OpenAI-compatible chat/completions probe.

**Tác động:** Tạo connection qua form hợp lệ nhưng Test thất bại; chưa có protocol adapter riêng chứng minh các provider được liệt kê đều được hỗ trợ.

**Hướng xử lý:** Provider registry khai báo protocol/default URL/auth/capabilities; UI chỉ liệt kê adapter được hỗ trợ; lỗi probe trả mã và thông điệp đã lọc secret.

**Tiêu chí đóng:** Dùng defaults của từng provider hoàn tất create/test/run; provider unsupported được nói rõ ngay.

## I14 — Capability probe có thể chứng nhận tính năng chưa được kiểm tra

**P1 · Nguồn + mock reproduction · K05.** Nguồn: [backend/control/src/toxagent/connections/service.py](../../backend/control/src/toxagent/connections/service.py).

**Bằng chứng và điều kiện:** Chỉ cần line bắt đầu data: để đặt streaming; tool_calls=True và structured_output=True vô điều kiện sau HTTP success.

**Tác động:** MockTransport trong audit trả duy nhất `data: [DONE]`; probe thực tế trả `streaming=True, tool_calls=True, structured_output=True`. Không có network request thật. Endpoint như vậy vẫn có thể được coi READY cho tool calling/JSON. Sai capability gây lỗi giữa run.

**Hướng xử lý:** Parse protocol; kiểm tra content JSON và tool call round-trip riêng; unknown khác unsupported; timeout và lỗi có cấu trúc.

**Tiêu chí đóng:** Fixtures DONE-only, malformed JSON, tools ignored, non-stream đều không được quảng cáo đủ capabilities.

## I15 — Probe URL do người dùng nhập chưa có policy truy cập mạng

**P1 · Nguồn; cần security test cô lập · K05.** Nguồn: [backend/control/src/toxagent/connections/service.py](../../backend/control/src/toxagent/connections/service.py), [backend/control/src/toxagent/api/routes.py](../../backend/control/src/toxagent/api/routes.py).

**Bằng chứng và điều kiện:** base_url được nối /chat/completions rồi gọi bằng httpx từ server; không thấy kiểm soát destination ở đường probe này.

**Tác động:** Người có quyền tạo connection có thể khiến server gửi POST tới địa chỉ nội bộ. Mức khai thác phụ thuộc egress và chế độ local/hosted; audit không gửi request vào mạng nội bộ để khai thác.

**Hướng xử lý:** Policy theo deployment: local self-host có opt-in rõ, hosted kiểm tra URL/DNS/IP/redirect và egress; áp quota/timeouts, không lộ secret trong lỗi.

**Tiêu chí đóng:** Mock DNS rebinding/private IPv4/IPv6/redirect; remote allowed vẫn dùng được; logs không chứa credential.

## I16 — Secret và attachment không có storage bền vững trong Compose

**P1 · Nguồn · K07.** Nguồn: [backend/control/src/toxagent/api/app.py](../../backend/control/src/toxagent/api/app.py), [backend/control/src/toxagent/config.py](../../backend/control/src/toxagent/config.py), [devops/compose/compose.yaml](../../devops/compose/compose.yaml).

**Bằng chứng và điều kiện:** App dùng filesystem object_store và model-secrets bên cạnh nó; control service không mount volume cho hai vùng này, trong khi PostgreSQL có volume.

**Tác động:** Recreate container mất file/key nhưng DB vẫn còn refs và trạng thái connection. Nhiều replica không thấy cùng attachment/secret.

**Hướng xử lý:** Local mount volume đúng quyền; hosted dùng object store và secret manager qua interface; backup/restore/retention đồng bộ metadata và object.

**Tiêu chí đóng:** Upload/create profile, recreate container, retrieve/use lại thành công; restore sang instance mới không có dangling refs.

## I17 — Khởi động replica mới có thể fail run đang chạy trên replica cũ

**P1 · Nguồn · K07.** Nguồn: [backend/control/src/toxagent/application/startup_reconciliation.py](../../backend/control/src/toxagent/application/startup_reconciliation.py), [backend/control/src/toxagent/api/app.py](../../backend/control/src/toxagent/api/app.py).

**Bằng chứng và điều kiện:** Startup list_non_terminal toàn DB và fail tất cả; suy luận worker cũ đã chết chỉ đúng cho single-process exclusive deployment.

**Tác động:** Rolling deploy hoặc scale-out gây mất run hợp lệ và có thể ghi potentially_billed dù worker vẫn thực thi.

**Hướng xử lý:** Lease/worker ownership/heartbeat và fencing token; chỉ reconcile lease hết hạn. Giữ replica=1 trước khi gate đa replica đạt.

**Tiêu chí đóng:** A đang chạy; B startup không thay trạng thái run A; kill A, hết lease thì chỉ một worker recover.

## I18 — Queue/cancel/recovery chưa đủ cho nhiều process

**P1 · Nguồn · K07.** Nguồn: [backend/control/src/toxagent/application/run_scheduler.py](../../backend/control/src/toxagent/application/run_scheduler.py).

**Bằng chứng và điều kiện:** Tasks nằm trong RAM; cancel remote ghi cờ nhưng không có local worker để task.cancel; startup không replay original request đã mất.

**Tác động:** Request accepted không có durable execution guarantee; cancel qua replica khác không bảo đảm dừng tác vụ deterministic kịp thời.

**Hướng xử lý:** Durable job envelope, claim atomically, heartbeat/fencing, cancel polling/notification và checkpoint idempotent.

**Tiêu chí đóng:** Kill sau accept/trước execute; kill sau predictor; cancel từ B; không chạy trùng hoặc double-commit và thông báo billing đúng.

## I19 — XAI bundle chưa checkpoint phần việc đã hoàn thành

**P1 · Nguồn · K06.** Nguồn: [backend/control/src/toxagent/application/create_analysis.py](../../backend/control/src/toxagent/application/create_analysis.py).

**Bằng chứng và điều kiện:** Predict và các explain được tính trước commit kết quả; lưu explanation rows chưa đồng nghĩa có bundle state bền vững theo từng target.

**Tác động:** Crash giữa nhiều assay mất phần đã tính; retry có thể tính lại toàn bộ và tăng latency. Deadline tổng cần được áp ở cả lane deterministic, không chỉ timeout từng HTTP.

**Hướng xử lý:** Persist bundle/prediction/checkpoint từng target với idempotency key và deadline còn lại; terminal failure vẫn giữ kết quả hợp lệ.

**Tiêu chí đóng:** Kill sau target thứ nhất; retry chỉ làm phần còn lại; tổng execution không vượt budget dù từng request chưa timeout.

## I20 — Session summary trả preview cũ và đếm run bị giới hạn 10

**P2 · Nguồn · K08.** Nguồn: [backend/control/src/toxagent/application/sessions.py](../../backend/control/src/toxagent/application/sessions.py), [backend/control/src/toxagent/persistence/sql/repositories.py](../../backend/control/src/toxagent/persistence/sql/repositories.py).

**Bằng chứng và điều kiện:** List lấy messages limit50 theo sequence tăng dần rồi messages[-1]; run_count=len(runs) với limit10.

**Tác động:** Session trên 50 messages vẫn preview message thứ 50; trên 10 runs trả count10. N+1 queries tăng chi phí theo số session.

**Hướng xử lý:** Query latest message riêng, COUNT đúng hoặc đổi tên trường thành recent_run_count; aggregate summary trong một query/index phù hợp.

**Tiêu chí đóng:** Session 61 messages/12 runs trả preview61/count12; query count không tăng tuyến tính.

## I21 — Enter chưa bảo vệ IME composition

**P2 · Nguồn · K03.** Nguồn: [frontend/src/components/workbench/MessageComposer.tsx](../../frontend/src/components/workbench/MessageComposer.tsx).

**Bằng chứng và điều kiện:** Keydown gửi khi Enter và không Shift; không kiểm tra isComposing.

**Tác động:** Một số bộ gõ dùng Enter xác nhận composition có thể gửi bản nháp chưa hoàn chỉnh.

**Hướng xử lý:** Chặn submit khi nativeEvent.isComposing/composition state; kiểm tra keyboard và focus contract.

**Tiêu chí đóng:** Browser test compositionstart → Enter không gửi; compositionend → Enter gửi một lần.

## I22 — Đường dẫn profile mặc định sai sau chuyển sang src layout

**P1 · Test · K01.** Nguồn: [backend/control/src/toxagent/config.py](../../backend/control/src/toxagent/config.py), [backend/control/tests/unit/test_opencode_profile.py](../../backend/control/tests/unit/test_opencode_profile.py).

**Bằng chứng và điều kiện:** PROJECT_ROOT trỏ backend/control/src; profiles_dir thành src/agent_profiles thay vì backend/control/agent_profiles. Hai unit tests FileNotFoundError.

**Tác động:** Local setup dùng defaults không tìm thấy profile; env override hoặc image packaging có thể che lỗi.

**Hướng xử lý:** Dùng package resources hoặc resolve service root rõ; kiểm tra source checkout, wheel và image riêng.

**Tiêu chí đóng:** Hai test profile pass và runtime bootstrap dùng đúng profile trong clean install.

## I23 — Predictor manifest mặc định trỏ thư mục không tồn tại

**P1 · Nguồn · K01.** Nguồn: [backend/predictor/src/toxpred/settings.py](../../backend/predictor/src/toxpred/settings.py).

**Bằng chứng và điều kiện:** REPO_ROOT là backend/predictor/src; default manifest nằm src/artifacts/predictor-manifest.yaml. Manifest thực thuộc registry; Docker override che lỗi.

**Tác động:** Developer chạy app mặc định có thể không load registry dù artifact đã provision.

**Hướng xử lý:** Một contract tìm manifest dùng package resource/explicit required setting, thông báo preflight có đường dẫn đã resolve.

**Tiêu chí đóng:** App khởi tạo trong checkout và clean wheel từ cwd bất kỳ, không dựa vào thư mục legacy.

## I24 — Control contract test đọc snapshot tại đường dẫn cũ

**P1 · Test · K01.** Nguồn: [backend/control/tests/contract/test_predictor_contract.py](../../backend/control/tests/contract/test_predictor_contract.py), [backend/control/src/toxagent/predictor/contract_snapshot.json](../../backend/control/src/toxagent/predictor/contract_snapshot.json).

**Bằng chứng và điều kiện:** Fixture tìm backend/control/toxagent/predictor/contract_snapshot.json, không phải src/toxagent/...; test -x xác nhận FileNotFoundError.

**Tác động:** Contract suite dừng vì fixture, không kiểm tra được compatibility API.

**Hướng xử lý:** Định vị snapshot theo package và thêm regeneration/diff thật, không chỉ git diff sau một test không sinh output.

**Tiêu chí đóng:** Contract suite chạy hết và thay đổi schema phá contract tạo failure có ý nghĩa.

## I25 — CI gom hai package tests gây lỗi collection

**P1 · Test · K01.** Nguồn: [.github/workflows/unified-ci.yml](../../.github/workflows/unified-ci.yml), [backend/control/tests/__init__.py](../../backend/control/tests/__init__.py), [backend/predictor/tests/__init__.py](../../backend/predictor/tests/__init__.py).

**Bằng chứng và điều kiện:** Lệnh collection cùng predictor/control unit+contract có 382 collected và 17 errors, gồm ModuleNotFoundError tests.unit.* do namespace tests trùng.

**Tác động:** Fast PR gate hỏng trước assertions; không thể dùng green lịch sử làm bằng chứng HEAD hiện tại.

**Hướng xử lý:** Chạy matrix mỗi service/cwd hoặc import-mode và namespace không đụng nhau; kiểm tra wheel không dựa vào PYTHONPATH tình cờ.

**Tiêu chí đóng:** Lệnh đúng như workflow trên clean env thu thập và chạy đủ số test kỳ vọng.

## I26 — CI thiếu các gate đã từng được báo là hoàn tất

**P1 · Nguồn + test giới hạn · K01.** Nguồn: [.github/workflows/unified-ci.yml](../../.github/workflows/unified-ci.yml), [frontend/e2e](../../frontend/e2e).

**Bằng chứng và điều kiện:** Fast gate không chạy FE/OCR/control integration/e2e/PG; container gate chỉ import predictor. Scientific job không provision artifact và có thể skip toàn bộ; benchmark chỉ --help.

**Tác động:** Regression UI mặc định và topology không bị bắt. Tám browser tests dùng mock, fixture hERG-only không bao phủ default herg+tox21. Golden local hiện đã pass7, không có lỗi đường dẫn golden.

**Hướng xử lý:** Bổ sung job theo service, full-stack smoke, PG migration, artifact admission/golden fail-on-unexpected-skip, frontend tests/build/policy/e2e và upload evidence.

**Tiêu chí đóng:** Chủ động làm hỏng defaults, contract, artifact provisioning hoặc migration phải khiến đúng gate đỏ.

## I27 — Fresh clone setup chưa provision predictor artifacts

**P1 · Nguồn · K02.** Nguồn: [bin/toxagent](../../bin/toxagent), [devops/compose/compose.yaml](../../devops/compose/compose.yaml), [backend/predictor/deploy/entrypoint.sh](../../backend/predictor/deploy/entrypoint.sh).

**Bằng chứng và điều kiện:** setup chỉ provision OCR; predictor mount .data/models ngoài Git. MODEL_ARTIFACTS_URI được khai báo nhưng entrypoint không gọi downloader để thực thi URI. Script `deploy/download_model_artifacts.py` có tồn tại nhưng không được Dockerfile hiện tại copy/nối vào startup.

**Tác động:** Máy đã có artifact chạy được không chứng minh khách hàng clone sạch chạy được; up có thể không đạt readiness.

**Hướng xử lý:** Provision immutable manifest bundle cho predictor/OCR, checksum, resumable cache và offline export; doctor kiểm tra đủ artifact.

**Tiêu chí đóng:** Clone sạch không dùng .data cũ: setup/up/smoke thành công; offline bundle và hash mismatch có đường xử lý đúng.

## I28 — Wrapper restore không nhận cú pháp được hướng dẫn

**P1 · Test + nguồn · K07.** Nguồn: [bin/toxagent](../../bin/toxagent).

**Bằng chứng và điều kiện:** Dispatcher shift command rồi restore yêu cầu $#=2 và đọc $2. Gọi restore với một file tồn tại trả usage trước mọi thao tác DB. Sau sửa argv, psql cũng cần ON_ERROR_STOP và semantics target rõ.

**Tác động:** Runbook khôi phục không thể sử dụng; tuyên bố replaces database chưa được implementation bảo đảm.

**Hướng xử lý:** Sửa parsing; kiểm tra gzip, xác định restore fresh DB hay replace có kiểm soát, fail atomic/exitcode; backup cả objects/secret refs.

**Tiêu chí đóng:** Backup rồi restore vào DB tạm, so sánh owners/messages/analyses/objects; lỗi SQL phải exit nonzero. Không thử phá DB thật.

## I29 — Wrapper logs/up/agent bootstrap chưa đúng contract thao tác

**P2 · Nguồn · K02.** Nguồn: [bin/toxagent](../../bin/toxagent).

**Bằng chứng và điều kiện:** logs dùng ${@:2} sau dispatcher shift nên bỏ service đầu. doctor chặn port đang listen kể cả stack của chính nó; agent_up --no-build phụ thuộc images có sẵn.

**Tác động:** Khó lấy log đúng service, up lặp không idempotent, agent fresh clone thiếu image. Dòng ready còn phụ thuộc health live của control.

**Hướng xử lý:** Parser nhất quán; doctor phân biệt port của stack và process khác; build/pull explicit; smoke capability trước ready.

**Tiêu chí đóng:** Shell harness mock docker/ss kiểm tra argv; up hai lần; clean image cache agent bootstrap; chỉ log service yêu cầu.

## I30 — Workflow deploy vẫn trỏ topology cũ và đích live từ agent_test

**P1 · Nguồn · K12.** Nguồn: `backend-autodeploy.yml` và `frontend-autodeploy.yml` tại thời điểm audit; hai file đã được thay bằng [.github/workflows/deploy.yml](../../.github/workflows/deploy.yml) và [devops/scripts/deploy_target.py](../../devops/scripts/deploy_target.py).

**Bằng chứng và điều kiện:** Backend build model_server/Dockerfile và deploy/cloudrun-env.yaml đã di chuyển; frontend deploy hosting live với URL hardcode, trong khi runbook yêu cầu preview/test service.

**Tác động:** Pipeline hiện không khớp layout; sửa mỗi path có thể đưa branch test vào đích dùng chung. Không có xác nhận cloud đang bị thay đổi trong audit này.

**Hướng xử lý:** Thiết kế lại deployment của các service, staging/prod tách biệt, validate manifests và environment protection; digest promotion sau gates.

**Tiêu chí đóng:** Dry-run target assertions với agent_test/main; staging smoke đủ auth/SSE/research; rollback về digest cũ đã thử.

## I31 — Kernel budget/coverage chưa đủ điều kiện cutover

**P2 · Nguồn; đường thử nghiệm · K09.** Nguồn: [backend/control/src/toxagent/agent/kernel.py](../../backend/control/src/toxagent/agent/kernel.py).

**Bằng chứng và điều kiện:** plan được gọi trước budget check; compose luôn gọi và cộng model_turns. Step hoàn tất đánh dấu question answered ngay cả observations rỗng, chưa kiểm success_condition. Stop reason có thể bị tính lại sau không đủ budget cho step.

**Tác động:** Kernel v2 có thể vượt budget hoặc báo coverage quá mức khi được nối vào production. Current app vẫn dùng compatibility gateway nên đây không phải nguyên nhân trực tiếp của sáu lỗi ban đầu.

**Hướng xử lý:** Reserve budget cho mọi model call, enforce execution usage thực, validate success/coverage và giữ stop reason; chỉ terminal completed sau answer admission.

**Tiêu chí đóng:** Limits model_turns=1 không tạo2calls; empty evidence không trả coverage sufficient; restart phục hồi observations; validator reject không completed.

## I32 — Tài liệu vận hành và bằng chứng lịch sử chưa theo layout hiện tại

**P2 · Nguồn · K01.** Nguồn: [docs/DEVELOPMENT.md](../../docs/DEVELOPMENT.md), [docs/runbooks/DEPLOY_FIREBASE_APP_RUNBOOK.md](../../docs/runbooks/DEPLOY_FIREBASE_APP_RUNBOOK.md), [docs/refactor/PREDICTOR_ONLY_STATUS_VI.md](../../docs/refactor/PREDICTOR_ONLY_STATUS_VI.md), [docs/unified-v2/BASELINE.md](../../docs/unified-v2/BASELINE.md).

**Bằng chứng và điều kiện:** Nhiều command/path còn model_server, deploy root, predictor-only hoặc artifact path cũ. Báo cáo progress tổng hợp nhiều thời điểm, có mục đã được mục sau sửa.

**Tác động:** Người triển khai làm theo tài liệu có thể dùng sai topology; số test/live score lịch sử bị hiểu nhầm là xác nhận HEAD.

**Hướng xử lý:** Một index canonical, đánh dấu superseded, command/link checker và release evidence có commit/image/config/dataset hashes. Giữ archive để truy vết.

**Tiêu chí đóng:** Tất cả command hỗ trợ chạy từ clean clone; reference artifact resolve được; không trộn score khác fixture/runtime.

## Những giới hạn còn lại được quản lý như backlog, không giả thành bug đã tái hiện

- Calibration, learned applicability domain và conformal uncertainty có contracts/infrastructure nhưng chưa có fitted artifact được admission. Raw probability và element rules vẫn phải được mô tả đúng nghĩa; không gọi đây là clinical risk hoặc OOD confidence.
- Model compare, identity/BioAssay enrichment, evidence provider thứ hai, safe web fetch, hosted OIDC/JWKS, export/deletion lifecycle, metrics/alerts và alpha/production gates chưa đủ bằng chứng hoàn tất. Kế hoạch đi kèm phân rã riêng.
- Kernel v2/case/observation/compiler đã có code; chưa có cutover production và paired runtime evaluation đủ để thay gateway hiện tại.
- Semantic activity/history/developer details và nhiều UI refinement đã có implementation. Chưa chạy visual/a11y đầy đủ ở năm viewport hoặc live recovery/research/provider switching; không kết luận toàn bộ UX mới là chưa làm.
- Không suy luận CVE, breach, wrong prediction trên mọi request hoặc cloud outage chỉ từ dependency ranges, URL cấu hình và test bị skip.

## Thứ tự xử lý đề xuất

1. K01 khôi phục baseline/CI; K02–K03 sửa topology, admission và đường nhập mặc định.
2. K04–K06 bảo toàn model/profile/XAI qua mọi luồng; kiểm tra isolation trước khi quảng bá BYOC.
3. K07 làm bền dữ liệu/run/recovery trước khi tăng replica; K08 hoàn thiện trải nghiệm và bằng chứng browser.
4. K09–K13 hoàn tất khoa học, release, runtime matrix và alpha theo gates trong [kế hoạch còn lại](REMAINING_IMPLEMENTATION_PLAN_VI.md).

Không đóng issue chỉ vì thêm test phản chiếu implementation. Bằng chứng đóng phải tái hiện trigger, xác nhận behavior sửa và chạy ở boundary có thể làm mất cấu hình/dữ liệu.

## Lệnh tái kiểm chứng các failure đã xác định

Chạy từ root workspace với môi trường đã có dependency; không cần khởi động stack hoặc cung cấp provider credential cho các lệnh dưới. Đây là lệnh chẩn đoán, không phải script sửa lỗi.

```bash
# Collection như fast-pr, dùng source paths thay cho editable install.
PYTHONPATH=backend/predictor/src:backend/control/src \
  /home/minhquang/miniconda3/envs/drug-tox-env/bin/python -m pytest --collect-only -q \
  backend/predictor/tests/unit backend/predictor/tests/contract \
  backend/control/tests/unit backend/control/tests/contract

# Hai fixture profile lỗi hiện tại; chạy trong backend/control.
cd /home/minhquang/tox-agent/backend/control
PYTHONPATH=src /home/minhquang/miniconda3/envs/drug-tox-env/bin/python -m pytest -q \
  tests/unit/test_opencode_profile.py

# Snapshot contract sai path; dừng ngay lỗi đầu tiên.
PYTHONPATH=src /home/minhquang/miniconda3/envs/drug-tox-env/bin/python -m pytest -q -x \
  tests/contract/test_predictor_contract.py

# Golden đã pass trong audit; cần artifact local như manifest.
cd /home/minhquang/tox-agent/backend/predictor
PYTHONPATH=src /home/minhquang/miniconda3/envs/drug-tox-env/bin/python -m pytest -q -rs tests/golden
```

Không dùng lệnh restore trên DB thật để tái hiện I28. Lỗi argv đã được kiểm tra bằng một file tạm tồn tại; acceptance restore sau sửa phải dùng database tạm và backup fixture.
