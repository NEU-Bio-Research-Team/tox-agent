# Rà soát ToxAgent local — 05/09/2026

Phạm vi: FE React/Vite, API control plane, kết nối predictor và OpenCode, routing, tools, validator, lưu trữ, SSE, transcript, audit UI và CI. Kiểm tra trên working tree hiện tại, bao gồm các thay đổi chưa commit có sẵn.

Không sửa mã nguồn, không restart/cancel dịch vụ hoặc run của người dùng, không gửi lượt LLM mới. Các ca cần tạo dữ liệu chạy trên SQLite tạm với predictor/runtime giả lập. Đọc HTTP/SSE của các phiên hiện có để đối chiếu. Chưa kiểm tra tương tác/layout bằng trình duyệt thật vì môi trường không có browser binary; các kiểm tra FE bên dưới dùng mã hiện tại, HTTP, TypeScript và React server rendering.

## Trạng thái đã xác minh

| Thành phần | Kết quả |
|---|---|
| FE 127.0.0.1:5173 | HTTP 200, trả trang Vite |
| Control plane 127.0.0.1:8000 | `/health/ready` HTTP 200, `ready=true` |
| Predictor 127.0.0.1:8080 | Ready, phục vụ hERG và Tox21 |
| OpenCode 127.0.0.1:4096 | `/global/health`: healthy, version 1.17.11 |
| CORS local | Cho phép cả http://localhost:5173 và http://127.0.0.1:5173 |
| Control plane tests | **463 passed**, 11 warnings, 100.48 giây |
| Predictor tests | **142 passed**, 30.12 giây, gồm contract/golden |
| FE | Typecheck, policy lint và production build đều qua |

Các test Python ban đầu đứng trong sandbox; đã dừng riêng các tiến trình test đó và chạy lại ngoài sandbox thành công. Đây không được tính là lỗi ứng dụng. Build kiểm tra đặt ở `/tmp/toxagent-audit-build`.

**Test hiện có qua không đồng nghĩa luồng FE–BE và các cam kết kiểm chứng đã đúng. Các ca bổ sung dưới đây tái hiện nhiều lỗi ngoài phạm vi của bộ test.**

P1: nên sửa trước khi dùng luồng chính một cách tin cậy. P2: lỗi chức năng, khả năng vận hành hoặc audit cần xử lý tiếp. Phân biệt rõ kết quả tái hiện, bằng chứng lịch sử và điều kiện chưa xuất hiện ở các phiên đang mở.

## A01 — P1 — Nội dung trả lời có thể chứa số liệu không được kiểm chứng

Đã gọi trực tiếp validator hiện tại với:

```json
{"answer_markdown":"The hERG probability is 99.99%.","claims":[],"limitations":[]}
```

Kết quả: `accepted=true`, `violations=[]`, không có observation nào. Một URL tài liệu tự viết trong Markdown cũng được nhận khi không có citation.

Nguyên nhân: kiểm chứng số liệu chạy trên `candidate.claims`; phần Markdown chỉ được quét một số mẫu từ ngữ cấm. Không có ràng buộc nối những phát biểu thực sự hiển thị với các claim đã qua kiểm chứng. `claims` được phép rỗng. Vì vậy nhãn “đã qua validator” không bảo đảm mọi con số người dùng đọc có nguồn.

Vị trí: [answer_validator.py:73](/home/minhquang/tox-agent/toxagent-control/toxagent/validation/answer_validator.py:73), [wire.py:96](/home/minhquang/tox-agent/toxagent-control/toxagent/validation/wire.py:96), [AnswerRenderer.tsx:70](/home/minhquang/tox-agent/frontend/src/components/answer/AnswerRenderer.tsx:70).

Hướng sửa: tạo các phần chứa dữ kiện từ claim đã xác thực, hoặc áp dụng cơ chế tham chiếu bắt buộc và đối chiếu coverage giữa văn bản và claim/citation; không chỉ yêu cầu `claims` khác rỗng.

## A02 — P1 — Có thể trả lời theo sai phân tích hoặc sai phân tử

Hai ca đã tái hiện trên app thật với dependency giả lập:

1. Tạo phân tích A (`CCO`), rồi B (`CCC`). Gửi `analysis_id=A` với `ask_report`: prompt không chứa ID A, lại ghim B đang active.
2. Khi B đang active, gửi `ask_report` kèm phân tử mới `CCN`: số lần gọi predictor mới bằng **0**, active SMILES vẫn là `CCC`, `CCN` không được đưa vào prompt.

Nguyên nhân: gateway ghim `session.active_analysis_id` và không sử dụng `context.analysis_id`. Một số nhánh router chỉ tạo snapshot mới nếu session chưa có active analysis, dù request có SMILES mới. User message chuyển vào runtime chỉ có `context.text`.

Vị trí: [gateway.py:279](/home/minhquang/tox-agent/toxagent-control/toxagent/harness/gateway.py:279), [router.py:160](/home/minhquang/tox-agent/toxagent-control/toxagent/application/router.py:160). Các nhánh research/attribution có cùng điều kiện ở dòng 130 và 146.

Hướng sửa: xác định và kiểm tra quyền của analysis mục tiêu một lần khi tiếp nhận request; mang mục tiêu đó xuyên suốt snapshot, prompt, tools và đáp án. SMILES mới phải được xử lý rõ ràng cả khi session đã có kết quả.

## A03 — P1 — FE không đọc được sự kiện realtime từ BE

Đã đọc stream của phiên `ses_fab103c35a3b4ddbbd3419ff0315833b`: nhận 22.017 byte, có **47** frame kết thúc bằng `\r\n\r\n`, có **0** dấu phân tách `\n\n`.

FE chỉ tìm `buffer.indexOf('\n\n')`, nên không dispatch các event này. Kết nối vẫn báo `live` ngay sau HTTP 200. Hệ quả: trạng thái chạy, kết quả, tool calls và lỗi không tự cập nhật; ô gửi có thể vẫn bị khoá theo active run cũ cho tới khi một REST refetch khác xảy ra. Buffer cũng tiếp tục tích luỹ.

Vị trí: [sse.ts:65](/home/minhquang/tox-agent/frontend/src/lib/events/sse.ts:65), [sse.ts:76](/home/minhquang/tox-agent/frontend/src/lib/events/sse.ts:76), [WorkbenchPage.tsx:85](/home/minhquang/tox-agent/frontend/src/pages/WorkbenchPage.tsx:85).

Hướng sửa: parser SSE hỗ trợ CRLF/LF và dấu phân tách bị cắt giữa các chunk; thêm kiểm tra tích hợp với wire output thực của BE, cùng cách phát hiện stream không còn tiến triển.

## A04 — P1 — Giới hạn tool có race và có thể chặn luôn nộp đáp án

Ca bổ sung đặt `max_calls_per_run=2`, gọi đồng thời 5 lượt `get_analysis_slice`: **cả 5 đều completed và được lưu**. Sau đó `submit_grounded_answer` bị trả `tool_denied` vì hết budget.

Nguyên nhân: kiểm tra số call và ghi nhận call mới ở hai transaction riêng, không giữ chỗ nguyên tử. Nộp đáp án dùng chung quota với các tool đọc dữ liệu. Các lần bị từ chối ở admission/schema được trả về trước bước ghi audit call.

Bằng chứng live: run `run_804c15fb3296442e80e4613c28ebef7f` ghi nhận **14** lượt đọc slice; tiến trình hiện tại không override `TOXAGENT_MAX_TOOL_CALLS`, mặc định là 12. Run kết thúc `runtime_protocol_error`: `the runtime reached a terminal event without submit_grounded_answer`. Không có event nộp đáp án được ghi nhận. Cơ chế chặn sau budget đã tái hiện; không khẳng định model live đã thử nộp vì các lần admission-denied không được lưu.

Vị trí: [runner.py:71](/home/minhquang/tox-agent/toxagent-control/toxagent/tools/runner.py:71), [runner.py:133](/home/minhquang/tox-agent/toxagent-control/toxagent/tools/runner.py:133), [config.py:96](/home/minhquang/tox-agent/toxagent-control/toxagent/config.py:96).

Hướng sửa: admission nguyên tử theo run; dành quota riêng cho final answer/correction; ghi audit cả call bị từ chối; cung cấp budget còn lại cho runtime.

## A05 — P1 — Run mất worker sau restart có thể kẹt vĩnh viễn

Tái hiện bằng cách khởi động app mới trên database tạm chứa run chưa kết thúc: `active_status=queued`, `workers=0`. Gọi cancel trả `cancellation_recorded_no_local_worker`; run vẫn `queued`.

Scheduler chỉ giữ task trong bộ nhớ, startup không đối soát các run còn dang dở. Không có worker xử lý cờ cancel hay deadline của orphan run. Kết hợp giới hạn một active run/session, session có thể không gửi tiếp được. Restart sạch có drain task, nhưng không giải quyết crash/kill đột ngột.

Vị trí: [run_scheduler.py:76](/home/minhquang/tox-agent/toxagent-control/toxagent/application/run_scheduler.py:76), [run_scheduler.py:211](/home/minhquang/tox-agent/toxagent-control/toxagent/application/run_scheduler.py:211), [app.py:70](/home/minhquang/tox-agent/toxagent-control/toxagent/api/app.py:70).

Hướng sửa: reconciliation lúc startup, lease/worker ownership và chính sách đưa orphan về terminal hoặc recovery run mới; cancel không có worker phải có đường xử lý cuối cùng.

## A06 — P1 — Luồng tìm tài liệu đang được cung cấp trên UI nhưng chưa có công cụ thực thi

FE cho chọn “Tìm evidence”, router có intent `evidence_research`. Registry thực tế cho profile đó chỉ có:

```text
get_analysis_slice
submit_grounded_answer
```

Không có `search_toxicology_evidence` hoặc `get_evidence_record`; bootstrap không wiring provider research. Ngoài ra, câu đã xuất hiện trong phiên live “Thử research xem, có nên xuất phân tử này không?” bị route thành `report_qa` vì keyword `research` không nằm trong danh sách.

Vị trí: [bootstrap.py:17](/home/minhquang/tox-agent/toxagent-control/toxagent/tools/bootstrap.py:17), [router.py:27](/home/minhquang/tox-agent/toxagent-control/toxagent/application/router.py:27), [MessageComposer.tsx:18](/home/minhquang/tox-agent/frontend/src/components/workbench/MessageComposer.tsx:18).

Hướng sửa: wiring provider và capability thực sự, hoặc phản hồi “chưa hỗ trợ” trước khi gọi LLM; UI dựa trên capabilities của deployment.

## A07 — P2 — Không mở được nguồn của số liệu trong đáp án

`linkifyClaims` tạo liên kết `[0.213](claim:...)`. ReactMarkdown mặc định loại scheme này trước khi gọi component `a`. Tái hiện với đúng package đang cài: component nhận `href=""`, chỉ render chữ `0.213`, không tạo ClaimChip/ObservationDialog.

Vị trí: [AnswerRenderer.tsx:26](/home/minhquang/tox-agent/frontend/src/components/answer/AnswerRenderer.tsx:26), [AnswerRenderer.tsx:70](/home/minhquang/tox-agent/frontend/src/components/answer/AnswerRenderer.tsx:70).

Hướng sửa: chuyển claim thành node có cấu trúc, hoặc URL transform cho phép chính xác scheme nội bộ và giữ bộ lọc cho URL khác.

## A08 — P2 — Các nút hỏi làm rõ quay vòng

Nút `select_analysis`/`submit_smiles` gửi nguyên chuỗi đó thành tin nhắn. Backend không xử lý chúng như hành động chọn phân tích hoặc điền SMILES. Phiên live đã ghi hai lần `molecule_missing`, lần sau do bấm `select_analysis`.

Vị trí: [ClarificationCard.tsx:22](/home/minhquang/tox-agent/frontend/src/components/transcript/ClarificationCard.tsx:22), [WorkbenchPage.tsx:117](/home/minhquang/tox-agent/frontend/src/pages/WorkbenchPage.tsx:117).

Hướng sửa: gắn action có nghĩa: focus ô SMILES, mở bộ chọn analysis, gửi `analysis_id` hợp lệ; dịch nhãn nút sang ngôn ngữ người dùng.

## A09 — P2 — Ô chính mời nhập SMILES nhưng backend không nhận SMILES từ text

Placeholder ghi “Nhập SMILES hoặc mô tả yêu cầu…”, nhưng nội dung ô này luôn gửi trong `content.text`. Router chỉ coi `molecule.smiles` là cấu trúc phân tử. Tái hiện `RouteRequest(text='CCO')` trả `clarification_required`. Với phiên đã có analysis, chuỗi đó có thể trở thành câu hỏi về phân tích cũ.

Vị trí: [MessageComposer.tsx:43](/home/minhquang/tox-agent/frontend/src/components/workbench/MessageComposer.tsx:43), [router.py:188](/home/minhquang/tox-agent/toxagent-control/toxagent/application/router.py:188).

Hướng sửa: quy định rõ input SMILES và text; cập nhật hướng dẫn hoặc thêm bước parse/xác nhận có cấu trúc. Nhập tên chất như “Thalidomide” cũng chưa có name-to-SMILES resolver trong luồng hiện tại; đó là thiếu capability, không phải model đã dự đoán ra kết quả sai.

## A10 — P2 — Truy cập FE qua IP/URL forward có thể trỏ API nhầm cổng

Config local là `VITE_API_BASE_URL=http://127.0.0.1:8000`. Khi hostname trình duyệt khác localhost/127.0.0.1, resolver bỏ config này và dùng `/v1/...` cùng origin FE. Vite không có proxy. Đã kiểm tra `GET http://127.0.0.1:5173/v1/sessions`: **HTTP 200 nhưng body là HTML của FE**.

Đã thực thi resolver với hostname LAN và URL preview: API URL trở thành `/v1/sessions`. Đây là lỗi có điều kiện: nếu đang dùng `localhost:5173` với port forwarding tới máy local thì không gặp nhánh này; CORS cho hai origin local hiện hoạt động.

Vị trí: [client.ts:13](/home/minhquang/tox-agent/frontend/src/lib/api/client.ts:13), [vite.config.ts:15](/home/minhquang/tox-agent/frontend/vite.config.ts:15).

Hướng sửa: cấu hình API URL/public origin rõ ràng hoặc reverse proxy cho `/v1` và `/health`.

## A11 — P2 — Phiên dài mất dữ liệu hiển thị và ngữ cảnh mới

- FE đọc cố định **200 tin đầu**, không phân trang. Ca tạm chứa 210 tin: trả sequence 1–200, không có 201–210.
- Gateway đọc **100 tin đầu**, rồi chọn 12 tin cuối của nhóm đó, nên sau ngưỡng 100 sẽ dùng ngữ cảnh cũ.
- `recent_runs` chỉ lấy 10; transcript cần danh sách này để ghép status/analysis card, nên run cũ mất card.
- Session list chỉ lấy 50 session, không dùng `next_offset`; tìm kiếm chỉ trong trang đầu.
- Backend lấy 50 tin đầu để làm “last message preview”, `run_count=len(last_10_runs)`, không phải tổng.
- `updated_at` không được cập nhật ở đường gửi tin/hỏi đáp thông thường; phiên live có tin tới 23:45 UTC nhưng updated_at vẫn 23:42 UTC.

Vị trí: [WorkbenchPage.tsx:63](/home/minhquang/tox-agent/frontend/src/pages/WorkbenchPage.tsx:63), [gateway.py:279](/home/minhquang/tox-agent/toxagent-control/toxagent/harness/gateway.py:279), [sessions.py:109](/home/minhquang/tox-agent/toxagent-control/toxagent/application/sessions.py:109), [SessionsPage.tsx:32](/home/minhquang/tox-agent/frontend/src/pages/SessionsPage.tsx:32).

Hướng sửa: pagination cho lịch sử, truy vấn recent theo thứ tự giảm dần rồi trả đúng thứ tự hội thoại; count/preview/update time lấy từ dữ liệu đúng ngữ nghĩa.

## A12 — P2 — Nội dung soạn bị mất ngay cả khi gửi thất bại

Composer gọi mutation rồi lập tức xoá text và SMILES, trước khi BE xác nhận. Với 401/403/409, lỗi mạng hoặc payload không hợp lệ, UI chỉ báo lỗi và không khôi phục nội dung. Gửi lại tạo client_message_id mới, nên trường hợp BE đã nhận nhưng response bị mất có thể tạo lượt trùng.

Vị trí: [MessageComposer.tsx:58](/home/minhquang/tox-agent/frontend/src/components/workbench/MessageComposer.tsx:58), [WorkbenchPage.tsx:68](/home/minhquang/tox-agent/frontend/src/pages/WorkbenchPage.tsx:68).

Hướng sửa: chỉ clear sau khi được chấp nhận; giữ nội dung và idempotency key khi retry. Các helper draft hiện chưa được nối vào composer.

## A13 — P2 — Timeline tool bị lệch do timestamp thiếu timezone

Run live có `started_at=2026-09-04T23:43:44.771376+00:00`; tool có `started_at=2026-09-04T23:44:03.099686` không có offset. SQLite trả datetime naive, route gọi thẳng `isoformat()`. JavaScript hiểu chuỗi tool theo timezone máy người xem.

Tái hiện tại Asia/Shanghai: tool offset **−28.781.672 ms**, CSS `left` khoảng **−30.968%**, thay vì khoảng 18,3 giây sau lúc bắt đầu. Múi giờ khác UTC cũng bị ảnh hưởng.

Vị trí: [routes.py:42](/home/minhquang/tox-agent/toxagent-control/toxagent/api/routes.py:42), [RunTimelineTab.tsx:35](/home/minhquang/tox-agent/frontend/src/components/inspector/RunTimelineTab.tsx:35).

Hướng sửa: chuẩn hoá UTC có offset trước khi serialize mọi timestamp.

## A14 — P2 — Metadata provenance ánh xạ sai contract thực tế

Payload live có `provenance.predictor_version`, nhưng adapter chỉ đọc `service_version` hoặc `version`, làm `predictor_service_version=null`. `artifact_hashes` chứa chuỗi biểu diễn cả Python dict thay vì các hash có cấu trúc, vì adapter gọi `str()` trên từng phần tử `artifacts`.

Raw payload vẫn được giữ, nên bằng chứng gốc chưa mất; các trường chuẩn hoá dùng cho audit/cache bị sai hình dạng hoặc thiếu dữ liệu. Fixture test đang dùng kiểu provenance cũ nên không bắt lỗi này.

Vị trí: [client.py:135](/home/minhquang/tox-agent/toxagent-control/toxagent/predictor/client.py:135), [predictor fixture](/home/minhquang/tox-agent/toxagent-control/tests/support/predictor.py:23).

Hướng sửa: ánh xạ theo contract thực của ToxPred và dùng captured payload trong contract test.

## A15 — P2 — Log có lỗi cleanup DB khi SSE bị ngắt

Log hiện có `Exception terminating connection`, `CancelledError`, `Task exception was never retrieved`, và `sqlite3.OperationalError: no active connection`; stack chỉ vào cancel scope của `sse-starlette` và rollback/close aiosqlite. Xuất hiện quanh lúc mở/chuyển phiên FE.

Vị trí bằng chứng: [control.log:64](/home/minhquang/tox-agent/.data/logs/control.log:64). Đường DB liên quan: [database.py:151](/home/minhquang/tox-agent/toxagent-control/toxagent/persistence/sql/database.py:151), [database.py:204](/home/minhquang/tox-agent/toxagent-control/toxagent/persistence/sql/database.py:204).

Đây là lỗi đã có trong log; chưa chạy stress test để kết luận có rò pool hay ảnh hưởng tới mọi request. Cần xử lý cleanup dưới cancellation và test connect/disconnect lặp lại.

## A16 — P2 — Health ready không chứng minh luồng hỏi đáp hoạt động

Endpoint chỉ probe predictor và trả `runtime.kind`; không kiểm tra runtime availability/handler. Ca app không có report_qa handler vẫn trả `ready=true`. Dịch vụ OpenCode hiện tại có health tốt, nhưng tình trạng đó được xác minh riêng, không phải nhờ ready của control plane.

Vị trí: [routes.py:86](/home/minhquang/tox-agent/toxagent-control/toxagent/api/routes.py:86).

Hướng sửa: trả trạng thái dependency/capability riêng, có probe runtime và phân biệt khả năng đọc session, phân tích và hỏi đáp.

## A17 — P2 — Transcript/audit còn các đường hiển thị sai hoặc thiếu

Phát hiện từ mã và payload hiện có:

- `system_event`/part `error` không được render trong Transcript. Run lỗi chỉ hiện failure_code chung ở RunBlock, mất thông điệp chi tiết BE đã lưu.
- `runByTrigger` ghép theo ID tin nhắn user; đến tin assistant thì không tìm ra run, nên `onOpenValidation` của answer luôn thiếu trong luồng bình thường.
- Khi có recovery cùng trigger_message_id, danh sách run mới trước/cũ sau bị ghi đè trong Map; có thể hiện run thất bại gốc và ẩn recovery.
- `AnalysisSystemCard` không xử lý riêng `cancelled`, nên rơi xuống nhánh spinner “Đang gọi predictor…”.
- Cache `run-events` của tab kiểm định không được invalidated khi nhận event mới; mở tab trong lúc run chạy có thể giữ kết quả cũ.

Vị trí: [Transcript.tsx:32](/home/minhquang/tox-agent/frontend/src/components/transcript/Transcript.tsx:32), [Transcript.tsx:59](/home/minhquang/tox-agent/frontend/src/components/transcript/Transcript.tsx:59), [AnalysisSystemCard.tsx:44](/home/minhquang/tox-agent/frontend/src/components/transcript/AnalysisSystemCard.tsx:44), [ValidationTab.tsx:19](/home/minhquang/tox-agent/frontend/src/components/inspector/ValidationTab.tsx:19).

Hướng sửa: liên kết answer/run bằng ID bền vững, render đầy đủ terminal/system events, giữ các recovery dưới cùng một trigger, invalidation đúng query key.

## A18 — Khoảng trống kiểm tra tự động

605 test Python và các kiểm tra FE hiện có đều qua trong lần rà soát này. FE không có test script hoặc bộ test cho luồng SSE/Markdown/session đang xét. CI hiện chạy predictor, không có job build/test frontend hoặc `toxagent-control`.

Vị trí: [package.json](/home/minhquang/tox-agent/frontend/package.json), [ci.yml:16](/home/minhquang/tox-agent/.github/workflows/ci.yml:16).

Cần đưa các ca đã tái hiện vào kiểm tra tích hợp theo hành vi: số hiển thị phải có nguồn; đúng molecule/analysis; wire SSE thực tế; final answer còn khả năng submit; restart không khoá phiên; lịch sử dài hiển thị được tin mới.

## Capability hiện chưa sẵn sàng: ClinTox

`GET :8080/v1/models` xác nhận ClinTox `loaded=false`, thiếu `models/smilesgnn_model/tokenizer.pkl`. hERG và Tox21 vẫn hoạt động và kiểm tra golden qua. Đây là thiếu artifact đã được khai báo, không phải toàn bộ predictor đang chết.

FE vẫn có checkbox ClinTox; cần lấy capabilities và giải thích rõ endpoint chưa khả dụng trước khi người dùng gửi yêu cầu. Không thể sửa bằng cách đổi nhãn hay dùng output hERG thay thế.

Vị trí: [manifest.yaml:70](/home/minhquang/tox-agent/artifacts/predictor-manifest.yaml:70), [MessageComposer.tsx:22](/home/minhquang/tox-agent/frontend/src/components/workbench/MessageComposer.tsx:22).

## Thứ tự xử lý đề xuất

1. Khép ràng buộc đáp án–claim (A01), cố định analysis mục tiêu (A02), sửa SSE (A03).
2. Sửa budget/admission/final submit và audit từ chối (A04); recovery sau crash và cancel orphan (A05).
3. Làm rõ capability research/ClinTox; sửa các đường nhập SMILES, clarification và xem nguồn.
4. Hoàn thiện lịch sử dài, retry draft, timezone, provenance, readiness và cleanup SSE.
5. Bổ sung các kiểm tra hành vi tương ứng vào CI.

Script tái hiện backend: [toxagent-audit-repro.py](/tmp/toxagent-audit-repro.py). Script tạo và tự dọn database tạm, không gọi LLM hoặc API đang chạy. Các findings FE/live có bằng chứng và vị trí mã riêng ở trên.
