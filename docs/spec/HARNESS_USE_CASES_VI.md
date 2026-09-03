# ToxAgent Harness — Đặc tả tình huống sử dụng

## 1. Mục đích và phạm vi

Tài liệu này mô tả các tình huống sử dụng của kiến trúc harness mục tiêu cho ToxAgent. Đây là đặc tả **đề xuất**, không mô tả toàn bộ hành vi production hiện tại. Nguồn kiến trúc chuẩn là [HARNESS_ARCHITECTURE.md](HARNESS_ARCHITECTURE.md).

Phạm vi gồm phân tích độc tính theo từng phân tử, hỏi đáp trên báo cáo, bằng chứng có thể kiểm toán, quan sát tiến trình và quản trị ngữ cảnh. Không thuộc phạm vi: chạy mã do người dùng cung cấp, Code Mode, sandbox thực thi mã, subagent, hay workflow do LLM tự tạo.

## 2. Vai trò

| Vai trò | Mục tiêu chính |
| --- | --- |
| Nhà nghiên cứu | Phân tích một hợp chất, hiểu nguy cơ và bằng chứng đi kèm. |
| Nhà nghiên cứu đánh giá nhiều mẫu | Chạy batch hoặc benchmark có thể tái lập. |
| Người đọc báo cáo | Đặt câu hỏi cụ thể trên một báo cáo đã mở. |
| Người kiểm định khoa học | Truy ngược số liệu, nguồn và từng bước đã tạo ra báo cáo. |
| Vận hành hệ thống | Theo dõi lỗi/độ trễ/chi phí, cấu hình công cụ và giới hạn runtime. |

## 3. Quy ước luồng

- **Làn A**: workflow xác định (deterministic) cho phân tích hoặc batch. Kết quả phân tích phải tái lập với cùng dữ liệu, model và cấu hình.
- **Làn B**: vòng lặp agent hỏi đáp trên một báo cáo đã mở. Agent chỉ gọi công cụ đã đăng ký và có thể gọi Làn A qua `run_full_analysis(smiles)` khi thật sự cần phân tích mới.
- Mọi tool call được lưu dưới một `part` có `callID`, trạng thái, input, output hiển thị cho model, metadata và attachment.
- Các số trong câu trả lời chỉ được chấp nhận khi có provenance từ observation của chính lượt đó hoặc từ phép biến đổi đã khai báo.

## 4. Tình huống sử dụng chính

### UC-01 — Phân tích một chuỗi SMILES

**Tác nhân chính:** Nhà nghiên cứu
**Mục tiêu:** Nhận báo cáo độc tính đầy đủ cho một phân tử.

**Tiền điều kiện:** Người dùng cung cấp SMILES hợp lệ hoặc chuỗi có thể được kiểm tra/canonicalize.

**Luồng thành công:**

1. Người dùng nhập SMILES và yêu cầu phân tích.
2. Router nhận diện đây là SMILES hợp lệ không kèm câu hỏi, chọn Làn A.
3. Hệ thống kiểm tra và canonicalize SMILES trước khi chạy bất kỳ tool nào.
4. Làn A chạy screening và nghiên cứu theo workflow xác định; các nhánh độc lập có thể chạy song song.
5. Hệ thống tạo báo cáo gồm dự đoán lâm sàng, cơ chế, giải thích cấu trúc (nếu được bật), dữ liệu literature và khuyến nghị.
6. Hệ thống lưu session, message và part; kết quả tool lớn/ảnh được lưu attachment thay vì đưa vào context LLM.
7. UI nhận trạng thái/tiến trình qua SSE và hiển thị báo cáo cuối.

**Luồng thay thế:**

- SMILES không hợp lệ: dừng sớm, trả lỗi có cấu trúc; không gọi inference hay literature.
- Một nguồn research hoặc explainer lỗi/quá thời gian: ghi lỗi vào part và `failure_registry`; báo cáo còn lại vẫn nêu rõ phần bằng chứng không có.

**Hậu điều kiện:** Có một session có thể kiểm toán; report và mọi số liệu nguồn của nó truy được tới kết quả tool/observation tương ứng.

### UC-02 — Phân tích hợp chất từ tên hoặc ảnh

**Tác nhân chính:** Nhà nghiên cứu
**Mục tiêu:** Không cần biết SMILES vẫn có thể bắt đầu phân tích.

**Luồng thành công:**

1. Người dùng nhập tên hợp chất hoặc tải ảnh cấu trúc.
2. Với ảnh, hệ thống trích xuất cấu trúc bằng MolScribe; với tên, gọi `resolve_compound` để tìm hợp chất.
3. Hệ thống hiển thị SMILES đã phân giải/canonicalize để người dùng xác nhận khi cần.
4. Router chuyển yêu cầu sang Làn A và thực hiện UC-01.

**Ngoại lệ:** Không phân giải được tên/ảnh, hoặc SMILES sau phân giải không hợp lệ: trả nguyên nhân và không suy đoán thay thế im lặng.

### UC-03 — Phân tích batch hoặc benchmark

**Tác nhân chính:** Nhà nghiên cứu đánh giá nhiều mẫu
**Mục tiêu:** Phân tích nhiều phân tử theo pipeline nhất quán, có thể tái lập.

**Luồng thành công:**

1. Người dùng gửi danh sách phân tử hoặc yêu cầu benchmark.
2. Router luôn chọn Làn A; batch không được đi qua Làn B.
3. Hệ thống kiểm tra từng input, chạy workflow đã định nghĩa và lưu trạng thái/bằng chứng của từng phần tử.
4. UI/API trả kết quả từng phần tử, các lỗi riêng lẻ và tổng hợp benchmark nếu có.

**Hậu điều kiện:** Không có LLM tự do quyết định thứ tự phân tích hoặc diễn giải làm thay đổi kết quả định lượng của batch.

### UC-04 — Hỏi đáp trên báo cáo đang mở

**Tác nhân chính:** Người đọc báo cáo
**Mục tiêu:** Hiểu hoặc khai thác sâu báo cáo mà không phải phân tích lại không cần thiết.

**Tiền điều kiện:** Có report/session đang mở và người dùng gửi một câu hỏi.

**Luồng thành công:**

1. Router chọn Làn B vì đây là câu hỏi gắn với report hiện có.
2. Harness dựng context từ system/tool schema tĩnh và các observation liên quan, phần biến động của session và ngôn ngữ đầu ra.
3. Model trả lời trực tiếp hoặc yêu cầu tool được đăng ký, ví dụ tìm literature/analogs hay xem giải thích cấu trúc.
4. Mỗi tool đi qua pipeline validate → execute → post-execute và tạo tool part.
5. Validator provenance kiểm tra các số trong câu trả lời trước khi kết thúc lượt.
6. Hệ thống ghi message/part, cập nhật SSE và trả câu trả lời cho người dùng.

**Ngoại lệ:** Không đủ dữ liệu để trả lời thì agent nói rõ giới hạn, có thể đề nghị/ gọi `run_full_analysis` khi cần. Không được bịa số hoặc trích dẫn.

### UC-05 — Hỏi về một phân tử mới khi đang chat

**Tác nhân chính:** Người đọc báo cáo
**Mục tiêu:** Phân tích một SMILES mới từ trong cuộc hội thoại.

**Luồng thành công:**

1. Người dùng đưa SMILES mới hoặc yêu cầu so sánh cần dữ liệu chưa có.
2. Làn B xác định không thể trả lời chỉ từ observation hiện hữu.
3. Agent gọi tool `run_full_analysis(smiles)`; tool này bọc Làn A và tuân theo deadline chung của lượt.
4. Kết quả được lưu như một tool part/observation, sau đó agent trả lời dựa trên kết quả đó.

**Quy tắc:** Không cho phép một tool `rerun_screening` độc lập chỉ để yêu cầu model tuân theo prompt; nếu cần chạy lại, dùng đường Làn A được kiểm soát.

### UC-06 — Tra cứu bằng chứng literature và hợp chất tương tự

**Tác nhân chính:** Người đọc báo cáo
**Mục tiêu:** Kiểm tra bối cảnh khoa học của một nhận định.

**Luồng thành công:**

1. Agent gọi `search_literature`, `resolve_compound`, `find_analogs` hoặc nguồn research phù hợp.
2. Post-execute chiếu dữ liệu có cấu trúc: ví dụ PMID, tiêu đề, năm và đoạn abstract ngắn; top-5 analog và điểm Tanimoto.
3. Dữ liệu thô đầy đủ được giữ trong metadata/store, không làm phình context.
4. Câu trả lời chỉ dùng con số/citation tồn tại trong projection hoặc numeric index của observation.

### UC-07 — Xem giải thích cấu trúc và ảnh heatmap

**Tác nhân chính:** Nhà nghiên cứu
**Mục tiêu:** Xem các atom/bond đóng góp vào dự đoán mà không gửi dữ liệu ảnh base64 vào LLM context.

**Luồng thành công:**

1. Hệ thống hoặc agent gọi `explain_prediction` khi điều kiện giải thích được thỏa.
2. Tool trả `output` gồm top atom/top bond, `metadata` cho chỉ số cấu trúc và `attachments` chứa heatmap/molecule PNG.
3. UI lấy attachment qua URL có kiểm soát quyền truy cập để hiển thị.
4. Attachment và raw result sống ít nhất bằng vòng đời session; không được coi là cache có thể dọn tùy ý.

### UC-08 — Xử lý số liệu không có nguồn hợp lệ

**Tác nhân chính:** Harness tự động
**Mục tiêu:** Ngăn câu trả lời khoa học chứa số bịa hoặc biến đổi không được phép.

**Luồng thành công:**

1. Sau khi LLM sinh câu trả lời, validator trích xuất token số, hỗ trợ dấu `,` hoặc `.` thập phân tiếng Việt.
2. Validator so với `numeric_index`, projection và whitelist (PMID, năm, số thứ tự) của observation trong lượt.
3. Số hợp lệ nếu khớp dung sai làm tròn, hoặc là phép `round`, `percent`, `ratio`, `diff`, `unit_convert` đã khai báo.
4. Khi vi phạm, system retry tối đa một lần cùng danh sách số vi phạm.
5. Nếu vẫn vi phạm, hệ thống trả lời bằng fallback deterministic, không vá chuỗi sau khi sinh, và lưu verdict kiểm tra.

**Ghi chú triển khai:** Giai đoạn đầu chạy shadow mode: ghi verdict nhưng chưa chặn câu trả lời để đo tỉ lệ vi phạm thực tế.

### UC-09 — Tiếp tục phiên và kiểm toán báo cáo

**Tác nhân chính:** Người kiểm định khoa học
**Mục tiêu:** Đọc lại tiến trình đã tạo một kết quả và truy ra bằng chứng.

**Luồng thành công:**

1. Người dùng mở session theo quyền sở hữu.
2. Hệ thống trả transcript đầy đủ gồm message/part và trạng thái tool cuối cùng.
3. Người dùng xem input, output model nhìn thấy, metadata, thời gian, token/chi phí, attachment và verdict provenance theo `callID`.
4. Nếu context từng bị nén, transcript vẫn giữ lịch sử gốc; surface cho LLM chỉ dùng baseline cộng phần tail theo con trỏ.

**Hậu điều kiện:** Có thể giải thích “con số này đến từ tool nào, với input nào, vào thời điểm nào” mà không phải suy diễn từ log streaming.

### UC-10 — Theo dõi tiến trình, lỗi và chi phí

**Tác nhân chính:** Nhà nghiên cứu và vận hành hệ thống
**Mục tiêu:** Biết hệ thống đang làm gì và chẩn đoán chậm/lỗi dựa trên dữ liệu.

**Luồng thành công:**

1. Khi turn chạy, tool part chuyển trạng thái `pending → running → completed | error`.
2. SSE được dẫn xuất từ thay đổi state/part, không phải một luồng sự thật độc lập.
3. UI hiển thị tiến độ và lỗi có nghĩa; không chờ spinner câm cho các tác vụ dài.
4. Hệ thống lưu thời gian start/end từng tool và token/chi phí của từng step, đồng thời tổng hợp lên session.

### UC-11 — Giữ cuộc hội thoại trong ngân sách context

**Tác nhân chính:** Harness tự động
**Mục tiêu:** Duy trì chat dài mà không mất audit trail hoặc gửi JSON/blob lớn vào model.

**Luồng thành công:**

1. Hệ thống dùng `count_tokens` của provider để đo context thay vì ước lượng ký tự.
2. Trước hết chỉ chiếu những trường cần cho từng tool, sau đó đẩy observation cũ ra khỏi surface nhưng vẫn giữ trong store.
3. Nếu còn vượt ngưỡng, hệ thống tạo baseline tóm tắt và lưu `tail_start_id`, `baseline`, `baseline_seq`, cùng cờ `auto`/`overflow`.
4. Transcript không bị sửa; lần gọi sau dựng surface từ baseline cộng phần tail.

## 5. Ràng buộc áp dụng cho mọi use case

- Router phải xác định và có unit test; LLM không được tự chọn Làn A/B.
- Mỗi lượt có một deadline chung truyền tới mọi tool, retry và fallback.
- Tool catalog là bề mặt quyền lực: chỉ công cụ hiện diện mới có thể được model gọi. Tắt một khả năng bằng cách bỏ nó khỏi catalog, không chỉ bằng prompt.
- Cấu hình không hợp lệ hoặc không rõ phải fail fast.
- Dữ liệu model nhìn thấy, tool schema và cấu hình request phải đủ để tái dựng request phục vụ audit.
- Mọi dữ liệu thuộc session, gồm raw observation và attachment, phải có vòng đời bền vững bằng session.

## 6. Liên kết đặc tả

| Nhu cầu | Tài liệu chi tiết |
| --- | --- |
| User story, ưu tiên và tiêu chí nghiệm thu | [HARNESS_USER_STORIES_VI.md](HARNESS_USER_STORIES_VI.md) |
| Thành phần, dữ liệu, contract và lộ trình kỹ thuật | [HARNESS_SYSTEM_DESIGN_VI.md](HARNESS_SYSTEM_DESIGN_VI.md) |
| Cơ sở quyết định kiến trúc | [HARNESS_ARCHITECTURE.md](HARNESS_ARCHITECTURE.md) |
