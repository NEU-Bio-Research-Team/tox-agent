# ToxAgent Harness — Câu chuyện người dùng và tiêu chí nghiệm thu

## 1. Mục đích

Tài liệu này chuyển kiến trúc harness đề xuất thành backlog hướng người dùng. Mỗi câu chuyện người dùng mô tả **hành vi mong muốn của hệ thống đích**, không phải cam kết rằng tính năng đã có trong production. Các tình huống sử dụng chi tiết nằm tại [HARNESS_USE_CASES_VI.md](HARNESS_USE_CASES_VI.md).

## 2. Nhóm người dùng và nhóm tính năng

| Nhóm người dùng | Nhóm tính năng | Giá trị nhận được |
| --- | --- | --- |
| Nhà nghiên cứu | E1 — Phân tích xác định | Có báo cáo độc tính tái lập và dễ hiểu. |
| Người đọc báo cáo | E2 — Hỏi đáp có căn cứ | Khai thác báo cáo mà không nhận số liệu bịa. |
| Người kiểm định khoa học | E3 — Audit và evidence | Truy ngược kết quả tới input, tool và nguồn. |
| Người dùng cần xử lý dài | E4 — Trải nghiệm phiên bền vững | Không mất ngữ cảnh hoặc tiến trình khi chat dài. |
| Vận hành hệ thống | E5 — Điều hành có thể đo | Biết lỗi, độ trễ, token và chi phí để cải thiện có dữ liệu. |

## 3. Backlog ưu tiên

| Mã | Ưu tiên | Giai đoạn | Liên kết use case | Tóm tắt |
| --- | --- | --- | --- | --- |
| US-01 | Must | 1 | UC-01 | Phân tích một SMILES qua Làn A. |
| US-02 | Must | 1 | UC-03 | Chạy batch/benchmark không qua LLM chat. |
| US-03 | Must | 1.5 | UC-09 | Lưu audit trail và evidence của Làn A. |
| US-04 | Must | 1.5 | UC-10 | Quan sát tiến trình tool có trạng thái và thời gian. |
| US-05 | Must | 2 | UC-04 | Hỏi đáp trên report bằng tool calling native. |
| US-06 | Must | 2 | UC-08 | Kiểm chứng provenance cho số liệu trong câu trả lời. |
| US-07 | Should | 2 | UC-05 | Phân tích phân tử mới ngay trong chat qua Làn A. |
| US-08 | Should | 2 | UC-11 | Nén context mà vẫn giữ transcript/audit. |
| US-09 | Should | 2 | UC-06, UC-07 | Dùng literature/analogs/heatmap theo projection và attachment. |
| US-10 | Should | 3 | UC-02 | Nhập tên hợp chất hoặc ảnh cấu trúc. |
| US-11 | Should | 3 | UC-10 | Quản trị bề mặt tool, skill và cấu hình khai báo. |

## 4. User stories và tiêu chí nghiệm thu

### E1 — Phân tích xác định

#### US-01 — Phân tích một SMILES

**Là** nhà nghiên cứu, **tôi muốn** gửi một SMILES và nhận báo cáo độc tính, **để** đánh giá hợp chất trên cùng pipeline khoa học có thể tái lập.

**Tiêu chí nghiệm thu:**

- Với một SMILES hợp lệ không kèm câu hỏi, khi gửi yêu cầu, thì router chọn Làn A.
- Với Làn A đã hoàn tất, khi đọc kết quả, thì báo cáo có verdict, dữ liệu lâm sàng/cơ chế, OOD, research và các phần giải thích được phép có mặt.
- Với cùng SMILES, model artifact và cấu hình, khi chạy lại Làn A, thì kết quả định lượng theo contract của workflow không đổi.
- Với SMILES sai, khi gửi yêu cầu, thì hệ thống trả lỗi có cấu trúc trước khi inference/research chạy.

#### US-02 — Batch và benchmark

**Là** nhà nghiên cứu đánh giá nhiều mẫu, **tôi muốn** gửi batch, **để** so sánh kết quả với quy trình nhất quán.

**Tiêu chí nghiệm thu:**

- Với đầu vào là batch hoặc benchmark, khi router phân làn, thì luôn chọn Làn A.
- Với một phần tử batch lỗi, khi batch còn phần tử hợp lệ, thì lỗi được ghi riêng theo phần tử và không che kết quả hợp lệ khác.
- Với batch đã hoàn tất, thì từng kết quả có session/part/observation để audit.

#### US-03 — Phân giải tên hoặc ảnh thành phân tử

**Là** nhà nghiên cứu, **tôi muốn** nhập tên hợp chất hoặc ảnh cấu trúc, **để** bắt đầu phân tích khi chưa có SMILES.

**Tiêu chí nghiệm thu:**

- Với tên hợp chất phân giải được hoặc ảnh đọc được, khi xử lý input, thì hệ thống tạo SMILES canonical trước khi vào Làn A.
- Với input không thể phân giải, thì hệ thống thông báo rõ thất bại và không tự thay thế bằng một hợp chất khác.

### E2 — Hỏi đáp có căn cứ

#### US-04 — Hỏi về báo cáo đang mở

**Là** người đọc báo cáo, **tôi muốn** đặt câu hỏi theo ngữ cảnh báo cáo, **để** hiểu nguy cơ và bằng chứng mà không phải tự đọc toàn bộ dữ liệu thô.

**Tiêu chí nghiệm thu:**

- Với một session report đang mở và câu hỏi, khi router xử lý, thì chọn Làn B.
- Với agent cần dữ liệu thêm, khi gọi tool, thì chỉ có tool trong registry của session được gọi.
- Với tool thành công, thì output model nhìn thấy, metadata và trạng thái call được lưu trong một tool part.
- Với dữ liệu không đủ trả lời, thì assistant nêu giới hạn thay vì suy đoán.

#### US-05 — Hỏi về phân tử mới trong chat

**Là** người đọc báo cáo, **tôi muốn** yêu cầu phân tích SMILES mới trong chat, **để** tiếp tục nghiên cứu mà không mất dòng suy nghĩ.

**Tiêu chí nghiệm thu:**

- Với câu hỏi yêu cầu dữ liệu cho SMILES chưa có, khi agent cần phân tích, thì agent gọi `run_full_analysis(smiles)` thay vì gọi những bước inference rời rạc.
- Với tool đã hoàn tất, thì kết quả của Làn A là observation có thể trích dẫn trong câu trả lời.
- Với deadline lượt đã hết, thì tool dừng theo deadline chung và trả lỗi/partial result có cấu trúc.

#### US-06 — Kiểm chứng số liệu khoa học

**Là** người đọc báo cáo, **tôi muốn** các số trong câu trả lời có nguồn rõ ràng, **để** không đưa nhận định sai vào quyết định nghiên cứu.

**Tiêu chí nghiệm thu:**

- Với câu trả lời chứa số, khi validator chạy, thì mỗi số khớp `numeric_index`, projection, whitelist hoặc phép biến đổi đã khai báo của observation trong lượt.
- Với model dùng dấu phẩy thập phân tiếng Việt, thì validator so khớp đúng với giá trị số chuẩn hóa.
- Với số vi phạm, khi lần sinh đầu kết thúc, thì hệ thống retry một lần với phản hồi vi phạm cụ thể.
- Với retry vẫn vi phạm và strict mode bật, thì hệ thống trả fallback deterministic, không sửa/che câu trả lời sai bằng hậu xử lý chuỗi.
- Với shadow mode, verdict vẫn được lưu nhưng câu trả lời chưa bị chặn.

#### US-07 — Dùng evidence và visual attachment

**Là** nhà nghiên cứu, **tôi muốn** xem literature, analog và heatmap gắn với câu trả lời, **để** đánh giá căn cứ của dự đoán.

**Tiêu chí nghiệm thu:**

- Với tool literature, projection chỉ chứa các trường đã khai báo, còn abstract/raw đầy đủ ở metadata/store.
- Với tool giải thích tạo PNG/heatmap, UI nhận attachment qua URL; base64 không xuất hiện trong LLM context.
- Với một attachment được trả cho session, attachment còn truy cập được trong suốt vòng đời audit của session.

### E3 — Audit và evidence

#### US-08 — Đọc lại phiên phân tích

**Là** người kiểm định khoa học, **tôi muốn** mở lại một session và truy ngược từng kết quả, **để** xác nhận báo cáo có căn cứ.

**Tiêu chí nghiệm thu:**

- Với quyền truy cập session, khi xem transcript, tôi thấy message/part gốc, input tool, output, metadata, attachment và thời gian gọi.
- Với một số trong report, khi truy evidence, có thể xác định `callID`/observation và input tool sinh nó.
- Với context đã compact, transcript gốc không bị mất hay bị viết lại.
- Với session bị restart hoặc instance đổi, audit trail không phụ thuộc vào dict in-memory của process.

#### US-09 — Bảo vệ ranh giới bề mặt tool

**Là** người kiểm định hệ thống, **tôi muốn** chỉ các khả năng cần thiết được expose cho model, **để** luật quan trọng được cưỡng chế bằng hệ thống thay vì prompt.

**Tiêu chí nghiệm thu:**

- Với một tool không được phép trong profile/session, schema của nó không xuất hiện trong request gửi model và không thể bị gọi.
- Với cấu hình tool/skill không hợp lệ, khởi động hoặc nạp profile thất bại rõ ràng.
- Với `steps` thay đổi theo profile, giới hạn vòng lặp thay đổi không cần sửa hằng số trong mã.

### E4 — Phiên bền vững và ngữ cảnh kiểm soát được

#### US-10 — Chat dài nhưng không mất bằng chứng

**Là** người đọc báo cáo, **tôi muốn** tiếp tục chat dài mà hệ thống vẫn phản hồi trong budget, **để** không phải mở phiên mới và mất bối cảnh.

**Tiêu chí nghiệm thu:**

- Với context sắp chạm ngưỡng, khi harness chuẩn bị request, thì dùng số token do provider trả về thay vì `len(text) // 4`.
- Với nhu cầu giảm context, hệ thống chiếu theo field và bỏ observation cũ khỏi surface trước khi summary.
- Với trường hợp vẫn cần compact, lưu baseline, `tail_start_id`, `baseline_seq`, `auto` và `overflow`.
- Với compact đã hoàn tất, các message/part gốc vẫn có trong transcript/audit.

### E5 — Điều hành có thể đo

#### US-11 — Xem tiến độ và sự cố có dữ liệu

**Là** nhà nghiên cứu, **tôi muốn** thấy từng bước hệ thống đang chạy, **để** biết chờ, thử lại hay xử lý input.

**Tiêu chí nghiệm thu:**

- Với tool bắt đầu/chạy/kết thúc/lỗi, tool part biểu diễn đúng một trong `pending`, `running`, `completed`, `error` cùng thời gian start/end.
- Với UI đang subscribe SSE, SSE được suy ra từ thay đổi state đã lưu và khớp transcript cuối cùng.
- Với tool lỗi, UI có tiêu đề/lỗi có nghĩa thay vì chỉ spinner hoặc timeout chung chung.

#### US-12 — Đo token, cache và chi phí

**Là** người vận hành, **tôi muốn** đo token và chi phí theo step lẫn session, **để** quyết định có cần cache/compaction tối ưu hay không.

**Tiêu chí nghiệm thu:**

- Với một step LLM kết thúc, lưu `input`, `output`, `reasoning`, `cache.read`, `cache.write` và `cost` nếu provider hỗ trợ.
- Với một session kết thúc, các chỉ số tổng được denormalize lên session để truy vấn nhanh.
- Với provider không trả một chỉ số, trường đó được đánh dấu không khả dụng/0 theo contract, không được tự ước đoán thành số thật.

## 5. Định nghĩa hoàn thành chung

Một story chỉ hoàn thành khi:

- Có unit test cho router/validator/contract quyết định hành vi.
- Có integration test cho lưu `session/message/part`, tool pipeline hoặc endpoint tương ứng.
- Có E2E/smoke test cho đường người dùng áp dụng, gồm ít nhất một đường lỗi chính.
- Log/metadata không chứa secret, base64 ảnh lớn hoặc dữ liệu ngoài chính sách lưu trữ.
- Tài liệu API và migration dữ liệu được cập nhật nếu story đổi contract.

## 6. Không làm trong phạm vi backlog này

- Cho model chạy shell hoặc mã người dùng.
- Dựng plugin runtime đa gói, waterfall hook phức tạp hoặc subagent framework.
- Bật strict provenance trước khi có dữ liệu shadow mode đủ để đánh giá tỉ lệ vi phạm.
- Phụ thuộc vào cache prefix best-effort của nhà cung cấp để đảm bảo đúng đắn hay khả dụng.

## 7. Tài liệu liên quan

- [HARNESS_USE_CASES_VI.md](HARNESS_USE_CASES_VI.md)
- [HARNESS_SYSTEM_DESIGN_VI.md](HARNESS_SYSTEM_DESIGN_VI.md)
- [HARNESS_ARCHITECTURE.md](HARNESS_ARCHITECTURE.md)
