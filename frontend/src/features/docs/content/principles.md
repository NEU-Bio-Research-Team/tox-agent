Đây là những điều hệ thống tự thực thi, không phải lời hứa. Mỗi ràng buộc dưới đây đi kèm chỗ bạn có thể nhìn thấy nó trong sản phẩm.

## Ba ranh giới triển khai

Ba thành phần được tách ra để mỗi phần chỉ có đúng quyền nó cần.

- **ToxPred** đo. Nó stateless và chạy offline sau khi cấp phát.
- **Control plane** của ToxAgent sở hữu product state, session, validator và audit trail.
- **Agent runtime** (OpenCode/DSH) chỉ chạy vòng model-tool. Nó không public ra internet và không có quyền gì ngoài MCP tool đã khai báo cho đúng một run.

## Không có điểm tổng về độc tính

hERG channel blockade, Tox21 assay activity và ClinTox clinical-trial signal là ba phép đo độc lập. Không màn hình nào trong sản phẩm gộp chúng thành một nhãn, một màu hay một con số duy nhất.

**Bạn thấy ở đâu:** mỗi phép đo một thẻ riêng trong Phân tích nhanh. Xem [Đọc kết quả](/docs/reading-results).

## Mỗi con số neo vào một observation

Một claim trong câu trả lời luôn mang `observation_id` và `field_path` trỏ về đúng giá trị predictor đã trả. Validator từ chối bất kỳ con số nào không đối chiếu được với nguồn, trước khi câu trả lời được lưu.

**Bạn thấy ở đâu:** liên kết "Nguồn và kiểm chứng" dưới mỗi câu trả lời của agent.

## Model không tự commit câu trả lời

Text của model chỉ thành nội dung sản phẩm sau khi qua validator: số liệu, phân loại, giới hạn bắt buộc và claim bị cấm. Model được sửa tối đa hai lượt. Nếu vẫn sai, hệ thống tự dựng một câu trả lời dự phòng và đánh dấu rõ đó là bản dự phòng.

**Bạn thấy ở đâu:** banner trạng thái phát hành ở đầu mỗi câu trả lời.
