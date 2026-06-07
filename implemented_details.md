# Nhật Ký Thực Hiện Implementation Plan — Local LLM Integration & Fine-tuning

Tài liệu này ghi nhận toàn bộ những thay đổi và cài đặt đã được thực hiện trong quá trình triển khai cấu hình chạy Local LLM và chuẩn bị hạ tầng fine-tuning cho hệ thống ToxAgent.

---

## 1. Adapter Runtime Cho Local LLM (`local_llm_runtime.py`)
- **Tạo mới file [services/local_llm_runtime.py](file:///home/minhquang/tox-agent/services/local_llm_runtime.py):**
  - Định nghĩa class `LocalLLMClient` và `LocalLLMModels` giả lập hoàn toàn giao diện của `google.genai.Client` nhằm giảm thiểu tối đa sự thay đổi trong code của các agents hiện tại.
  - Implement các method `generate_content` và `generate_content_stream` tương thích với API của vLLM `/v1/chat/completions`.
  - Tích hợp tính năng **Auto-fallback sang Gemini API** khi `LLM_RUNTIME=auto`: Hệ thống sẽ gửi request đến local vLLM trước, nếu phát hiện server offline hoặc lỗi thì tự động chuyển hướng request sang Gemini API mà không làm gián đoạn luồng chạy của agent.
  - Xử lý chuyển đổi linh hoạt cấu hình từ Gemini SDK sang OpenAI format (system instructions, response_schema, structured JSON, temperature).

---

## 2. Tích Hợp Vào Runtime Chung Của Hệ Thống
- **Cập nhật [services/genai_runtime.py](file:///home/minhquang/tox-agent/services/genai_runtime.py):**
  - Sửa đổi hàm `build_genai_client_candidates()` để kiểm soát thông qua biến môi trường `LLM_RUNTIME`.
  - Nếu `LLM_RUNTIME=local`, trả về duy nhất local client candidate.
  - Nếu `LLM_RUNTIME=auto`, xếp local client candidate lên đầu danh sách candidates và xếp các Gemini clients xuống dưới để làm fallback tự động trong loop thử lỗi của agent.
- **Cập nhật [model_server/main.py](file:///home/minhquang/tox-agent/model_server/main.py):**
  - Sửa đổi hàm `_build_report_chat_client()` để khởi tạo và trả về local client trong phiên chat tương tác báo cáo khi cấu hình `local` hoặc `auto` được chọn.
- **Cập nhật cấu hình môi trường:**
  - Bổ sung cấu hình chi tiết cho local LLM ở cuối file [.env.example](file:///home/minhquang/tox-agent/.env.example) và [.env](file:///home/minhquang/tox-agent/.env).

---

## 3. Hạ Tầng Fine-tuning (Unsloth & TRL)
Đã triển khai đầy đủ các file python phục vụ fine-tuning tại thư mục `scripts/`:

- **[scripts/finetune_group_a.py](file:///home/minhquang/tox-agent/scripts/finetune_group_a.py):**
  - SFT training script cho Group A (Tool calling & Validation) sử dụng Qwen2.5-7B-Instruct.
  - Hỗ trợ lưu adapter và xuất trực tiếp ra GGUF format Q4_K_M bằng tính năng tối ưu của Unsloth.
- **[scripts/finetune_group_b_sft.py](file:///home/minhquang/tox-agent/scripts/finetune_group_b_sft.py):**
  - SFT training script cho Group B (MolRAG Reasoner) sử dụng Mistral-7B-Instruct-v0.3.
  - Áp dụng curriculum learning 2 giai đoạn: Stage 1 chuyên sâu về cơ chế suy luận (`mechanism_chain`), Stage 2 tối ưu định dạng đầu ra strict JSON.
  - Thêm SMILES-specific tokens vào tokenizer để tối ưu hóa quá trình học cấu trúc phân tử hóa học.
- **[scripts/grpo_rewards.py](file:///home/minhquang/tox-agent/scripts/grpo_rewards.py):**
  - Triển khai các hàm tính điểm reward cho GRPOTrainer:
    - `toxicity_label_reward`: Điểm cho độ chính xác của nhãn phân loại.
    - `json_schema_reward`: Kiểm tra và cộng điểm khi đầu ra tuân thủ strict JSON schema.
    - `mechanism_chain_quality`: Đánh giá chiều sâu của chuỗi cơ chế, sự xuất hiện của từ khóa assays và SMARTS.
    - `confidence_calibration`: Phạt độ tự tin thái quá khi độ tương đồng của analogs thấp (ECE reduction).
- **[scripts/finetune_group_b_grpo.py](file:///home/minhquang/tox-agent/scripts/finetune_group_b_grpo.py):**
  - Script huấn luyện RL cho MolRAG sử dụng GRPOTrainer và tích hợp với các hàm reward trên.
- **[scripts/finetune_group_c_d.py](file:///home/minhquang/tox-agent/scripts/finetune_group_c_d.py):**
  - Script huấn luyện ORPO preference learning cho Writer Agent (Group C) báo cáo độc tính.

---

## 4. Công Cụ Đánh Giá (Evaluation Benchmark)
- **Tạo mới file [scripts/eval_e2e_benchmark.py](file:///home/minhquang/tox-agent/scripts/eval_e2e_benchmark.py):**
  - Công cụ chạy đánh giá tự động end-to-end trên tập dữ liệu kiểm thử [test_data/full_test_set.csv](file:///home/minhquang/tox-agent/test_data/full_test_set.csv).
  - Thu thập và đo đạc các chỉ số: Độ chính xác phân loại (Accuracy), tỷ lệ lỗi cú pháp JSON, tỷ lệ gọi tool đúng, và độ trễ sinh từ trung bình (Latency).
  - Xuất báo cáo tổng hợp chi tiết ra file JSON để theo dõi sự cải thiện của mô hình qua các đợt fine-tune.
