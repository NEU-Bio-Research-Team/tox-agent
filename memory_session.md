# Session Memory — ToxAgent Local LLM Integration

Tài liệu này lưu trữ thông tin ngữ cảnh dự án, trạng thái hiện tại, cấu trúc kiến trúc và các bước tiếp theo để phục vụ cho các phiên làm việc (session) kế tiếp.

---

## 1. Ngữ Cảnh Dự Án (Context)
- **ToxAgent** là hệ thống AI đa tác tử (Multi-Agent) hỗ trợ sàng lọc và lập báo cáo độc tính học của các hợp chất hóa học (thông qua chuỗi SMILES).
- Hệ thống có 2 luồng hoạt động chính:
  1. Luồng chạy tác tử qua ADK (`TOX_AGENT_ANALYZE_RUNTIME=adk`).
  2. Luồng chạy tác tử deterministic thông qua Python (`TOX_AGENT_ANALYZE_RUNTIME=deterministic` - mặc định).
- Mục tiêu chính hiện tại là **chuyển đổi (swap) toàn bộ luồng gọi Gemini API sang Local LLM** phục vụ trên card đồ họa RTX 3090 (24GB VRAM) và thực hiện các quy trình fine-tuning để tối ưu hóa độ chính xác và bảo mật dữ liệu.

---

## 2. Trạng Thái Hiện Tại (Current State)
Hạ tầng tích hợp local LLM và chuẩn bị cho việc huấn luyện (Fine-tuning) đã được **triển khai và cấu hình hoàn tất** (Phase 0 hoàn thành):
- **Local Client Adapter:** Đã triển khai [services/local_llm_runtime.py](file:///home/minhquang/tox-agent/services/local_llm_runtime.py) đóng vai trò làm proxy trung gian. Nó đóng gói các API OpenAI/vLLM thành giao diện tương thích hoàn toàn với Google GenAI SDK (`client.models.generate_content`).
- **Auto-Fallback:** Tích hợp cơ chế tự động fallback sang Gemini API trong chế độ `LLM_RUNTIME=auto` nếu vLLM server gặp lỗi.
- **Serving & Routing Integration:** Tích hợp adapter vào hàm `build_genai_client_candidates()` của `genai_runtime.py` và hàm `_build_report_chat_client()` của `model_server/main.py`.
- **Huấn luyện mô hình:** Triển khai 4 scripts fine-tuning chính (`finetune_group_a.py`, `finetune_group_b_sft.py`, `finetune_group_b_grpo.py`, `finetune_group_c_d.py`) và module tính điểm reward `grpo_rewards.py` cho GRPO.
- **Công cụ Đánh giá:** Triển khai công cụ đo hiệu năng và độ chính xác end-to-end `eval_e2e_benchmark.py`.

---

## 3. Bản Đồ Code Quan Trọng (Key Code Map)
- **[services/local_llm_runtime.py](file:///home/minhquang/tox-agent/services/local_llm_runtime.py)**: Chứa mock client thực hiện gọi API local vLLM.
- **[services/genai_runtime.py](file:///home/minhquang/tox-agent/services/genai_runtime.py)**: Nơi các agents gọi để lấy candidate client candidates.
- **[model_server/main.py](file:///home/minhquang/tox-agent/model_server/main.py)**: File máy chủ chính của model server, tích hợp chat client.
- **[scripts/](file:///home/minhquang/tox-agent/scripts)**: Thư mục chứa toàn bộ các script huấn luyện và đánh giá.

---

## 4. Hướng Dẫn Các Bước Tiếp Theo (Next Steps)
Trong các session tiếp theo, bạn có thể thực hiện theo lộ trình:

1. **Chuẩn bị Dữ liệu:**
   - Soạn thảo và đặt các file dataset vào thư mục `data/` theo đúng tên file cấu hình trong các script training (ví dụ: `group_a_tool_calling.json`, `group_b_stage1.json`, `group_b_stage2.json`).
2. **Khởi động vLLM và test smoke:**
   - Chạy lệnh khởi động vLLM (xem chi tiết tại `config_and_run.md`).
   - Đặt `LLM_RUNTIME=local` hoặc `LLM_RUNTIME=auto` trong `.env` để kiểm tra hoạt động của adapter.
3. **Thực thi các script fine-tuning:**
   - Chạy các tiến trình training trên GPU RTX 3090: SFT trước cho Group A và B, sau đó chạy GRPO cho Group B.
4. **Đánh giá & So sánh hiệu năng:**
   - Thực thi `python scripts/eval_e2e_benchmark.py` để lấy baseline của mô hình local chưa fine-tune so với Gemini, sau đó so sánh lại sau khi hoàn tất training.
