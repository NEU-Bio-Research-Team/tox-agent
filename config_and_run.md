# Hướng Dẫn Cấu Hình Và Chạy Local LLM / Fine-tuning cho ToxAgent

Tài liệu này hướng dẫn chi tiết cách cấu hình và chạy hệ thống ToxAgent sử dụng Local LLM (qua vLLM serving) và thực hiện các quy trình fine-tuning (SFT & GRPO).

---

## 1. Cấu Hình Environment Variables

Tất cả các cấu hình được quản lý qua file `.env`. Hãy thêm hoặc cập nhật các biến sau ở cuối file `.env` của bạn:

```env
# ============================================================
# LOCAL LLM CONFIGURATION (vLLM / Custom)
# ============================================================
LLM_RUNTIME=local                   # Lựa chọn: local | gemini | auto
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL_FAST=model-fast     # Model phục vụ Group A + D (Qwen2.5-7B)
LOCAL_LLM_MODEL_PRO=model-reasoning # Model phục vụ Group B + C (Mistral-7B)
```

### Chế độ hoạt động (`LLM_RUNTIME`):
- **`local`**: Chỉ sử dụng Local LLM server qua vLLM.
- **`gemini`**: Mặc định sử dụng Gemini API.
- **`auto`**: Thử gọi Local LLM server trước; nếu server bị offline hoặc lỗi, hệ thống tự động fallback về Gemini API.

---

## 2. Thiết Lập vLLM Serving Environment (RTX 3090 / CUDA)

Khởi động vLLM server OpenAI-compatible để serve các models đã fine-tune.

### Cài đặt vLLM:
```bash
pip install vllm
# Hoặc chạy qua Docker container: vllm/vllm-openai
```

### Khởi động vLLM Server:
Để serve model, ví dụ với base Qwen2.5-7B-Instruct:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --structured-outputs-config.backend xgrammar \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85
```

- `--structured-outputs-config.backend xgrammar`: Enforce JSON Schema output cho các task của MolRAG.
- `--gpu-memory-utilization 0.85`: Dành 15% VRAM trống cho các tiến trình khác trên GPU 24GB.

---

## 3. Thực Hiện Fine-tuning (SFT & GRPO)

Các scripts fine-tuning đã được cấu hình hoàn chỉnh tại thư mục `scripts/` sử dụng thư viện **Unsloth** để tối ưu hóa tốc độ và giảm VRAM tiêu thụ.

### Yêu cầu cài đặt:
```bash
pip install unsloth trl peft datasets accelerate bitsandbytes requests
```

### Chạy Fine-tuning từng Group:

1. **Group A (Tool Calling / SMILES Validation):**
   ```bash
   python scripts/finetune_group_a.py
   ```
   *Sử dụng base model Qwen2.5-7B-Instruct, train trên tập tool calling và xuất ra GGUF Q4_K_M.*

2. **Group B (MolRAG Reasoner - SFT):**
   ```bash
   python scripts/finetune_group_b_sft.py
   ```
   *Sử dụng base model Mistral-7B-Instruct-v0.3 với curriculum learning 2 giai đoạn (tập trung vào mechanism chain trước, sau đó là full response schema) và xuất ra GGUF.*

3. **Group B (MolRAG Reasoner - GRPO Reinforcement Learning):**
   ```bash
   python scripts/finetune_group_b_grpo.py
   ```
   *Sử dụng base model là checkpoint SFT thu được ở bước trên. Thực hiện reinforcement learning qua GRPOTrainer sử dụng các hàm reward tại `scripts/grpo_rewards.py` (bao gồm schema compliance, label target, quality chain và ECE calibration).*

4. **Group C & D (Writer Agent & QA):**
   ```bash
   python scripts/finetune_group_c_d.py
   ```
   *Sử dụng base model Qwen2.5-7B-Instruct thực hiện ORPO preference training cho Writer Agent tạo báo cáo.*

---

## 4. Chạy Evaluation Benchmark

Để đo đạc hiệu năng và độ chính xác của hệ thống ToxAgent sử dụng Local LLM:

```bash
# Chạy đánh giá đầy đủ end-to-end trên tập test
python scripts/eval_e2e_benchmark.py --test-set test_data/full_test_set.csv --output results/e2e_benchmark_results.json

# Chạy thử nghiệm nhanh (Giới hạn 5 mẫu)
python scripts/eval_e2e_benchmark.py --test-set test_data/full_test_set.csv --limit 5
```

Kết quả benchmark (Accuracy, Latency, Compliance rate) sẽ được lưu tại file `results/e2e_benchmark_results.json`.
