Bây giờ tôi đã có đủ thông tin từ codebase để đưa ra quyết định cụ thể, kỹ thuật, và lý giải chắc chắn. Tôi sẽ trả lời trực tiếp 5 câu hỏi.

***

Dưới đây là quyết định cho từng câu hỏi trong phần **Open Questions**, lý giải trực tiếp từ việc đọc codebase của bạn.

***

## Q1: Base model cho Nhóm B (MolRAG) — `Mistral-7B-Instruct-v0.3` hay `BioMistral-7B`?

**Chọn: `Mistral-7B-Instruct-v0.3` làm base, fine-tune trên BioMistral's data.**

Lý do kỹ thuật rõ ràng: `BioMistral-7B` là *continual pretraining* của Mistral trên PubMed Central — nó tốt cho factual recall nhưng **không có instruction-following format**. Cụ thể với ToxAgent, `run_molrag_reasoning()` trong `molrag_reasoner.py` yêu cầu model output strict JSON theo `_MOLRAG_RESPONSE_SCHEMA` với nhiều nested fields (`mechanism_chain`, `key_substructures`, `confidence_rationale`) . BioMistral không được align cho structured output, sẽ khó fine-tune hơn vì phải đồng thời học cả instruction format lẫn chemistry domain.

**Chiến lược tối ưu hơn:** Dùng `Mistral-7B-Instruct-v0.3` (đã có instruction format), sau đó **inject biomedical knowledge qua fine-tuning data** — tức là mix dataset SMolInstruct + hERG + Tox21 của bạn (đã có trong repo tại `test_data/`) thay vì dùng BioMistral làm base. LlaSMol đã validate rõ Mistral-7B-Instruct fine-tune trên chemistry data beats GPT-4. Không cần BioMistral pretrain vì knowledge đó sẽ được inject qua SFT. [openreview](https://openreview.net/forum?id=lY6XTF9tPv)

***

## Q2: vLLM vs Ollama cho dev/staging — Cả hai hay chỉ vLLM?

**Chọn: Chỉ vLLM từ đầu, không setup Ollama song song.**

Đọc `adk_compat.py` cho thấy `LlmAgent` trong codebase của bạn là một abstraction layer — `google.adk.agents` được import với try/except fallback . Điều này có nghĩa pipeline thực chất không phụ thuộc vào model serving layer cụ thể của Ollama hay vLLM; tất cả đi qua `genai_runtime.py` .

Lý do không cần Ollama:

- **Structured output là blocker thực sự:** `_MOLRAG_RESPONSE_SCHEMA` cần `response_format` với JSON schema strict enforcement. Ollama hỗ trợ `format: json` nhưng **không enforce schema** — bạn sẽ phải giữ nguyên `_safe_json_parse()` repair logic phức tạp. vLLM với xgrammar **constrained decoding** loại bỏ hoàn toàn vấn đề này ngay từ dev.
- **Overhead 2 stack:** Setup và maintain 2 serving stack (Ollama dev + vLLM prod) tạo ra discrepancy khó debug — behavior khác nhau giữa môi trường.
- **RTX 3090 VRAM:** 24GB đủ để vLLM serve Qwen2.5-7B-Instruct ở bfloat16 (14GB), còn dư 10GB margin. Không cần trade-off dùng Ollama để nhẹ hơn.

Dev với vLLM dùng `--gpu-memory-utilization 0.7` thay vì 0.85 là đủ nhẹ cho local development.

***

## Q3: Multi-model serving — Multi-LoRA adapters hay export GGUF riêng?

**Chọn: Export GGUF riêng từng nhóm (2 models tổng), không dùng multi-LoRA.**

Lý do từ codebase: `genai_runtime.py` cho thấy có 2 tier models — `AGENT_MODEL_FAST` (screening, validator, molrag fast) và `AGENT_MODEL_PRO` (researcher, molrag fallback) . Mapping tự nhiên nhất là **2 GGUF models**:

- `model-fast`: Qwen2.5-7B fine-tuned cho Nhóm A + D (tool calling + QA)
- `model-reasoning`: Mistral-7B fine-tuned cho Nhóm B + C (chemistry reasoning + writing)

**Tại sao không dùng multi-LoRA của vLLM:**

| | Multi-LoRA | Export GGUF riêng |
|--|--|--|
| VRAM overhead | 1 base model + N adapters đồng thời | 2 models load sequentially |
| Adapter switching latency | ~50ms per request | Zero (model đã loaded) |
| Stability | Experimental feature, adapter contamination risk | Production-tested |
| RTX 3090 feasibility | 1 base (14GB) + 4 adapters (~1GB/adapter) ≈ 18GB | 2x ~7GB Q4_K_M = ~14GB total, serve 1 at a time |

Với throughput không cao (toxicology screening, không phải chat app), serving sequential theo route request là hoàn toàn đủ. `orchestrator_agent.py` đã có `ANALYSIS_AGENT_MODE = sequential` làm default .

***

## Q4: Dataset priority — Synthetic từ Gemini traces hay public datasets trước?

**Chọn: Public datasets trước (SMolInstruct + Tox21), sau đó mới bổ sung synthetic traces.**

Lý do thực tiễn từ repository của bạn: Repo không có production logs folder — không có sẵn `(input_smiles, retrieved_examples) → molrag_output` traces từ usage thực tế. Test data có (`test_data/full_test_set.csv`, `reference_panel.csv`, `screening_library.csv`) nhưng đây là ground truth test sets, không phải training data .

**Pipeline đúng:**

1. **Tuần 1:** Download SMolInstruct subset (`osunlp/SMolInstruct`, filter property prediction + toxicity tasks) → format theo `_MOLRAG_RESPONSE_SCHEMA` → đây là SFT corpus chính. Size ~5K samples là đủ để bootstrap.
2. **Song song tuần 1:** Dùng Gemini API (bạn đang có key sẵn) để generate `mechanism_chain` annotations cho hERG và Tox21 compounds trong `test_data/` → đây là teacher distillation, không phải production logs.
3. **Tuần 2+:** Sau khi có local model chạy được, bắt đầu capture inference traces từ local model + filter by confidence → dùng làm continual fine-tuning data.

Synthetic từ Gemini traces **rất valuable** nhưng cần infrastructure local model hoạt động trước. Public datasets không có dependency đó.

***

## Q5: Qwen2.5-3B cho Nhóm D hay giữ 7B?

**Chọn: Giữ 7B cho tất cả, nhưng dùng chung 1 fine-tuned 7B cho Nhóm A + D.**

Lý do từ codebase: `EvidenceQAAgent` (`evidence_qa_agent.py`) và `ReportChatAgent` (`report_chat_agent.py`) có instruction phức tạp — đặc biệt `report_chat_agent.py` dài 1009 lines với 9 tools (`get_article_detail`, `check_claim_support`, `rerun_screening`, `query_molrag_live`, ...) . Với 3B model, tool selection accuracy cho 9 tools song song sẽ drop đáng kể so với 7B.

**Tuy nhiên, không cần fine-tune riêng Nhóm D:** Qwen2.5-7B-Instruct **base model chưa fine-tune** đã rất tốt cho conversational QA. Benchmark cho thấy Qwen2.5-7B đứng top trong multilingual instruction following bao gồm Vietnamese. Chiến lược thực tế:

1. Phase 0: Test Qwen2.5-7B-Instruct base model cho EvidenceQA và ReportChat ngay lập tức.
2. Nếu baseline đủ tốt (>80% answer relevance) → không cần fine-tune Nhóm D riêng.
3. Chỉ fine-tune Nhóm D nếu baseline thất bại trên multi-turn toxicology-specific conversations.

3B chỉ hợp lý nếu bạn cần serve **4+ sessions đồng thời** trên 24GB — với usecase toxicology screening hiện tại, đây không phải bottleneck.

***

## Tóm Tắt Quyết Định

| # | Quyết Định | Lựa Chọn |
|---|-----------|----------|
| Q1 | Base model Nhóm B | `Mistral-7B-Instruct-v0.3` + chemistry SFT data |
| Q2 | Serving stack | **Chỉ vLLM** từ đầu (xgrammar constrained decoding là bắt buộc) |
| Q3 | Multi-model strategy | **2 GGUF exports riêng**: model-fast + model-reasoning |
| Q4 | Dataset priority | **Public first** (SMolInstruct → Tox21 → synthetic) |
| Q5 | Model size Nhóm D | **7B cho tất cả**, test base model trước khi fine-tune |