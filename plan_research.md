<aside>
✅

Tài liệu này đã được **kiểm chứng trực tiếp** với mã nguồn thật trên branch `agent_test` của repo NEU-Bio-Research-Team/tox-agent. Mỗi nhận định về bug/gap đều ghi rõ trạng thái: **ĐÚNG (xác nhận)**, **CẦN SỬA LẠI**, hoặc **bổ sung mới**. Một số kết luận trong bản nháp gốc của bạn (đặc biệt Gap #3) **không khớp với code thực tế** và đã được chỉnh lại bên dưới.

</aside>

## Tóm tắt kiểm chứng nhanh

| Nhận định trong bản nháp | Trạng thái sau khi đọc code | Ghi chú |
| --- | --- | --- |
| Bug #1 — `import json` sai scope trong `finetune_group_a.py` | ✅ **ĐÚNG** | Crash thật khi `dataset.map()` chạy `format_prompts` ở subprocess worker. |
| Gap #2 — `REQUIRED_KEYS` (7) lệch `_MOLRAG_RESPONSE_SCHEMA` (11) | ✅ **ĐÚNG** (và còn tệ hơn) | Nhánh thưởng `1.5` cho "perfect match" gần như **không bao giờ đạt được** nếu model xuất đủ 11 field. Xem phân tích bên dưới. |
| Gap #3 — `genai_runtime.py` trả về `LocalLLMClient` khi `LLM_RUNTIME=local` | ❌ **SAI — cần sửa lại** | `build_genai_client_candidates()` **không hề** tham chiếu `LocalLLMClient`. Local runtime là **dead code chưa được nối dây**. Đây mới là gap thật. |
| Gap #3 — `config = genai.types...` sẽ crash khi `genai = None` | ❌ **SAI** | Đã có guard `if genai is None ... return result` **trước** dòng đó, nên không bao giờ chạm tới. "Fix" đề xuất là thừa. |
| Group A tool gồm `validate_smiles`  • `predict_toxicity` | ⚠️ **MỘT PHẦN** | Tool thật là `validate_smiles` và **`analyze_molecule`** (không phải `predict_toxicity`). |
| GRPO: `use_vllm=True`, `vllm_gpu_memory_utilization=0.4`, `num_generations=4` | ✅ **ĐÚNG** | Xác nhận trong `finetune_group_b_grpo.py`. |

---

## Phần 1 — Fine-tuning LLM là gì và quy trình tổng quát

### Khái niệm nền tảng

Fine-tuning là quá trình tiếp tục huấn luyện một pretrained LLM trên tập dữ liệu nhỏ, chuyên biệt hơn, nhằm "khắc" vào model những hành vi cụ thể mà pretrained weights chưa có sẵn. Pretrained LLM (Qwen2.5-7B, Mistral-7B) đã nắm ngôn ngữ, lý luận chung và nhiều kiến thức chemistry/biology. Fine-tuning **không dạy lại từ đầu** — nó *điều hướng lại* model theo domain của bạn.

Repo của bạn dùng **3 paradigm**, mỗi cái cho một nhóm agent:

| Paradigm | Script (đã xác nhận) | Base model | Cơ chế |
| --- | --- | --- | --- |
| **SFT** (Supervised Fine-Tuning) | `finetune_group_a.py`, `finetune_group_b_sft.py` | Qwen2.5-7B-Instruct / Mistral-7B-Instruct-v0.3 | Minimize cross-entropy trên cặp (prompt, response) — bắt chước teacher. |
| **GRPO** (Group Relative Policy Optimization) | `finetune_group_b_grpo.py` | Checkpoint SFT của Group B | Sinh G completions/prompt, dùng reward functions điều chỉnh policy. |
| **ORPO** (Odds-Ratio Preference Optimization) | `finetune_group_c_d.py` | Qwen2.5-7B-Instruct | Gộp SFT loss + contrastive preference loss (chosen vs rejected) trong một pass, không cần reference model. |

**QLoRA** được dùng ở cả 3 script (`load_in_4bit=True` + LoRA adapters qua Unsloth `FastLanguageModel`): model gốc quantize 4-bit (đóng băng), chỉ train ~0.3–1% tham số. Đây là yếu tố giúp fit RTX 3090 24GB.

### Quy trình từ A đến Z

```
[1] Thu thập raw data (SMILES + labels, ví dụ Tox21)
        ↓
[2] Generate teacher responses (Gemini Flash) — knowledge distillation
        ↓
[3] Preprocess → JSON đúng schema từng script
        ↓
[4] Sửa bug/gap trong repo (xem Phần 2)
        ↓
[5] SFT: Group A → Group B (Stage 1 → Stage 2) → Group C
        ↓
[6] GRPO: chạy TỪ checkpoint SFT của Group B
        ↓
[7] Serve GGUF qua vLLM (OpenAI-compatible endpoint)
        ↓
[8] *** NỐI DÂY local runtime *** rồi đặt LLM_RUNTIME=local (xem Gap #3)
```

<aside>
⚠️

Bước [8] trong bản nháp giả định chỉ cần `export LLM_RUNTIME=local` là agent sẽ dùng model local. **Điều này hiện KHÔNG đúng** với code trên `agent_test` — phải sửa code nối dây trước (chi tiết ở Gap #3). Nếu không, dù train model local xong, các agent vẫn không bao giờ gọi tới nó.

</aside>

---

## Phần 2 — Bug/Gap cần sửa trước khi train

### Bug #1 (CRASH) — `import json` sai scope trong `finetune_group_a.py` — ✅ ĐÚNG

Đã xác nhận trong scripts/finetune_group_a.py: `import json` nằm **bên trong** `if __name__ == "__main__":` ở cuối file, trong khi `format_prompts()` (được gọi qua `dataset.map(format_prompts, batched=True)`) dùng `json.dumps(...)` ở module scope. Khi HuggingFace `datasets` chạy `format_prompts` trong subprocess worker (`dataset_num_proc`), `json` chưa được import ở scope đó → **NameError / crash**.

**Fix** — đưa `import json` lên đầu file:

```python
# Đầu file finetune_group_a.py
import json  # ← phải ở module level, KHÔNG để trong __main__
import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
```

<aside>
💡

**Lưu ý đối chiếu**: `finetune_group_b_sft.py`, `finetune_group_b_grpo.py`, `finetune_group_c_d.py` đều đã có `import json` ở đầu file → KHÔNG dính bug này. Chỉ `finetune_group_a.py` cần sửa.

</aside>

### Gap #2 (CRITICAL) — `REQUIRED_KEYS` lệch schema — ✅ ĐÚNG (và nghiêm trọng hơn bản nháp mô tả)

Xác nhận: trong scripts/grpo_rewards.py, `REQUIRED_KEYS` có đúng **7 key**. Trong agents/molrag_reasoner.py, `_MOLRAG_RESPONSE_SCHEMA` có **11 `properties`** nhưng phần `required` của schema lại chỉ liệt kê **5 key**. Tức là có **lệch ba chiều**:

| Nguồn | Số key | Danh sách |
| --- | --- | --- |
| `_MOLRAG_RESPONSE_SCHEMA.properties` | 11 | evidence_overview, longform_summary, mechanism_chain, key_substructures, confidence_rationale, analogy_reasoning, risk_modifiers, knowledge_highlights, literature_highlights, suggested_label, confidence |
| `_MOLRAG_RESPONSE_SCHEMA.required` | 5 | evidence_overview, longform_summary, mechanism_chain, suggested_label, confidence |
| `REQUIRED_KEYS` (reward) | 7 | 5 key trên + key_substructures, confidence_rationale |

**Hệ quả nghiêm trọng hơn bản nháp nói**: trong `json_schema_reward`, nhánh thưởng `1.5` chỉ kích hoạt khi `len(overlap) == len(REQUIRED_KEYS)` **VÀ** `len(present_keys) == len(REQUIRED_KEYS)` — tức model phải xuất **đúng 7 key, không thừa không thiếu**. Nhưng nếu bạn train model (qua SFT Stage 2) để xuất **đủ 11 field theo schema**, thì `present_keys = 11 ≠ 7` → **không bao giờ** chạm nhánh `1.5`, chỉ nhận `score = 7/11 ≈ 0.636`. Reward signal vừa sai vừa mâu thuẫn với mục tiêu SFT.

**Fix khuyến nghị** — đồng bộ `REQUIRED_KEYS` lên đủ 11 và làm reward mượt:

```python
REQUIRED_KEYS = {
    "evidence_overview", "longform_summary", "mechanism_chain",
    "key_substructures", "confidence_rationale", "suggested_label", "confidence",
    # 4 key còn thiếu:
    "analogy_reasoning", "risk_modifiers", "knowledge_highlights", "literature_highlights",
}

def json_schema_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            payload = json.loads(_strip_fence(completion))
            present = set(payload.keys())
            overlap = REQUIRED_KEYS & present
            partial = len(overlap) / len(REQUIRED_KEYS)
            # Thưởng theo tỉ lệ, nhân hệ số khi đủ — gradient mượt, không nhị phân
            rewards.append(partial * 1.5)
        except Exception:
            rewards.append(-1.0)
    return rewards
```

<aside>
📌

Quyết định thiết kế cần bạn chốt: model nên xuất **7 hay 11 field**? Nếu giữ schema 11 field, đồng bộ `REQUIRED_KEYS=11`. Nếu chỉ cần 7 field cốt lõi, rút gọn `_MOLRAG_RESPONSE_SCHEMA.properties` xuống 7. Quan trọng là **schema sinh dữ liệu, schema response, và REQUIRED_KEYS phải khớp nhau**.

</aside>

### Gap #3 — ❌ BẢN NHÁP SAI, đây là phân tích đúng

Bản nháp kết luận: *"`genai_runtime.py` đã handle `LLM_RUNTIME=local` bằng cách trả về `LocalLLMClient`"* và *"`config = genai.types...` sẽ crash khi `genai=None`"*. **Cả hai đều không khớp code thực tế.**

**Sự thật 1 — `genai_runtime.py` KHÔNG hề biết tới local runtime.**

Đọc services/genai_runtime.py: `build_genai_client_candidates()` chỉ tạo client Gemini (API key) hoặc Vertex AI. Nó **không import** và **không bao giờ trả về** `LocalLLMClient`. Nếu `genai is None` → trả về `[]`. Không có một dòng nào kiểm tra `LLM_RUNTIME`.

**Sự thật 2 — local runtime là dead code chưa nối dây.**

services/local_llm_runtime.py định nghĩa `LocalLLMClient`, `LocalLLMModels.generate_content()` (có đọc `LLM_RUNTIME`, có convert `response_schema` → `response_format` cho vLLM) và `build_local_client()`. **Nhưng không file nào gọi `build_local_client()`** trong luồng chọn client của agent. Tức là dù `LLM_RUNTIME=local`, MolRAG/Writer vẫn đi qua `build_genai_client_candidates()` → vẫn cố gọi Gemini, **không bao giờ chạm model local của bạn**.

**Sự thật 3 — không có crash `genai.types` như mô tả.**

Đọc `run_molrag_reasoning()` trong `molrag_reasoner.py`, thứ tự thực thi là:

```python
result = _deterministic_reasoning(...)   # 1) Luôn tính kết quả DETERMINISTIC trước
result["prompt_preview"] = prompt[:1800]

if genai is None or not MOLRAG_MODEL:    # 2) GUARD — thoát sớm nếu không có genai
    result["llm_status"] = "llm_unavailable"
    return result

client_candidates = build_genai_client_candidates()
if not client_candidates:
    result["llm_status"] = "llm_client_unavailable"
    return result

config = genai.types.GenerateContentConfig(...)  # 3) Chỉ tới đây khi genai != None
```

Vì có guard `if genai is None: return result` **trước** dòng `genai.types...`, nên **không thể** xảy ra `AttributeError: 'NoneType'`. "Fix" wrap `if genai is not None` mà bản nháp đề xuất là **thừa**.

**Sự thật 4 (quan trọng nhất cho fine-tuning) — MolRAG về bản chất là DETERMINISTIC.**

`run_molrag_reasoning()` luôn sinh đầy đủ output (`mechanism_chain`, `key_substructures`, `confidence`, ...) bằng **Python thuần** (`_deterministic_reasoning`, các hàm `_build_*`). LLM chỉ là lớp *tùy chọn ghi đè* (`result.update(llm_out)`) khi có Gemini. Nghĩa là: fine-tune Group B chỉ có tác dụng **khi luồng LLM được kích hoạt** (`molrag_enabled=True` + có client). Hiện `molrag_enabled` mặc định `False` trong `run_screening`.

**Gap #3 thật sự cần sửa** — nối dây local runtime vào bộ chọn client. Ví dụ trong `services/genai_runtime.py`:

```python
def build_genai_client_candidates(location_override=None):
    runtime = (os.getenv("LLM_RUNTIME") or "gemini").strip().lower()
    if runtime in ("local", "auto"):
        try:
            from services.local_llm_runtime import build_local_client
            local = [(build_local_client(), "local_llm")]
            if runtime == "local":
                return local            # chỉ dùng local
            # auto: ưu tiên local rồi fallback gemini bên dưới
        except Exception as exc:
            LOG.warning("Local client init failed: %s", exc)
            local = []
    else:
        local = []
    # ... phần build Gemini/Vertex hiện có ...
    return local + gemini_candidates
```

<aside>
🔧

Thêm một điểm khớp tốt: `LocalLLMModels.generate_content()` đã xử lý cả `config` dạng **dict lẫn object** (`getattr`/`.get`), và `molrag_reasoner` truyền `genai.types.GenerateContentConfig(...)` (object) trong khi `writer_agent` truyền **dict**. Cả hai đều tương thích với local client. Nhưng vì `genai.types.GenerateContentConfig` chỉ tồn tại khi thư viện `google-genai` được cài, bạn vẫn cần `genai` installed để đi qua guard — kể cả khi muốn dùng local. Cân nhắc bỏ điều kiện `genai is None` ra khỏi guard cho nhánh local, hoặc tạo config bằng dict khi chạy local.

</aside>

### Các vấn đề bổ sung phát hiện thêm khi đọc code

<aside>
🆕

Những mục dưới đây **không có trong bản nháp** nhưng là rủi ro thật, nên xử lý trước khi train/serve.

</aside>

- **Import tương đối của reward functions.** `finetune_group_b_grpo.py` dùng `from grpo_rewards import (...)` (không phải `from scripts.grpo_rewards`). Phải **chạy từ trong thư mục `scripts/`** (hoặc thêm `scripts/` vào `PYTHONPATH`), nếu không sẽ `ModuleNotFoundError`.
- **Group C target lệch với hành vi Writer thật.** `finetune_group_c_d.py` train model xuất **toàn bộ report JSON** (`executive_summary`, `risk_level`, `sections`...). Nhưng trong agents/writer_agent.py, LLM **chỉ sinh phần `recommendations`** (JSON nhỏ), còn `risk_level` và phần khung report do Python tính (`_compute_risk_level`, `_default_recommendations`). → Nếu mục tiêu là cải thiện Writer trong pipeline hiện tại, nên train theo **schema `recommendations`** thật, không phải full report.
- **Tên tool Group A.** Screening dùng tool thật là **`analyze_molecule`** (+ `validate_smiles`), không phải `predict_toxicity`. Dữ liệu tool-calling phải khớp tên + chữ ký thật, nếu không model học sai tool.
- **`add_tokens` + `resize_token_embeddings` ở Group B.** `finetune_group_b_sft.py` thêm token SMILES rồi resize embedding. Khi **export GGUF và serve bằng vLLM**, tokenizer phục vụ phải đồng bộ các token mới này, nếu không inference sẽ lệch. Kiểm tra kỹ khâu này.

---

## Phần 3 — Data cần thu thập & cách preprocess từng Group

Chiến lược chung: dùng **Tox21** làm nguồn SMILES + label, và **Gemini Flash** làm *teacher* để distill xuống model local.

### Group A — `data/group_a_tool_calling.json` (SFT tool-calling, Qwen2.5-7B)

**Mục đích**: dạy InputValidator + ScreeningAgent gọi đúng tool với đúng arguments từ input SMILES.

**Schema mẫu (đã sửa tên tool theo code thật)**:

```json
{
  "tools": [
    {"name": "validate_smiles",
     "description": "Verify SMILES validity and canonicalize.",
     "parameters": {"type": "object", "properties": {"smiles": {"type": "string"}}}},
    {"name": "analyze_molecule",
     "description": "Run clinical + mechanism toxicity analysis on a canonical SMILES.",
     "parameters": {"type": "object", "properties": {
        "smiles": {"type": "string"},
        "clinical_threshold": {"type": "number"},
        "mechanism_threshold": {"type": "number"}}}}
  ],
  "query": "Analyze the toxicity of molecule CC(=O)Oc1ccccc1C(=O)O",
  "response": {"name": "validate_smiles", "arguments": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}}
}
```

**Target**: 1000–2000 samples. Muốn nhanh: 500 samples + `num_train_epochs=5`.

### Group B Stage 1 — `data/group_b_stage1.json` (SFT mechanism_chain, Mistral-7B)

**Mục đích**: dạy MolRAG sinh `mechanism_chain` (chuỗi lý luận cơ chế). Stage 1 tập trung mechanism, chưa cần full 11-field.

**Schema mẫu (khớp `format_prompts` trong `finetune_group_b_sft.py`)** — cần các field `smiles`, `baseline`, `contexts`, `response`:

```json
{
  "smiles": "c1ccc(N)cc1",
  "baseline": {"label": "Toxic", "score": 0.82},
  "contexts": ["Aniline derivatives cause methemoglobinemia...",
               "Primary aromatic amines undergo N-oxidation..."],
  "response": {
    "evidence_overview": "3 analogs retrieved, top similarity=0.87",
    "longform_summary": "Aniline is a prototypical aromatic amine...",
    "mechanism_chain": [
      "SMARTS match: Primary Aromatic Amine — methemoglobin formation",
      "Mechanism: N-oxidation → hydroxylamine → reactive intermediate",
      "Analog vote: 2.1 toxic / 0.3 non-toxic → leans Toxic"],
    "key_substructures": ["Primary aromatic amine", "Benzene ring"],
    "confidence_rationale": "...",
    "suggested_label": "Toxic",
    "confidence": 0.85
  }
}
```

**Nguồn `contexts` tốt nhất**: tái dùng chính pipeline production. `screening_agent.py` gọi `retrieve_similar_molecules(...)` (từ `services`) để lấy analog + similarity, và `molrag_reasoner.py` gọi `retrieve_knowledge_context(...)`. Chạy hai hàm này trên mỗi SMILES Tox21 để sinh `contexts` đúng phân phối mà model sẽ gặp lúc inference. Bổ sung SMolInstruct (lọc các task toxicity) nếu cần thêm lượng.

### Group B Stage 2 — `data/group_b_stage2.json` (SFT full schema)

**Mục đích**: dạy model xuất **đủ schema** (xem lại Gap #2 để chốt 7 hay 11 field). Dùng Gemini Flash distill ra response đầy đủ. **Target**: 500–1000 samples (học trên nền Stage 1).

### Group B GRPO — `data/group_b_grpo.json`

**Khác biệt**: GRPO **không cần** field `response`. Theo `finetune_group_b_grpo.py` và các reward, mỗi sample cần: `smiles`, `baseline`, `contexts`, và metadata cho reward: `label_targets`, `max_similarities`.

```json
{
  "smiles": "O=C(O)c1ccccc1",
  "baseline": {"label": "Non-toxic", "score": 0.15},
  "contexts": ["Benzoic acid is a common food preservative..."],
  "label_targets": "non-toxic",
  "max_similarities": 0.91
}
```

`max_similarities` = Tanimoto similarity giữa SMILES và analog top-1, lấy từ `retrieve_similar_molecules(...)`. **Target**: 200–500 samples.

<aside>
⚠️

`confidence_calibration` reward thực tế tính `reward = max(1.0 - abs(confidence - sim), 0.0)` — tức là phạt theo khoảng cách giữa `confidence` model xuất và `max_similarities`. Đây là **proxy calibration đơn giản**, chưa phải ECE đầy đủ. Bản nháp gọi là "ECE-based reward" là **gần đúng nhưng nói quá**; nếu định publish (Contribution 4) thì cần đo ECE/Brier riêng, đừng tuyên bố reward này = ECE.

</aside>

### Group C — `data/group_c_writer_preference.json` (ORPO, Qwen2.5-7B)

**Schema khớp `format_prompts`**: cần `screening`, `research`, `language`, `chosen`, `rejected`.

```json
{
  "screening": {"summary": "NR-AR: positive, NR-AhR: positive", "final_verdict": "TOXIC"},
  "research": {"consensus_mechanisms": ["hERG blocking", "mitochondrial uncoupling"]},
  "language": "vi",
  "chosen": {"...báo cáo chi tiết, đúng cấu trúc..."},
  "rejected": {"...báo cáo mơ hồ, thiếu section..."}
}
```

**Cách tạo cặp**: cùng một molecule, sinh `chosen` (Gemini `temperature=0.1`, đầy đủ) vs `rejected` (`temperature=1.2` hoặc rule-based degradation: xóa field, thay text generic, sai `risk_level`).

<aside>
📌

Nhắc lại từ Phần 2: nếu mục tiêu là cải thiện Writer **đang chạy trong repo**, hãy cân nhắc đổi target sang schema `recommendations` (priority / action_type / action / rationale) — vì đó mới là phần LLM thật sự sinh ra trong `writer_agent.py`.

</aside>

---

## Phần 4 — Thứ tự train & hardware (RTX 3090 24GB)

### Cấu hình đã xác nhận từ code

| Script | Model | LoRA | seq len | epochs / lr |
| --- | --- | --- | --- | --- |
| `finetune_group_a.py` | Qwen2.5-7B-Instruct | r=32, alpha=16 | 4096 | 3 ep / 2e-4 |
| `finetune_group_b_sft.py` | Mistral-7B-Instruct-v0.3 | r=64, alpha=32 | 8192 | S1: 3ep/1e-4 · S2: 2ep/5e-5 |
| `finetune_group_b_grpo.py` | checkpoint SFT Group B | r=64, alpha=32 | 8192 | 1 ep / 2e-6 · num_generations=4 |
| `finetune_group_c_d.py` | Qwen2.5-7B-Instruct | r=32, alpha=16 | 8192 | 2 ep / 5e-6 · beta=0.1 |

### Thứ tự chạy

```bash
pip install unsloth trl transformers datasets accelerate bitsandbytes vllm

# 1) Group A (Qwen2.5-7B tool-calling)
python scripts/finetune_group_a.py            # NHỚ sửa Bug #1 trước

# 2) Group B SFT (Stage 1 → Stage 2 tự động trong 1 script)
python scripts/finetune_group_b_sft.py

# 3) Group B GRPO — chạy SAU SFT, TỪ TRONG thư mục scripts/
cd scripts && python finetune_group_b_grpo.py && cd ..

# 4) Group C ORPO
python scripts/finetune_group_c_d.py
```

<aside>
⚠️

**Rủi ro OOM trên 1 GPU.** `finetune_group_b_grpo.py` đặt `use_vllm=True`, `vllm_device="cuda:0"`, `vllm_gpu_memory_utilization=0.4`. Chạy đồng thời training model + vLLM inference trên cùng RTX 3090 24GB rất dễ OOM (model 7B 4-bit + KV cache + vLLM). Khuyến nghị: giảm `num_generations` từ 4 → 2, hoặc `use_vllm=False`, hoặc giảm `vllm_gpu_memory_utilization` xuống ~0.25. `max_prompt_length=4096` + `max_completion_length=4096` cũng rất nặng — cân nhắc hạ xuống 2048 nếu nội dung cho phép.

</aside>

---

## Phần 5 — Serve model & nối dây local runtime

Mỗi script export GGUF `q4_k_m` (qua `model.save_pretrained_gguf`). Serve bằng vLLM (OpenAI-compatible):

```bash
# Serve Group B GRPO (MolRAG reasoning)
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/mistral-7b-group-b-grpo-gguf --port 8000 --dtype bfloat16
```

Đặt env theo đúng tên biến trong `local_llm_runtime.py`:

```bash
export LLM_RUNTIME=local
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
export LOCAL_LLM_MODEL_FAST=qwen2.5-7b-group-a-sft
export LOCAL_LLM_MODEL_PRO=mistral-7b-group-b-grpo
```

<aside>
🚨

**Điều kiện tiên quyết (Gap #3):** các env trên **chưa đủ**. Vì `build_genai_client_candidates()` không trả về `LocalLLMClient`, bạn **bắt buộc** phải sửa code nối dây `build_local_client()` (xem Gap #3) trước. Nếu không, `LLM_RUNTIME=local` sẽ bị bỏ qua và agent vẫn cố gọi Gemini (hoặc chỉ trả deterministic output). Đây là điểm khác biệt lớn nhất so với bản nháp gốc.

</aside>

Lưu ý mapping model trong `LocalLLMModels._resolve_model`: tên chứa `flash`/`fast` → `LOCAL_LLM_MODEL_FAST`; chứa `pro`/`reasoning`/`writer`/`molrag`/`mistral` → `LOCAL_LLM_MODEL_PRO`. Đặt tên model phục vụ cho khớp logic này.

---

## Phần 6 — Novelty contributions (có thể publish)

| # | Contribution | Novelty | Đo lường |
| --- | --- | --- | --- |
| 1 | **Curriculum GRPO**: warm-start GRPO từ checkpoint SFT domain-specific (thay vì base model) cho molecular reasoning. | Cao | So sánh reward curves: (a) base Mistral, (b) SFT-S1, (c) SFT-S2. Tham chiếu DeepSeekMath. |
| 2 | **Agent-Specific Evaluation Harness**: đo hành vi agent thay vì accuracy đơn lẻ. | Trung bình–Cao | Tool-calling fidelity (Group A), schema compliance (Group B, đủ field), report/recommendation quality win-rate (Group C). |
| 3 | **Distillation-then-RL cho toxicology**: Gemini → QLoRA SFT → GRPO, so với chỉ-SFT và chỉ-GRPO. | Cao | Benchmark Tox21 (12 task, AUROC). |
| 4 | **Confidence calibration reward trong GRPO**. | Hẹp nhưng publishable | So calibration (ECE, Brier) có/không reward. **Lưu ý: reward hiện tại là proxy `1 - |conf - sim|`, KHÔNG phải ECE** — cần đo ECE riêng. |

<aside>
💡

Gợi ý khác biệt hóa mạnh hơn cho paper: tận dụng đặc điểm **MolRAG deterministic + LLM ghi đè** của repo để nghiên cứu "LLM-as-refiner over a deterministic evidence baseline" — so sánh chất lượng khi LLM ghi đè vs giữ deterministic. Đây là setup ít gặp và rất hợp với kiến trúc thực tế của bạn.

</aside>

---

## Checklist thực thi (theo thứ tự, đã hiệu chỉnh)

- [ ]  **Bug #1**: chuyển `import json` lên module-level trong `finetune_group_a.py`.
- [ ]  **Gap #2**: chốt số field (7 hay 11) và đồng bộ `REQUIRED_KEYS` + `_MOLRAG_RESPONSE_SCHEMA` + dữ liệu; đổi `json_schema_reward` sang `partial * 1.5`.
- [ ]  **Gap #3 (THẬT)**: nối dây `build_local_client()` vào `build_genai_client_candidates()` theo `LLM_RUNTIME`. (Bỏ qua "fix" `genai.types` trong bản nháp — không cần.)
- [ ]  **Bổ sung**: chạy GRPO từ trong `scripts/` (import tương đối); kiểm tra target Group C khớp Writer; sửa tên tool Group A thành `analyze_molecule`; đồng bộ token SMILES khi serve.
- [ ]  **DATA**: `group_a_tool_calling.json` (Tox21 + Gemini, tool đúng tên).
- [ ]  **DATA**: `group_b_stage1.json` (dùng `retrieve_similar_molecules` + `retrieve_knowledge_context` để sinh `contexts`).
- [ ]  **DATA**: `group_b_stage2.json` (full schema, distill Gemini).
- [ ]  **DATA**: `group_b_grpo.json` (`label_targets`, `max_similarities`).
- [ ]  **DATA**: `group_c_writer_preference.json` (chosen/rejected).
- [ ]  **TRAIN**: Group A → B SFT → B GRPO (giảm `num_generations`/`use_vllm` nếu 1 GPU) → C ORPO.
- [ ]  **SERVE**: vLLM + GGUF; set `LOCAL_LLM_*` env.
- [ ]  **DEPLOY**: sau khi đã nối dây Gap #3 → `LLM_RUNTIME=local`, test end-to-end (nhớ bật `molrag_enabled=True` nếu muốn dùng nhánh LLM của MolRAG).