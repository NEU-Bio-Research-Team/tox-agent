# Fine-tuning Local LLM cho ToxAgent: SOTA Research & Harness Architecture (RTX 3090 24GB)

## 1. Executive Summary

Repo `tox-agent` hiện có **7 agents LLM** tất cả đều gọi Gemini API qua `google.genai`: `InputValidator`, `ScreeningAgent`, `ResearcherAgent`, `WriterAgent`, `MolRAG Reasoner`, `EvidenceQAAgent`, và `ReportChatAgent`. Mỗi agent có profile nhiệm vụ khác nhau — từ tool-calling JSON-structured (Validator, Screener) đến long-form scientific reasoning (MolRAG, Writer) đến conversational QA (EvidenceQA, ReportChat). Chiến lược tối ưu là **không fine-tune một model duy nhất cho tất cả**, mà phân nhóm agents theo đặc tính và áp dụng recipe fine-tuning phù hợp với từng nhóm. Với budget RTX 3090 24GB, QLoRA + Unsloth là bắt buộc cho tất cả fine-tuning runs.

***

## 2. Phân Tích Agents — Nhóm Theo Đặc Tính LLM

### 2.1 Agent Inventory từ Codebase

Dựa trên đọc source code trực tiếp từ repo:

| Agent | File | Env Var Model | Task Profile | Output Type |
|-------|------|--------------|-------------|-------------|
| **InputValidator** | `orchestrator_agent.py` | `AGENT_MODEL_FAST` | Tool calling: `validate_smiles()` → JSON output | Structured JSON |
| **ScreeningAgent** | `screening_agent.py` | `AGENT_MODEL_FAST` | Tool calling: `analyze_molecule()` → JSON | Structured JSON |
| **ResearcherAgent** | `researcher_agent.py` | (default fast) | Literature/PubMed retrieval reasoning | Semi-structured |
| **WriterAgent** | `writer_agent.py` | (default) | Long-form report synthesis từ evidence | Long-form text |
| **MolRAG Reasoner** | `molrag_reasoner.py` | `AGENT_MODEL_FAST` / `AGENT_MODEL_PRO` | Chemical analogy reasoning + JSON (`evidence_overview`, `mechanism_chain`, `confidence`) | Structured JSON |
| **EvidenceQAAgent** | `evidence_qa_agent.py` | (default) | Multi-turn Q&A trên evidence retrieved | Conversational |
| **ReportChatAgent** | `report_chat_agent.py` | (default) | Chat về báo cáo đã sinh | Conversational |

Tất cả models đều được inject qua `AGENT_MODEL_FAST`/`AGENT_MODEL_PRO` env vars, defaulting về `gemini-2.5-flash`/`gemini-2.5-pro`. Đây là điểm injection để swap sang local LLM.

### 2.2 Phân Nhóm Fine-tuning

**Nhóm A — Structured Tool Calling (InputValidator, ScreeningAgent, MolRAG Reasoner):** Output bắt buộc là strict JSON, agent phải gọi tool đúng schema. Fine-tuning cần tập trung vào JSON schema adherence + tool-call formatting. MolRAG Reasoner có schema phức tạp nhất với các fields như `mechanism_chain`, `key_substructures`, `confidence`.

**Nhóm B — Chemical Domain Reasoning (MolRAG Reasoner, ResearcherAgent):** Đây là nhóm domain-critical nhất. Model cần hiểu SMILES, cơ chế độc tính, SMARTS patterns, similarity reasoning, và trích xuất evidence từ literature. Fine-tuning domain-specific lên chemistry/toxicology corpus là cần thiết.[^1][^2]

**Nhóm C — Long-form Synthesis (WriterAgent):** Cần instruction-following tốt và khả năng tổng hợp multi-source evidence thành báo cáo có cấu trúc. Ít tool-calling nhưng cần coherent long-context reasoning.

**Nhóm D — Conversational QA (EvidenceQAAgent, ReportChatAgent):** Cần multi-turn coherence và khả năng trả lời từ context document. Đây là nhóm dễ fine-tune nhất và có thể chia sẻ một base model.

***

## 3. SOTA Fine-tuning Methods — Lựa Chọn Theo Task

### 3.1 SFT (Supervised Fine-tuning) với QLoRA — Baseline bắt buộc

QLoRA là phương pháp **không thể bỏ qua** với RTX 3090 24GB. Kỹ thuật này freeze toàn bộ model weights, chỉ train low-rank adapter matrices nhỏ. Với Unsloth, Llama-3.1-8B QLoRA 4-bit tiêu thụ khoảng 15GB VRAM, Qwen2.5-7B tương tự. Quy trình SFT với TRL/PEFT:[^3][^4][^5]

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=8192,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,           # LoRA rank; tăng lên 64 cho Nhóm B (chemistry)
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

Unsloth cung cấp tới 2x tốc độ training và giảm 70% memory so với vanilla HuggingFace. Model được export sang GGUF và serve qua Ollama hoặc vLLM sau khi train.[^4]

### 3.2 GRPO (Group Relative Policy Optimization) — Cho Nhóm B & MolRAG

GRPO là method được phát triển bởi DeepSeek team (dùng cho R1 model) và hiện đang được áp dụng rộng rãi cho fine-tuning LLM reasoning. GRPO đặc biệt phù hợp với **Nhóm B** (MolRAG Reasoner) vì:[^6][^7]

1. Không cần large labeled dataset — hoạt động với <100 examples nếu reward function được thiết kế tốt[^7]
2. Reward function viết bằng Python → có thể viết rule-based reward cho toxicity classification (binary label verification, JSON schema check, SMARTS hit validation)
3. Cải thiện multi-step reasoning — chính xác những gì MolRAG Reasoner cần cho `mechanism_chain` generation

Với GRPO + LoRA, Llama3.1-8B chỉ cần 15GB VRAM. Reward functions cho MolRAG có thể bao gồm:[^5]

```python
def toxicity_label_reward(completions, ground_truth_labels, **kwargs) -> list[float]:
    rewards = []
    for completion, gt_label in zip(completions, ground_truth_labels):
        try:
            parsed = json.loads(completion)
            predicted = parsed.get("suggested_label", "").upper()
            rewards.append(1.0 if predicted == gt_label.upper() else -0.5)
        except:
            rewards.append(-1.0)  # Penalize non-JSON
    return rewards

def json_schema_reward(completions, **kwargs) -> list[float]:
    required_keys = {"evidence_overview", "mechanism_chain", "suggested_label", "confidence"}
    rewards = []
    for comp in completions:
        try:
            parsed = json.loads(comp)
            coverage = len(required_keys & set(parsed.keys())) / len(required_keys)
            rewards.append(coverage)
        except:
            rewards.append(-1.0)
    return rewards
```

### 3.3 DPO/ORPO — Cho Nhóm C & D (Writer, QA)

Direct Preference Optimization (DPO) và ORPO (Odds Ratio Preference Optimization) phù hợp cho WriterAgent và conversational agents vì chúng không có verifiable outcomes rõ ràng như classification. ORPO đặc biệt efficient hơn DPO vì gộp SFT và preference learning trong một pass, giảm memory footprint. Dataset cần pairs của (báo cáo tốt, báo cáo kém) cho DPO.[^8]

### 3.4 Retriever-Aware Training (RAT) — Cho ResearcherAgent

Gorilla (Berkeley) đề xuất RAT: trong khi training, model được expose với retrieved documents thay vì chỉ static knowledge. Điều này cực kỳ phù hợp với `ResearcherAgent` vì agent này cần grounding vào retrieved PubMed abstracts. RAT giảm hallucination đáng kể khi model học cách dùng context động thay vì memorized facts.[^9][^10]

***

## 4. Domain-Specific Fine-tuning: Chemistry & Toxicology

### 4.1 Dataset SOTA cho Domain này

| Dataset/Model | Size | Tasks | Relevance cho ToxAgent |
|--------------|------|-------|----------------------|
| **SMolInstruct** (LlaSMol) | 3.3M samples, 1.6M molecules | 14 chemistry tasks incl. property prediction | Rất cao — bao gồm molecular property prediction, SMILES processing[^2][^11] |
| **ChemBERTa-2** pretrain data | 77M SMILES (PubChem) | Self-supervised SMILES MLM | Tốt cho tokenizer/embedding fine-tuning[^12] |
| **Tox21 + ToxCast** | ~12K compounds, 617 assays | Multi-label toxicity binary classification | Trực tiếp relevant — chính là task của ScreeningAgent |
| **SMILES-BERT tasks** | Downstream classification | Toxicity, bioactivity | Fine-tune cho binary SMILES classification[^13] |
| **OpenBioLLM training data** | 3K+ healthcare topics | Medical QA, clinical tasks | Tốt cho ResearcherAgent literature reasoning[^14][^15] |
| **PubMed Central (BioMistral)** | 65M articles continual pretraining | Biomedical domain | Background cho ResearcherAgent/WriterAgent[^16] |

**Key insight từ LlaSMol:** Fine-tuning Mistral-7B trên SMolInstruct cho kết quả 93.2% EM trên SMILES-to-Formula, vượt GPT-4 (4.8%) và Claude 3 Opus (9.2%). Đây là bằng chứng mạnh rằng **domain fine-tuning nhỏ beats large generalist API** cho chemistry tasks.[^11]

### 4.2 Custom Dataset Construction cho ToxAgent

Dataset construction theo AgentInstruct pipeline — dùng teacher model (Gemini hoặc GPT-4) để generate training data:[^17]

**Bước 1 — Mining từ repo hiện tại:**
- Extract tất cả `(input_smiles, retrieved_examples, baseline_prediction) → molrag_output` traces từ production logs
- Mỗi trace là một training sample hoàn chỉnh cho MolRAG Reasoner
- Lọc bằng confidence score (chỉ giữ samples với `confidence >= 0.7`)

**Bước 2 — Augmentation từ public datasets:**
- Sample từ TDC (Therapeutics Data Commons) toxicity benchmarks: ClinTox, SIDER, hERG
- Với mỗi SMILES, gọi teacher LLM để generate `mechanism_chain` explanation
- Tạo negative examples (sai label hoặc sai JSON) cho DPO/ORPO pairs

**Bước 3 — Tool-call traces cho Nhóm A:**
- Format theo Berkeley Function Calling Leaderboard (BFCL) schema[^18]
- Mỗi sample: `(system_prompt_with_tool_schema, user_message) → {"tool_call": "validate_smiles", "arguments": {"smiles": "..."}, "output": {...}}`

Curriculum-inspired approach từ EMNLP 2025 paper: train bằng structured reasoning templates trước, sau đó full fine-tuning. Cụ thể cho MolRAG: step 1 training trên `mechanism_chain` structure, step 2 training trên full JSON output.[^19][^20]

***

## 5. Base Model Selection — RTX 3090 24GB Budget

### 5.1 Model Comparison

| Model | Params | 4-bit VRAM | Tool Calling | Chemistry | Vietnamese | Recommendation |
|-------|--------|-----------|--------------|-----------|-----------|----------------|
| **Qwen2.5-7B-Instruct** | 7B | ~6GB | ✅ Native | Good | ✅ 29 langs | **Best overall** |
| **Llama-3.1-8B-Instruct** | 8B | ~7GB | ✅ (fine-tune needed) | Moderate | Limited | Good for Nhóm C/D |
| **Mistral-7B-Instruct-v0.3** | 7B | ~6GB | ✅ | **Best (LlaSMol)** | Limited | Best for chemistry Nhóm B |
| **BioMistral-7B** | 7B | ~6GB | ❌ | ✅ PubMed pretrain | Limited | Good base for Nhóm B/C [^16] |
| **Qwen2.5-3B-Instruct** | 3B | ~3GB | ✅ | Moderate | ✅ | Backup cho Nhóm A (nhẹ) |
| **OpenBioLLM-Llama3-8B** | 8B | ~7GB | ❌ | ✅ Medical | Limited | Good base for Nhóm C/D[^14] |

**Khuyến nghị theo nhóm:**
- **Nhóm A (Validator, Screener):** `Qwen2.5-7B-Instruct` — native JSON/tool calling, structured output tốt nhất[^21]
- **Nhóm B (MolRAG):** `Mistral-7B` base → fine-tune trên SMolInstruct subset + Tox21 — đây là combo được LlaSMol validate[^2][^11]
- **Nhóm C (Writer):** `Qwen2.5-7B-Instruct` hoặc `Llama-3.1-8B` — long-context, coherent generation
- **Nhóm D (QA Agents):** `Qwen2.5-3B-Instruct` — đủ nhẹ để serve nhiều sessions song song, Qwen native multilingual hỗ trợ Vietnamese[^21]

### 5.2 VRAM Budget Analysis

Với RTX 3090 24GB:
- **Inference serving:** Qwen2.5-7B-Instruct GGUF Q4_K_M ~ 4.8GB. Có thể serve 2 models song song (ví dụ Nhóm A + Nhóm D cùng lúc).
- **Fine-tuning:** QLoRA 4-bit rank-32 trên Qwen2.5-7B tiêu thụ ~10-12GB, còn 12GB cho batch size và gradients. GRPO thêm rollout buffer, nên batch size phải nhỏ (2-4).
- **Multi-model fine-tuning approach:** Fine-tune từng model một, export GGUF xong rồi chuyển sang model tiếp theo. Không cần serve và train đồng thời.

***

## 6. Local LLM Harness Infrastructure

### 6.1 Serving: vLLM vs Ollama cho ToxAgent

Vì ToxAgent đang dùng `google.genai` client với structured output schema (`response_schema` param trong GenAI SDK), việc chuyển sang local cần một server **OpenAI-compatible** với structured output support.[^22]

| Feature | vLLM | Ollama |
|---------|------|--------|
| OpenAI `/v1/chat/completions` | ✅ | ✅ |
| Structured JSON schema (`response_format`) | ✅ xgrammar backend[^23] | Limited |
| Tool calling schema | ✅ hermes parser[^23] | Limited |
| Throughput | **20x cao hơn**[^24] | Thấp |
| Setup complexity | Medium | Very easy |
| Concurrent requests | ✅ PagedAttention | Single-user |

**Kết luận:** Dùng **vLLM** cho production serving của ToxAgent. Lý do: MolRAG Reasoner và ScreeningAgent đòi hỏi strict JSON schema (`_MOLRAG_RESPONSE_SCHEMA` trong code), vLLM với xgrammar backend đảm bảo constrained decoding. Ollama phù hợp cho development/testing.[^23][^22]

```bash
# Serve Qwen2.5-7B với structured output + tool calling
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --structured-outputs-config.backend xgrammar \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000 \
  --max-model-len 32768
```

### 6.2 Thay thế Gemini API trong codebase

Hiện tại `genai_runtime.py` build Gemini clients. Cần tạo một `local_llm_runtime.py` adapter:

```python
# services/local_llm_runtime.py
from openai import OpenAI

_LOCAL_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")

def build_local_client():
    return OpenAI(base_url=_LOCAL_BASE_URL, api_key="local")

def call_local_with_schema(
    prompt: str,
    response_schema: dict,
    model: str = "Qwen2.5-7B-Instruct",
) -> dict:
    client = build_local_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": response_schema, "strict": True}
        }
    )
    return json.loads(response.choices.message.content)
```

### 6.3 Constrained Decoding để đảm bảo JSON validity

MolRAG Reasoner hiện dùng `_safe_json_parse()` với JSON repair logic phức tạp vì Gemini đôi khi trả về malformed JSON. Với vLLM xgrammar, **constrained decoding** đảm bảo output luôn valid JSON matching `_MOLRAG_RESPONSE_SCHEMA` — loại bỏ hoàn toàn nhu cầu repair logic này. Đây là lợi thế kỹ thuật quan trọng của local deployment.

### 6.4 RAG Harness: MAIN-RAG cho ResearcherAgent

MAIN-RAG (EMNLP/ACL 2025) đề xuất dùng multiple LLM agents để filter và score retrieved documents, với adaptive threshold dựa trên score distribution. Cải thiện 2-11% accuracy trên QA benchmarks. Cho ResearcherAgent, có thể implement lightweight MAIN-RAG với 2 local Qwen2.5-3B judges để filter PubMed retrieval results trước khi đưa vào main reasoner.[^25]

### 6.5 Chemical-Specific Tokenizer Augmentation

Paper từ ScienceDirect 2024 chỉ ra rằng standard tokenization không capture hết SMILES grammar. Nên thêm custom SMILES tokens vào tokenizer trước khi fine-tuning Nhóm B:[^26]

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# Add chemistry-specific tokens
smiles_tokens = ["[NH]", "[NH2]", "[NH3+]", "[OH]", "[CH2]", "C#N", "C=O", "S=O", "P=O"]
tokenizer.add_tokens(smiles_tokens)
# Resize embedding matrix accordingly
model.resize_token_embeddings(len(tokenizer))
```

***

## 7. Fine-tuning Roadmap Cụ Thể cho ToxAgent

### 7.1 Phase 1: Nhóm A — Tool Calling Agents (2-3 ngày)

**Target:** InputValidator, ScreeningAgent
**Method:** SFT + QLoRA
**Base model:** Qwen2.5-7B-Instruct (native JSON/tool calling)[^21]
**Dataset:**
- Giai đoạn 1: Dùng Gorilla APIBench tool-calling format[^27][^9]
- Giai đoạn 2: Custom dataset từ validate_smiles + analyze_molecule call traces
- Size: 500-2000 examples là đủ cho task đơn giản này[^28]

**Training config:**
```python
SFTConfig(
    max_seq_length=4096,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_8bit",
)
```

**Evaluation:** BFCL v2 benchmark và custom test với ToxAgent SMILES inputs. Target: >90% valid JSON, >95% correct tool selection.[^19]

### 7.2 Phase 2: Nhóm B — MolRAG Reasoner (1-2 tuần)

**Target:** MolRAG Reasoner (file lớn nhất, logic phức tạp nhất)
**Method:** SFT QLoRA → GRPO fine-tuning
**Base model:** Mistral-7B (best chemistry base per LlaSMol) hoặc BioMistral-7B[^16][^2]

**Dataset construction:**
1. Download SMolInstruct từ HuggingFace (`osunlp/SMolInstruct`) — lọc các tasks liên quan: property prediction, toxicity classification[^29]
2. Download Tox21 dataset, format thành instruction pairs
3. Augment với teacher-generated `mechanism_chain` explanations
4. Total target: ~10K samples

**GRPO reward functions:**
- `toxicity_label_reward`: binary correct/incorrect label → reward ±1.0
- `json_schema_reward`: schema coverage → reward 0.0-1.0
- `mechanism_chain_quality`: chain length và keyword presence (SMARTS, assay names) → heuristic reward
- `confidence_calibration`: penalize extreme confidence on low-similarity cases

**Evaluation:** So sánh với baseline Gemini flash trên Tox21 test set và custom ToxAgent benchmarks.

### 7.3 Phase 3: Nhóm C & D — Writer + QA Agents (3-5 ngày)

**Target:** WriterAgent, EvidenceQAAgent, ReportChatAgent
**Method:** SFT + ORPO (writer), SFT (QA)
**Base model:** Qwen2.5-7B-Instruct (multilingual, long-context)[^21]

**Dataset:**
- WriterAgent: (screening_result + research_result) → expert_report pairs. Có thể generate với teacher LLM từ real ToxAgent outputs.
- EvidenceQA/ReportChat: Multi-turn conversation datasets trên biomedical QA (BioASQ, MedQuAD) + custom toxicology QA pairs.

**Key challenge:** WriterAgent cần maintain long coherent context (~8K tokens). Dùng Unsloth's extended context support (342K context cho Llama 3.1 8B).[^4]

### 7.4 Evaluation Framework

Dùng AgentBench framework để đánh giá toàn bộ pipeline sau khi fine-tune. AgentBench đánh giá LLM agents trên POMDP framework gồm reasoning, decision-making, và instruction following. Phát hiện từ AgentBench: "Improving instruction following and training on high quality multi-round alignment data could improve agent performance" — trực tiếp relevant với tất cả agents trong ToxAgent.[^30][^31]

Custom eval metrics cho ToxAgent cụ thể:
- **End-to-end accuracy:** SMILES in → correct toxicity verdict out
- **JSON validity rate:** % responses passing `_safe_json_parse()`
- **Tool call accuracy:** % correct tool selections và arguments
- **Latency:** Time-to-first-token và total generation time

***

## 8. Self-Play & Multiagent Fine-tuning (Phương Pháp Nâng Cao)

### 8.1 Multiagent Self-Improvement (ICLR 2025)

Paper ICLR 2025 về "Multiagent Fine-tuning" đề xuất fine-tune mỗi agent trên data generated qua multiagent interactions. Applied vào ToxAgent: chạy pipeline với teacher model (Gemini) để generate trajectories, rồi fine-tune từng local agent trên sub-trajectories liên quan đến nó. Ưu điểm là **specialization và diversification** tự nhiên — mỗi agent học từ những interactions mà nó tham gia trực tiếp.[^32]

### 8.2 SPA (Self-Play Agentic) Framework (ICLR 2026)

SPA kết hợp self-play SFT (học world model từ environment interactions) với RL-based policy optimization. Áp dụng cho MolRAG Reasoner: "environment" là chemical space retrieval + SMARTS matching, "world model" là dự đoán outcome của reasoning path. SPA boost performance từ 25.6% → 59.8% trên Sokoban với Qwen2.5-1.5B — cho thấy tiềm năng lớn với reasoning agents nhỏ.[^33]

### 8.3 Agent Data Protocol (ADP) — Unifying Training Datasets

ADP (arXiv 2025) là "interlingua" giữa các agent datasets dạng khác nhau, cho phép train unified agent pipeline từ diverse sources. Với ToxAgent có 7 agents, ADP giúp chuẩn hóa format từ APIBench (Nhóm A) + SMolInstruct (Nhóm B) + BioASQ (Nhóm D) thành một training pipeline thống nhất.[^34]

***

## 9. Failure Modes & Mitigation

### 9.1 Tool Calling Degradation sau Fine-tuning

Một vấn đề được ghi nhận: fine-tuning có thể làm model quên tool-calling patterns nếu training data không include tool call examples. **Mitigation:** Always include 10-20% general tool-calling examples (từ BFCL dataset) trong mọi fine-tuning run, kể cả khi train cho domain-specific task. Đây là catastrophic forgetting prevention.[^35]

### 9.2 SMILES Tokenization Issues

Standard BPE tokenizer có thể tokenize SMILES không optimal, tách sai bonds. **Mitigation:** Test với canonical SMILES trước và sau fine-tuning. Nếu có regression, add SMILES-specific tokens vào vocabulary và retrain embedding layer.[^26]

### 9.3 JSON Schema Hallucination

Small local models (7B) có thể generate JSON với keys không đúng schema, đặc biệt với nested schemas như `_MOLRAG_RESPONSE_SCHEMA`. **Mitigation chính:** vLLM xgrammar constrained decoding. **Mitigation phụ:** Schema-aware loss masking trong training — only compute loss trên keys của schema, không phải arbitrary text.

### 9.4 Chemical Hallucination

Augmented LLM Prompts paper (J. Chem. Inf. Model. 2025) chỉ ra RAG + ML-optimized prompts giảm chemical hallucination từ 62.34 RMSE xuống 11.76 RMSE. **Mitigation:** Với ResearcherAgent và MolRAG, always pipe retrieved chemical evidence vào context; không để model "remember" chemistry facts từ pretraining.[^36]

***

## 10. Recommended Implementation Stack

```
Fine-tuning:
  - Unsloth + TRL (SFT, DPO, ORPO, GRPO)
  - PEFT (LoRA/QLoRA)
  - HuggingFace Datasets (SMolInstruct, Tox21 formatted)
  - Weights & Biases (tracking)

Serving:
  - vLLM (production) với xgrammar + hermes tool parser
  - Ollama (dev/testing)
  - OpenAI-compatible Python client (drop-in for google.genai)

Domain Data:
  - SMolInstruct (osunlp/SMolInstruct on HuggingFace)
  - TDC Toxicity benchmarks (ClinTox, hERG, SIDER)
  - BioMistral PubMed pretraining data
  - Tox21 SMILES datasets

Evaluation:
  - AgentBench (agentic pipeline eval)
  - BFCL v2 (function calling)
  - Custom ToxAgent end-to-end benchmark
  - Tox21 test split (domain accuracy)
```

***

## 11. Kết Luận & Priority

**Độ ưu tiên thực hiện:**

1. **Ngay lập tức:** Setup vLLM server + `local_llm_runtime.py` adapter để swap Gemini API. Test với Qwen2.5-7B-Instruct chưa fine-tune — đây là baseline local model.
2. **Week 1:** Fine-tune Nhóm A (Validator, Screener) với SFT QLoRA. Dataset nhỏ, kết quả nhanh.
3. **Week 2-3:** Fine-tune Nhóm B (MolRAG) với SFT → GRPO. Đây là agent quan trọng nhất và phức tạp nhất.
4. **Week 4:** Fine-tune Nhóm C/D (Writer, QA). Ít urgent hơn vì không cần structured output.
5. **Ongoing:** Monitor với AgentBench + custom ToxAgent eval; iterate với GRPO reward shaping.

Điểm mấu chốt: Nghiên cứu từ LlaSMol và SLM tool-calling paper (AAAI 2026) đều xác nhận **targeted fine-tuning của small models beats API-accessed large models** cho domain-specific tasks. ToxAgent với chemistry + toxicology domain hoàn toàn phù hợp với pattern này.[^37][^28][^11]

---

## References

1. [SmileyLlama: Modifying Large Language Models for Directed ... - arXiv](https://arxiv.org/html/2409.02231v2) - Here we show that a Large Language Model (LLM) can serve as a foundation model for a Chemical Langua...

2. [LlaSMol: Advancing Large Language Models for Chemistry with a...](https://openreview.net/forum?id=lY6XTF9tPv) - This paper proposes SMolInstruct, a large-scale instruction-tuning dataset designed with over 3 mill...

3. [Fine-Tuning Large Language Models for Function Calling with LoRA](https://pub.aimind.so/fine-tuning-large-language-models-for-function-calling-with-lora-d26f22910043) - In this blog post, we'll explore how to fine-tune large language models (LLMs) for function calling ...

4. [unsloth 2025.1.8 - PyPI](https://pypi.org/project/unsloth/2025.1.8/) - We tested Llama 3.1 (8B) Instruct and did 4bit QLoRA on all linear layers (Q, K, V, O, gate, up and ...

5. [7GB VRAM is all you need to train your own reasoning model ...](https://www.facebook.com/0xSojalSec/posts/7gb-vram-is-all-you-need-to-train-your-own-reasoning-model-unsloth-made-some-gre/1360371072283959/) - Unsloth made some great points: - GRPO is now optimized to use 80% less VRAM - Qwen2.5(1.5B) can be ...

6. [Learn Reinforcement Fine-Tuning with GRPO for LLMs - LinkedIn](https://www.linkedin.com/posts/andrewyng_new-course-reinforcement-fine-tuning-llms-activity-7330979772581269506-RQ3b) - Learn to use reinforcement learning to improve your LLM performance in this short course, built in c...

7. [Reinforcement Fine-Tuning LLMs with GRPO - DeepLearning.AI](https://www.deeplearning.ai/alpha/short-courses/reinforcement-fine-tuning-llms-grpo) - Using RFT to adapt small, open-source models can lead to competitive performance on reasoning tasks,...

8. [LLM Fine‑Tuning in 2025: A Hands‑On, Test‑Driven Blueprint](https://medium.com/@tabers77/llm-fine-tuning-in-2025-a-hands-on-test-driven-blueprint-dd1c7887bb99) - TL;DR: Most posts cover LoRA/QLoRA quickstarts. This article goes beyond: a practical decision tree ...

9. [Introduction to Gorilla LLM](https://gorilla.cs.berkeley.edu/blogs/1_gorilla_intro.html)

10. [Large Language Model Connected with Massive APIs](https://openreview.net/forum?id=tBRNC6YemY) - Large Language Models (LLMs) have seen an impressive wave of advances, with models now excelling in ...

11. [LlaSMol](https://osu-nlp-group.github.io/LLM4Chem/) - LlaSMol is fine-tuned on SMolInstruct, an instruction dataset of 14 meticulously selected tasks. Lla...

12. [No-Code Fine-tuning of Chemical Foundation Models with Prithvi](https://deepforestsci.com/blog/8) - The pretraining improvements in ChemBERTa-2 shows that scaling pre-training datasets can significant...

13. [LLM Finetuning w/ SMILES-BERT - Stephen Z. Lu](https://thematrixmaster.github.io/blog/2023/finetuning-llm/) - I want to explore the alternative method of improving llm performance through finetuning on a downst...

14. [Saama's Medical-Domain LLMs Release](https://www.saama.com/openbiollm-llama3-saama-medical-llms/) - Introducing OpenBioLLM-Llama3-70B & 8B: Saama's AI Research Lab Released the Most Openly Available M...

15. [Researchers Introduce OpenBioLLM-Llama3-70B & 8B - LinkedIn](https://www.linkedin.com/pulse/researchers-introduce-openbiollm-llama3-70b-8b-llms-ai-alchemist-jzj2c) - These new open-source LLMs set the bar for medical language models, outperforming commercial giants ...

16. [cniongolo/biomistral - Ollama](https://ollama.com/cniongolo/biomistral) - We introduce BioMistral, an open-source LLM tailored for the biomedical domain, utilizing Mistral as...

17. [AgentInstruct, a Framework for Generating Diverse Synthetic Data ...](https://www.deeplearning.ai/the-batch/researchers-increasingly-fine-tune-models-on-synthetic-data-but-generated-datasets-may-not-be-sufficiently-diverse-new-work-used-agentic-workflows-to-produce-diverse-synthetic-datasets) - Researchers increasingly fine-tune models on synthetic data, but generated datasets may not be suffi...

18. [Gorilla](https://gorilla.cs.berkeley.edu)

19. [Improving Large Language Models Function Calling and ... - arXiv](https://arxiv.org/html/2509.18076v1) - To address this, we introduce a curriculum-inspired framework that leverages structured reasoning te...

20. [[PDF] Improving Large Language Models Function Calling and ...](https://aclanthology.org/2025.emnlp-main.1242.pdf) - To address this, we introduce a curriculum-inspired framework that lever- ages structured reasoning ...

21. [qwen2.5:7b-instruct - Ollama](https://ollama.com/library/qwen2.5:7b-instruct) - Qwen2.5 models are pretrained on Alibaba's latest large-scale dataset, encompassing up to 18 trillio...

22. [OpenAI-Compatible Server - vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)

23. [vllm/examples/tool_calling/openai_responses_client_with_tools.py ...](https://github.com/vllm-project/vllm/blob/main/examples/tool_calling/openai_responses_client_with_tools.py) - A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm

24. [Ollama vs vLLM: Choosing the Right LLM Framework for 2025](https://www.linkedin.com/posts/om-umarvaishya_ai-llm-deeplearning-activity-7367215410724704256-nzRA) - Ollama vs vLLM: Choosing the Right Local LLM Framework in 2025 With the rise of local and edge large...

25. [MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](https://aclanthology.org/2025.acl-long.131/) - Retrieval-Augmented Generation (RAG) addresses this issue by incorporating external, real-time infor...

26. [A novel approach to unlocking the synergy of large language ...](https://www.sciencedirect.com/science/article/abs/pii/S1746809424014460) - This work explores the potential of using the pre-trained large language model Llama2 to address cha...

27. [Gorilla APIBench benchmark | Tool use agent evaluation | Steel.dev](https://leaderboard.steel.dev/registry/benchmarks/gorilla-apibench/) - Gorilla APIBench is a public tool use benchmark. Compare its evaluation method, task count, top mode...

28. [Paper page - Small Language Models for Efficient Agentic Tool Calling](https://huggingface.co/papers/2512.15943) - Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted F...

29. [GitHub - OSU-NLP-Group/LLM4Chem: Official code repo for the ...](https://github.com/osu-nlp-group/llm4chem) - Official code repo for the paper "LlaSMol: Advancing Large Language Models for Chemistry with a Larg...

30. [AgentBench: Evaluating LLMs as Agents - alphaXiv](https://www.alphaxiv.org/overview/2308.03688v3) - View recent discussion. Abstract: The potential of Large Language Model (LLM) as agents has been wid...

31. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) - The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Thus, t...

32. [Published as a conference paper at ICLR 2025](https://openreview.net/pdf?id=JtGPIZpOrz)

33. [Internalizing World Models via Self-Play Finetuning for Agentic RL](https://openreview.net/forum?id=K8wCGMzeuY) - This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-...

34. [Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents](https://arxiv.org/html/2510.24702v1) - To this end, we introduce the agent data protocol (ADP), a light-weight representation language that...

35. [Is there a way to retain tool calling ability after LLM fine-tuning?](https://www.reddit.com/r/AgentsOfAI/comments/1nm3p0o/is_there_a_way_to_retain_tool_calling_ability/) - One is to include tool call examples in your fine tuning data so the model continues to see the patt...

36. [Augmented and Programmatically Optimized LLM Prompts Reduce ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12076503/) - LLMs are opening new possibilities for leveraging natural language processing in chemistry and other...

37. [Small Language Models for Efficient Agentic Tool Calling - arXiv](https://arxiv.org/abs/2512.15943) - As organizations scale adoption of generative AI, model cost optimization and operational efficiency...

