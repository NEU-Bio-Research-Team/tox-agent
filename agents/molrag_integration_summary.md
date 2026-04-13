# Tóm tắt phần MolRAG prototype đã thêm

## 1. Mục tiêu của bản prototype

Bản code mới được thêm theo hướng MVP trong file `agents/MoIRAG`:

- không viết lại toàn bộ pipeline hiện tại
- không thay baseline model bằng LLM
- không deploy
- chỉ bổ sung lớp:
  - retrieval các phân tử tương tự
  - reasoning kiểu MolRAG
  - evidence thêm vào report cuối

Hướng này tương ứng với **Option A**:

- baseline model vẫn là nguồn quyết định chính
- MolRAG đóng vai trò giải thích và bổ sung bằng chứng

---

## 2. Đã thêm file mới ở đâu

### 2.1. `services/fingerprint_service.py`

Vai trò:

- canonicalize SMILES
- tạo Morgan fingerprint bằng RDKit
- tính độ tương đồng Tanimoto

Hàm chính:

- `canonicalize_smiles(...)`
- `fingerprint_from_smiles(...)`
- `tanimoto_similarity(...)`

Đây là lớp nền cho retrieval.

---

### 2.2. `services/molecule_retriever.py`

Vai trò:

- đọc database prototype từ:
  - `test_data/reference_panel.csv`
  - `test_data/screening_library.csv`
- tạo fingerprint cho các phân tử trong database
- tìm top-k phân tử tương tự với input

Hàm chính:

- `retrieve_similar_molecules(...)`

Đầu ra gồm:

- `matches`
- `similarity`
- `label`
- `source`
- `notes`

Đây là phần retrieval chính của MolRAG.

---

### 2.3. `services/prompt_builder.py`

Vai trò:

- tạo prompt theo kiểu MolRAG / Sim-CoT
- ghép:
  - input smiles
  - baseline prediction
  - retrieved examples

Hàm chính:

- `build_molrag_prompt(...)`

Hiện tại prompt này được tạo ra để sẵn sàng cho bước LLM reasoning sau này.

---

### 2.4. `services/result_fusion.py`

Vai trò:

- đóng gói kết quả baseline và MolRAG vào một object chung
- đánh dấu mức độ đồng thuận giữa 2 nguồn

Hàm chính:

- `fuse_molrag_with_baseline(...)`

Trạng thái hiện tại:

- đang ở mode `evidence_only`
- baseline vẫn là kết quả cuối
- MolRAG chưa override prediction

---

### 2.5. `services/__init__.py`

Vai trò:

- export các service mới để import gọn hơn

---

### 2.6. `agents/molrag_reasoner.py`

Vai trò:

- là layer reasoning mới của MolRAG
- nhận:
  - input smiles
  - baseline prediction
  - danh sách analog retrieved
- sinh:
  - `evidence_summary`
  - `reasoning_summary`
  - `suggested_label`
  - `confidence`

Hàm chính:

- `_deterministic_reasoning(...)`
- `run_molrag_reasoning(...)`

Trạng thái hiện tại:

- đây là prototype deterministic
- có tạo `prompt_preview`
- chưa gọi LLM thật làm đường chạy mặc định

Lý do:

- để chạy local an toàn
- để giữ pipeline ổn định
- để dễ debug hơn ở giai đoạn đầu

---

## 3. Đã sửa file cũ ở đâu

### 3.1. `agents/screening_agent.py`

Đây là nơi được chọn để **gắn MolRAG chính**.

Đã thêm vào `run_screening(...)`:

- `molrag_enabled`
- `molrag_top_k`
- `molrag_min_similarity`

Khi `molrag_enabled=True`, flow mới trong `ScreeningAgent` là:

1. validate input như cũ
2. chạy `analyze_molecule(...)` lấy baseline prediction
3. tạo `baseline_prediction`
4. gọi `retrieve_similar_molecules(...)`
5. gọi `run_molrag_reasoning(...)`
6. gọi `fuse_molrag_with_baseline(...)`
7. gán kết quả vào `screening_result`

Đã thêm 2 block mới trong `screening_result`:

- `molrag`
- `fusion_result`

Nghĩa là `screening_result` bây giờ không chỉ có:

- `clinical`
- `mechanism`
- `explanation`

mà còn có thêm:

- evidence từ analog molecules
- reasoning của MolRAG
- trạng thái fusion với baseline

---

### 3.2. `agents/orchestrator_agent.py`

Vai trò sửa:

- cho phép bật / tắt MolRAG từ cấp orchestrator

Đã thêm tham số vào:

- `run_orchestrator_flow(...)`
- `run_orchestrator_from_text(...)`

Thông tin mới được truyền xuống `ScreeningAgent`:

- `molrag_enabled`
- `molrag_top_k`
- `molrag_min_similarity`

Ý nghĩa:

- `OrchestratorAgent` giờ có thể điều phối baseline-only hoặc baseline + MolRAG
- đúng hướng đề xuất trong file `MoIRAG`

---

### 3.3. `agents/writer_agent.py`

Đã mở rộng report cuối để hiển thị phần MolRAG.

Trong `sections`, đã thêm:

- `molrag_evidence`
- `fusion_result`

Ý nghĩa:

- report cuối có thể hiển thị top analog molecules
- có reasoning summary
- có suggested label của MolRAG
- có decision note cho biết baseline vẫn là source of truth

---

## 4. Luồng dữ liệu mới sau khi thêm MolRAG

Flow hiện tại sau khi sửa:

`SMILES -> Validation -> Screening(baseline + MolRAG) -> Researcher -> Writer -> Final report`

Chi tiết:

1. User gửi SMILES
2. `InputValidator` kiểm tra input
3. `ScreeningAgent` chạy baseline prediction
4. Nếu bật MolRAG:
   - retrieval top-k analogs
   - tạo prompt MolRAG
   - chạy deterministic reasoning
   - tạo fusion result
5. `ResearcherAgent` vẫn chạy như cũ cho external knowledge
6. `WriterAgent` tổng hợp tất cả vào report cuối

---

## 5. Output mới được thêm vào hệ thống

Trong `screening_result`, đã có thêm object:

```json
{
  "baseline_prediction": {
    "label": "...",
    "score": 0.81,
    "confidence": 0.76,
    "ood_flag": false
  },
  "molrag": {
    "enabled": true,
    "strategy": "sim_cot",
    "retrieved_examples": [],
    "evidence_summary": "...",
    "reasoning_summary": "...",
    "suggested_label": "...",
    "confidence": 0.72
  },
  "fusion_result": {
    "mode": "evidence_only",
    "baseline_label": "...",
    "molrag_label": "...",
    "final_label": "...",
    "decision_note": "Baseline model remains the source of truth in MVP mode."
  }
}
```

Ý nghĩa:

- `baseline_prediction`: kết quả model hiện tại
- `molrag`: bằng chứng và reasoning từ MolRAG
- `fusion_result`: cách ghép 2 nguồn

---

## 6. Phần nào đang là prototype, phần nào đã sẵn sàng

### Đã sẵn sàng về mặt code

- có retrieval service
- có fingerprint service
- có reasoning layer riêng
- có fusion layer
- có tích hợp vào `ScreeningAgent`
- có tích hợp vào `OrchestratorAgent`
- có đưa vào `WriterAgent`

### Mới ở mức prototype

- retrieval database đang là DB nhỏ, đọc từ file CSV trong `test_data`
- reasoner đang là deterministic, chưa dùng LLM thật làm đường chạy mặc định
- fusion đang ở chế độ `evidence_only`
- chưa có UI riêng cho panel MolRAG
- chưa benchmark retrieval quality

---

## 7. Điểm quan trọng nhất cần nhớ

- Mình không thay pipeline cũ.
- Mình không đưa MolRAG vào `ResearcherAgent`.
- Mình gắn MolRAG vào `ScreeningAgent`, đúng theo đề xuất trong `MoIRAG`.
- `OrchestratorAgent` chỉ đóng vai trò bật/tắt và truyền config.
- `WriterAgent` chỉ hiển thị và tổng hợp kết quả.
- Bản này là bước đầu để chứng minh MolRAG có thể chèn vào repo hiện tại mà không phá vỡ kiến trúc cũ.

---

## 8. Trạng thái kiểm tra hiện tại

Đã xác nhận:

- các file mới và file sửa đều parse được
- logic đã nối vào pipeline đúng hướng

Chưa chạy demo end-to-end thành công trong environment hiện tại vì:

- Python env đang thiếu `rdkit`

Nghĩa là:

- code integration đã có
- môi trường local hiện tại chưa đủ dependency để chạy thử thật

---

## 9. Nếu muốn đi tiếp, bước hợp lý nhất

Có 3 hướng tiếp theo để làm:

1. Thêm script demo local cho MolRAG
2. Hiển thị `molrag_evidence` đẹp hơn trong UI/report
3. Nâng từ deterministic reasoning lên LLM reasoning thật

Hướng hợp lý nhất cho bước tiếp theo:

- viết script demo local trước
- sau đó mới nâng UI hoặc LLM
