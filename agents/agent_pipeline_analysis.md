# ToxAgent Pipeline Analysis

## 1) Mục tiêu triển khai hiện tại

Pipeline hiện được triển khai theo hướng **deterministic orchestration** (chạy bằng hàm Python), không phụ thuộc hoàn toàn vào ADK runtime:

- Validate đầu vào SMILES và health của model server.
- Chạy song song 2 nhánh:
  - **Screening**: dự đoán độc tính lâm sàng/cơ chế từ model server.
  - **Research**: thu thập ngữ cảnh từ PubChem/PubMed.
- Có thêm một tầng **Evidence QA** để quality-gate literature evidence ở mức module riêng.
- Tổng hợp thành báo cáo chuẩn hóa (`final_report`) bằng logic rule-based.

Lưu ý quan trọng: trong code hiện tại, `evidence_qa_agent.py` đã tồn tại và được export, nhưng **chưa được nối vào `run_orchestrator_flow(...)` hoặc `SequentialAgent orchestrator`**. Nghĩa là pipeline chạy thực tế hiện vẫn là:

`validation -> parallel(screening + research) -> writer`

chứ chưa phải:

`validation -> parallel(screening + research) -> evidence_qa -> writer`

Kết luận ngắn: với backend ổn định, pipeline có thể cho ra output tốt cho screening ban đầu; đã có nền tảng để thêm quality gate cho evidence, nhưng hiện chưa phải mức production-robust cho workload lớn hoặc audit nghiêm ngặt.

---

## 2) Kiến trúc agent và vai trò từng tầng

### 2.1 `adk_compat.py`

- Cố gắng import `google.adk.agents`.
- Nếu không có ADK, fallback sang lớp local (`LlmAgent`, `ParallelAgent`, `SequentialAgent`) để giữ cấu trúc agent metadata.
- Ý nghĩa: pipeline vẫn chạy được bằng code Python ngay cả khi thiếu ADK runtime.

### 2.2 `orchestrator_agent.py` (tầng điều phối chính)

- `extract_smiles_from_text(user_text)`: tách candidate SMILES từ text tự do, validate lần lượt.
- `run_input_validation(smiles_input)`:
  - `check_model_server_health()`
  - `validate_smiles(smiles_input)`
  - Trả `VALID/INVALID` + lỗi + canonical SMILES.
- `run_orchestrator_flow(smiles_input, max_literature_results=5)`:
  - Gate validation trước.
  - Nếu invalid: trả `final_report` lỗi sớm.
  - Nếu valid: chạy song song `run_screening` và `run_research` bằng `ThreadPoolExecutor(max_workers=2)`.
  - Gọi `build_final_report(...)` để tổng hợp đầu ra cuối.
- `run_orchestrator_from_text(user_text, ...)`:
  - Parse text -> SMILES.
  - Nếu không parse được, trả report lỗi `smiles_not_found`.

### 2.3 `screening_agent.py` (nhánh phân tích model)

- `run_screening(smiles_input)`:
  - Validate SMILES.
  - Gọi `analyze_molecule(canonical_smiles)` đến model server (`/analyze`).
  - Chuẩn hóa kết quả về `screening_result`:
    - `clinical`, `mechanism`, `explanation`, `final_verdict`, `summary`.
  - Nếu lỗi: trả `screening_error` + payload thô (`analysis_raw`) khi có.

### 2.4 `researcher_agent.py` (nhánh nghiên cứu dữ liệu ngoài)

- `run_research(smiles_input, max_results=5)`:
  - Gọi `get_compound_info_pubchem(smiles)` để lấy CID + tên ưu tiên.
  - Gọi `search_toxicity_literature(preferred_name, max_results)`.
  - Nếu có CID: gọi thêm `get_pubchem_bioassay_data(cid)`.
  - Trả `research_result` gồm:
    - `compound_info`, `literature`, `bioassay_summary`, `query_name_used`.

### 2.5 `evidence_qa_agent.py` (tầng quality-gate evidence)

- `run_evidence_qa(research_result, top_k=5, high_relevance_threshold=0.55)`:
  - Không gọi LLM thực tế; là deterministic Python function để chuẩn hóa và kiểm tra chất lượng evidence.
  - Input chính là `research_result` từ `researcher_agent`.
  - Nếu thiếu `research_result`: trả `evidence_confidence = LOW`, cờ `research_result_missing`, và `evidence_qa_error`.
- Các bước xử lý chính:
  - Chuẩn hóa article fields:
    - `pmid`, `title`, `authors`, `year`, `journal`, `pubmed_url`.
  - Khử trùng lặp:
    - ưu tiên dedupe theo `pmid`,
    - fallback theo normalized `title`.
  - Trích xuất `compound_terms` từ:
    - `query_name_used`,
    - `compound_info.common_name`,
    - `compound_info.iupac_name`.
  - Chấm điểm từng bài bằng `_score_article(...)` theo heuristic:
    - có `pmid`, `title`, `journal`,
    - có toxicity terms trong title/journal,
    - có compound terms,
    - độ mới theo năm xuất bản.
  - Gán `relevance_score`, `relevance_level`, `qa_reasons`.
  - Sort theo score rồi giữ lại `top_k` bài tốt nhất.
  - Tính:
    - `high_relevance_count`,
    - `evidence_confidence` (`HIGH/MEDIUM/LOW`),
    - `research_quality_flags`.
- Output chính:
  - `research_result_sanitized`
  - `curated_articles`
  - `total_articles_in`
  - `total_articles_curated`
  - `high_relevance_count`
  - `evidence_confidence`
  - `research_quality_flags`
- Các quality flags có thể xuất hiện:
  - `compound_info_missing`
  - `literature_missing`
  - `literature_error_present`
  - `no_articles_found`
  - `duplicate_articles_removed:<n>`
  - `low_relevance_evidence`
- Ý nghĩa trong pipeline:
  - Đây là lớp kiểm soát chất lượng research evidence trước khi ghi report.
  - Giảm nguy cơ đưa toàn bộ literature thô hoặc trùng lặp vào report cuối.
  - Tạo confidence signal để downstream quyết định mức tin cậy evidence.
- Trạng thái tích hợp hiện tại:
  - `evidence_qa_agent` và `run_evidence_qa` đã được export trong `agents/__init__.py`.
  - Nhưng **chưa được gọi trong `orchestrator_agent.py`**.
  - Vì vậy writer hiện vẫn đang nhận trực tiếp `research_result` gốc, chưa nhận `research_result_sanitized`.

### 2.6 `writer_agent.py` (tầng tổng hợp báo cáo)

- `build_final_report(smiles_input, screening_result, research_result)`:
  - Nếu thiếu screening: trả report lỗi `screening_result_missing`.
  - Tính `risk_level` theo policy:
    - `CRITICAL`: `p_toxic > 0.8` và `assay_hits >= 3`
    - `HIGH`: `p_toxic > 0.6` hoặc `assay_hits >= 2`
    - `MODERATE`: `p_toxic > 0.4` hoặc `assay_hits >= 1`
    - `LOW`: còn lại
  - Build report cuối gồm:
    - `report_metadata`
    - `executive_summary`
    - `risk_level`
    - `sections`: clinical/mechanism/explanation/literature/recommendations

---

## 3) Luồng chạy chi tiết (step-by-step)

1. Nhận input SMILES trực tiếp hoặc text tự do.
2. Nếu là text tự do: extract token giống SMILES và validate từng candidate.
3. Chạy gate validation:
   - Health model server.
   - Validate SMILES bằng RDKit.
4. Nếu fail ở gate: dừng sớm, trả `final_report` với error.
5. Nếu pass:
   - Song song nhánh A (`run_screening`) và nhánh B (`run_research`).
6. Thu kết quả 2 nhánh vào `state`.
7. Hiện tại writer nhận trực tiếp `research_result` và tổng hợp thành `final_report`.
8. Nếu tích hợp `evidence_qa_agent` trong tương lai, bước trung gian hợp lý sẽ là:
   - `run_evidence_qa(research_result)`
   - truyền `research_result_sanitized` sang writer thay cho literature thô.
9. Trả output đầy đủ: validation + screening + research + final report.

---

## 4) Đánh giá tốc độ xử lý theo code (nhanh/chậm)

## 4.1 Input validation: **nhanh**

- `validate_smiles`: chạy local RDKit, thường rất nhanh.
- `check_model_server_health`: HTTP timeout mặc định `5s`.

## 4.2 Screening branch: **trung bình đến chậm**

- `analyze_molecule` gọi model server `/analyze`.
- Timeout mặc định `MODEL_SERVER_TIMEOUT = 30s`.
- Tốc độ phụ thuộc chủ yếu vào inference backend.

## 4.3 Research branch: **trung bình**

- Gồm nhiều external HTTP call:
  - PubChem (CID + property + synonyms): timeout `10s/call`.
  - PubMed search + summary: timeout `15s/call`.
  - PubChem bioassay: timeout `15s` khi có CID.
- Không có retry/backoff hiện tại.

## 4.4 Evidence QA branch: **nhanh**

- `run_evidence_qa` là pure Python:
  - normalize dữ liệu,
  - dedupe,
  - heuristic scoring,
  - sort + top-k.
- Không gọi network, không phụ thuộc model server.
- Chi phí thường nhỏ hơn nhiều so với screening/research; chủ yếu tỷ lệ theo số bài literature nhận vào.

## 4.5 Writer/report: **rất nhanh**

- Pure Python transform + rule logic, không gọi network.

## 4.6 Tổng thời gian end-to-end

Với flow đang chạy thực tế, do song song 2 nhánh chính, latency tổng gần đúng:

`T_total ~= T_validate + max(T_screening, T_research) + T_writer`

Nếu tích hợp thêm Evidence QA vào critical path, công thức gần đúng sẽ thành:

`T_total ~= T_validate + max(T_screening, T_research) + T_evidence_qa + T_writer`

Trong đó `T_evidence_qa` thường nhỏ, nên ít khi là bottleneck.

Nút thắt cổ chai thường là:

- model inference `/analyze`, hoặc
- external API PubMed/PubChem khi mạng chậm.

---

## 5) Chất lượng output: điểm mạnh và giới hạn

### Điểm mạnh

- Có gate kiểm tra đầu vào rõ ràng trước khi chạy nặng.
- Song song hóa screening/research để giảm thời gian tổng.
- Report schema cố định, dễ tích hợp downstream.
- Có cơ chế degrade khi một phần dữ liệu thiếu.
- Đã có sẵn module `evidence_qa_agent` để làm sạch và chấm chất lượng literature theo hướng deterministic.

### Giới hạn hiện tại

- `research_error` ở orchestration hầu như không được set meaningful (đa phần lỗi nằm trong field con của payload).
- Literature đang dùng title làm `abstract_snippet` (chưa lấy abstract thật), nên chiều sâu evidence còn hạn chế.
- Validate SMILES bị lặp (gate + screening), tăng nhẹ overhead.
- Chưa có retry/circuit-breaker cho external APIs.
- Nếu `screening_result` thiếu thì writer fail cứng, chưa có chế độ “partial report có cảnh báo”.
- `evidence_qa_agent` chưa được nối vào orchestrator nên lợi ích dedupe/scoring/confidence chưa đi vào `final_report`.
- `writer_agent` hiện chưa đọc các field như `evidence_confidence` hay `research_quality_flags`.

---

## 6) Kỳ vọng output có đạt “tốt” không?

Với mục tiêu **screening và tổng hợp nhanh**: đạt mức tốt.

Với mục tiêu **ra quyết định production hoặc regulatory-grade**: chưa đủ, dù đã có nền để thêm evidence quality gate. Cần tăng:

- độ tin cậy nguồn research (abstract/full-text metadata tốt hơn),
- quan sát lỗi rõ ràng hơn ở cấp pipeline,
- tích hợp `evidence_qa_agent` thật sự vào flow trước writer,
- kiểm soát timeout/retry cho network call,
- chiến lược fallback khi thiếu screening hoặc research.

---

## 7) Gợi ý cải tiến ưu tiên (theo tác động)

1. Chuẩn hóa propagation lỗi (`screening_error`, `research_error`) ở cấp orchestrator.
2. Tích hợp `run_evidence_qa(research_result)` vào `run_orchestrator_flow(...)` trước bước writer.
3. Cho `writer_agent` đọc `research_result_sanitized`, `evidence_confidence`, `research_quality_flags`.
4. Bổ sung retry + backoff cho PubChem/PubMed.
5. Loại validate SMILES lặp ở screening khi đã có canonical từ gate.
6. Nâng quality literature context (lấy abstract thực, score relevance giàu ngữ cảnh hơn).
7. Cho writer hỗ trợ chế độ partial report tốt hơn khi thiếu một nhánh.
