Từ việc đọc toàn bộ cấu trúc repo [branch `agent_test`](https://github.com/NEU-Bio-Research-Team/tox-agent/tree/agent_test), tôi đã có đủ thông tin để phân tích kỹ theo từng tiêu chí **Độ hoàn thiện & ổn định sản phẩm (40%)** của rubric vòng 3.

***

## Phân tích tổng quan kiến trúc

Hệ thống **Tox-Agent** có kiến trúc đa tầng khá rõ ràng: 
- **`agents/`** — Multi-agent orchestration (orchestrator, screening, writer, report_chat, molrag_reasoner, evidence_qa) 
- **`backend/`** — ML core (GNN/AttentiveFP/GPS models, inference, pipelines, explainability) 
- **`services/`** — Infrastructure layer (Firestore, GenAI runtime, RAG retrievers, knowledge base) 
- **`frontend/`** — UI layer, deploy qua Firebase Hosting

Đây là một hệ thống tương đối phức tạp và có độ sâu kỹ thuật tốt. Tuy nhiên, qua phân tích cấu trúc, có **5 điểm yếu rõ ràng** mà BGK sẽ nhìn thấy ngay.

***

## Điểm yếu 1: Coverage test cực kỳ mỏng (Risk cao nhất)

Thư mục [`tests/`](https://github.com/NEU-Bio-Research-Team/tox-agent/tree/agent_test/tests) **chỉ có đúng 1 file**: `test_report_chat_agent.py` (9.3 KB).  Trong khi đó hệ thống có:

- `backend/inference.py` (53 KB) — core inference pipeline, **không có test**
- `backend/pipelines.py` (17 KB) — end-to-end pipeline, **không có test**
- `agents/orchestrator_agent.py` (11 KB) — điều phối toàn bộ luồng, **không có test**
- `agents/screening_agent.py`, `writer_agent.py` — **không có test**

**Hệ quả thực tế**: Trong buổi demo live, nếu một luồng chức năng bị lỗi (timeout của Gemini API, SMILES không hợp lệ, Firestore connection fail), hệ thống không có safety net nào để fallback gracefully. BGK kỹ thuật sẽ hỏi thẳng: *"Team có test gì để đảm bảo system stability không?"* — đây là điểm trừ trực tiếp vào tiêu chí **Hiệu năng, độ ổn định và an toàn vận hành (5%)**.

***

## Điểm yếu 2: Error handling và graceful degradation thiếu nhất quán

File `services/molrag_fallback_data.py`  tồn tại — chứng tỏ team đã nhận ra vấn đề fallback với MolRAG. Nhưng **fallback logic chưa được đồng bộ toàn hệ thống**:

- `ood_guard.py` (2.6 KB) trong backend  — OOD detection module rất nhỏ, chưa tích hợp đủ sâu vào inference pipeline chính
- File `output.json` (311 KB) và `temp_out.json` (1.2 MB) được commit thẳng lên branch  — cho thấy pipeline output chưa có structured logging/storage layer riêng biệt, dễ gây data contamination khi demo

**Rủi ro demo**: Nếu user nhập một SMILES lạ hoặc molecule nằm ngoài training domain, OOD guard không catch được → model output garbage → BGK mất tin tưởng vào độ tin cậy hệ thống.

***

## Điểm yếu 3: Luồng End-to-End phức tạp, dễ đứt giữa chừng

Orchestrator (`orchestrator_agent.py`, 11 KB) điều phối nhiều agent , nhưng flow hiện tại có **nhiều điểm có thể timeout hoặc deadlock**:

- `molrag_reasoner.py` (41 KB) là agent nặng nhất — gọi cả external RAG + Gemini GenAI 
- `writer_agent.py` (39 KB) + `report_chat_agent.py` (35 KB) đều rất nặng, không có cơ chế streaming trả về partial result 
- `genai_runtime.py` trong services  wrap Gemini API nhưng không rõ có retry với exponential backoff chưa

**Kịch bản fail thường gặp nhất trong demo**: Người dùng submit một molecule → screening agent gọi backend inference → molrag_reasoner gọi RAG + GenAI → writer_agent generate report → mỗi bước mất 5-15 giây → tổng flow có thể mất >60 giây trước khi UI phản hồi, gây ra cảm giác "hệ thống bị treo."

***

## Điểm yếu 4: Kiến trúc module bị chồng lấp (Scalability concern)

Có sự trùng lặp rõ ràng trong backend :

| File | Size | Chức năng trùng lắp |
|---|---|---|
| `graph_models.py` | 12 KB | GNN base |
| `graph_models_gin.py` | 10 KB | GIN variant |
| `graph_models_hybrid.py` | 23 KB | Hybrid GNN |
| `attentivefp_model.py` | 4 KB | AttentiveFP |
| `gps_model.py` | 4 KB | GPS |
| `pretrained_gnn.py` | 8 KB | Pretrained wrapper |
| `pretrained_mol_model.py` | 10 KB | Another pretrained wrapper |

Có **2 pretrained wrapper** riêng biệt và nhiều model file không được abstract qua một interface chung (`BaseModel`). Trong buổi demo nếu BGK hỏi *"Architecture của model pipeline trông như thế nào?"*, diagram sẽ rất khó giải thích rõ ràng vì không có factory pattern hay unified model registry.

***

## Điểm yếu 5: Không có health check / monitoring endpoint

Hệ thống deploy qua GCP (có `cloudbuild.tox-agent.yaml`, `deploy/` folder)  nhưng không có evidence của:
- `/health` endpoint trong `app.py`
- Rate limiting cho API calls đến Gemini
- Caching layer cho inference kết quả giống nhau (cùng một SMILES được query nhiều lần)

Tiêu chí **Hiệu năng, độ ổn định và an toàn vận hành** có trọng số 5% nhưng trong bối cảnh **phản biện vòng 2**, các đội khác hoàn toàn có thể hỏi: *"Nếu có 100 users đồng thời submit molecule, hệ thống xử lý như thế nào?"* — và hiện tại không có câu trả lời thuyết phục từ code.

***

## Ưu tiên cải thiện trước deadline 23h59 ngày 07/05

Xếp theo ROI cao nhất trong thời gian còn lại:

1. **Thêm error handling + loading state rõ ràng trên frontend** — tác động trực tiếp đến trải nghiệm demo live, dễ làm nhất
2. **Thêm 2-3 unit test cho inference pipeline** — chứng minh với BGK rằng team có test coverage, không cần test toàn bộ
3. **Xóa `output.json` và `temp_out.json` khỏi branch** — professional hygiene, BGK sẽ nhìn vào repo
4. **Thêm timeout + retry wrapper** trong `genai_runtime.py` — giảm risk demo bị treo
5. **Viết một sơ đồ kiến trúc đơn giản trên slide** giải thích rõ từng agent làm gì — bù đắp cho việc code phức tạp khó explain live