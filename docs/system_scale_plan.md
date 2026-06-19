# Kế hoạch scale hệ thống ToxAgent (2026-05-28)

## 1) Bằng chứng kỹ thuật từ repo
- Backend triển khai Cloud Run service tox-agent-cpu, region asia-southeast1 (firebase.json).
- FastAPI chạy uvicorn 1 worker (model_server/scripts/entrypoint.sh).
- Inference bị serialize bởi model lock: async with _model_lock() trong /predict và /explain, và with _model_lock_sync() trong /analyze (model_server/main.py).
- Orchestrator chạy song song 2 nhánh screening + research (ThreadPoolExecutor max_workers=2) với công thức latency: T_total ~= T_validate + max(T_screening, T_research) + T_writer (agents/orchestrator_agent.py, agents/agent_pipeline_analysis.md).
- Timeout hiện tại: MODEL_SERVER_TIMEOUT 30s, PubChem 10s/call, PubMed 15s/call (agents/agent_pipeline_analysis.md).
- Kích thước model artifacts local hiện có tổng 33.2 MB; lớn nhất tox21_gatv2_model ~14.2 MB (đo trực tiếp thư mục models/).
- Dockerfile hỗ trợ CPU/GPU qua build arg TORCH_VARIANT=cpu|cu121 (model_server/Dockerfile).

## 2) Mô hình sức chứa cho 100 request đồng thời
Định nghĩa:
- S = latency trung bình của 1 request /agent/analyze.
- Theo ghi chú nội bộ, E2E latency ~3–6s (temp_outline_extracted.txt).
- Do inference bị khóa theo model lock, concurrency hiệu dụng trên 1 instance ~1.

Công thức xếp hàng đồng thời (burst đến cùng lúc):
- Với N instance và 100 request đến cùng lúc, mỗi instance xử lý k ~ ceil(100 / N) request.
- Thời gian hoàn thành trung bình ~ (k + 1) / 2 * S; worst-case ~ k * S.

Bảng tính nhanh với S=6s:
- N=100: mean ~6s, worst ~6s (không queue).
- N=50: k=2 -> mean ~9s, worst ~12s.
- N=25: k=4 -> mean ~15s, worst ~24s.
- N=20: k=5 -> mean ~18s, worst ~30s.

Kết luận: nếu yêu cầu 100 request đồng thời và muốn gần như không queue, cần ~100 instance.

## 3) Phương án đề xuất (đã Việt hóa hoàn toàn)
### Phương án A — Ưu tiên tốc độ, ít queue
Mục tiêu: đáp ứng burst 100 request đồng thời với P95 thấp.
- concurrency=1.
- max instances >= 100.
- min instances 5–10 (giảm cold start).
- CPU/instance: 4 vCPU.
- RAM/instance: 8 GB.

Lý do:
- Uvicorn 1 worker => mỗi instance chỉ có 1 process.
- Model lock => 1 inference tại một thời điểm.
- Model artifacts chỉ ~33 MB; RAM chủ yếu cho torch + rdkit + OCR + explainer, 8 GB là ngưỡng an toàn.

### Phương án B — Tối ưu chi phí, chấp nhận queue
Mục tiêu: tiết kiệm chi phí, P95 cao hơn.
- concurrency=1.
- max instances 25–50.
- Kỳ vọng P95 ~15–24s (với S=6s) theo mô hình mục 2.

### Phương án C — Dùng GPU để giảm S
Mục tiêu: giảm latency để giảm số instance cần thiết.
- Build image với TORCH_VARIANT=cu121.
- GPU: 1x T4 16 GB hoặc L4 24 GB.
- CPU: 4 vCPU, RAM: 8–16 GB.
- concurrency=1.

Ghi chú: cần benchmark thực tế để xác nhận S giảm bao nhiêu và GPU utilization có đủ cao để đáng đầu tư.

## 4) Tác động của 500 người truy cập đồng thời (frontend)
- Frontend chạy Firebase Hosting (static), không cần tự cấp CPU/RAM.
- Backend chỉ bị tải khi nhiều người cùng gọi /analyze hoặc /agent/analyze.
- Nếu 20% trong 500 người chạy analyze (100 người), áp dụng mô hình ở mục 2 và cấu hình ở mục 3.

## 5) Thay đổi thiết kế nên làm (giữ nguyên logic core)
1) Tách dịch vụ theo profile tài nguyên
- Tách OCR (/extract-smiles-from-image) thành service riêng (RAM 16 GB) để tránh làm chậm inference.
- Tách /predict và /agent/analyze thành 2 service riêng; service inference đặt concurrency=1.

2) Chuyển /agent/analyze sang xử lý bất đồng bộ
- Request -> enqueue -> worker xử lý -> lưu Firestore -> client poll/stream.
- Giảm rủi ro timeout khi PubChem/PubMed chậm.

3) Cache theo canonical SMILES
- Cache kết quả /predict và /analyze theo key: canonical_smiles + backend + threshold.
- Cache PubChem/PubMed theo query + TTL để giảm gọi ngoài.

4) Retry/backoff + circuit breaker
- Hiện timeout 10–15s cho PubChem/PubMed, chưa có backoff.
- Thêm retry có giới hạn + fallback để không nghẽn pipeline.

## 6) Đo lường bắt buộc để chốt con số chính xác nhất
- Đo P50/P95 cho /analyze và /agent/analyze trên Cloud Run (có/không explainer).
- Đo peak RAM khi bật/tắt MolScribe preload.
- Đo GPU utilization nếu chạy GPU.

## 7) Kết luận ngắn
Với thiết kế hiện tại (1 worker + model lock), số instance là tham số quyết định để đáp ứng burst 100 request đồng thời. Cấu hình CPU/RAM đề xuất ở mục 3 là mức an toàn để chạy ổn định; muốn giảm instance cần GPU hoặc tối ưu để giảm S.
