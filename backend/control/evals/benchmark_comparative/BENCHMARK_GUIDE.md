# HƯỚNG DẪN CHẠY BENCHMARK VÀ CHẤM ĐIỂM (TOXBENCH)

Tài liệu này hướng dẫn cách chạy chấm điểm độc lập hoặc đối đầu (ToxAgent vs ChatGPT vs Gemini) cho 95 test case benchmark.

---

## 1. Cấu trúc thư mục

```
evals/benchmark_comparative/
├── dataset/
│   └── toxbench_dataset.json             # 95 ca kiểm thử chuẩn (Ground Truth)
├── prompts_web/                          # 6 file batch prompt để copy lên web
│   ├── batch_01_cases_01_to_16.md
│   ├── batch_02_cases_17_to_32.md
│   └── ... (đến batch 06)
├── web_results/                          # Nơi dán kết quả JSON từ Web
│   ├── chatgpt_batch_01.json ... 06.json
│   └── gemini_batch_01.json ... 06.json
├── results/                              # Kết quả sau khi chấm điểm
│   ├── scorecard_chatgpt_*.json
│   ├── scorecard_gemini_*.json
│   ├── scorecard_toxagent_*.json
│   └── comparative_report.md             # Báo cáo so sánh Markdown tổng hợp
├── evaluate_web_results.py               # Script chấm điểm kết quả từ Web
├── compare_scorecards.py                 # Script xuất bảng đối chiếu & báo cáo
└── run_benchmark.py                      # Script chạy trực tiếp ToxAgent API
```

---

## 2. Quy trình chấm điểm cho những lần sau

### Bước 1: Thu thập kết quả từ Web (nếu có cập nhật prompt mới)
- Mở các file trong `prompts_web/batch_*.md`.
- Copy đoạn trong khối code và dán vào ChatGPT Web hoặc Gemini Web.
- Lưu kết quả JSON trả về vào thư mục `web_results/` với quy tắc tên:
  - Cho ChatGPT: `chatgpt_batch_01.json`, `chatgpt_batch_02.json`, ...
  - Cho Gemini: `gemini_batch_01.json`, `gemini_batch_02.json`, ...

---

### Bước 2: Chạy lệnh chấm điểm tự động

Mở Terminal tại thư mục `backend/control`:

#### A. Chấm điểm Gemini:
```powershell
python -m evals.benchmark_comparative.evaluate_web_results --input "evals/benchmark_comparative/web_results/gemini_batch_*.json" --system gemini
```

#### B. Chấm điểm ChatGPT:
```powershell
python -m evals.benchmark_comparative.evaluate_web_results --input "evals/benchmark_comparative/web_results/chatgpt_batch_*.json" --system chatgpt
```

#### C. Chấm điểm ToxAgent (Chạy trực tiếp từ backend ToxAgent):
*Yêu cầu backend ToxAgent đang chạy (`uvicorn app.main:app --port 8000`)*:
```powershell
python -m evals.benchmark_comparative.run_benchmark --token dev-local --systems toxagent --trials 1
```

---

### Bước 3: Xuất bảng so sánh và báo cáo tổng hợp

Chạy lệnh sau để tự động đọc tất cả các scorecard mới nhất trong `results/` và xuất ra báo cáo:

```powershell
python -m evals.benchmark_comparative.compare_scorecards
```

- Bảng so sánh sẽ hiển thị trực tiếp trên màn hình terminal.
- File báo cáo markdown sẽ được lưu tại:
  `evals/benchmark_comparative/results/comparative_report.md`
