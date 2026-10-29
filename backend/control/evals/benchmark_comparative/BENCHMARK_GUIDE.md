# HƯỚNG DẪN CHẠY BENCHMARK VÀ CHẤM ĐIỂM (TOXBENCH)

Tài liệu này hướng dẫn cách chạy chấm điểm độc lập hoặc đối đầu (ToxAgent vs ChatGPT vs Gemini) cho 95 test case benchmark.

---

## 0. Cần biết trước khi đọc số

- **95 case này là dev/regression set, không phải test cuối.** Hệ thống được phát triển dựa
  trên chúng và kết quả từng case đã được xem rồi điều chỉnh theo. Dùng để bắt regression thì
  đúng; dùng làm con số công bố thì không. Test khóa độc lập là artifact riêng (issue #56).
- **Nhãn gold chưa đủ provenance.** `source_id` rỗng ở **cả 95/95** case; không case nào có
  `assay_id`, `readout`, `unit`, `comparator`, `label_rule` hay `verified_by`. Chỉ 50 case có
  nhãn hERG và 20 case có nhãn Tox21, nên mẫu số của từng endpoint **không phải 95** — đọc
  `*_labelled_count` trong scorecard. Xem `provenance_audit` trong file dataset (issue #37).
- **Bộ `prompts_web/` gốc lộ đáp án.** Case ID (`herg_pos_01`, `herg_neg_05`…) và tên thuốc nằm
  trong prompt, vignette borderline ghi sẵn "borderline activity". Kết quả `web_results/` hiện có
  vì vậy đo khả năng *nhớ* chứ không phải *dự đoán*, và không so công bằng với ToxAgent (chỉ nhận SMILES).
  Thu lại bằng `prompts_web_blind/` (mục 2, Bước 1).
- **Nhánh ToxAgent gọi `POST /v1/predict`** — tức là predictor, không phải agent. Văn bản trả về
  là dòng tóm tắt do driver tự dựng, nên bẫy hallucination và safety gate đạt gần như mặc định.
- Các chỉ số là proxy tự động; điểm chuyên gia đến từ phòng lab chấm mù.

### Ba con số accuracy, không phải một

Một hệ thống từ chối trả lời các case khó sẽ có accuracy cao giả. Luôn đọc cùng nhau:

| Chỉ số | Nghĩa |
|---|---|
| `herg_coverage` | số case có dự đoán / số case có nhãn |
| `herg_classification_accuracy` | accuracy **khi cam kết** — chỉ so được giữa các hệ cùng coverage |
| `herg_strict_accuracy` | số đúng / **toàn bộ** case có nhãn; abstention tính là sai |

36/36 đúng với coverage 36/50 là accuracy khi cam kết `1.0` nhưng strict `0.72`; không được
đem con số `1.0` đó so với hệ đã trả lời đủ 50 case.

- `compare_scorecards` in cả ba dòng trong bảng đối chiếu.
- `run_benchmark` in cả ba, và nhãn `← winner` nằm trên **strict accuracy**.
- `build_comparative_report` xếp hạng `winners["predictive_accuracy"]` theo
  `herg_strict_accuracy`, và thêm khối `predictive_accuracy_basis` ghi cả ba số cho từng hệ.
  Trước đây nó xếp hạng theo accuracy khi cam kết, nên hệ càng từ chối trả lời càng dễ thắng.

### Bước 0: kiểm scorer trước khi tin điểm của nó

```bash
python -m evals.benchmark_comparative.positive_controls
```

Lệnh này chạy các câu trả lời **cố ý sai** qua đúng scorer đang dùng và báo false
negative/false positive của chính evaluator, cùng khối `known_gaps` liệt kê các lớp defect
scorer **không** phát hiện được. Chạy nó khi sửa scorer hoặc sửa dataset.

## 1. Cấu trúc thư mục

```
evals/benchmark_comparative/
├── dataset/
│   └── toxbench_dataset.json             # 95 ca kiểm thử chuẩn (Ground Truth)
├── prompts_web/                          # 6 file batch prompt (bản gốc, LỘ nhãn — xem mục 0)
├── prompts_web_blind/                    # Bản mù + unblinding_key.json (dùng cho lần thu sau)
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
- Tạo bản mù: `python -m evals.benchmark_comparative.export_web_prompts --blind`
- Mở các file trong `prompts_web_blind/batch_*.md`, mỗi batch một đoạn chat mới.
- Ghi tên model hiển thị trên giao diện và ngày giờ vào `web_results_blind/<system>_meta.json`.
- Copy đoạn trong khối code và dán vào ChatGPT Web hoặc Gemini Web.
- Lưu kết quả JSON trả về vào thư mục `web_results/` với quy tắc tên:
  - Cho ChatGPT: `chatgpt_batch_01.json`, `chatgpt_batch_02.json`, ...
  - Cho Gemini: `gemini_batch_01.json`, `gemini_batch_02.json`, ...
  - Kết quả bản mù lưu vào `web_results_blind/` và chấm kèm `--key evals/benchmark_comparative/prompts_web_blind/unblinding_key.json`.

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

#### C. Chấm điểm ToxAgent (chạy trực tiếp trên stack):
*Yêu cầu stack đang chạy (`./bin/toxagent up`); control nằm sau frontend ở cổng 8088*:
```bash
export TOXAGENT_STATIC_TOKENS="$(grep '^TOXAGENT_STATIC_TOKENS=' ../../.env | cut -d= -f2-)"
python -m evals.benchmark_comparative.run_benchmark --base-url http://localhost:8088 --systems toxagent --trials 1
```
Lệnh ghi `results/toxagent-<ts>.json` (phản hồi thô + đánh giá từng case) và `results/scorecard_toxagent_<ts>.json`.

---

### Bước 3: Xuất bảng so sánh và báo cáo tổng hợp

Chạy lệnh sau để tự động đọc tất cả các scorecard mới nhất trong `results/` và xuất ra báo cáo:

```powershell
python -m evals.benchmark_comparative.compare_scorecards
```

- Bảng so sánh sẽ hiển thị trực tiếp trên màn hình terminal.
- `evaluate_web_results` ghi thêm `results/cases_<system>_<ts>.json` (đánh giá từng case) để kiểm tra lại.
- File báo cáo markdown sẽ được lưu tại:
  `evals/benchmark_comparative/results/comparative_report.md`
