# Web Evaluation Prompts (ToxBench)

Đã chia toàn bộ 95 cases thành 6 batches để paste vào ChatGPT / Gemini Web.

## Danh sách các batch:
- [`batch_01_cases_01_to_16.md`](./batch_01_cases_01_to_16.md)
- [`batch_02_cases_17_to_32.md`](./batch_02_cases_17_to_32.md)
- [`batch_03_cases_33_to_48.md`](./batch_03_cases_33_to_48.md)
- [`batch_04_cases_49_to_64.md`](./batch_04_cases_49_to_64.md)
- [`batch_05_cases_65_to_80.md`](./batch_05_cases_65_to_80.md)
- [`batch_06_cases_81_to_95.md`](./batch_06_cases_81_to_95.md)

## Cách thực hiện:
1. Mở lần lượt từng file `batch_*.md`.
2. Copy đoạn text bên trong khối ` ```text ... ``` `.
3. Dán vào ChatGPT Web hoặc Gemini Web trong một đoạn chat mới (nên dùng Model mới nhất: GPT-4o hoặc Gemini 2.5 Pro).
4. Lưu toàn bộ JSON model trả lời vào thư mục `evals/benchmark_comparative/web_results/`:
   - `chatgpt_batch_01.json`, `chatgpt_batch_02.json`, ... (hoặc gộp chung thành `chatgpt_results.json`)
   - `gemini_batch_01.json`, `gemini_batch_02.json`, ... (hoặc gộp chung thành `gemini_results.json`)
5. Chạy lệnh:
   ```powershell
   python -m evals.benchmark_comparative.evaluate_web_results --input evals/benchmark_comparative/web_results/chatgpt_results.json --system gpt
   ```
