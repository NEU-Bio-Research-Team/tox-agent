# Thư mục lưu kết quả từ ChatGPT / Gemini Web

Bạn có thể lưu kết quả theo 1 trong 2 cách:

### Cách 1: Lưu theo từng batch (Khuyên dùng)
- Copy JSON trả lời từ Batch 1 -> Lưu vào file `chatgpt_batch_01.json`
- Copy JSON trả lời từ Batch 2 -> Lưu vào file `chatgpt_batch_02.json`
- ...
- Tương tự cho Gemini: `gemini_batch_01.json`, `gemini_batch_02.json`, ...

Sau đó chạy lệnh đánh giá tất cả các batch cùng lúc:
```powershell
python -m evals.benchmark_comparative.evaluate_web_results --input "evals/benchmark_comparative/web_results/chatgpt_batch_*.json" --system chatgpt
```

### Cách 2: Lưu chung vào một file
- Gộp chung toàn bộ JSON vào `chatgpt_results.json` hoặc `gemini_results.json`
- Chạy lệnh:
```powershell
python -m evals.benchmark_comparative.evaluate_web_results --input evals/benchmark_comparative/web_results/chatgpt_results.json --system chatgpt
```
