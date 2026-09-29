"""Export ToxBench cases into clean batch prompts for ChatGPT / Gemini Web.

Divides the 95 dataset cases into 5-6 batches (~16-19 cases each), formatted
with explicit instructions asking the web model to return a strict JSON array.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset" / "toxbench_dataset.json"
OUTPUT_DIR = HERE / "prompts_web"

SYSTEM_INSTRUCTION = """You are an expert toxicologist and computational pharmacology evaluator.
Analyze each of the following drug cases for hERG channel inhibition risk and Tox21 assay toxicity profile.

CRITICAL INSTRUCTION:
You MUST respond with ONLY a valid, raw JSON array (no markdown code blocks, no intro/outro conversational text).
Every object in the array must strictly follow this JSON schema:

[
  {
    "case_id": "<case_id exactly as given>",
    "herg_classification": "blocker" | "non-blocker" | "uncertain",
    "herg_probability": <float between 0.0 and 1.0, or null>,
    "tox21_active_assays": ["<assay_name_1>", ...],
    "limitations": ["<limitation_1>", ...],
    "response_text": "<Your full scientific interpretation and rationale. State clearly your reasoning, biological mechanisms, and any limitations. Do NOT overreach into human clinical efficacy or cardiotoxicity unless supported.>"
  }
]

Common Tox21 assays include: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53.
"""


def export_prompts(batch_size: int = 16) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    total = len(cases)
    print(f"Loaded {total} cases from {DATASET_PATH}")

    created_files: list[Path] = []
    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_cases = cases[start:end]

        batch_num = batch_idx + 1
        filename = f"batch_{batch_num:02d}_cases_{start+1:02d}_to_{end:02d}.md"
        out_file = OUTPUT_DIR / filename

        lines = [
            f"# BATCH {batch_num:02d} / {num_batches:02d} (Cases {start+1} - {end} of {total})",
            "",
            "> **HƯỚNG DẪN:**",
            "> 1. Copy toàn bộ nội dung trong khung code bên dưới.",
            "> 2. Dán vào ChatGPT (GPT-4o) hoặc Gemini (Gemini 2.5 Pro) bản Web.",
            "> 3. Copy toàn bộ JSON model trả về và lưu vào file kết quả tương ứng.",
            "",
            "```text",
            SYSTEM_INSTRUCTION.strip(),
            "",
            "--- CASES TO ANALYZE ---",
            "",
        ]

        for idx, c in enumerate(batch_cases, start=start + 1):
            lines.append(f"[{idx}] Case ID: {c['case_id']}")
            lines.append(f"Compound: {c.get('compound_name', 'Unknown')}")
            lines.append(f"SMILES: {c.get('smiles', '')}")
            lines.append(f"Question/Vignette: {c.get('vignette_en', '')}")
            lines.append("")

        lines.append("```")

        out_file.write_text("\n".join(lines), encoding="utf-8")
        created_files.append(out_file)
        print(f"Exported Batch {batch_num}: {out_file.name} ({len(batch_cases)} cases)")

    # Also create a README in prompts_web
    readme_content = f"""# Web Evaluation Prompts (ToxBench)

Đã chia toàn bộ {total} cases thành {num_batches} batches để paste vào ChatGPT / Gemini Web.

## Danh sách các batch:
"""
    for f in created_files:
        readme_content += f"- [`{f.name}`](./{f.name})\n"

    readme_content += """
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
"""
    (OUTPUT_DIR / "README.md").write_text(readme_content, encoding="utf-8")

    return created_files


if __name__ == "__main__":
    export_prompts(batch_size=16)
