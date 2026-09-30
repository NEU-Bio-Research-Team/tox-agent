"""Export ToxBench cases into clean batch prompts for ChatGPT / Gemini Web.

Divides the 95 dataset cases into 5-6 batches (~16-19 cases each), formatted
with explicit instructions asking the web model to return a strict JSON array.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset" / "toxbench_dataset.json"
OUTPUT_DIR = HERE / "prompts_web"
BLIND_OUTPUT_DIR = HERE / "prompts_web_blind"

#: Groups whose vignette names the compound or hints at the answer. In a blind
#: export they get this neutral task instead; adversarial and edge cases keep
#: their vignette, because the vignette is what those cases test.
NEUTRAL_GROUPS = ("herg_positive", "herg_negative", "herg_borderline", "tox21_active", "tox21_inactive")
NEUTRAL_TASK = (
    "Analyze the compound with the SMILES above for hERG channel blocking risk and its "
    "Tox21 assay profile. State the predicted probability, interpretation, and any limitations."
)

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


def blind_cases(cases: list[dict], seed: int) -> tuple[list[dict], dict[str, str]]:
    """Opaque ids, shuffled order, no compound name, neutral task text.

    The dataset ids (``herg_pos_01``) and names (Astemizole) give the answer
    away; a web model that reads them is scored on recall, not prediction.
    """
    order = list(cases)
    random.Random(seed).shuffle(order)
    key: dict[str, str] = {}
    blinded = []
    for n, c in enumerate(order, start=1):
        blind_id = f"B{n:03d}"
        key[blind_id] = c["case_id"]
        blinded.append({
            "case_id": blind_id,
            "compound_name": None,
            "smiles": c.get("smiles", ""),
            "vignette_en": NEUTRAL_TASK if c.get("group") in NEUTRAL_GROUPS else c.get("vignette_en", ""),
        })
    return blinded, key


def export_prompts(batch_size: int = 16, blind: bool = False, seed: int = 20260930) -> list[Path]:
    output_dir = BLIND_OUTPUT_DIR if blind else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if blind:
        cases, key = blind_cases(cases, seed)
        (output_dir / "unblinding_key.json").write_text(
            json.dumps({"seed": seed, "key": key}, indent=2) + "\n", encoding="utf-8"
        )
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
        out_file = output_dir / filename

        lines = [
            f"# BATCH {batch_num:02d} / {num_batches:02d} (Cases {start+1} - {end} of {total})",
            "",
            "> **HƯỚNG DẪN:**",
            "> 1. Copy toàn bộ nội dung trong khung code bên dưới.",
            "> 2. Dán vào ChatGPT hoặc Gemini bản Web, mỗi batch một đoạn chat mới.",
            "> 3. Copy toàn bộ JSON model trả về và lưu vào file kết quả tương ứng.",
            "> 4. Ghi lại tên model hiển thị trên giao diện và ngày giờ chạy vào `<system>_meta.json`.",
            "",
            "```text",
            SYSTEM_INSTRUCTION.strip(),
            "",
            "--- CASES TO ANALYZE ---",
            "",
        ]

        for idx, c in enumerate(batch_cases, start=start + 1):
            lines.append(f"[{idx}] Case ID: {c['case_id']}")
            if c.get("compound_name"):
                lines.append(f"Compound: {c['compound_name']}")
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
   python -m evals.benchmark_comparative.evaluate_web_results --input evals/benchmark_comparative/web_results/chatgpt_results.json --system chatgpt
   ```
"""
    if blind:
        readme_content += """
## Bản mù (blind)
ID case là mã mờ (`B001`…), thứ tự đã xáo trộn, không có tên hợp chất. Khi chấm, truyền khóa giải mù:
```powershell
python -m evals.benchmark_comparative.evaluate_web_results --input "evals/benchmark_comparative/web_results_blind/chatgpt_batch_*.json" --system chatgpt-blind --key evals/benchmark_comparative/prompts_web_blind/unblinding_key.json
```
"""
    (output_dir / "README.md").write_text(readme_content, encoding="utf-8")

    return created_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", action="store_true",
                        help="Opaque ids, shuffled order, no compound names (prompts_web_blind/)")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    export_prompts(batch_size=args.batch_size, blind=args.blind, seed=args.seed)
