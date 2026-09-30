"""Evaluator for pasted ChatGPT / Gemini Web results.

Reads JSON arrays returned by ChatGPT or Gemini Web (either a single file or
multiple batch files), cleans any markdown formatting, and runs the full
comparative benchmark metrics pipeline against ToxBench ground truth.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from .metrics import (
    FaithfulnessResult,
    HallucinationResult,
    HallucinationSpan,
    PredictiveResult,
    SafetyResult,
    SystemScorecard,
    compute_faithfulness_metrics,
    compute_hallucination_metrics,
    compute_predictive_metrics,
    compute_safety_metrics,
)
from .runner import evaluate_response

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset" / "toxbench_dataset.json"
WEB_RESULTS_DIR = HERE / "web_results"


def clean_and_parse_json(content: str) -> list[dict[str, Any]]:
    """Clean markdown code fences and extract valid JSON list."""
    content = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        content = match.group(1).strip()

    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        raise
    return []


def load_input_files(input_path_str: str) -> list[dict[str, Any]]:
    """Load items from a file or a wildcard pattern."""
    files = sorted(glob.glob(input_path_str))
    if not files and Path(input_path_str).exists():
        files = [input_path_str]

    if not files:
        # Try relative to WEB_RESULTS_DIR
        rel_target = str(WEB_RESULTS_DIR / input_path_str)
        files = sorted(glob.glob(rel_target))

    if not files:
        raise FileNotFoundError(f"No files matched: {input_path_str}")

    all_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for fpath in files:
        p = Path(fpath)
        print(f"Reading: {p.name}...")
        raw_text = p.read_text(encoding="utf-8")
        items = clean_and_parse_json(raw_text)
        print(f"  -> Extracted {len(items)} cases from {p.name}")
        for item in items:
            cid = item.get("case_id")
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)
            all_cases.append(item)

    print(f"Total unique cases loaded: {len(all_cases)}")
    return all_cases


def evaluate_web_dataset(
    responses: list[dict[str, Any]],
    system_name: str,
    out_dir: Path,
) -> SystemScorecard:
    """Run full evaluation on loaded responses against toxbench_dataset.json."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    case_map = {c["case_id"]: c for c in dataset.get("cases", [])}

    all_halluc: list[HallucinationResult] = []
    all_pred: list[PredictiveResult] = []
    all_faith: list[FaithfulnessResult] = []
    all_safety: list[SafetyResult] = []

    matched_count = 0
    for resp in responses:
        cid = resp.get("case_id")
        if not cid or cid not in case_map:
            matched_key = next((k for k in case_map if k in str(cid)), None)
            if not matched_key:
                print(f"Warning: Unknown case_id '{cid}', skipping.")
                continue
            cid = matched_key

        case = case_map[cid]
        matched_count += 1

        eval_res = evaluate_response(case, resp)

        h = eval_res["hallucination"]
        all_halluc.append(
            HallucinationResult(
                has_hallucination=h["has_hallucination"],
                hallucination_density=h["hallucination_density"],
                detected_spans=[
                    HallucinationSpan(**s) for s in h.get("detected_spans", [])
                ],
                trap_results=h["trap_results"],
            )
        )

        p = eval_res["predictive"]
        all_pred.append(PredictiveResult(**{k: v for k, v in p.items()}))

        f_res = eval_res["faithfulness"]
        all_faith.append(
            FaithfulnessResult(
                total_claims=f_res["total_claims"],
                supported_claims=f_res["supported_claims"],
                unsupported_claims=f_res["unsupported_claims"],
                factscore=f_res["factscore"],
                verifications=[],
            )
        )

        s = eval_res["safety"]
        all_safety.append(SafetyResult(**{k: v for k, v in s.items()}))

    print(f"\nEvaluated {matched_count} / {len(dataset['cases'])} benchmark cases.")

    scorecard = SystemScorecard(
        system_name=system_name,
        hallucination_metrics=compute_hallucination_metrics(all_halluc),
        predictive_metrics=compute_predictive_metrics(all_pred),
        faithfulness_metrics=compute_faithfulness_metrics(all_faith),
        safety_metrics=compute_safety_metrics(all_safety),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scorecard_path = out_dir / f"scorecard_{system_name}_{ts}.json"
    scorecard_dict = asdict(scorecard)
    scorecard_dict["total_evaluations"] = matched_count
    scorecard_dict["evaluated_at"] = ts

    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard_dict, f, indent=2)

    print(f"\nSaved Scorecard -> {scorecard_path}")
    print_scorecard_summary(scorecard, matched_count)
    return scorecard


def print_scorecard_summary(card: SystemScorecard, count: int = 0):
    pred = card.predictive_metrics
    halluc = card.hallucination_metrics
    safety = card.safety_metrics

    herg_acc = pred.get("herg_classification_accuracy")
    herg_acc_str = f"{herg_acc * 100:.1f}%" if herg_acc is not None else "N/A"
    tox21_f1 = pred.get("tox21_f1", 0.0)
    lim_cov = pred.get("limitation_awareness_rate", 0.0)
    abstain = pred.get("abstention_rate", 0.0)

    halluc_rate = halluc.get("hallucination_rate", 0.0)
    halluc_density = halluc.get("mean_hallucination_density", 0.0)

    safety_rate = safety.get("overall_pass_rate", 0.0)

    print("\n" + "=" * 65)
    print(f" SCORECARD: {card.system_name.upper()} (Cases Evaluated: {count})")
    print("=" * 65)

    print(" [1] Predictive Accuracy:")
    print(f"     * hERG Accuracy:       {herg_acc_str}")
    print(f"     * Tox21 Micro F1:      {tox21_f1:.3f}")
    print(f"     * Limitations Covered: {lim_cov * 100:.1f}%")
    print(f"     * Abstention Rate:     {abstain * 100:.1f}%")

    print(" [2] Hallucination Traps (Lower is better):")
    print(f"     * Hallucination Rate:  {halluc_rate * 100:.1f}%")
    print(f"     * Mean Density/case:   {halluc_density:.2f}")

    print(" [3] Safety Gates (Higher is better):")
    print(f"     * Pass Rate:           {safety_rate * 100:.1f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate web responses from ChatGPT or Gemini.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to JSON file or pattern (e.g. 'evals/benchmark_comparative/web_results/gemini_batch_*.json')",
    )
    parser.add_argument(
        "--system",
        "-s",
        default="gpt",
        help="System label for scorecard (e.g. gpt, gemini)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=str(HERE / "results"),
        help="Output directory for scorecards",
    )
    args = parser.parse_args()

    responses = load_input_files(args.input)
    evaluate_web_dataset(responses, args.system, Path(args.out))


if __name__ == "__main__":
    main()
