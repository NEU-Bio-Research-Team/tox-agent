import os
import time
import pandas as pd
import json
import argparse
from typing import Dict, Any
from agents.orchestrator_agent import run_orchestrator_flow

def evaluate_e2e(test_csv_path: str, output_json_path: str, limit: int = None):
    print(f"Loading test set from {test_csv_path}...")
    df = pd.read_csv(test_csv_path)
    
    if limit:
        print(f"Limiting evaluation to first {limit} records.")
        df = df.head(limit)
        
    results = []
    correct_predictions = 0
    total_valid = 0
    start_time = time.time()
    
    for idx, row in df.iterrows():
        name = row["name"]
        smiles = row["smiles"]
        ground_truth = float(row["label"]) # 1.0 for toxic, 0.0 for safe
        
        print(f"[{idx+1}/{len(df)}] Evaluating {name} (SMILES={smiles[:40]}...)")
        
        iter_start = time.time()
        try:
            # Run the deterministic orchestration flow
            state = run_orchestrator_flow(
                smiles_input=smiles,
                max_literature_results=5,
                language="vi",
                clinical_threshold=0.35,
                mechanism_threshold=0.5,
                inference_backend="xsmiles",
                molrag_enabled=True,
                molrag_top_k=5,
                molrag_min_similarity=0.15
            )
            
            iter_duration = time.time() - iter_start
            
            validation_status = state.get("validation_status", "INVALID")
            screening = state.get("screening_result") or {}
            final_report = state.get("final_report") or {}
            
            # Extract clinical label and check verdict accuracy
            clinical = screening.get("clinical", {})
            pred_label = clinical.get("label") # "Toxic" or "Non-toxic"
            
            # Map predictions to float
            pred_float = 1.0 if pred_label == "Toxic" else 0.0 if pred_label == "Non-toxic" else None
            
            is_correct = False
            if pred_float is not None and validation_status == "VALID":
                is_correct = (pred_float == ground_truth)
                if is_correct:
                    correct_predictions += 1
                total_valid += 1
                
            # Check JSON compliance
            report_is_valid_json = (final_report is not None and "report_metadata" in final_report)
            
            results.append({
                "name": name,
                "smiles": smiles,
                "ground_truth": ground_truth,
                "pred_label": pred_label,
                "pred_value": pred_float,
                "is_correct": is_correct,
                "validation_status": validation_status,
                "report_is_valid_json": report_is_valid_json,
                "latency_seconds": iter_duration,
                "error": state.get("screening_error")
            })
            
        except Exception as exc:
            print(f"Error evaluating {name}: {exc}")
            results.append({
                "name": name,
                "smiles": smiles,
                "ground_truth": ground_truth,
                "validation_status": "EXCEPTION",
                "latency_seconds": time.time() - iter_start,
                "error": str(exc)
            })
            
    total_time = time.time() - start_time
    accuracy = correct_predictions / total_valid if total_valid > 0 else 0.0
    
    summary = {
        "total_records": len(df),
        "total_valid_records": total_valid,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "total_time_seconds": total_time,
        "average_latency_seconds": total_time / len(df) if len(df) > 0 else 0.0,
        "results": results
    }
    
    print("\n--- Evaluation Summary ---")
    print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_valid})")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Latency: {summary['average_latency_seconds']:.2f} seconds")
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print(f"Results written to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Evaluation Benchmark for ToxAgent")
    parser.add_argument("--test-set", type=str, default="test_data/full_test_set.csv", help="Path to test set CSV")
    parser.add_argument("--output", type=str, default="results/e2e_benchmark_results.json", help="Path to write results JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluated compounds")
    
    args = parser.parse_args()
    evaluate_e2e(args.test_set, args.output, args.limit)
