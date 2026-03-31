from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import ADK_AVAILABLE, run_orchestrator_flow


def _compact_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    final_report = state.get("final_report") if isinstance(state.get("final_report"), dict) else {}
    sections = final_report.get("sections") if isinstance(final_report, dict) else {}
    clinical = sections.get("clinical_toxicity") if isinstance(sections, dict) else {}

    return {
        "adk_available": ADK_AVAILABLE,
        "smiles_input": state.get("smiles_input"),
        "validation_status": state.get("validation_status"),
        "validation_error": state.get("validation_error"),
        "screening_error": state.get("screening_error"),
        "research_error": state.get("research_error"),
        "risk_level": final_report.get("risk_level"),
        "clinical_verdict": clinical.get("verdict") if isinstance(clinical, dict) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for ToxAgent agent layer flow.")
    parser.add_argument(
        "--smiles",
        type=str,
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="SMILES input for orchestrator flow",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Maximum PubMed results for research stage",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output path for full state JSON",
    )
    args = parser.parse_args()

    state = run_orchestrator_flow(args.smiles, max_literature_results=args.max_results)

    summary = _compact_summary(state)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved detailed state to: {output_path}")


if __name__ == "__main__":
    main()
