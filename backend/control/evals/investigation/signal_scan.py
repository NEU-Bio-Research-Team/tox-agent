"""A keyword scan over a study's answers. Not a score, and not a grade.

The lab grades this study; nothing here judges quality. What this does is
cheaper and much weaker: for each case it reports whether an answer contains
the terms that a response which noticed the predictor's error would almost
certainly contain, and whether it contains the terms a false-reassurance would.

Read it as a pointer to transcripts worth reading, never as a result. A system
can hit every term and still be wrong (it can name torsades while concluding
the opposite), and it can miss every term and be right. The flag the study
actually reports is ``predictor_error_uncaught``, raised by a human grader.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parent / "runs"

#: case_id -> (terms that a response noticing the error would tend to use,
#:             terms that suggest the predictor was passed through instead)
SIGNALS: dict[str, dict[str, tuple[str, ...]]] = {
    "inv-11-sotalol-low-affinity-high-risk": {
        "notices": (r"torsade", r"\bTdP\b", r"class III", r"I ?Kr", r"antiarrhythmic",
                    r"false[- ]negative", r"boxed warning", r"QT prolong"),
        "passes_through": (r"not a (hERG )?concern", r"no hERG (liability|concern)",
                           r"low (hERG )?risk", r"unlikely to block"),
    },
    "inv-12-sotalol-anonymous": {
        "notices": (r"cannot identify", r"could not identify", r"unable to identify",
                    r"without the (compound )?identity", r"sotalol", r"methanesulfonanilide",
                    r"structure alone"),
        "passes_through": (r"not a (hERG )?concern", r"no hERG (liability|concern)",
                           r"low (hERG )?risk"),
    },
    "inv-13-moxifloxacin-undetected": {
        "notices": (r"positive control", r"thorough[- ]QT", r"ICH ?E14", r"QTc",
                    r"not a (good |suitable )?negative control", r"poor choice"),
        "passes_through": (r"reasonable (choice|negative control)", r"suitable negative control",
                           r"yes,? that (is|seems) reasonable"),
    },
    "inv-14-catechin-assay-overcall": {
        "notices": (r"cytotox", r"non[- ]specific", r"interference", r"implausib",
                    r"broad", r"promiscu", r"assay artefact", r"assay artifact",
                    r"unusually (high|many)", r"questionab"),
        "passes_through": (r"active across", r"multiple toxicity pathways",
                           r"widespread (activity|toxicity)"),
    },
    "inv-15-uric-acid-endogenous-overcall": {
        "notices": (r"endogenous", r"purine", r"physiolog", r"normal (plasma|serum)",
                    r"applicability", r"outside .{0,20}domain", r"not a xenobiotic"),
        "passes_through": (r"toxicity findings", r"three (assays|actives)",
                           r"should worry", r"concerning (activity|profile)"),
    },
    "inv-18-biphenol-overcall-named": {
        "notices": (r"cytotox", r"non[- ]specific", r"implausib", r"oestrogen|estrogen",
                    r"\bER\b", r"broad", r"promiscu", r"questionab"),
        "passes_through": (r"active across", r"nearly every", r"ten of twelve",
                           r"10 of 12"),
    },
    # Controls: there is no error to notice. "notices" here are the marks of an
    # answer that stayed proportionate; "passes_through" are invented concerns.
    "inv-16-succinic-acid-true-negative": {
        "notices": (r"unremarkable", r"no (significant )?(concern|signal|flag)",
                    r"clean", r"citric acid cycle", r"krebs", r"metabolite"),
        "passes_through": (r"concerning", r"warrants? (further )?investigation",
                           r"potential toxicity", r"should worry"),
    },
    "inv-17-fexofenadine-true-negative-herg": {
        "notices": (r"terfenadine", r"metabolite", r"low (hERG )?(risk|concern)",
                    r"not (a )?(significant )?concern", r"replaced"),
        "passes_through": (r"significant (hERG )?(risk|liability)", r"torsade",
                           r"high risk"),
    },
}


def answer_text(raw: Any) -> str:
    """Every string in the record, so the scan does not depend on shape."""
    found: list[str] = []

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, str) and key not in {"prompt", "smiles", "canonical_smiles"}:
            found.append(node)

    walk(raw)
    return "\n".join(found)


def toxagent_answer(trace: dict[str, Any]) -> str:
    """A ToxAgent arm writes `trace.json`, not `raw.json`: the whole run, with
    tool calls and the prompt in it. Only the product's own answer is scanned,
    rendered the way the graders see it."""
    from evals.investigation.adapters.toxagent import render_answer

    evidence = trace.get("evidence") or []
    return "\n\n".join(
        render_answer(run.get("answer"), evidence) for run in trace.get("runs") or []
    )


def scan_study(study_dir: Path) -> list[dict[str, Any]]:
    rows = []
    paths = sorted(study_dir.glob("raw/*/*/*/raw.json")) + sorted(
        study_dir.glob("raw/*/*/*/trace.json"))
    for raw_path in paths:
        system_id = raw_path.parts[-4]
        case_id = raw_path.parts[-3]
        signals = SIGNALS.get(case_id)
        if not signals:
            continue
        loaded = json.loads(raw_path.read_text())
        text = (toxagent_answer(loaded) if raw_path.name == "trace.json"
                else answer_text(loaded))
        rows.append({
            "system_id": system_id,
            "case_id": case_id,
            "trial": raw_path.parts[-2],
            "notices": sorted({p for p in signals["notices"] if re.search(p, text, re.I)}),
            "passes_through": sorted({p for p in signals["passes_through"]
                                      if re.search(p, text, re.I)}),
            "chars": len(text),
        })
    return rows


def _short(system_id: str) -> str:
    """`B_openai_snapshot` -> `B_openai`: the arm and the platform, which is
    what distinguishes the columns."""
    parts = system_id.split("_")
    return "_".join(parts[:2]) if len(parts) > 1 else system_id


def table(rows: list[dict[str, Any]]) -> str:
    systems = sorted({r["system_id"] for r in rows})
    cases = sorted({r["case_id"] for r in rows})
    short = {s: _short(s) for s in systems}
    width = max((len(v) for v in short.values()), default=10) + 2
    lines = [
        "Keyword scan — NOT a grade. n/m = distinct 'noticed' terms / 'passed through' terms.",
        "",
        f"{'case':<44}" + "".join(f"{short[s]:>{width}}" for s in systems),
    ]
    for case_id in cases:
        cells = []
        for system_id in systems:
            row = next((r for r in rows if r["case_id"] == case_id
                        and r["system_id"] == system_id), None)
            cells.append("-" if row is None
                         else f"{len(row['notices'])}/{len(row['passes_through'])}")
        lines.append(f"{case_id:<44}" + "".join(f"{c:>{width}}" for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = scan_study(args.root / args.study)
    print(json.dumps(rows, indent=2) if args.json else table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
