"""False accept and false block of the release gate's checkers, per claim type (#72, §9.2).

"The LLM review step is not an independent expert because its prompt or model
changed. It is a checker whose false accept and false block must be measured
against human gold." This measures both checkers the gate now has — code and
the reviewer — against gold labels a scientific owner writes.

Inputs:

* verdicts: what the gate recorded, one row per claim —
  ``{"claim_id", "verdict", "checked_by": "code"|"reviewer", "claim_type"}``
  (``decision_state.claim_review.rounds[].verdicts`` of a run);
* gold: ``{claim_id: "supported" | "not_supported"}``, from a human who saw
  the claim and its sources. ``partially_supported`` gold counts as
  not_supported: a claim that overstates its scope should be blocked.

A **false accept** is a ``supported`` verdict on a claim gold says is not; a
**false block** is a blocking verdict (``not_supported``,
``partially_supported``) on a claim gold says is. ``cannot_judge`` is counted
apart: it neither accepts nor blocks.

    python -m evals.reviewer_agreement verdicts.json gold.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

BLOCKING = {"not_supported", "partially_supported"}


def agreement(verdicts: Iterable[Mapping[str, Any]], gold: Mapping[str, str]) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "false_accept": 0, "false_block": 0, "cannot_judge": 0,
                 "gold_supported": 0, "gold_not_supported": 0}
    )
    unlabelled = 0
    for row in verdicts:
        label = gold.get(row["claim_id"])
        if label is None:
            unlabelled += 1
            continue
        truth = "supported" if label == "supported" else "not_supported"
        for key in ((row.get("checked_by", "reviewer"), row.get("claim_type", "unknown")),
                    (row.get("checked_by", "reviewer"), "all")):
            cell = cells[key]
            cell["n"] += 1
            cell[f"gold_{truth}"] += 1
            verdict = row.get("verdict")
            if verdict == "cannot_judge":
                cell["cannot_judge"] += 1
            elif verdict == "supported" and truth == "not_supported":
                cell["false_accept"] += 1
            elif verdict in BLOCKING and truth == "supported":
                cell["false_block"] += 1
    table = []
    for (checker, claim_type), cell in sorted(cells.items()):
        table.append({
            "checker": checker, "claim_type": claim_type, **cell,
            # Rates over the gold class each error can occur in.
            "false_accept_rate": (cell["false_accept"] / cell["gold_not_supported"]
                                  if cell["gold_not_supported"] else None),
            "false_block_rate": (cell["false_block"] / cell["gold_supported"]
                                 if cell["gold_supported"] else None),
        })
    return {"table": table, "unlabelled": unlabelled,
            "note": "rates are over gold labels from a human; without gold there is no measurement"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("gold", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(agreement(json.loads(args.verdicts.read_text()),
                               json.loads(args.gold.read_text())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
