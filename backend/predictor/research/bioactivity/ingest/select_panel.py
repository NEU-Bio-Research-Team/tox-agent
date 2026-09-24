"""Select the V1 target panel by pre-registered rule (data contract, 3.5).

The rule must be fixed before training, and must never consult a model score --
otherwise the reported benchmark is the outcome of a search over targets rather
than the outcome of a method.

So selection uses only dataset properties available from the P0 profile:

  * a record floor per (target, standard_type) task
  * at least `MIN_FAMILIES` distinct level-1 protein families
  * family-balanced round robin, so the panel is not 16 kinases

The round robin is deterministic: families are visited in a fixed order and
targets within a family are ranked by record count with the ChEMBL ID as
tie-break. Running this twice on the same profile yields the same panel.

Every target that *qualified* is written out alongside the ones selected, so a
reviewer can see what the rule passed over.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("select_panel")

ALLOWED_STANDARD_TYPES = ("Ki", "Kd", "IC50", "EC50")

#: Record floor for a (target, standard_type) task to be a candidate. This is
#: raw activity rows; the stricter "unique standardized parents" floor can only
#: be checked after standardization, and is re-verified in `build_dataset`.
MIN_TASK_RECORDS = 1200

#: Panel size band from the data contract.
PANEL_SIZE = 16
MIN_FAMILIES = 4

#: Not a family -- the absence of a ChEMBL protein-class annotation.
UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class PanelTarget:
    target_chembl_id: str
    pref_name: str
    gene_symbol: str
    protein_family: str
    uniprot: str
    hq_records: int
    qualifying_tasks: tuple[str, ...]
    task_counts: dict[str, int]


def load_profile(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def find_qualifying(
    rows: list[dict[str, str]], *, min_task_records: int
) -> list[PanelTarget]:
    """Every target with at least one task clearing the record floor."""
    qualifying: list[PanelTarget] = []
    for row in rows:
        counts = {st: _as_int(row.get(f"n_{st}")) for st in ALLOWED_STANDARD_TYPES}
        tasks = tuple(
            st for st in ALLOWED_STANDARD_TYPES if counts[st] >= min_task_records
        )
        if not tasks:
            continue
        qualifying.append(
            PanelTarget(
                target_chembl_id=row["target_chembl_id"],
                pref_name=row.get("pref_name", ""),
                gene_symbol=row.get("gene_symbol", ""),
                protein_family=row.get("protein_family") or "unclassified",
                uniprot=row.get("uniprot", ""),
                hq_records=_as_int(row.get("hq_records")),
                qualifying_tasks=tasks,
                task_counts=counts,
            )
        )
    return qualifying


def select_panel(
    qualifying: list[PanelTarget],
    *,
    panel_size: int = PANEL_SIZE,
    min_families: int = MIN_FAMILIES,
) -> list[PanelTarget]:
    """Family-balanced deterministic round robin over qualifying targets."""
    by_family: dict[str, list[PanelTarget]] = defaultdict(list)
    for target in qualifying:
        by_family[target.protein_family].append(target)

    # Rank within family by data volume, ChEMBL ID as a stable tie-break.
    for targets in by_family.values():
        targets.sort(key=lambda t: (-t.hq_records, t.target_chembl_id))

    # Families are visited richest-first, then alphabetically, so the order does
    # not depend on dict insertion. `unclassified` is not a family -- it is the
    # absence of an annotation -- so it is visited last and does not count
    # toward the diversity requirement. Letting it count would let a panel of
    # kinases plus unannotated targets claim broad family coverage.
    real_families = [f for f in by_family if f != UNCLASSIFIED]
    family_order = sorted(
        real_families,
        key=lambda f: (-sum(t.hq_records for t in by_family[f]), f),
    )
    if UNCLASSIFIED in by_family:
        family_order.append(UNCLASSIFIED)

    if len(real_families) < min_families:
        raise ValueError(
            f"only {len(real_families)} annotated protein families qualify, "
            f"contract requires >= {min_families}. Tighten nothing: reduce the "
            "panel or revisit the record floor deliberately."
        )

    selected: list[PanelTarget] = []
    cursors = {family: 0 for family in family_order}
    while len(selected) < panel_size:
        progressed = False
        for family in family_order:
            if len(selected) >= panel_size:
                break
            index = cursors[family]
            if index < len(by_family[family]):
                selected.append(by_family[family][index])
                cursors[family] = index + 1
                progressed = True
        if not progressed:  # every family exhausted
            break

    families = {t.protein_family for t in selected} - {UNCLASSIFIED}
    if len(families) < min_families:
        raise ValueError(
            f"panel spans only {len(families)} annotated families: "
            f"{sorted(families)}"
        )
    n_unclassified = sum(1 for t in selected if t.protein_family == UNCLASSIFIED)
    LOG.info(
        "selected %d targets across %d annotated families (%d unclassified)",
        len(selected), len(families), n_unclassified,
    )
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-task-records", type=int, default=MIN_TASK_RECORDS)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_profile(args.profile)
    qualifying = find_qualifying(rows, min_task_records=args.min_task_records)
    LOG.info(
        "%d/%d targets have a task with >= %d records",
        len(qualifying),
        len(rows),
        args.min_task_records,
    )

    selected = select_panel(qualifying, panel_size=args.panel_size)

    panel = {
        "panel_id": "chembl37-human-single-protein-v1",
        "selection_rule": {
            "min_task_records": args.min_task_records,
            "panel_size": args.panel_size,
            "min_families": MIN_FAMILIES,
            "method": "family-balanced deterministic round robin over "
            "qualifying targets ranked by HQ-Exact record count",
            "uses_model_scores": False,
        },
        "n_qualifying_targets": len(qualifying),
        "targets": [
            {
                "target_chembl_id": t.target_chembl_id,
                "pref_name": t.pref_name,
                "gene_symbol": t.gene_symbol,
                "protein_family": t.protein_family,
                "uniprot": t.uniprot,
                "hq_records": t.hq_records,
                "tasks": list(t.qualifying_tasks),
                "task_counts": t.task_counts,
            }
            for t in selected
        ],
    }
    (args.out / "panel-v1.json").write_text(json.dumps(panel, indent=2) + "\n")

    # The targets the rule passed over, for audit.
    with open(args.out / "qualifying_targets.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["target_chembl_id", "pref_name", "gene_symbol", "protein_family",
             "hq_records", "qualifying_tasks", "selected"]
        )
        chosen = {t.target_chembl_id for t in selected}
        for t in sorted(qualifying, key=lambda t: (-t.hq_records, t.target_chembl_id)):
            writer.writerow(
                [t.target_chembl_id, t.pref_name, t.gene_symbol, t.protein_family,
                 t.hq_records, "|".join(t.qualifying_tasks),
                 int(t.target_chembl_id in chosen)]
            )

    print(f"\nPanel: {len(selected)} targets")
    for t in selected:
        print(
            f"  {t.target_chembl_id:14} {t.protein_family:22} "
            f"{(t.gene_symbol or t.pref_name)[:28]:30} "
            f"n={t.hq_records:6} tasks={','.join(t.qualifying_tasks)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
