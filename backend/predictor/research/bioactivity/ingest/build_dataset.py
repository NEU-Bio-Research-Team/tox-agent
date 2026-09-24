"""P1: extract, standardize and aggregate `toxact-chembl37-hq-v1`.

Three stages, each of which records what it discarded:

  extract    page HQ-Exact activities for every (target, standard_type) task
  standardize   parent structure + connectivity key, once per unique SMILES
  aggregate     median pChEMBL per (parent, target, type, assay context)

The aggregation key includes assay context, never just the target. Collapsing
Ki/Kd/IC50/EC50 -- or pooling a binding assay with a cell-based functional one --
produces a label whose variance is protocol, not chemistry, and a model that
fits it looks accurate while predicting nothing useful.

Replicates that disagree by more than one log unit are flagged and held out of
HQ-Exact by default: a median over 5.0 and 8.5 is not a measurement.

Usage:
    python -m bioactivity.ingest.build_dataset --panel <panel-v1.json> --out <dir>
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import gzip
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bioactivity.ingest.chembl_client import (  # noqa: E402
    ACTIVITY_FIELDS,
    ChemblClient,
    filter_definition_sha256,
    hq_exact_activity_params,
    sha256_file,
)
from bioactivity.ingest.standardize import (  # noqa: E402
    STANDARDIZER_VERSION,
    standardize_many,
)

LOG = logging.getLogger("build_dataset")

#: Replicate spread above which the aggregate is not trusted (log10 units).
MAX_REPLICATE_RANGE = 1.0

#: Validity comments ChEMBL uses to mark records it does not stand behind.
ACCEPTED_VALIDITY = {None, "", "Manually validated"}

DATASET_ID = "toxact-chembl37-hq-v1"


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------

def extract_task(
    client: ChemblClient, target_id: str, standard_type: str
) -> list[dict[str, Any]]:
    """Page every HQ-Exact activity for one (target, standard_type) task."""
    params = hq_exact_activity_params(
        target_chembl_id=target_id, standard_type=standard_type
    )
    rows: list[dict[str, Any]] = []
    for page in client.iter_pages("activity", params, fields=ACTIVITY_FIELDS):
        rows.extend(page)
    return rows


def extract_panel(
    client: ChemblClient, panel: dict[str, Any], workers: int
) -> list[dict[str, Any]]:
    jobs = [
        (target["target_chembl_id"], standard_type)
        for target in panel["targets"]
        for standard_type in target["tasks"]
    ]
    LOG.info("extracting %d tasks with %d workers", len(jobs), workers)

    collected: list[dict[str, Any]] = []
    done = 0
    started = time.time()

    def run(job: tuple[str, str]) -> tuple[tuple[str, str], list[dict[str, Any]]]:
        return job, extract_task(client, job[0], job[1])

    with futures.ThreadPoolExecutor(workers) as pool:
        for job, rows in pool.map(run, jobs):
            collected.extend(rows)
            done += 1
            LOG.info(
                "  [%2d/%2d] %s %-5s -> %6d rows (%5.0fs elapsed, %d total)",
                done, len(jobs), job[0], job[1], len(rows),
                time.time() - started, len(collected),
            )
    return collected


# --------------------------------------------------------------------------
# assay context
# --------------------------------------------------------------------------

def assay_context_key(row: dict[str, Any]) -> str:
    """Deterministic measurement-context key.

    Combines assay format and protein variant, which are the two context axes
    that most change what a potency number means for a fixed target:
    `assay_type` separates binding from functional from ADME, `bao_label`
    separates single-protein from cell-based from organism formats, and a
    variant mutation means the construct is not wild type.
    """
    assay_type = (row.get("assay_type") or "NA").strip()
    bao = (row.get("bao_label") or "NA").strip().replace("|", "/")
    mutation = (row.get("assay_variant_mutation") or "WT").strip().replace("|", "/")
    return f"{assay_type}|{bao}|{mutation}"


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def build_table(
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Standardize then aggregate, returning the table and a filter ledger."""
    ledger: dict[str, int] = defaultdict(int)
    ledger["raw_rows"] = len(raw_rows)

    # -- record-level filters ------------------------------------------
    kept: list[dict[str, Any]] = []
    for row in raw_rows:
        validity = row.get("data_validity_comment")
        if validity not in ACCEPTED_VALIDITY:
            ledger[f"dropped_validity:{validity}"] += 1
            continue
        pchembl = row.get("pchembl_value")
        if pchembl in (None, ""):
            ledger["dropped_no_pchembl"] += 1
            continue
        try:
            value = float(pchembl)
        except (TypeError, ValueError):
            ledger["dropped_pchembl_unparseable"] += 1
            continue
        if not (0.0 < value < 20.0):  # physically implausible pActivity
            ledger["dropped_pchembl_out_of_range"] += 1
            continue
        try:
            standard_value = float(row.get("standard_value"))
        except (TypeError, ValueError):
            ledger["dropped_standard_value_unparseable"] += 1
            continue
        if standard_value <= 0:
            ledger["dropped_standard_value_nonpositive"] += 1
            continue
        row["_pchembl"] = value
        kept.append(row)
    ledger["after_record_filters"] = len(kept)

    # -- standardization ------------------------------------------------
    LOG.info("standardizing %d unique SMILES", len({r.get("canonical_smiles") for r in kept}))
    cache, reasons = standardize_many([r.get("canonical_smiles") for r in kept])
    for reason, count in reasons.items():
        ledger[f"structure:{reason}"] += count

    usable: list[dict[str, Any]] = []
    for row in kept:
        std = cache[str(row.get("canonical_smiles") or "").strip()]
        if std.status == "rejected":
            ledger["dropped_structure_rejected"] += 1
            continue
        if std.status == "flagged":
            # Kept out of HQ-Exact, but counted: these are applicability-domain
            # exclusions, not silent data loss.
            ledger["dropped_structure_flagged"] += 1
            continue
        if std.connectivity_key is None:
            ledger["dropped_no_connectivity_key"] += 1
            continue
        row["_std"] = std
        usable.append(row)
    ledger["after_standardization"] = len(usable)

    # -- aggregation ----------------------------------------------------
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        std = row["_std"]
        key = (
            std.standardized_smiles,
            row["target_chembl_id"],
            row["standard_type"],
            assay_context_key(row),
        )
        groups[key].append(row)

    table: list[dict[str, Any]] = []
    for (smiles, target_id, standard_type, context), rows in groups.items():
        values = [r["_pchembl"] for r in rows]
        spread = max(values) - min(values)
        mad = (
            statistics.median([abs(v - statistics.median(values)) for v in values])
            if len(values) > 1
            else 0.0
        )
        high_disagreement = len(values) > 1 and spread > MAX_REPLICATE_RANGE
        if high_disagreement:
            ledger["flagged_high_disagreement"] += 1

        years = [
            int(r["document_year"])
            for r in rows
            if str(r.get("document_year") or "").strip().isdigit()
        ]
        std = rows[0]["_std"]
        assay_type, bao_label, variant = context.split("|", 2)

        table.append(
            {
                "standardized_smiles": smiles,
                "connectivity_key": std.connectivity_key,
                "inchikey": std.inchikey,
                "murcko_scaffold": std.murcko_scaffold or "",
                "molecule_chembl_id": sorted({r["molecule_chembl_id"] for r in rows})[0],
                "target_chembl_id": target_id,
                "standard_type": standard_type,
                "assay_type": assay_type,
                "bao_label": bao_label,
                "assay_variant_mutation": variant,
                "assay_context_key": context,
                "task_key": f"{target_id}|{standard_type}|{context}",
                "pactivity": round(statistics.median(values), 4),
                "n_measurements": len(values),
                "pactivity_min": round(min(values), 4),
                "pactivity_max": round(max(values), 4),
                "pactivity_range": round(spread, 4),
                "pactivity_mad": round(mad, 4),
                "high_disagreement": int(high_disagreement),
                "document_year_min": min(years) if years else "",
                "document_year_max": max(years) if years else "",
                "n_heavy_atoms": std.num_heavy_atoms,
                "mol_weight": std.mol_weight,
                "assay_chembl_ids": ";".join(
                    sorted({r["assay_chembl_id"] for r in rows if r.get("assay_chembl_id")})
                )[:500],
                "n_source_records": len(rows),
            }
        )

    ledger["aggregated_rows"] = len(table)
    ledger["aggregated_rows_hq"] = sum(1 for r in table if not r["high_disagreement"])
    ledger["unique_compounds"] = len({r["connectivity_key"] for r in table})

    table.sort(key=lambda r: (r["target_chembl_id"], r["standard_type"],
                              r["assay_context_key"], r["standardized_smiles"]))
    return table, dict(ledger)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty dataset to {path}")
    fields = list(rows[0].keys())
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=None,
        help="reuse a previously extracted raw dump instead of re-querying",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    panel = json.loads(args.panel.read_text())

    client = ChemblClient()
    status = client.assert_release()

    raw_path = args.raw_cache or (args.out / "raw_activities.json.gz")
    if raw_path.exists():
        LOG.info("reusing raw extract %s", raw_path)
        with gzip.open(raw_path, "rt") as handle:
            raw_rows = json.load(handle)
    else:
        raw_rows = extract_panel(client, panel, args.workers)
        with gzip.open(raw_path, "wt") as handle:
            json.dump(raw_rows, handle)
        LOG.info("wrote raw extract %s (%d rows)", raw_path, len(raw_rows))

    table, ledger = build_table(raw_rows)

    dataset_path = args.out / "hq_exact.csv.gz"
    write_csv_gz(dataset_path, table)
    LOG.info("wrote %s (%d rows)", dataset_path, len(table))

    manifest = {
        "dataset_id": DATASET_ID,
        "source": {
            "name": "ChEMBL",
            "release": status["chembl_db_version"],
            "release_date": status["chembl_release_date"],
            "doi": "10.6019/CHEMBL.database.37",
            "access": "REST API (www.ebi.ac.uk/chembl/api/data)",
            "api_activities_total": status["activities"],
        },
        "extraction": {
            "filter_definition_sha256": filter_definition_sha256(),
            "fields": list(ACTIVITY_FIELDS),
        },
        "chemistry": {"standardizer_version": STANDARDIZER_VERSION},
        "labels": {
            "primary": "pactivity",
            "scale": "pChEMBL (-log10 M)",
            "aggregation": "median over replicates within "
            "(parent, target, standard_type, assay_context)",
            "max_replicate_range": MAX_REPLICATE_RANGE,
        },
        "panel": panel["panel_id"],
        "filter_ledger": ledger,
        "artifacts": {
            "hq_exact.csv.gz": {
                "sha256": sha256_file(dataset_path),
                "bytes": dataset_path.stat().st_size,
                "rows": len(table),
            }
        },
        "request_stats": client.stats.as_dict(),
    }
    (args.out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nFilter ledger:")
    for key, value in ledger.items():
        print(f"  {key:44} {value:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
