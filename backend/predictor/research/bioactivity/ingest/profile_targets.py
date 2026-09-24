"""P0: profile every human single-protein target, then select the panel by rule.

The panel must be frozen *before* any model is trained, and selected by criteria
that never look at a model score -- otherwise the benchmark reports the result of
a search over targets rather than the result of a method. This script therefore
runs in two passes:

  Pass 1  count HQ-Exact records for all human SINGLE PROTEIN targets
  Pass 2  break the survivors down per (target, standard_type) task

It writes the full profile for every target it examined, not just the winners,
so the selection can be audited and re-derived.

Usage:
    python -m bioactivity.ingest.profile_targets --out <dir> [--workers 16]
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow direct `python profile_targets.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bioactivity.ingest.chembl_client import (  # noqa: E402
    ALLOWED_STANDARD_TYPES,
    ChemblClient,
    filter_definition_sha256,
    hq_exact_activity_params,
)
from bioactivity.ingest.target_annotation import annotate_targets  # noqa: E402

LOG = logging.getLogger("profile_targets")

#: Coarse gate for pass 2. Deliberately far below the real task floor so that
#: pass 2 sees every target that could plausibly qualify.
COARSE_MIN_RECORDS = 400


def fetch_target_universe(client: ChemblClient) -> list[dict[str, Any]]:
    """All human single-protein targets, annotated with family and gene symbol."""
    LOG.info("fetching human SINGLE PROTEIN target universe")
    records = client.fetch_all(
        "target",
        {"target_type": "SINGLE PROTEIN", "organism": "Homo sapiens"},
        fields=["target_chembl_id", "pref_name", "target_components"],
    )
    annotations = annotate_targets(client, records)

    universe = []
    for rec in records:
        annotation = annotations[rec["target_chembl_id"]]
        universe.append(
            {
                "target_chembl_id": annotation.target_chembl_id,
                "pref_name": annotation.pref_name,
                "gene_symbol": annotation.gene_symbol,
                "protein_family": annotation.protein_family,
                "protein_class_path": annotation.protein_class_path,
                "uniprot": annotation.uniprot,
            }
        )
    LOG.info("target universe: %d targets", len(universe))
    return universe


def count_target(client: ChemblClient, target_id: str) -> int:
    return client.count("activity", hq_exact_activity_params(target_chembl_id=target_id))


def count_task(client: ChemblClient, target_id: str, standard_type: str) -> int:
    return client.count(
        "activity",
        hq_exact_activity_params(target_chembl_id=target_id, standard_type=standard_type),
    )


def _run_parallel(fn, items: list[Any], workers: int, label: str) -> list[Any]:
    results: list[Any] = []
    done = 0
    started = time.time()
    with futures.ThreadPoolExecutor(workers) as pool:
        for result in pool.map(fn, items):
            results.append(result)
            done += 1
            if done % 250 == 0 or done == len(items):
                rate = done / max(time.time() - started, 1e-9)
                LOG.info("%s %d/%d (%.1f q/s)", label, done, len(items), rate)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--coarse-min",
        type=int,
        default=COARSE_MIN_RECORDS,
        help="pass-1 record floor for entering the per-task breakdown",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    client = ChemblClient()

    status = client.assert_release()
    LOG.info(
        "ChEMBL release pin OK: %s (%s), %s activities",
        status["chembl_db_version"],
        status["chembl_release_date"],
        f"{status['activities']:,}",
    )

    universe = fetch_target_universe(client)

    # -- pass 1: one count per target ------------------------------------
    LOG.info("pass 1: counting HQ-Exact records for %d targets", len(universe))
    ids = [t["target_chembl_id"] for t in universe]
    counts = _run_parallel(
        lambda tid: (tid, count_target(client, tid)), ids, args.workers, "pass 1"
    )
    total_by_target = dict(counts)
    for target in universe:
        target["hq_records"] = total_by_target.get(target["target_chembl_id"], 0)

    survivors = [t for t in universe if t["hq_records"] >= args.coarse_min]
    LOG.info(
        "pass 1 done: %d/%d targets have >=%d HQ-Exact records",
        len(survivors),
        len(universe),
        args.coarse_min,
    )

    # -- pass 2: per (target, standard_type) breakdown --------------------
    jobs = [(t["target_chembl_id"], st) for t in survivors for st in ALLOWED_STANDARD_TYPES]
    LOG.info("pass 2: %d (target, standard_type) task counts", len(jobs))
    task_counts = _run_parallel(
        lambda job: (job[0], job[1], count_task(client, job[0], job[1])),
        jobs,
        args.workers,
        "pass 2",
    )

    by_target: dict[str, dict[str, int]] = {}
    for target_id, standard_type, n in task_counts:
        by_target.setdefault(target_id, {})[standard_type] = n

    profile_path = args.out / "target_profile.csv"
    with open(profile_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_chembl_id",
                "pref_name",
                "gene_symbol",
                "protein_family",
                "protein_class_path",
                "uniprot",
                "hq_records",
                *[f"n_{st}" for st in ALLOWED_STANDARD_TYPES],
            ]
        )
        for target in sorted(universe, key=lambda t: -t["hq_records"]):
            per_type = by_target.get(target["target_chembl_id"], {})
            writer.writerow(
                [
                    target["target_chembl_id"],
                    target["pref_name"],
                    target["gene_symbol"],
                    target["protein_family"],
                    target["protein_class_path"],
                    target["uniprot"],
                    target["hq_records"],
                    *[per_type.get(st, "") for st in ALLOWED_STANDARD_TYPES],
                ]
            )
    LOG.info("wrote %s (%d rows)", profile_path, len(universe))

    summary = {
        "release": status["chembl_db_version"],
        "release_date": status["chembl_release_date"],
        "api_activities_total": status["activities"],
        "filter_definition_sha256": filter_definition_sha256(),
        "target_universe": len(universe),
        "coarse_min_records": args.coarse_min,
        "targets_profiled_per_task": len(survivors),
        "request_stats": client.stats.as_dict(),
    }
    (args.out / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOG.info("summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
