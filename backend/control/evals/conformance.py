"""Control-Plane Conformance suite (review P2-02): a report, not a score.

Worker topology, leases, fencing, cancellation and backpressure are properties
of the control plane, not of the agent, and they do not belong in a semantic
agent score. This module maps each drill the review requires to the tests that
exercise it, runs them, and writes ``control-plane-conformance-v1``:

* ``pass`` / ``fail`` — the mapped tests ran and their outcome;
* ``not_covered`` — no in-repository test exercises the drill; it needs a
  multi-process or multi-replica environment (``requires`` says which), and it
  blocks an external-worker cutover until that drill has been run there.

    python -m evals.conformance --out conformance.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE_ROOT = HERE.parent
SCHEMA_VERSION = "control-plane-conformance-v1"

W = "tests/integration/test_external_workers.py"
L = "tests/integration/test_run_lease_ownership.py"
Q = "tests/integration/test_worker_quotas_and_drain.py"

DRILLS: dict[str, dict[str, object]] = {
    "api_restart_does_not_cancel_active_run": {
        "tests": [f"{L}::test_startup_reconciliation_leaves_a_run_under_a_live_lease_alone",
                  f"{L}::test_a_fresh_worker_executes_a_run_left_by_a_dead_one"],
    },
    "lease_expiry_adoption_without_duplicate_side_effect": {
        "tests": [f"{L}::test_an_expired_lease_is_claimed_by_exactly_one_of_two_workers",
                  f"{W}::test_two_workers_sweeping_at_once_execute_each_run_exactly_once",
                  f"{L}::test_a_fenced_worker_stops_executing_the_run_it_lost"],
    },
    "report_burst_does_not_starve_interactive_queue": {
        "tests": [f"{W}::test_a_saturated_report_worker_does_not_hold_up_a_question"],
    },
    "cancellation_across_processes": {
        "tests": [f"{L}::test_a_cancellation_through_another_replica_reaches_the_owning_worker",
                  f"{W}::test_cancelling_a_queued_job_settles_it_without_a_worker"],
    },
    "concurrency_caps_and_backpressure": {
        "tests": [f"{Q}::test_a_global_cap_holds_across_three_workers",
                  f"{Q}::test_a_tenant_cap_keeps_one_owner_from_occupying_the_fleet",
                  f"{Q}::test_racing_for_the_last_slots_grants_exactly_the_limit",
                  f"{Q}::test_a_deferred_job_reports_why_and_when_it_can_run"],
    },
    "drain_hands_off_instead_of_cancelling": {
        "tests": [f"{Q}::test_a_draining_worker_hands_its_run_on_instead_of_cancelling_it",
                  f"{Q}::test_a_dead_workers_lease_and_slot_are_reclaimed_by_the_next"],
    },
    "poisoned_job_is_bounded": {
        "tests": [f"{Q}::test_recovery_is_bounded_to_one_generation"],
    },
    "provider_rate_limit_backpressure_live": {
        "tests": [], "requires": "live provider or snapshot fault under concurrent load",
    },
    "rolling_deploy_with_mixed_worker_versions": {
        "tests": [], "requires": "two image versions against one PostgreSQL",
    },
    "api_and_workers_as_separate_os_processes": {
        "tests": [f"{W}::test_an_api_process_and_a_worker_process_over_one_database"],
        "requires": "the in-repo test shares one event loop; a real drill kills OS processes",
    },
}


def run(out: Path | None = None) -> dict[str, object]:
    results: dict[str, object] = {}
    for name, drill in DRILLS.items():
        tests = list(drill["tests"])  # type: ignore[arg-type]
        if not tests:
            results[name] = {"status": "not_covered", "requires": drill.get("requires")}
            continue
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", *tests],
            cwd=SERVICE_ROOT, capture_output=True, text=True,
        )
        results[name] = {
            "status": "pass" if completed.returncode == 0 else "fail",
            "tests": tests,
            **({"requires": drill["requires"]} if drill.get("requires") else {}),
            **({"output_tail": completed.stdout.splitlines()[-5:]} if completed.returncode else {}),
        }
    statuses = [r["status"] for r in results.values()]  # type: ignore[index]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drills": results,
        "summary": {s: statuses.count(s) for s in ("pass", "fail", "not_covered")},
        "external_worker_cutover_supported": all(s == "pass" for s in statuses),
    }
    if out is not None:
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(args.out)
    print(json.dumps(report["summary"]))
    return 1 if report["summary"]["fail"] else 0  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
