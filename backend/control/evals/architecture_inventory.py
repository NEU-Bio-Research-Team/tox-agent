"""The generated architecture inventory: what the product *is*, from code.

Architecture documents explain why. They are bad at saying what is currently
true — the 2026-09-16 review found two architecture files differing only by
case, a configuration page naming a retired compose path, and a progress log
contradicting itself. This module answers "what is it now" from the code the
product actually composes itself from, and ``docs/architecture-inventory.json``
is its checked-in output. ``tests/unit/test_architecture_inventory.py`` fails
when the two differ, so a new intent, flag, profile, tool, provider, queue or
eval pack is a reviewed diff to that file rather than a silent change.

The output is deterministic: sorted, with no timestamp.

    python -m evals.architecture_inventory           # print
    python -m evals.architecture_inventory --write   # regenerate the checked-in file
    python -m evals.architecture_inventory --check   # exit 1 on drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
INVENTORY_PATH = REPO_ROOT / "docs" / "architecture-inventory.json"
SCHEMA_VERSION = "architecture-inventory-v1"

#: Retired by ADR 0010; still deserialisable, never produced for new requests.
HISTORICAL_INTENTS = ("attribution", "evidence_research", "report_qa")


def build_inventory() -> dict[str, Any]:
    from toxagent import flags as rollout
    from toxagent.api import effective_product
    from toxagent.application import run_budget
    from toxagent.application.queues import PRIORITY, QueueClass, queue_for_intent
    from toxagent.domain.run import Intent
    from toxagent.tools.registry import PROFILES, ToolRegistry

    from evals import manifest, task_packs, trace
    from evals.graders import GRADER_REGISTRY

    registry = ToolRegistry()
    live = [i.value for i in Intent if i.value not in HISTORICAL_INTENTS]
    discovery = task_packs.discover(
        tuple(n for n, m in task_packs.load_pack_manifests().items() if not m.external)
    )
    packs = task_packs.load_pack_manifests()

    schemas = {
        "effective_product": effective_product.SCHEMA_VERSION,
        "effective_run_budget": run_budget.SCHEMA_VERSION,
        "eval_manifest": manifest.SCHEMA_VERSION,
        "eval_trace": trace.SCHEMA_VERSION,
        "eval_tasks": sorted(task_packs.SCHEMAS),
    }
    try:
        from toxagent.domain import decision_state

        schemas["decision_support_state"] = decision_state.SCHEMA_VERSION
    except ImportError:  # pragma: no cover - present from Wave 2 on
        pass
    try:
        from toxagent.domain import scientific_case

        schemas["scientific_case"] = scientific_case.SCHEMA_VERSION
        schemas["decision_dossier"] = scientific_case.DOSSIER_SCHEMA_VERSION
    except ImportError:  # pragma: no cover
        pass
    try:
        from toxagent.domain import evidence_ontology

        schemas["evidence_ontology"] = evidence_ontology.ONTOLOGY_VERSION
    except ImportError:  # pragma: no cover
        pass
    try:
        from toxagent.tools import trust

        schemas["trust_envelope"] = trust.SCHEMA_VERSION
    except ImportError:  # pragma: no cover
        pass

    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_source_roots": ["backend/control", "backend/ocr", "backend/predictor",
                                   "devops", "docs", "frontend"],
        "live_runtime_path": "AgentRuntimeGateway",
        "intents": {
            intent: {
                "default_lane": effective_product.intent_lane(
                    intent, report_orchestrator_v2=rollout.flag("report_orchestrator_v2").default
                ),
                "lane_with_report_orchestrator_v2": effective_product.intent_lane(
                    intent, report_orchestrator_v2=True
                ),
                "capability_profile": (
                    None if intent in effective_product._DETERMINISTIC_INTENTS
                    else registry.profile_for_intent(intent)
                ),
                "queue": queue_for_intent(intent).value,
            }
            for intent in sorted(live)
        },
        "historical_intents": list(HISTORICAL_INTENTS),
        "rollout_flags": {
            f.name: {"default": f.default, "owner": f.owner, "remove_by": f.remove_by.isoformat()}
            for f in sorted(rollout.FLAGS, key=lambda f: f.name)
        },
        "capability_profiles": {name: sorted(tools) for name, tools in sorted(PROFILES.items())},
        "queues": {queue.value: PRIORITY[queue] for queue in QueueClass},
        "runtime_kinds": ["dsh", "opencode", "scripted"],
        "providers": {"research": ["europepmc"], "compound": ["pubchem"], "ocr": ["molscribe"]},
        "eval_packs": {
            name: {
                "owner": m.owner,
                "external": m.external,
                "suites": list(m.suites),
                "tasks": (
                    None if m.external else discovery.packs[name].loaded
                ),
            }
            for name, m in sorted(packs.items())
        },
        "graders": {n: {"version": s.version, "kind": s.kind} for n, s in sorted(GRADER_REGISTRY.items())},
        "schemas": schemas,
    }


def render(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(build_inventory())
    if args.write:
        INVENTORY_PATH.write_text(text)
        return 0
    if args.check:
        current = INVENTORY_PATH.read_text() if INVENTORY_PATH.exists() else ""
        if current != text:
            print(
                f"{INVENTORY_PATH.relative_to(REPO_ROOT)} is stale; run "
                "`python -m evals.architecture_inventory --write` and review the diff",
                file=sys.stderr,
            )
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
