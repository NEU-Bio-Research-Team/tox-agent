"""One entry point for the agent benchmark (master plan §17.2).

    python scripts/agent_benchmark.py preflight --case-set <dir> [--arm ID=URL ...]
    python scripts/agent_benchmark.py run     --study <id> --case-set <dir> --arm ID=URL ...
    python scripts/agent_benchmark.py packet  --study <id> --case-set <dir> --packet-id <id>
    python scripts/agent_benchmark.py score   --study <id> --case-set <dir> --packet-id <id> \\
        --grades a.csv b.csv [--compare D2_investigator:B2_fixed_workflow]
    python scripts/agent_benchmark.py freeze  --study <id> --case-set <dir>

It wraps the ``evals.investigation`` modules; there is no second runner, no
second scorer and no second definition of fairness. Each subcommand prints
JSON and returns non-zero when what it checked does not hold.

**preflight** is the gate before a study runs. It audits the case set (every
schema, hash, artifact, span, pairing, polarity and gold-isolation rule),
confirms the set's pinned replay and evidence files are the bytes it names,
and -- for each arm whose deployment URL is given -- that the deployment is
that arm (flags, skills mode, budget preset, release policy), that it replays
the set's own files, and that the arms differ in nothing but their declared
treatments. It never writes to a study.

**run** executes the arms given, one deployment at a time, and is safe to
invoke again: a (case, arm, trial) already attempted is not attempted again,
and the study refuses an arm whose deployment disagrees with what the study
began with.

**freeze** writes the repro bundle (§11.14): the study manifest, every arm's
effective product, the case-set pins, the scorecards and the commands, with
the sha256 of each file, so the study can be checked later or re-run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:  # the evals package lives beside src/
    sys.path.insert(0, str(HERE.parent))

from evals.investigation import run as runner  # noqa: E402
from evals.investigation.case_inputs import (  # noqa: E402
    audit_set,
    load_case,
    load_case_set,
    stores_from_files,
)
from evals.investigation.record import StudyStore  # noqa: E402
from evals.investigation.systems import SYSTEMS, fairness_problems, product_mismatch  # noqa: E402

BUNDLE_SCHEMA = "agent-benchmark-freeze-v1"


def _arms(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        system_id, separator, url = value.partition("=")
        if not separator or system_id not in SYSTEMS:
            raise SystemExit(f"--arm expects <system_id>=<base url>, got {value!r}")
        out[system_id] = url
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case_set(case_set_dir: Path) -> dict[str, Any]:
    """Every rule the set and its cases can break, with the gold read."""
    case_set = load_case_set(case_set_dir)
    loaded = [load_case(c.path, with_gold=True) for c in case_set.cases]
    stores = stores_from_files(
        replay_path=case_set.replay_path, snapshot_path=case_set.evidence_path)
    findings = audit_set(loaded, **stores)
    groups = sorted({str(c.case["task_group"]) for c in loaded})
    contexts = sorted({str(c.case.get("workflow_context")) for c in loaded})
    return {
        "case_set": case_set.pins(), "cases": len(loaded),
        "task_groups": groups, "workflow_contexts": contexts,
        "gold_status": sorted({str((c.gold or {}).get("status")) for c in loaded}),
        "findings": [{"case_id": f.case_id, "code": f.code, "detail": f.detail}
                     for f in findings],
    }


async def _products(arms: dict[str, str], token: str) -> dict[str, dict[str, Any]]:
    from evals.investigation.adapters.toxagent_api import ToxAgentAPI

    products: dict[str, dict[str, Any]] = {}
    for system_id, url in sorted(arms.items()):
        async with ToxAgentAPI(url, token) as api:
            products[system_id] = await api.effective_product()
    return products


def preflight(*, case_set_dir: Path, arms: dict[str, str], token: str) -> dict[str, Any]:
    audit = audit_case_set(case_set_dir)
    case_set = load_case_set(case_set_dir)
    report: dict[str, Any] = {"check": "preflight", **audit, "arms": {}, "problems": []}
    report["problems"] += [f"case set: {f['code']} ({f['case_id']})" for f in audit["findings"]]
    if not arms:
        report["arms_note"] = "no deployment URL given; the arms were not checked"
        report["ok"] = not report["problems"]
        return report
    products = asyncio.run(_products(arms, token))
    for system_id, product in sorted(products.items()):
        mismatch = product_mismatch(SYSTEMS[system_id], product)
        report["arms"][system_id] = {
            "effective_product_hash": product.get("effective_product_hash"),
            "model_id": (product.get("runtime") or {}).get("model_id"),
            "budget_preset": (((product.get("intents") or {}).get("decision_support") or {})
                              .get("budget") or {}).get("preset"),
            "release_policy": (product.get("contracts") or {}).get("release_policy"),
            "replay_sha256": ((product.get("providers") or {}).get("activity_replay")
                              or {}).get("sha256"),
            "evidence_sha256": ((product.get("providers") or {}).get("research_snapshot")
                                or {}).get("sha256"),
            "not_this_arm": mismatch,
        }
        report["problems"] += [f"{system_id}: {problem}" for problem in mismatch]
    report["problems"] += runner.deployment_pin_problems(case_set, products)
    report["problems"] += fairness_problems(
        [SYSTEMS[system_id] for system_id in sorted(products)], products)
    report["ok"] = not report["problems"]
    return report


def freeze(*, study_dir: Path, case_set_dir: Path) -> dict[str, Any]:
    store = StudyStore(study_dir)
    manifest = store.manifest()
    if not manifest.get("protocol_v2"):
        raise SystemExit(f"{study_dir} is not a protocol v2 study")
    case_set = load_case_set(case_set_dir)
    files: dict[str, str] = {}
    for path in sorted(study_dir.rglob("*")):
        if path.is_file() and "packets" not in path.parts:
            files[str(path.relative_to(study_dir))] = _sha256(path)
    bundle = {
        "schema_version": BUNDLE_SCHEMA, "study_id": manifest.get("study_id"),
        "runner_version": manifest.get("runner_version"),
        "turn_rendering": (manifest["protocol_v2"]).get("turn_rendering"),
        "order_seed": (manifest["protocol_v2"]).get("order_seed"),
        "arms": {system_id: {k: arm.get(k) for k in
                             ("rethink_arm", "adapter", "required_flags", "skills_mode",
                              "budget_preset", "required_product", "treatment_fields",
                              "effective_product_hash")}
                 for system_id, arm in (manifest["protocol_v2"].get("arms") or {}).items()},
        "case_set": case_set.pins(),
        "cases": manifest["protocol_v2"].get("cases"),
        "environment": manifest.get("environment"),
        "denominators": manifest.get("denominators"),
        "files_sha256": files,
        "commands": [
            f"python scripts/agent_benchmark.py preflight --case-set {case_set_dir}",
            f"python scripts/agent_benchmark.py run --study {manifest.get('study_id')} "
            f"--case-set {case_set_dir} --arm <ID>=<URL>",
            f"python scripts/agent_benchmark.py packet --study {manifest.get('study_id')} "
            f"--case-set {case_set_dir} --packet-id <ID>",
            f"python scripts/agent_benchmark.py score --study {manifest.get('study_id')} "
            f"--case-set {case_set_dir} --packet-id <ID> --grades <CSV>",
        ],
        "note": "Packets and unblinding keys are deliberately left out of this bundle: a "
                "packet is sent to graders and a key must not travel with it.",
    }
    path = study_dir / "freeze-bundle.json"
    path.write_text(json.dumps(bundle, indent=2, default=str) + "\n", encoding="utf-8")
    return {"bundle": str(path), "files": len(files),
            "arms": sorted(bundle["arms"]), "study_id": bundle["study_id"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, *, study: bool = True) -> None:
        if study:
            p.add_argument("--study", required=True)
            p.add_argument("--root", type=Path, default=runner.DEFAULT_ROOT)
        p.add_argument("--case-set", type=Path, required=True)

    check = sub.add_parser("preflight", help="audit the case set and the deployments")
    common(check, study=False)
    check.add_argument("--arm", action="append", default=[], metavar="SYSTEM=URL")
    check.add_argument("--token", default="")

    go = sub.add_parser("run", help="run the arms on the case set")
    common(go)
    go.add_argument("--arm", action="append", default=[], metavar="SYSTEM=URL")
    go.add_argument("--systems", default="", help="comma-separated; default: every --arm plus A")
    go.add_argument("--cases", default="")
    go.add_argument("--trials", type=int, default=1)
    go.add_argument("--token", default="")
    go.add_argument("--order-seed", type=int, default=runner.DEFAULT_ORDER_SEED)
    go.add_argument("--parallel", type=int, default=1)
    go.add_argument("--rerun-errors", action="store_true",
                    help="add a diagnostic rerun of each failed primary; never replaces it")

    pack = sub.add_parser("packet", help="build the blinded grading packet")
    common(pack)
    pack.add_argument("--packet-id", required=True)
    pack.add_argument("--seed", type=int, default=20261022)
    pack.add_argument("--systems", default="")

    card = sub.add_parser("score", help="score a packet's grades")
    common(card)
    card.add_argument("--packet-id", required=True)
    card.add_argument("--grades", type=Path, nargs="+", required=True)
    card.add_argument("--compare", action="append", default=[], metavar="SYSTEM:BASELINE")
    card.add_argument("--reps", type=int, default=2000)

    bundle = sub.add_parser("freeze", help="write the repro bundle")
    common(bundle)

    args = parser.parse_args(argv)

    if args.command == "preflight":
        report = preflight(case_set_dir=args.case_set, arms=_arms(args.arm), token=args.token)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if args.command == "run":
        arms = _arms(args.arm)
        systems = [s for s in args.systems.split(",") if s] or [*sorted(arms), "A2_template"]
        checked = preflight(case_set_dir=args.case_set, arms=arms, token=args.token)
        if not checked["ok"]:
            print(json.dumps(checked, indent=2))
            return 1
        manifest = asyncio.run(runner.run_protocol_study(
            study_id=args.study, root=args.root, case_set_dir=args.case_set,
            system_ids=systems, case_ids=[c for c in args.cases.split(",") if c] or None,
            trials=args.trials, toxagent_urls=arms, token=args.token,
            order_seed=args.order_seed, rerun_errors=args.rerun_errors,
            parallel=args.parallel,
        ))
        print(json.dumps(manifest["denominators"], indent=2))
        return 0

    if args.command == "packet":
        from evals.investigation.packet_v2 import build_packet_v2

        result = build_packet_v2(
            study_dir=args.root / args.study, case_set_dir=args.case_set,
            packet_id=args.packet_id, seed=args.seed,
            systems=[s for s in args.systems.split(",") if s] or None)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "score":
        from evals.investigation.scorecard_v2 import render_markdown, score_study, write_card

        study_dir = args.root / args.study
        result = score_study(
            study_dir=study_dir, case_set_dir=args.case_set, packet_id=args.packet_id,
            grades=args.grades, reps=args.reps,
            compare=[tuple(item.split(":", 1)) for item in args.compare])
        written = write_card(result, study_dir / "scorecards" / args.packet_id)
        print(render_markdown(result))
        print(json.dumps(written, indent=2))
        return 0

    result = freeze(study_dir=args.root / args.study, case_set_dir=args.case_set)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
