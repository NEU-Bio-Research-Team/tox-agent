"""Run the comparison study: every case through every chosen system, all logged.

    python -m evals.investigation.run --study pilot-2026-09 \\
        --systems A_predictor_template,C_toxagent_current,D_toxagent_investigator,P_openai_bare \\
        --toxagent C_toxagent_current=http://127.0.0.1:8011 \\
        --toxagent D_toxagent_investigator=http://127.0.0.1:8012 \\
        --snapshot-from http://127.0.0.1:8011 --trials 1

Each ToxAgent arm is a deployment with its own flags, so each gets its own base
URL; the adapter checks the deployment's effective product before recording
anything under the arm's name. The predictor snapshot the template and the
``with_snapshot`` arms use is computed once per case, stored under
``snapshots/``, and reused, so those arms see byte-identical predictor facts.

Re-running is safe: a (case, system, trial) with an ``ok`` record is skipped
(``--rerun-errors`` retries failed ones; pending manual ones are always
re-checked). Records are appended, never rewritten.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from evals.investigation import cases as case_module
from evals.investigation import prompts
from evals.investigation.adapters import AdapterResult
from evals.investigation.adapters.manual import ManualAdapter
from evals.investigation.adapters.platform_cli import ClaudeCLIAdapter, CodexCLIAdapter, GeminiMCPAdapter
from evals.investigation.adapters.predictor_template import PredictorTemplateAdapter
from evals.investigation.adapters.toxagent import ToxAgentAdapter
from evals.investigation.adapters.toxagent_api import ToxAgentAPI, snapshot_from_analysis
from evals.investigation.record import (
    STATUS_ERROR, STATUS_OK, STATUS_PENDING, MANIFEST_SCHEMA, RunRecord, StudyStore, environment,
    now_iso, summarise,
)
from evals.investigation.systems import fairness_problems, SYSTEMS, SystemSpec, resolve

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "runs"
RUNNER_VERSION = "investigation-runner-v1"


async def compute_snapshots(
    cases: list[dict[str, Any]], store: StudyStore, *, base_url: str | None, token: str,
    transport=None,
) -> dict[str, dict[str, Any]]:
    """One predictor snapshot per case, computed once and reused on every re-run."""
    directory = store.root / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, dict[str, Any]] = {}
    missing = [c for c in cases if not (directory / f"{c['case_id']}.json").exists()]
    if missing and base_url is None:
        raise SystemExit("a snapshot is needed but no --snapshot-from deployment was given")
    if missing:
        async with ToxAgentAPI(base_url, token, transport=transport) as api:
            product = await api.effective_product()
            for case in missing:
                session_id = await api.new_session()
                result = await api.analyse(session_id, case["compound"]["smiles"])
                snapshot = snapshot_from_analysis(result["analysis"])
                (directory / f"{case['case_id']}.json").write_text(json.dumps({
                    "case_id": case["case_id"], "computed_at": now_iso(),
                    "deployment_effective_product_hash": product.get("effective_product_hash"),
                    "analysis_run_status": result["run"].get("status"),
                    "snapshot": snapshot,
                }, indent=2, ensure_ascii=False) + "\n")
    for case in cases:
        path = directory / f"{case['case_id']}.json"
        if path.exists():
            snapshots[case["case_id"]] = json.loads(path.read_text())["snapshot"]
    return snapshots


def build_adapters(specs: list[SystemSpec], *, toxagent_urls: dict[str, str], token: str,
                   claude_model: str | None, transport=None, google_channel: str = "mcp",
                   gemini_mcp_command: list[str] | None = None,
                   gemini_mcp_env: dict[str, str] | None = None,
                   gemini_model: str | None = None) -> dict[str, Any]:
    adapters: dict[str, Any] = {}
    for spec in specs:
        if spec.adapter == "toxagent":
            if spec.system_id not in toxagent_urls:
                raise SystemExit(f"{spec.system_id} needs --toxagent {spec.system_id}=<base url>")
            adapters[spec.system_id] = ToxAgentAdapter(toxagent_urls[spec.system_id], token,
                                                       transport=transport)
        elif spec.adapter == "predictor_template":
            adapters[spec.system_id] = PredictorTemplateAdapter()
        elif spec.adapter == "codex_cli":
            adapters[spec.system_id] = CodexCLIAdapter()
        elif spec.adapter == "claude_cli":
            adapters[spec.system_id] = ClaudeCLIAdapter(model=claude_model)
        elif spec.adapter == "google" and google_channel == "manual":
            adapters[spec.system_id] = ManualAdapter()
        elif spec.adapter == "google":
            if not gemini_mcp_command:
                raise SystemExit(f"{spec.system_id} needs --gemini-mcp-command (or --google-channel manual)")
            adapters[spec.system_id] = GeminiMCPAdapter(gemini_mcp_command, gemini_mcp_env,
                                                        model=gemini_model)
        elif spec.adapter == "manual":
            adapters[spec.system_id] = ManualAdapter()
        else:  # pragma: no cover - SYSTEMS is closed
            raise SystemExit(f"no adapter {spec.adapter}")
    return adapters


async def run_study(
    *, study_id: str, root: Path, system_ids: list[str], case_ids: list[str] | None,
    trials: int, toxagent_urls: dict[str, str], snapshot_from: str | None, token: str,
    claude_model: str | None = None, rerun_errors: bool = False, transport=None,
    adapters: dict[str, Any] | None = None, cases_dir: Path = case_module.CASES_DIR,
    parallel: int = 1, google: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases = case_module.load_cases(cases_dir, only=case_ids)
    specs = resolve(system_ids)
    store = StudyStore(root / study_id)
    adapters = adapters or build_adapters(specs, toxagent_urls=toxagent_urls, token=token,
                                          claude_model=claude_model, transport=transport,
                                          **(google or {}))
    products: dict[str, Any] = {}
    for system_id, adapter in adapters.items():
        if hasattr(adapter, "prepare"):
            await adapter.prepare()
        if hasattr(adapter, "effective_product"):
            products[system_id] = await adapter.effective_product()
    # #50: arms of one protocol run on the same backbone, budget and pinned
    # sources, or their difference is not the variable under study.
    unfair = fairness_problems(specs, products)
    if unfair:
        raise ValueError("the arms are not comparable: " + "; ".join(unfair))

    needs_snapshot = any(s.uses_snapshot or s.adapter == "predictor_template" for s in specs)
    snapshots = (
        await compute_snapshots(cases, store, base_url=snapshot_from or next(iter(toxagent_urls.values()), None),
                                token=token, transport=transport)
        if needs_snapshot else {}
    )

    new_systems = {s.system_id: {**s.to_dict(), "adapter_config": adapters[s.system_id].describe()}
                   for s in specs}

    def begin(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            **manifest,
            "schema_version": MANIFEST_SCHEMA,
            "study_id": study_id,
            "created_at": manifest.get("created_at") or now_iso(),
            "updated_at": now_iso(),
            "runner_version": RUNNER_VERSION,
            "case_set": {"set_id": cases[0]["set_id"] if cases else None,
                         "sha256": case_module.case_set_sha256(cases),
                         "cases": {c["case_id"]: case_module.sha256(c) for c in cases}},
            "systems": {**manifest.get("systems", {}), **new_systems},
            "platform_prompt": {"version": prompts.PROMPT_VERSION, "preamble": prompts.PREAMBLE,
                                "preamble_sha256": prompts.preamble_sha256(),
                                "snapshot_header": prompts.SNAPSHOT_HEADER},
            "trials": trials,
            "environment": environment(),
        }

    store.update_manifest(begin)

    latest = store.latest_records()
    work = []
    for case in cases:
        for spec in specs:
            for trial in range(1, trials + 1):
                previous = latest.get((case["case_id"], spec.system_id, trial))
                if previous and previous.status == STATUS_OK:
                    continue
                if previous and previous.status == STATUS_ERROR and not rerun_errors:
                    continue
                work.append((case, spec, trial))

    semaphore = asyncio.Semaphore(max(1, parallel))

    async def one(case: dict[str, Any], spec: SystemSpec, trial: int) -> RunRecord:
        async with semaphore:
            started = now_iso()
            try:
                result: AdapterResult = await adapters[spec.system_id].run(
                    case, spec, trial=trial, snapshot=snapshots.get(case["case_id"]), store=store,
                )
            except Exception as exc:  # noqa: BLE001 - one failure never stops the study
                result = AdapterResult(status=STATUS_ERROR, turns=[],
                                       error=f"{type(exc).__name__}: {exc}")
            record = RunRecord(
                study_id=study_id, case_id=case["case_id"], case_sha256=case_module.sha256(case),
                system_id=spec.system_id, arm=spec.to_dict(), trial=trial, status=result.status,
                started_at=started, ended_at=now_iso(), model=result.model, turns=result.turns,
                final_text=result.final_text, usage=result.usage, toxagent=result.toxagent,
                artifacts=result.artifacts, error=result.error,
            )
            store.append(record)
            print(f"[{record.status:7}] {spec.system_id:32} {case['case_id']:40} t{trial}"
                  + (f"  {record.error}" if record.error else ""), flush=True)
            return record

    await asyncio.gather(*(one(*item) for item in work))

    def finish(manifest: dict[str, Any]) -> dict[str, Any]:
        # Over every system the study has, not only this runner's: a second
        # runner on the same study must not shrink the denominators.
        summary = summarise(store.latest_records().values())
        systems = manifest.get("systems", {})
        return {**manifest, "updated_at": now_iso(), "denominators": {
            "cases": len(cases), "systems": len(systems), "trials": trials,
            "expected_records": len(cases) * len(systems) * trials, "by_system": summary,
        }}

    return store.update_manifest(finish)


# ---------------------------------------------------------------- protocol v2

PROTOCOL_V2 = "investigation-protocol-v2"
PROTOCOL_RUNNER_VERSION = "investigation-runner-v2"
#: The order arms run in per (case, trial) is shuffled with this seed unless
#: the study states another; it is recorded either way.
DEFAULT_ORDER_SEED = 20261022


class StudyManifestMismatch(ValueError):
    """A resumed study whose cases, pins, arms or deployments are not the ones
    it started with. A changed study is a new study id."""


def arm_order(case_ids: list[str], system_ids: list[str], trials: int,
              seed: int) -> list[tuple[str, str, int]]:
    """Every (case, system, trial), with the arms of each (case, trial) in a
    seeded random order: no arm always runs first against a cold cache or a
    fresh provider quota."""
    import random

    rng = random.Random(seed)
    order: list[tuple[str, str, int]] = []
    for case_id in case_ids:
        for trial in range(1, trials + 1):
            arms = list(system_ids)
            rng.shuffle(arms)
            order.extend((case_id, system_id, trial) for system_id in arms)
    return order


def build_protocol_adapters(specs: list[SystemSpec], *, case_set, toxagent_urls: dict[str, str],
                            token: str, transport=None) -> dict[str, Any]:
    from evals.investigation.adapters.template_v2 import TemplateV2Adapter
    from evals.investigation.adapters.toxagent_replay import ToxAgentReplayAdapter

    adapters: dict[str, Any] = {}
    for spec in specs:
        if spec.adapter == "template_v2":
            if case_set.replay_path is None:
                raise SystemExit(f"{spec.system_id} reads the set's replay file; the set pins none")
            adapters[spec.system_id] = TemplateV2Adapter(case_set.replay_path,
                                                         sha256=str(case_set.replay_sha256))
        elif spec.adapter == "toxagent":
            if spec.system_id not in toxagent_urls:
                raise SystemExit(f"{spec.system_id} needs --toxagent {spec.system_id}=<base url>")
            adapters[spec.system_id] = ToxAgentReplayAdapter(
                toxagent_urls[spec.system_id], token, transport=transport)
        else:
            raise SystemExit(f"{spec.system_id}: adapter {spec.adapter} does not run v3 cases")
    return adapters


def deployment_pin_problems(case_set, products: dict[str, dict[str, Any]]) -> list[str]:
    """A deployment that does not replay the set's own frozen files is not
    running this study's inputs, whatever its arm config says."""
    problems: list[str] = []
    for system_id, product in sorted(products.items()):
        providers = product.get("providers") or {}
        replay = (providers.get("activity_replay") or {}).get("sha256")
        if case_set.replay_sha256 and replay != case_set.replay_sha256:
            problems.append(f"{system_id} replays {replay!r}, the set pins "
                            f"{case_set.replay_sha256!r}")
        snapshot = (providers.get("research_snapshot") or {}).get("sha256")
        if case_set.evidence_sha256 and snapshot != case_set.evidence_sha256:
            problems.append(f"{system_id} reads evidence {snapshot!r}, the set pins "
                            f"{case_set.evidence_sha256!r}")
    return problems


def _case_fingerprints(case_set) -> dict[str, dict[str, str]]:
    from evals.investigation import turn_actions
    from evals.investigation.case_inputs import file_sha256

    return {
        loaded.case_id: {
            "case_sha256": file_sha256(loaded.path),
            "inputs_sha256": str((loaded.case.get("inputs_ref") or {}).get("sha256")),
            "rendered_turns_sha256": turn_actions.rendered_turns_sha256(loaded.inputs),
            "task_group": str(loaded.case.get("task_group")),
            "workflow_context": str(loaded.case.get("workflow_context")),
        }
        for loaded in case_set.cases
    }


def _resume_problems(manifest: dict[str, Any], fingerprint: dict[str, Any],
                     products: dict[str, dict[str, Any]]) -> list[str]:
    previous = manifest.get("protocol_v2")
    if not previous:
        return []
    problems: list[str] = []
    for key in ("protocol", "case_set", "turn_rendering", "order_seed", "trials"):
        if previous.get(key) != fingerprint.get(key):
            problems.append(f"{key} changed since the study began")
    for case_id, prints in fingerprint["cases"].items():
        before = (previous.get("cases") or {}).get(case_id)
        if before is not None and before != prints:
            problems.append(f"case {case_id} changed since the study began")
    for system_id, product in products.items():
        before = ((previous.get("arms") or {}).get(system_id) or {}).get("effective_product_hash")
        if before is not None and before != product.get("effective_product_hash"):
            problems.append(f"{system_id}'s deployment changed since the study began")
    return problems


async def run_protocol_study(
    *, study_id: str, root: Path, case_set_dir: Path, system_ids: list[str],
    case_ids: list[str] | None = None, trials: int = 1, toxagent_urls: dict[str, str] | None = None,
    token: str = "", transport=None, adapters: dict[str, Any] | None = None,
    order_seed: int = DEFAULT_ORDER_SEED, rerun_errors: bool = False, parallel: int = 1,
) -> dict[str, Any]:
    """Protocol v2 (#50): every arm on one case set's frozen inputs.

    Arms may be run in several invocations -- on a laptop, one deployment at
    a time -- and every invocation is held to the first: each arm's
    effective product is kept under ``products/`` and compared with every
    other arm already in the study, the case set and its pins are compared
    with what the study began with, and a (case, arm, trial) already
    attempted is never attempted again as its primary. ``rerun_errors`` adds
    diagnostic reruns of failed primaries; they are recorded and costed and
    never scored in the primary's place."""
    import uuid

    from evals.investigation import turn_actions
    from evals.investigation.case_inputs import load_case_set
    from evals.investigation.record import ATTEMPT_DIAGNOSTIC, ATTEMPT_PRIMARY

    case_set = load_case_set(case_set_dir, only=case_ids)
    specs = resolve(system_ids)
    stray = [s.system_id for s in specs if s.protocol != PROTOCOL_V2]
    if stray:
        raise ValueError(f"{stray} are not arms of {PROTOCOL_V2}")
    store = StudyStore(root / study_id)
    adapters = adapters or build_protocol_adapters(
        specs, case_set=case_set, toxagent_urls=toxagent_urls or {}, token=token,
        transport=transport)
    products: dict[str, dict[str, Any]] = {}
    for system_id, adapter in adapters.items():
        if hasattr(adapter, "effective_product"):
            products[system_id] = await adapter.effective_product()

    pinned = deployment_pin_problems(case_set, products)
    if pinned:
        raise ValueError("the deployments do not replay this case set: " + "; ".join(pinned))
    manifest = store.manifest()
    products_dir = store.root / "products"
    recorded: dict[str, dict[str, Any]] = {}
    for path in sorted(products_dir.glob("*.json")) if products_dir.exists() else ():
        if path.stem not in products:
            recorded[path.stem] = json.loads(path.read_text())
    every = {**recorded, **products}
    unfair = fairness_problems(
        [SYSTEMS[sid] for sid in sorted(every) if sid in SYSTEMS], every)
    if unfair:
        raise ValueError("the arms are not comparable: " + "; ".join(unfair))

    fingerprint = {
        "protocol": PROTOCOL_V2, "case_set": case_set.pins(),
        "turn_rendering": turn_actions.RENDERING_VERSION, "order_seed": order_seed,
        "trials": trials, "cases": _case_fingerprints(case_set),
    }
    stale = _resume_problems(manifest, fingerprint, products)
    if stale:
        raise StudyManifestMismatch("study_manifest_mismatch: " + "; ".join(stale)
                                    + " -- start a new study id")
    products_dir.mkdir(parents=True, exist_ok=True)
    for system_id, product in products.items():
        path = products_dir / f"{system_id}.json"
        if not path.exists():
            path.write_text(json.dumps(product, indent=2, sort_keys=True, default=str) + "\n")

    loaded = {c.case_id: c for c in case_set.cases}
    order = arm_order(sorted(loaded), [s.system_id for s in specs], trials, order_seed)

    def begin(previous: dict[str, Any]) -> dict[str, Any]:
        before = previous.get("protocol_v2") or {}
        arms = dict(before.get("arms") or {})
        for spec in specs:
            arms[spec.system_id] = {
                **spec.to_dict(), "adapter_config": adapters[spec.system_id].describe(),
                "effective_product_hash": (products.get(spec.system_id) or {})
                .get("effective_product_hash"),
            }
        cases = {**(before.get("cases") or {}), **fingerprint["cases"]}
        return {
            **previous, "schema_version": MANIFEST_SCHEMA, "study_id": study_id,
            "created_at": previous.get("created_at") or now_iso(), "updated_at": now_iso(),
            "runner_version": PROTOCOL_RUNNER_VERSION, "trials": trials,
            "case_set": {"set_id": case_set.set_id,
                         "sha256": case_module.sha256(
                             {k: v["case_sha256"] for k, v in sorted(cases.items())}),
                         "cases": {k: v["case_sha256"] for k, v in sorted(cases.items())}},
            "systems": {**previous.get("systems", {}),
                        **{sid: arms[sid] for sid in arms}},
            "protocol_v2": {
                **{k: v for k, v in fingerprint.items() if k != "cases"},
                "cases": cases, "arms": arms,
                "execution_order": [*before.get("execution_order", []),
                                    *[list(item) for item in order]],
                "parallel": parallel,
            },
            "environment": environment(),
        }

    store.update_manifest(begin)

    primaries = store.primary_records()
    work: list[tuple[str, str, int, str]] = []
    for case_id, system_id, trial in order:
        primary = primaries.get((case_id, system_id, trial))
        if primary is None:
            work.append((case_id, system_id, trial, ATTEMPT_PRIMARY))
        elif primary.status == STATUS_ERROR and rerun_errors:
            work.append((case_id, system_id, trial, ATTEMPT_DIAGNOSTIC))

    semaphore = asyncio.Semaphore(max(1, parallel))

    async def one(case_id: str, system_id: str, trial: int, kind: str) -> RunRecord:
        case = loaded[case_id]
        spec = SYSTEMS[system_id]
        attempt_id = f"{'p' if kind == ATTEMPT_PRIMARY else 'd'}-{uuid.uuid4().hex[:12]}"
        async with semaphore:
            started = now_iso()
            try:
                result: AdapterResult = await adapters[system_id].run(
                    case.case, spec, trial=trial, snapshot=None, store=store,
                    inputs=case.inputs, attempt_id=attempt_id,
                )
            except Exception as exc:  # noqa: BLE001 - one failure never stops the study
                result = AdapterResult(status=STATUS_ERROR, turns=[],
                                       error=f"{type(exc).__name__}: {exc}")
            record = RunRecord(
                study_id=study_id, case_id=case_id,
                case_sha256=fingerprint["cases"][case_id]["case_sha256"],
                system_id=system_id, arm=spec.to_dict(), trial=trial, status=result.status,
                started_at=started, ended_at=now_iso(), model=result.model, turns=result.turns,
                final_text=result.final_text, usage=result.usage, toxagent=result.toxagent,
                artifacts=result.artifacts, error=result.error, attempt_id=attempt_id,
                attempt_kind=kind,
                inputs_sha256=fingerprint["cases"][case_id]["inputs_sha256"],
            )
            store.append(record)
            print(f"[{record.status:7}] {system_id:22} {case_id:42} t{trial} {kind}"
                  + (f"  {record.error}" if record.error else ""), flush=True)
            return record

    if parallel <= 1:
        for item in work:
            await one(*item)
    else:
        await asyncio.gather(*(one(*item) for item in work))

    def finish(current: dict[str, Any]) -> dict[str, Any]:
        primary = store.primary_records()
        systems = (current.get("protocol_v2") or {}).get("arms") or {}
        cases = (current.get("protocol_v2") or {}).get("cases") or {}
        by_system: dict[str, dict[str, int]] = {}
        for system_id in systems:
            rows = [r for (c, s, t), r in primary.items() if s == system_id]
            by_system[system_id] = {
                "ok": sum(r.status == STATUS_OK for r in rows),
                "error": sum(r.status == STATUS_ERROR for r in rows),
                "not_run": len(cases) * trials - len(rows),
                "diagnostic_reruns": sum(
                    1 for r in store.records()
                    if r.system_id == system_id and r.attempt_kind == ATTEMPT_DIAGNOSTIC),
            }
        return {**current, "updated_at": now_iso(), "denominators": {
            "cases": len(cases), "systems": len(systems), "trials": trials,
            "expected_records": len(cases) * len(systems) * trials,
            "scored": "primary attempt per (case, system, trial)",
            "by_system": by_system,
        }}

    return store.update_manifest(finish)


def _pairs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        system_id, sep, url = value.partition("=")
        if not sep or system_id not in SYSTEMS:
            raise SystemExit(f"--toxagent expects <system_id>=<base url>, got {value!r}")
        out[system_id] = url
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--systems", required=True, help="comma-separated system ids")
    parser.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--cases-dir", type=Path, default=case_module.CASES_DIR)
    parser.add_argument("--toxagent", action="append", default=[], metavar="SYSTEM=URL")
    parser.add_argument("--snapshot-from", default=None)
    parser.add_argument("--token", default=os.environ.get("TOXAGENT_STUDY_TOKEN", ""))
    parser.add_argument("--claude-model", default="opus",
                        help="alias passed to claude --model; recorded with the model the CLI reports")
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--google-channel", choices=("mcp", "manual"), default="mcp")
    parser.add_argument("--gemini-mcp-command", default="",
                        help="the operator's Gemini MCP server command, e.g. 'python3 /path/server.py'")
    parser.add_argument("--gemini-mcp-env", action="append", default=[], metavar="KEY=VALUE",
                        help="environment for that server (never a credential; the server finds its own)")
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--case-set", type=Path, default=None,
                        help="protocol v2: a v3 case set directory (cases/, inputs/, a set "
                             "manifest); runs the investigation-protocol-v2 arms on its inputs")
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    args = parser.parse_args(argv)
    if args.case_set is not None:
        manifest = asyncio.run(run_protocol_study(
            study_id=args.study, root=args.root, case_set_dir=args.case_set,
            system_ids=[s for s in args.systems.split(",") if s],
            case_ids=[c for c in args.cases.split(",") if c] or None, trials=args.trials,
            toxagent_urls=_pairs(args.toxagent), token=args.token, order_seed=args.order_seed,
            rerun_errors=args.rerun_errors, parallel=args.parallel,
        ))
        print(json.dumps(manifest["denominators"], indent=2))
        return 0
    manifest = asyncio.run(run_study(
        study_id=args.study, root=args.root,
        system_ids=[s for s in args.systems.split(",") if s],
        case_ids=[c for c in args.cases.split(",") if c] or None, trials=args.trials,
        cases_dir=args.cases_dir,
        toxagent_urls=_pairs(args.toxagent), snapshot_from=args.snapshot_from, token=args.token,
        claude_model=args.claude_model, rerun_errors=args.rerun_errors, parallel=args.parallel,
        google={
            "google_channel": args.google_channel,
            "gemini_mcp_command": args.gemini_mcp_command.split() or None,
            "gemini_mcp_env": dict(kv.split("=", 1) for kv in args.gemini_mcp_env),
            "gemini_model": args.gemini_model,
        },
    ))
    print(json.dumps(manifest["denominators"], indent=2))
    pending = sum(row.get(STATUS_PENDING, 0) for row in manifest["denominators"]["by_system"].values())
    if pending:
        print(f"{pending} record(s) pending manual answers; see {args.root / args.study / 'manual'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
