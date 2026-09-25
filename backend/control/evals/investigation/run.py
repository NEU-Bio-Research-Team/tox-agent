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
from evals.investigation.adapters.platform_cli import ClaudeCLIAdapter, CodexCLIAdapter
from evals.investigation.adapters.predictor_template import PredictorTemplateAdapter
from evals.investigation.adapters.toxagent import ToxAgentAdapter
from evals.investigation.adapters.toxagent_api import ToxAgentAPI, snapshot_from_analysis
from evals.investigation.record import (
    STATUS_ERROR, STATUS_OK, STATUS_PENDING, MANIFEST_SCHEMA, RunRecord, StudyStore, environment,
    now_iso, summarise,
)
from evals.investigation.systems import SYSTEMS, SystemSpec, resolve

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
                   claude_model: str | None, transport=None) -> dict[str, Any]:
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
    parallel: int = 1,
) -> dict[str, Any]:
    cases = case_module.load_cases(cases_dir, only=case_ids)
    specs = resolve(system_ids)
    store = StudyStore(root / study_id)
    adapters = adapters or build_adapters(specs, toxagent_urls=toxagent_urls, token=token,
                                          claude_model=claude_model, transport=transport)
    for adapter in adapters.values():
        if hasattr(adapter, "prepare"):
            await adapter.prepare()
        if hasattr(adapter, "effective_product"):
            await adapter.effective_product()

    needs_snapshot = any(s.uses_snapshot or s.adapter == "predictor_template" for s in specs)
    snapshots = (
        await compute_snapshots(cases, store, base_url=snapshot_from or next(iter(toxagent_urls.values()), None),
                                token=token, transport=transport)
        if needs_snapshot else {}
    )

    manifest = store.manifest()
    manifest.update({
        "schema_version": MANIFEST_SCHEMA,
        "study_id": study_id,
        "created_at": manifest.get("created_at") or now_iso(),
        "updated_at": now_iso(),
        "runner_version": RUNNER_VERSION,
        "case_set": {"set_id": cases[0]["set_id"] if cases else None,
                     "sha256": case_module.case_set_sha256(cases),
                     "cases": {c["case_id"]: case_module.sha256(c) for c in cases}},
        "systems": {**manifest.get("systems", {}),
                    **{s.system_id: {**s.to_dict(), "adapter_config": adapters[s.system_id].describe()}
                       for s in specs}},
        "platform_prompt": {"version": prompts.PROMPT_VERSION, "preamble": prompts.PREAMBLE,
                            "preamble_sha256": prompts.preamble_sha256(),
                            "snapshot_header": prompts.SNAPSHOT_HEADER},
        "trials": trials,
        "environment": environment(),
    })
    store.write_manifest(manifest)

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
    summary = summarise(store.latest_records().values())
    manifest["denominators"] = {
        "cases": len(cases), "systems": len(specs), "trials": trials,
        "expected_records": len(cases) * len(specs) * trials, "by_system": summary,
    }
    store.write_manifest(manifest)
    return manifest


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
    parser.add_argument("--toxagent", action="append", default=[], metavar="SYSTEM=URL")
    parser.add_argument("--snapshot-from", default=None)
    parser.add_argument("--token", default=os.environ.get("TOXAGENT_STUDY_TOKEN", ""))
    parser.add_argument("--claude-model", default="opus",
                        help="alias passed to claude --model; recorded with the model the CLI reports")
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = asyncio.run(run_study(
        study_id=args.study, root=args.root,
        system_ids=[s for s in args.systems.split(",") if s],
        case_ids=[c for c in args.cases.split(",") if c] or None, trials=args.trials,
        toxagent_urls=_pairs(args.toxagent), snapshot_from=args.snapshot_from, token=args.token,
        claude_model=args.claude_model, rerun_errors=args.rerun_errors, parallel=args.parallel,
    ))
    print(json.dumps(manifest["denominators"], indent=2))
    pending = sum(row.get(STATUS_PENDING, 0) for row in manifest["denominators"]["by_system"].values())
    if pending:
        print(f"{pending} record(s) pending manual answers; see {args.root / args.study / 'manual'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
