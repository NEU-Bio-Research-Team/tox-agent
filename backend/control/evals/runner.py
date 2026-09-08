"""The eval runner (plan sections 16.7, 16.9).

Loads the task set, executes each task's conversation against an in-process
control plane wired to the task's frozen fixture, gathers a
:class:`~evals.graders.model.TaskOutcome` over REST, applies the deterministic
graders, and reports ``pass@1`` / ``pass^k`` overall, per category, and for the
critical subset (never averaged — plan 16.5).

Runtimes:

* ``--runtime scripted`` (default, CI): in-process control plane, frozen
  fixture predictor, no model. Only deterministic-lane tasks execute —
  analysis-failure, routing (out_of_scope / clarification). Everything else is
  reported ``needs_runtime`` and excluded from the rate.
* ``--runtime opencode`` / ``dsh``: drives an already-running live stack
  (``scripts/run_local_phase3.sh``, or an equivalent set of independently
  started services) at ``--base-url`` over real HTTP — a real model, a real
  OpenCode/DSH turn. That stack's predictor is normally the real ToxPred, not
  a frozen fixture, so a task pinned to exact frozen numbers is skipped
  (``is_live_compatible``) rather than graded against a mismatched real
  prediction; wording/limitation/hard-gate checks still apply in full.

Every run writes ``<out>/manifest-<ts>.json`` (section 16.9) and
``<out>/results-<ts>.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from evals.frozen import FrozenPredictor, load_fixture
from evals.graders import TaskOutcome, TaskReport, grade_task

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
SCHEMA_PATH = HERE / "schema" / "task.schema.json"
DEFAULT_OUT = HERE / "manifests"

_DETERMINISTIC_INTENTS = {"out_of_scope", "clarification_required"}


# --------------------------------------------------------------------- loading

def load_tasks(tasks_dir: Path = TASKS_DIR) -> list[dict[str, Any]]:
    tasks = [json.loads(p.read_text()) for p in sorted(tasks_dir.glob("*.json"))]
    _validate(tasks)
    return tasks


def _validate(tasks: list[dict[str, Any]]) -> None:
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a dev dependency
        return
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    problems: list[str] = []
    for task in tasks:
        for error in validator.iter_errors(task):
            problems.append(f"{task.get('task_id', '?')}: {list(error.path)} {error.message}")
    if problems:
        raise ValueError("invalid eval tasks:\n" + "\n".join(problems))


def is_deterministic(task: dict[str, Any]) -> bool:
    """A task the scripted (no-LLM) driver can execute and grade."""
    run_expect = task.get("expect", {}).get("run", {})
    if run_expect.get("lane") == "deterministic":
        return True
    if run_expect.get("intent") in _DETERMINISTIC_INTENTS:
        return True
    # An analysis that must fail at the predictor never reaches a runtime.
    if run_expect.get("intent") == "analysis" and run_expect.get("status") == "failed":
        return True
    if task.get("expect", {}).get("error_code") in {
        "invalid_smiles", "predictor_not_ready", "predictor_protocol_error"
    }:
        return True
    return False


# ------------------------------------------------------------------- execution

@dataclass
class TaskResult:
    task_id: str
    category: str
    critical: bool
    executed: bool
    passed: bool
    reasons: list[str] = field(default_factory=list)
    deferred_graders: list[str] = field(default_factory=list)
    skipped_reason: str | None = None


class ScriptedDriver:
    """In-process control plane + frozen predictor, no model."""

    def __init__(self) -> None:
        self._tmp: list[Path] = []

    @asynccontextmanager
    async def _app(self, fixture: dict[str, Any], db_path: Path) -> AsyncIterator[httpx.AsyncClient]:
        from toxagent.api.app import create_app
        from toxagent.config import (
            OcrSettings, PolicySettings, PredictorSettings, PredictSettings, ResearchSettings,
            RuntimeSettings, SecuritySettings, Settings,
        )
        from toxagent.persistence.sql.database import Database

        settings = Settings(
            database_url=f"sqlite+aiosqlite:///{db_path}",
            predictor=PredictorSettings(base_url="http://predictor.frozen"),
            policy=PolicySettings(),
            predict=PredictSettings(),
            runtime=RuntimeSettings(kind="scripted"),
            research=ResearchSettings(),
            ocr=OcrSettings(),
            security=SecuritySettings(
                capability_secret="eval-secret-not-for-production",
                static_tokens=("eval-user-token:eval-user", "eval-other-token:eval-other"),
            ),
        )
        database = Database(settings.database_url)
        await database.create_schema()
        predictor = FrozenPredictor(fixture["predictor"])
        app = create_app(settings, database=database, predictor=predictor.client())
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://eval.test"
                ) as client:
                    client.app = app
                    yield client
        finally:
            await database.dispose()

    async def run(self, task: dict[str, Any], tmp_dir: Path) -> TaskOutcome:
        fixture = load_fixture(task["fixture"])
        db_path = tmp_dir / f"{task['task_id']}.db"
        auth = {"authorization": "Bearer eval-user-token"}
        async with self._app(fixture, db_path) as client:
            session = await client.post(
                "/v1/sessions",
                json={"preferred_language": task.get("language", "en")},
                headers=auth,
            )
            session.raise_for_status()
            session_id = session.json()["session_id"]

            last_run_id: str | None = None
            error_envelope: dict[str, Any] | None = None
            for turn in task["conversation"]:
                body: dict[str, Any] = {"intent_hint": turn.get("intent_hint", "auto")}
                if turn.get("content"):
                    body["content"] = [{"type": "text", "text": turn["content"]}]
                if "molecule" in turn:
                    body["molecule"] = turn["molecule"]
                if "analysis_options" in turn:
                    body["analysis_options"] = turn["analysis_options"]
                if "analysis_id" in turn:
                    body["analysis_id"] = turn["analysis_id"]
                response = await client.post(
                    f"/v1/sessions/{session_id}/messages", json=body, headers=auth
                )
                if response.status_code >= 400:
                    error_envelope = response.json()
                    continue
                last_run_id = response.json().get("run_id")
                if last_run_id:
                    await _await_run(client, session_id, last_run_id, auth)

            outcome = await gather_outcome(client, session_id, last_run_id, auth, error_envelope)

        if task.get("expect", {}).get("state", {}).get("reconstructable_after_restart"):
            outcome = await self._reconstruct(fixture, db_path, session_id, outcome, auth)
        return outcome

    async def _reconstruct(
        self, fixture, db_path, session_id, outcome: TaskOutcome, auth
    ) -> TaskOutcome:
        """Restart the control plane on the same database and confirm the
        session still reads back (PROD-04/05, hard gate #10)."""
        try:
            async with self._app(fixture, db_path) as client:
                session = await client.get(f"/v1/sessions/{session_id}", headers=auth)
                ok = session.status_code == 200 and bool(session.json().get("session_id"))
                if ok and outcome.answer:
                    claims_ok = True
                    for claim in outcome.answer.get("claims", []):
                        obs = claim.get("observation_id")
                        if obs and obs not in outcome.session_observation_ids:
                            claims_ok = False
                    ok = ok and claims_ok
        except Exception:  # pragma: no cover - restart failure is the signal
            ok = False
        from dataclasses import replace

        return replace(outcome, reconstructed_ok=ok)


async def _await_run(client, session_id, run_id, auth, *, tries: int = 300, delay: float = 0.01) -> None:
    for _ in range(tries):
        response = await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=auth)
        if response.status_code == 200 and response.json()["status"] in (
            "completed", "failed", "cancelled"
        ):
            return
        await asyncio.sleep(delay)


async def _fetch_all_evidence(client, session_id, auth) -> list[dict[str, Any]]:
    """``GET .../evidence`` pages (default ``limit=50``); a single unpaginated
    call silently drops older accepted records past the first page. A
    live task whose model searched enough times to pass 50 accepted records
    in one session hit exactly this — citations_resolve then flagged a
    genuinely-accepted evidence_id as unresolved only because it fell off
    page one, not because the product ever lost track of it."""
    records: list[dict[str, Any]] = []
    offset = 0
    limit = 200
    while True:
        response = await client.get(
            f"/v1/sessions/{session_id}/evidence", headers=auth,
            params={"limit": limit, "offset": offset},
        )
        page = response.json().get("evidence", [])
        records.extend(page)
        if len(page) < limit:
            return records
        offset += limit


async def gather_outcome(client, session_id, run_id, auth, error_envelope) -> TaskOutcome:
    """Read a :class:`TaskOutcome` back over the product REST API. Shared by
    every driver — scripted (ASGI transport) and remote (a real live stack,
    plan section 16.8 Track A/B) alike, since both are just an
    ``httpx.AsyncClient`` pointed at ``/v1/...`` paths."""
    session = (await client.get(f"/v1/sessions/{session_id}", headers=auth)).json()
    run: dict[str, Any] = {}
    if run_id:
        run_response = await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=auth)
        if run_response.status_code == 200:
            run = run_response.json()

    analyses: list[dict[str, Any]] = []
    active = session.get("active_analysis")
    if active:
        analyses.append(active)

    messages = (
        await client.get(f"/v1/sessions/{session_id}/messages", headers=auth)
    ).json().get("messages", [])

    answer = None
    for message in messages:
        for part in message.get("parts", []):
            if part.get("type") == "answer_ref":
                answer_id = part.get("content", {}).get("answer_id")
                if answer_id:
                    a = await client.get(
                        f"/v1/sessions/{session_id}/answers/{answer_id}", headers=auth
                    )
                    if a.status_code == 200:
                        answer = a.json()

    evidence = await _fetch_all_evidence(client, session_id, auth)

    observation_ids: set[str] = set()
    observation_values: dict[str, Any] = {}
    for snapshot in analyses:
        raw = await client.get(
            f"/v1/sessions/{session_id}/analyses/{snapshot['analysis_id']}",
            headers=auth, params={"include_raw": "true"},
        )
        if raw.status_code == 200:
            payload = raw.json()
            for obs in payload.get("observations", []):
                oid = obs.get("observation_id") or obs.get("id")
                if oid:
                    observation_ids.add(oid)
                    observation_values[oid] = obs.get("canonical_payload") or payload.get(
                        "predictor_response", {}
                    )

    return TaskOutcome(
        run=run,
        session=session,
        answer=answer,
        analyses=analyses,
        evidence=evidence,
        tool_calls=run.get("tool_calls", []),
        messages=messages,
        error=error_envelope,
        session_observation_ids=frozenset(observation_ids),
        session_evidence_ids=frozenset(e.get("evidence_id") or e.get("id") for e in evidence),
        observation_values=observation_values,
    )


class RemoteHTTPDriver:
    """Drives a task's conversation against an already-running product stack
    (``scripts/run_local_phase3.sh``) instead of an in-process app.

    Unlike :class:`ScriptedDriver` this talks to whatever predictor that stack
    is actually configured with — normally the *real* ToxPred, not a frozen
    fixture (plan section 16.3's "predictor integration mode", not "frozen
    mode"). A task whose ``expect.answer.required_claims`` pins an exact
    ``source_value``/``rendered_value`` would spuriously fail against real
    predictor output that does not match the frozen fixture, so
    :func:`is_live_compatible` filters those out before this driver ever sees
    them; what runs here is graded on structure and wording (required/forbidden
    limitations, must/must-not-mention, hard gates), which holds regardless of
    the exact numbers.
    """

    def __init__(
        self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = {"authorization": f"Bearer {token}"}
        #: Injectable so the contract suite can prove this driver's HTTP calls
        #: against a transport double, the same pattern the OpenCode adapter
        #: itself is tested with — no live stack needed in CI.
        self._transport = transport

    async def run(self, task: dict[str, Any], tmp_dir: Path) -> TaskOutcome:
        del tmp_dir  # no local scratch database against a live stack
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=30.0, transport=self._transport
        ) as client:
            session = await client.post(
                "/v1/sessions",
                json={"preferred_language": task.get("language", "en")},
                headers=self._auth,
            )
            session.raise_for_status()
            session_id = session.json()["session_id"]

            last_run_id: str | None = None
            error_envelope: dict[str, Any] | None = None
            for turn in task["conversation"]:
                body: dict[str, Any] = {"intent_hint": turn.get("intent_hint", "auto")}
                if turn.get("content"):
                    body["content"] = [{"type": "text", "text": turn["content"]}]
                if "molecule" in turn:
                    body["molecule"] = turn["molecule"]
                if "analysis_options" in turn:
                    body["analysis_options"] = turn["analysis_options"]
                if "analysis_id" in turn:
                    body["analysis_id"] = turn["analysis_id"]
                response = await client.post(
                    f"/v1/sessions/{session_id}/messages", json=body, headers=self._auth
                )
                if response.status_code >= 400:
                    error_envelope = response.json()
                    continue
                last_run_id = response.json().get("run_id")
                if last_run_id:
                    # A live agentic turn takes real wall-clock time (an actual
                    # model round trip), unlike the scripted driver's in-process
                    # turn — poll patiently rather than in a tight loop.
                    await _await_run(
                        client, session_id, last_run_id, self._auth, tries=180, delay=1.0
                    )

            return await gather_outcome(client, session_id, last_run_id, self._auth, error_envelope)


#: Fixtures that exist specifically to make ToxPred answer broken (a 503, a
#: malformed body). A live stack's real predictor is healthy, so a task
#: pinned to one of these can only ever fail for the wrong reason — it never
#: gets the failure it is testing for (found live 2026-09-05: fail-01/fail-02
#: both completed normally instead of failing).
_BROKEN_PREDICTOR_FIXTURES = frozenset({"predictor-503", "predictor-malformed"})


#: A task whose expectations are tied to a specific frozen fixture's numbers
#: cannot be graded honestly against a live, real predictor (see
#: RemoteHTTPDriver's docstring). The same is true, for a different reason,
#: of a task that needs an actual runtime/control-plane process to go down —
#: a bare HTTP driver cannot kill or restart the stack it is talking to, so
#: "the runtime was already unavailable" or "reconstruct after a restart"
#: can never be genuinely exercised this way (found live 2026-09-05: these
#: were attempted and counted as failures for a condition the driver never
#: actually created, rather than skipped as untestable — the eval-runner
#: mirror of §3.9's must_not_mention negation-blindness: a check the harness
#: cannot honestly perform must not silently read as the product having
#: failed it).
def is_live_compatible(task: dict[str, Any]) -> bool:
    for claim in (task.get("expect", {}).get("answer", {}) or {}).get("required_claims", []):
        if "rendered_value" in claim or "source_value" in claim:
            return False
    if task.get("fixture") in _BROKEN_PREDICTOR_FIXTURES:
        return False
    expect = task.get("expect", {})
    if expect.get("error_code") == "runtime_unavailable":
        return False
    if expect.get("state", {}).get("reconstructable_after_restart"):
        return False
    return True


# ---------------------------------------------------------------------- suite

async def run_suite(
    tasks: list[dict[str, Any]],
    *,
    runtime: str,
    trials: int,
    out_dir: Path,
    only: set[str] | None = None,
    base_url: str = "http://127.0.0.1:8000",
    token: str = "dev-local",
) -> dict[str, Any]:
    if runtime == "scripted":
        driver: Any = ScriptedDriver()
        eligible = is_deterministic
        skip_reason = "needs an agentic runtime"
    elif runtime in ("opencode", "dsh"):
        # Live: scripts/run_local_phase3.sh (or an equivalent independently
        # started stack) must already be running and reachable at base_url.
        # Unlike the scripted driver this is not frozen-fixture mode (plan
        # section 16.3) — it uses whatever predictor that stack is configured
        # with, so a task pinned to exact frozen numbers is skipped rather
        # than graded against a mismatched real prediction.
        driver = RemoteHTTPDriver(base_url, token)
        eligible = is_live_compatible
        skip_reason = "not live-compatible (pins exact frozen-fixture numbers)"
    else:
        raise SystemExit(f"unknown runtime {runtime!r}")
    tmp_dir = out_dir / "_work"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: list[TaskResult] = []
    for task in tasks:
        if only and task["task_id"] not in only:
            continue
        if not eligible(task):
            results.append(
                TaskResult(
                    task["task_id"], task["category"], task.get("critical", False),
                    executed=False, passed=False, skipped_reason=skip_reason,
                )
            )
            continue
        trial_reports: list[TaskReport] = []
        for _ in range(trials):
            outcome = await driver.run(task, tmp_dir)
            trial_reports.append(grade_task(task, outcome))
        passed = all(r.passed for r in trial_reports)  # pass^k, never averaged
        reasons: list[str] = []
        for report in trial_reports:
            reasons.extend(report.reasons())
        results.append(
            TaskResult(
                task["task_id"], task["category"], task.get("critical", False),
                executed=True, passed=passed, reasons=sorted(set(reasons)),
                deferred_graders=list(trial_reports[0].deferred_graders),
            )
        )

    summary = _summarise(results, trials)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results-{stamp}.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n"
    )
    (out_dir / f"manifest-{stamp}.json").write_text(
        json.dumps(_manifest(runtime, trials, summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def _summarise(results: list[TaskResult], trials: int) -> dict[str, Any]:
    executed = [r for r in results if r.executed]
    passed = [r for r in executed if r.passed]
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_category.setdefault(r.category, {"executed": 0, "passed": 0, "skipped": 0})
        if r.executed:
            bucket["executed"] += 1
            bucket["passed"] += int(r.passed)
        else:
            bucket["skipped"] += 1
    critical = [r for r in executed if r.critical]
    return {
        "trials": trials,
        "metric": "pass^%d" % trials if trials > 1 else "pass@1",
        "total_tasks": len(results),
        "executed": len(executed),
        "skipped_needs_runtime": len(results) - len(executed),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(executed), 4) if executed else None,
        "critical_executed": len(critical),
        "critical_passed": sum(r.passed for r in critical),
        "critical_all_pass": all(r.passed for r in critical) if critical else None,
        "by_category": by_category,
        "failures": [
            {"task_id": r.task_id, "critical": r.critical, "reasons": r.reasons}
            for r in executed if not r.passed
        ],
    }


def _manifest(runtime: str, trials: int, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_suite_hash": _suite_hash(),
        "toxagent_commit": _git("HEAD"),
        "toxpred_commit": _pinned_predictor_commit(),
        "runtime_kind": runtime,
        "runtime_version": "in-process-scripted" if runtime == "scripted" else "live-stack",
        "trial_count": trials,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }


def _suite_hash() -> str:
    from toxagent.domain.provenance import content_sha256

    payload = {
        p.name: p.read_text()
        for p in sorted(list(TASKS_DIR.glob("*.json")) + list((HERE / "fixtures").glob("*.json")))
    }
    payload["schema"] = SCHEMA_PATH.read_text()
    return content_sha256(payload)


def _git(ref: str) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", ref], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pinned_predictor_commit() -> str:
    snapshot = HERE.parent / "toxagent" / "predictor" / "contract_snapshot.json"
    try:
        return json.loads(snapshot.read_text()).get("captured_at_commit", "unknown")
    except (OSError, ValueError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="scripted", choices=["scripted", "opencode", "dsh"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task", action="append", dest="tasks", help="run only these task ids")
    parser.add_argument("--list", action="store_true", help="list tasks and exit")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="live stack URL, for --runtime opencode/dsh (default: the local Phase 3 stack)",
    )
    parser.add_argument(
        "--token", default="dev-local",
        help="bearer token for --runtime opencode/dsh (default: the local dev token)",
    )
    args = parser.parse_args(argv)

    tasks = load_tasks()
    if args.list:
        eligible = is_deterministic if args.runtime == "scripted" else is_live_compatible
        for task in tasks:
            mark = "yes" if eligible(task) else "no "
            print(f"{mark}  {task['category']:20s}  {task['task_id']}")
        return 0

    summary = asyncio.run(
        run_suite(
            tasks, runtime=args.runtime, trials=args.trials, out_dir=args.out,
            only=set(args.tasks) if args.tasks else None,
            base_url=args.base_url, token=args.token,
        )
    )
    print(json.dumps(summary, indent=2))
    if summary["critical_all_pass"] is False:
        return 1
    if summary["pass_rate"] is not None and summary["pass_rate"] < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
