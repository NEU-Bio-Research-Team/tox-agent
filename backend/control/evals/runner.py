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
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from evals import task_packs
from evals.frozen import FrozenPredictor, load_fixture
from evals.graders import GradeResult, TaskOutcome, TaskReport, grade_task, grader_versions
from evals.graders.semantic import CommandJudge, RecordedJudge, grade_semantic
from evals.graders.outcome_split import breakdown as outcome_breakdown
from evals.manifest import build_manifest
from evals.trace import project as project_trace

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
SCHEMA_PATH = HERE / "schema" / "task.schema.json"
DEFAULT_OUT = HERE / "manifests"

_DETERMINISTIC_INTENTS = {"out_of_scope", "clarification_required"}


# ------------------------------------------------------- W1-01 fixture modes

#: What supplied the numbers and the evidence a run graded against. It is not
#: the same question as which runtime drove the turn, and conflating them is
#: how two manifests get compared that were never measuring the same thing: a
#: `scripted` run is always frozen, but a live run may be reading a real
#: ToxPred with stubbed evidence or reaching a real provider over the network,
#: and those three produce different, non-interchangeable numbers.
#:
#: * `frozen` — a content-hashed fixture supplies predictor, evidence and
#:   runtime responses. No network, reproducible, and the only mode in which a
#:   task may pin an exact predicted value.
#: * `predictor_integration` — a real ToxPred answers; evidence is still
#:   stubbed. Wording, hard gates and semantics are graded; exact numbers are
#:   not, because they are the model's, not the fixture's.
#: * `live_evidence` — a real evidence provider is reached over the network as
#:   well. Retrieval is genuinely exercised and the run is, by construction,
#:   not reproducible from the repository alone.
FIXTURE_MODES = ("frozen", "predictor_integration", "live_evidence")

#: Modes in which the numbers come from the fixture, so a task may pin them.
_FROZEN_NUMBER_MODES = frozenset({"frozen"})


# ------------------------------------------------- W1-05 skipped_reason types

#: Why a task did not execute, as a closed set rather than a sentence.
#:
#: The summary used to report every skip as `skipped_needs_runtime`, which was
#: true of one of these and false of the rest — a task pinned to frozen
#: numbers does not need a runtime, it needs a different fixture mode, and a
#: task needing a process killed needs an orchestrator no driver here has. A
#: reader counting "skipped because we have no model" was counting four other
#: things as well, and could not tell which of them a credential would fix.
SKIPPED_REASONS = {
    "needs_agentic_runtime": (
        "the scripted driver runs the deterministic lane only; this task needs a model"
    ),
    "pins_frozen_numbers": (
        "expectations name exact frozen-fixture values, which a real predictor will not reproduce"
    ),
    "needs_broken_predictor_fixture": (
        "the task tests a predictor failure that a healthy live predictor never produces"
    ),
    "needs_runtime_outage_injection": (
        "the task needs the runtime to be unavailable; an HTTP driver cannot take it down"
    ),
    "needs_control_plane_restart": (
        "the task needs the control plane restarted mid-run; an HTTP driver cannot restart it"
    ),
    "needs_live_evidence": (
        "the task needs a real evidence provider; this run's fixture mode does not reach one"
    ),
    "feature_requirements_unmet": (
        "the deployment's effective flags/profile/topology differ from what the task requires"
    ),
    "not_selected": "excluded by --task on this invocation",
}

#: Result statuses (Wave 1). `pass`/`fail` are product verdicts; the other
#: three are facts about the run, and none of them is a pass.
STATUSES = ("pass", "fail", "invalid", "skipped", "infra_error")

#: Run failure codes that describe the environment rather than the product,
#: when the task did not ask for that failure.
INFRA_FAILURE_CODES = frozenset(
    {"runtime_unavailable", "predictor_not_ready", "internal_error", "provider_unavailable",
     "database_unavailable", "evidence_unavailable"}
)


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
    status: str = "skipped"
    capability_pack: str = "core"
    risk_tier: str = "medium"
    infra_errors: list[str] = field(default_factory=list)
    #: Per trial: answer outcome (first pass / corrected / fallback), stop
    #: reason and trace counters — capability and containment kept apart.
    trials: list[dict[str, Any]] = field(default_factory=list)


class ScriptedDriver:
    """In-process control plane + frozen predictor, no model."""

    def __init__(self) -> None:
        self._tmp: list[Path] = []

    @staticmethod
    def settings(db_path: Path | str = ":memory:"):
        """The deployment the scripted driver composes. Also what its
        manifest's effective_product describes, so the two cannot differ."""
        from toxagent.config import (
            CompoundSettings, OcrSettings, PolicySettings, PredictorSettings, PredictSettings,
            ResearchSettings, RuntimeSettings, SecuritySettings, Settings,
        )

        return Settings(
            database_url=f"sqlite+aiosqlite:///{db_path}",
            predictor=PredictorSettings(base_url="http://predictor.frozen"),
            policy=PolicySettings(),
            predict=PredictSettings(),
            runtime=RuntimeSettings(kind="scripted"),
            research=ResearchSettings(),
            compound=CompoundSettings(),
            ocr=OcrSettings(),
            security=SecuritySettings(
                capability_secret="eval-secret-not-for-production",
                static_tokens=("eval-user-token:eval-user", "eval-other-token:eval-other"),
            ),
        )

    @asynccontextmanager
    async def _app(self, fixture: dict[str, Any], db_path: Path) -> AsyncIterator[httpx.AsyncClient]:
        from toxagent.api.app import create_app
        from toxagent.persistence.sql.database import Database

        settings = self.settings(db_path)
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

            last_run_id, error_envelope = await drive_conversation(
                client, session_id, task, auth
            )

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


async def drive_conversation(
    client, session_id: str, task: dict[str, Any], auth: dict[str, str], *,
    tries: int = 300, delay: float = 0.01,
) -> tuple[str | None, dict[str, Any] | None]:
    """Send every turn of a task's conversation; return the last run id and
    the last synchronous error envelope. Shared by every driver.

    A v3 turn may be ``action: create_report``: it requests a report build for
    the session's active analysis through ``POST .../reports``, the same
    durable run path the product UI uses.
    """
    last_run_id: str | None = None
    error_envelope: dict[str, Any] | None = None
    for turn in task["conversation"]:
        if turn.get("action") == "create_report":
            session = (await client.get(f"/v1/sessions/{session_id}", headers=auth)).json()
            active = (session.get("active_analysis") or {}).get("analysis_id")
            body = {"analysis_id": active, **(turn.get("report") or {})}
            url = f"/v1/sessions/{session_id}/reports"
        else:
            body = {"intent_hint": turn.get("intent_hint", "auto")}
            if turn.get("content"):
                body["content"] = [{"type": "text", "text": turn["content"]}]
            for key in ("molecule", "analysis_options", "analysis_id"):
                if key in turn:
                    body[key] = turn[key]
            url = f"/v1/sessions/{session_id}/messages"
        response = await client.post(url, json=body, headers=auth)
        if response.status_code >= 400:
            error_envelope = response.json()
            continue
        last_run_id = response.json().get("run_id")
        if last_run_id:
            await _await_run(client, session_id, last_run_id, auth, tries=tries, delay=delay)
    return last_run_id, error_envelope


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

    reports_response = await client.get(f"/v1/sessions/{session_id}/reports", headers=auth)
    reports = (
        reports_response.json().get("reports", []) if reports_response.status_code == 200 else []
    )

    decision_state = None
    if run_id:
        state_response = await client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/decision-state", headers=auth
        )
        if state_response.status_code == 200:
            decision_state = state_response.json()

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
        decision_state=decision_state,
        reports=reports,
        budget=((run.get("configuration_snapshot") or {}).get("effective_budget")
                if isinstance(run.get("configuration_snapshot"), dict) else None),
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

            # A live agentic turn takes real wall-clock time (an actual model
            # round trip), unlike the scripted driver's in-process turn — poll
            # patiently rather than in a tight loop.
            last_run_id, error_envelope = await drive_conversation(
                client, session_id, task, self._auth, tries=180, delay=1.0
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
    """Kept as the boolean the older callers ask for; the reason is below."""
    return live_skip_reason(task) is None


def live_skip_reason(task: dict[str, Any]) -> str | None:
    """Which of the typed reasons excludes this task from a live run, if any."""
    for claim in (task.get("expect", {}).get("answer", {}) or {}).get("required_claims", []):
        if "rendered_value" in claim or "source_value" in claim:
            return "pins_frozen_numbers"
    if task.get("fixture") in _BROKEN_PREDICTOR_FIXTURES:
        return "needs_broken_predictor_fixture"
    expect = task.get("expect", {})
    if expect.get("error_code") == "runtime_unavailable":
        return "needs_runtime_outage_injection"
    if expect.get("state", {}).get("reconstructable_after_restart"):
        return "needs_control_plane_restart"
    return None


# ---------------------------------------------------------------------- suite

def resolve_fixture_mode(runtime: str, declared: str | None) -> str:
    """The declared mode, checked against what the runtime can actually be.

    The scripted driver *is* the frozen fixture — it builds its predictor from
    one — so letting a run label itself `live_evidence` while reading a JSON
    file would put a false provenance on the manifest. A live run defaults to
    `predictor_integration` because that is the weaker claim of the two it
    could make; reaching a real provider has to be stated, never assumed.
    """
    if declared is not None and declared not in FIXTURE_MODES:
        raise SystemExit(f"unknown fixture mode {declared!r}")
    if runtime == "scripted":
        if declared not in (None, "frozen"):
            raise SystemExit(
                f"--runtime scripted is frozen by construction; it cannot report {declared!r}"
            )
        return "frozen"
    if declared == "frozen":
        raise SystemExit(
            f"--runtime {runtime} drives a live stack, whose predictor is not the frozen fixture"
        )
    return declared or "predictor_integration"


async def run_suite(
    tasks: list[dict[str, Any]] | None = None,
    *,
    runtime: str,
    trials: int,
    out_dir: Path,
    only: set[str] | None = None,
    base_url: str = "http://127.0.0.1:8000",
    token: str = "dev-local",
    fixture_mode: str | None = None,
    discovery: task_packs.Discovery | None = None,
    suite: str | None = None,
    driver: Any = None,
    effective_product: dict[str, Any] | None = None,
    judge: Any = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a task set and write results, traces and an eval-manifest-v2.

    ``discovery`` is the pack-aware path. Passing bare ``tasks`` (the older
    call shape) treats them as one ad-hoc selection of the core pack.
    """
    mode = resolve_fixture_mode(runtime, fixture_mode)
    if discovery is None:
        if tasks is None:
            discovery = task_packs.discover(task_packs.DEFAULT_PACKS)
        else:
            discovery = _adhoc_discovery(tasks)
    timeout_policy: dict[str, Any]
    if runtime == "scripted":
        driver = driver or ScriptedDriver()
        timeout_policy = {"poll_tries": 300, "poll_delay_s": 0.01, "http_timeout_s": None}
    elif runtime in ("opencode", "dsh"):
        # Live: scripts/run_local_phase3.sh (or an equivalent independently
        # started stack) must already be running and reachable at base_url.
        # Unlike the scripted driver this is not frozen-fixture mode (plan
        # section 16.3) — it uses whatever predictor that stack is configured
        # with, so a task pinned to exact frozen numbers is skipped rather
        # than graded against a mismatched real prediction.
        driver = driver or RemoteHTTPDriver(base_url, token)
        timeout_policy = {"poll_tries": 180, "poll_delay_s": 1.0, "http_timeout_s": 30.0}
    else:
        raise SystemExit(f"unknown runtime {runtime!r}")

    if effective_product is None:
        effective_product = await _effective_product(runtime, base_url, token)

    def skip_reason_for(task: dict[str, Any]) -> str | None:
        if only and task["task_id"] not in only:
            return "not_selected"
        if runtime == "scripted":
            if not runs_scripted(task):
                return "needs_agentic_runtime"
        else:
            reason = live_skip_reason(task)
            if reason is not None:
                return reason
            if task.get("runtime_requirement") == "live_evidence" and mode != "live_evidence":
                return "needs_live_evidence"
        if not features_satisfied(task, effective_product):
            return "feature_requirements_unmet"
        return None

    tmp_dir = out_dir / "_work"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: list[TaskResult] = []
    traces: list[dict[str, Any]] = []
    for invalid in discovery.invalid:
        results.append(
            TaskResult(
                task_id=f"<invalid>/{invalid.path}", category="invalid", critical=False,
                executed=False, passed=False, status="invalid",
                capability_pack=invalid.pack, reasons=list(invalid.problems),
            )
        )
    for task in discovery.tasks:
        base = dict(
            task_id=task["task_id"], category=task["category"],
            critical=task.get("critical", False),
            capability_pack=task.get("capability_pack", "core"),
            risk_tier=task.get("risk_tier", "medium"),
        )
        reason = skip_reason_for(task)
        if reason is not None:
            assert reason in SKIPPED_REASONS, reason
            results.append(
                TaskResult(**base, executed=False, passed=False, skipped_reason=reason,
                           status="skipped")
            )
            continue
        trial_count = max(trials, (task.get("trial_policy") or {}).get("min_trials", 1))
        trial_reports: list[TaskReport] = []
        infra: list[str] = []
        trial_rows: list[dict[str, Any]] = []
        for trial in range(trial_count):
            try:
                outcome = await driver.run(task, tmp_dir)
            except (httpx.HTTPError, OSError, asyncio.TimeoutError) as exc:
                infra.append(f"trial {trial}: driver error {type(exc).__name__}: {exc}")
                trial_rows.append({"trial": trial, "status": "infra_error"})
                continue
            infra_reason = infra_failure(task, outcome)
            trace = project_trace(outcome)
            traces.append({"task_id": task["task_id"], "trial": trial, **trace.to_dict()})
            if infra_reason is not None:
                infra.append(f"trial {trial}: {infra_reason}")
                trial_rows.append({"trial": trial, "status": "infra_error"})
                continue
            report = grade_task(task, outcome)
            semantic = await grade_semantic(task, outcome, judge, calibration)
            if semantic.get("gating") and semantic.get("status") != "pass":
                # A calibrated rubric gates. Abstaining on a blocking dimension,
                # or a verdict that cannot be used, is not a pass either.
                report = replace(report, results=report.results + (
                    GradeResult.fail("semantic", f"semantic judge: {semantic.get('status')}"),
                ))
            trial_reports.append(report)
            trial_rows.append(
                {"trial": trial, "status": "pass" if report.passed else "fail",
                 "semantic": semantic, **outcome_breakdown(outcome)}
            )
        reasons: list[str] = []
        for report in trial_reports:
            reasons.extend(report.reasons())
        any_failed = any(not r.passed for r in trial_reports)
        if any_failed:
            status = "fail"  # a product failure is real whatever else happened
        elif infra or not trial_reports:
            status = "infra_error"
        else:
            status = "pass"
        results.append(
            TaskResult(
                **base, executed=status in ("pass", "fail"), passed=status == "pass",
                reasons=sorted(set(reasons)), status=status, infra_errors=infra,
                deferred_graders=list(trial_reports[0].deferred_graders) if trial_reports else [],
                trials=trial_rows,
            )
        )

    summary = _summarise(results, trials, mode, discovery)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results-{stamp}.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n"
    )
    (out_dir / f"traces-{stamp}.jsonl").write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in traces)
    )
    manifest = build_manifest(
        runtime=runtime, trials=trials, fixture_mode=mode, summary=summary,
        suite_hash=task_packs.suite_hash(discovery),
        discovery=discovery.summary(),
        effective_product=effective_product,
        grader_versions=grader_versions(),
        timeout_policy=timeout_policy,
        predictor_commit=_pinned_predictor_commit(),
        suite=suite,
    )
    manifest["semantic_judge"] = {
        "judge": getattr(judge, "name", None),
        "calibration": calibration.get("schema_version") if calibration else None,
        "gating_rubrics": sorted(
            key for key, r in ((calibration or {}).get("rubrics") or {}).items() if r.get("calibrated")
        ),
    }
    (out_dir / f"manifest-{stamp}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    summary["release_evidence"] = manifest["release_evidence"]
    return summary


def _adhoc_discovery(tasks: list[dict[str, Any]]) -> task_packs.Discovery:
    """Wrap a caller-supplied task list so it flows through the same path."""
    normalised = []
    for task in tasks:
        pack = task.get("capability_pack", "core")
        item = task if "source_schema_version" in task else task_packs.normalise(task, pack)
        item.setdefault("_source_path", f"tasks/{task['task_id']}.json")
        normalised.append(item)
    packs = tuple(dict.fromkeys(t["capability_pack"] for t in normalised)) or ("core",)
    records = {
        name: task_packs.PackDiscovery(
            name, available=True,
            files=[t["_source_path"] for t in normalised if t["capability_pack"] == name],
            loaded=sum(1 for t in normalised if t["capability_pack"] == name),
        )
        for name in packs
    }
    return task_packs.Discovery(packs, normalised, [], records)


def runs_scripted(task: dict[str, Any]) -> bool:
    requirement = task.get("runtime_requirement")
    if requirement is None:
        return is_deterministic(task)
    return requirement == "scripted"


def features_satisfied(task: dict[str, Any], product: dict[str, Any] | None) -> bool:
    """Whether the effective product is the one the task was written for.

    An unknown product satisfies nothing but an empty requirement: guessing
    that a flag was on is how a task grades a path that never ran.
    """
    requirements = task.get("feature_requirements") or {}
    if not requirements:
        return True
    if not product or product.get("unavailable"):
        return False
    flags = product.get("flags") or {}
    for name, wanted in (requirements.get("flags") or {}).items():
        if (flags.get(name) or {}).get("enabled") is not wanted:
            return False
    topology = requirements.get("topology")
    if topology and topology != "any":
        external = (product.get("topology") or {}).get("external_worker_mode")
        if external is not (topology == "external_workers"):
            return False
    kinds = requirements.get("runtime_kind")
    if kinds and (product.get("runtime") or {}).get("kind") not in kinds:
        return False
    providers = requirements.get("provider")
    if providers and (product.get("runtime") or {}).get("provider_id") not in providers:
        return False
    if requirements.get("research_provider") is True and not (
        (product.get("providers") or {}).get("research_provider")
    ):
        return False
    if requirements.get("ocr") is True and not (product.get("providers") or {}).get("ocr_configured"):
        return False
    wanted_snapshot = requirements.get("research_snapshot")
    if wanted_snapshot:
        # The stack must serve exactly this frozen evidence, with exactly the
        # declared fault (none when the task declares none).
        snapshot = (product.get("providers") or {}).get("research_snapshot") or {}
        if snapshot.get("path_name") != f"{wanted_snapshot}.json":
            return False
        if snapshot.get("fault") != requirements.get("research_snapshot_fault"):
            return False
    profile = requirements.get("capability_profile")
    intent = task.get("intent")
    if profile and intent:
        got = ((product.get("intents") or {}).get(intent) or {}).get("capability_profile")
        if got != profile:
            return False
    return True


def infra_failure(task: dict[str, Any], outcome: TaskOutcome) -> str | None:
    """The environment, not the product, ended this trial — or ``None``."""
    code = (outcome.run or {}).get("failure_code")
    if code is None or code not in INFRA_FAILURE_CODES:
        return None
    expect = task.get("expect", {})
    if code in (expect.get("error_code"), expect.get("run", {}).get("failure_code")):
        return None
    if (task.get("fault_injection") or {}) or task.get("inject"):
        # The task injected a fault; how the product handled it is the verdict.
        return None
    return f"run failed with infrastructure code {code!r}"


async def _effective_product(runtime: str, base_url: str, token: str) -> dict[str, Any]:
    from toxagent.application.effective_product import describe_effective_product

    if runtime == "scripted":
        return describe_effective_product(ScriptedDriver.settings())
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=15.0) as client:
            response = await client.get(
                "/v1/system/effective-product", headers={"authorization": f"Bearer {token}"}
            )
    except httpx.HTTPError as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
    if response.status_code != 200:
        return {"unavailable": f"HTTP {response.status_code} from /v1/system/effective-product"}
    return response.json()


def _summarise(
    results: list[TaskResult], trials: int, fixture_mode: str,
    discovery: task_packs.Discovery | None = None,
) -> dict[str, Any]:
    real = [r for r in results if r.status != "invalid"]
    executed = [r for r in real if r.status in ("pass", "fail")]
    passed = [r for r in executed if r.passed]
    by_category: dict[str, dict[str, int]] = {}
    by_pack: dict[str, dict[str, int]] = {}
    for r in results:
        if r.status != "invalid":
            if r.skipped_reason == "not_selected":
                continue
            bucket = by_category.setdefault(r.category, {"executed": 0, "passed": 0, "skipped": 0})
            if r.executed:
                bucket["executed"] += 1
                bucket["passed"] += int(r.passed)
            elif r.status == "skipped":
                bucket["skipped"] += 1
        pack = by_pack.setdefault(r.capability_pack, {s: 0 for s in STATUSES})
        pack[r.status] += 1
    critical = [r for r in executed if r.critical]
    skipped_by_reason = {reason: 0 for reason in SKIPPED_REASONS}
    for r in results:
        if r.status == "skipped" and r.skipped_reason is not None:
            skipped_by_reason[r.skipped_reason] += 1
    # Deselected by --task: counted for conservation, but not a reason a task
    # "could not run", so it stays out of skipped_by_reason.
    not_selected = skipped_by_reason.pop("not_selected")
    counts = {s: sum(1 for r in results if r.status == s) for s in STATUSES}

    fallback_trials = first_pass = corrected = answered = 0
    for r in executed:
        for trial in r.trials:
            outcome = trial.get("answer_outcome")
            if outcome in (None, "none"):
                continue
            answered += 1
            first_pass += outcome == "first_pass"
            corrected += outcome == "accepted_after_correction"
            fallback_trials += outcome == "fallback"

    conservation: list[str] = []
    not_evaluated: list[str] = []
    if discovery is not None:
        conservation = task_packs.check_conservation(
            discovery, executed=counts["pass"] + counts["fail"], skipped=counts["skipped"],
            invalid=counts["invalid"], infra_error=counts["infra_error"],
        )
        not_evaluated = sorted(n for n, p in discovery.packs.items() if not p.available)
    return {
        "trials": trials,
        "metric": "pass^%d" % trials if trials > 1 else "pass@1",
        "fixture_mode": fixture_mode,
        "total_tasks": len(real),
        "executed": len(executed),
        "skipped": counts["skipped"],
        "invalid": counts["invalid"],
        "infra_error": counts["infra_error"],
        "status_counts": counts,
        # W1-05: which skips a credential would fix, and which ones no
        # credential ever will. `skipped_needs_runtime` counted all five
        # reasons under the name of one of them.
        "skipped_by_reason": {k: v for k, v in skipped_by_reason.items() if v},
        "skipped_needs_runtime": skipped_by_reason["needs_agentic_runtime"],
        "not_selected": not_selected,
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(executed), 4) if executed else None,
        "critical_executed": len(critical),
        "critical_passed": sum(r.passed for r in critical),
        "critical_all_pass": all(r.passed for r in critical) if critical else None,
        "by_category": by_category,
        "by_pack": by_pack,
        # Capability and containment, never folded together (P1-07).
        "answer_outcomes": {
            "answered_trials": answered,
            "first_pass": first_pass,
            "accepted_after_correction": corrected,
            "fallback": fallback_trials,
            "first_pass_rate": round(first_pass / answered, 4) if answered else None,
        },
        "not_evaluated_packs": not_evaluated,
        "conservation_violations": conservation,
        "failures": [
            {"task_id": r.task_id, "critical": r.critical, "reasons": r.reasons}
            for r in executed if not r.passed
        ],
        "infra_errors": [
            {"task_id": r.task_id, "errors": r.infra_errors}
            for r in results if r.status == "infra_error"
        ],
        "invalid_tasks": [
            {"path": r.task_id, "reasons": r.reasons} for r in results if r.status == "invalid"
        ],
    }


def _suite_hash() -> str:
    """The default PR selection's hash (kept for older callers)."""
    return task_packs.suite_hash(task_packs.discover(task_packs.DEFAULT_PACKS))


def _git(ref: str) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", ref], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pinned_predictor_commit() -> str:
    snapshot = HERE.parent / "src" / "toxagent" / "predictor" / "contract_snapshot.json"
    try:
        return json.loads(snapshot.read_text()).get("captured_at_commit", "unknown")
    except (OSError, ValueError):
        return "unknown"


def exit_code(summary: dict[str, Any]) -> int:
    """Non-zero for anything that is not a clean pass of what was selected."""
    if summary.get("invalid") or summary.get("conservation_violations"):
        return 1
    if summary.get("infra_error"):
        return 1
    if summary["critical_all_pass"] is False:
        return 1
    if summary["pass_rate"] is not None and summary["pass_rate"] < 1.0:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="scripted", choices=["scripted", "opencode", "dsh"])
    parser.add_argument(
        "--fixture-mode", default=None, choices=list(FIXTURE_MODES),
        help="what supplies the numbers and evidence (default: frozen for scripted, "
             "predictor_integration for a live stack). Recorded in the manifest; two runs "
             "are comparable only in the same mode.",
    )
    parser.add_argument(
        "--packs", default=None,
        help="comma-separated task packs, 'all', or 'suite:pr|nightly|release' "
             f"(default: {','.join(task_packs.DEFAULT_PACKS)})",
    )
    parser.add_argument("--trials", type=int, default=1)
    judges = parser.add_mutually_exclusive_group()
    judges.add_argument(
        "--judge-command", default=None,
        help="external semantic judge: reads {request, verdict_schema} JSON on stdin, writes a "
             "verdict on stdout. Must not be the product's own model.",
    )
    judges.add_argument("--judge-recorded", type=Path, default=None,
                        help="directory of recorded verdicts, <task_id>.json")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="judge-calibration-v1 report; only calibrated rubrics gate")
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

    packs = task_packs.parse_packs(args.packs)
    suite = args.packs.removeprefix("suite:") if args.packs and args.packs.startswith("suite:") else None
    discovery = task_packs.discover(packs)
    if args.list:
        mode = resolve_fixture_mode(args.runtime, args.fixture_mode)
        print(f"# runtime={args.runtime} fixture_mode={mode} packs={','.join(packs)}")
        print(f"# suite_hash={task_packs.suite_hash(discovery)}")
        for name, record in discovery.packs.items():
            state = "available" if record.available else f"unavailable ({record.unavailable_reason})"
            print(f"# pack {name}: {len(record.files)} file(s), {record.loaded} loaded, "
                  f"{record.invalid} invalid, {state}")
        for task in discovery.tasks:
            if args.runtime == "scripted":
                reason = None if runs_scripted(task) else "needs_agentic_runtime"
            else:
                reason = live_skip_reason(task)
            mark = "yes" if reason is None else "no "
            print(f"{mark}  {task['capability_pack']:10s}  {task['category']:20s}  "
                  f"{task['task_id']:44s}  {reason or ''}")
        for invalid in discovery.invalid:
            print(f"INV  {invalid.pack:10s}  {invalid.path}: {'; '.join(invalid.problems)}")
        return 1 if discovery.invalid else 0

    summary = asyncio.run(
        run_suite(
            runtime=args.runtime, trials=args.trials, out_dir=args.out,
            only=set(args.tasks) if args.tasks else None,
            base_url=args.base_url, token=args.token, fixture_mode=args.fixture_mode,
            discovery=discovery, suite=suite,
            judge=(
                CommandJudge(args.judge_command) if args.judge_command
                else RecordedJudge(args.judge_recorded) if args.judge_recorded else None
            ),
            calibration=json.loads(args.calibration.read_text()) if args.calibration else None,
        )
    )
    print(json.dumps(summary, indent=2))
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
