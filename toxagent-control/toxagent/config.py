"""Runtime configuration.

The one module that reads the environment. Everything else receives resolved
values, so no code path can invent a default halfway down a call stack — the
failure mode that left the predictor's predecessor with five different clinical
thresholds in five different files.

Operational numbers (timeouts, caps) are versioned config, not scientific
semantics: changing one changes cost and latency, never what a prediction means.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env(name)
    return tuple(p.strip() for p in raw.split(",") if p.strip()) if raw else default


@dataclass(frozen=True)
class PredictorSettings:
    base_url: str = "http://127.0.0.1:8080"
    #: Stable identifier for *which* predictor deployment answered. Recorded in
    #: every snapshot; a URL alone would not survive a DNS change.
    base_url_id: str = "toxpred-local"
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 120.0
    attribution_read_timeout_s: float = 180.0
    max_batch_size: int = 256

    @classmethod
    def from_env(cls) -> "PredictorSettings":
        return cls(
            base_url=_env("TOXAGENT_PREDICTOR_URL", cls.base_url).rstrip("/"),
            base_url_id=_env("TOXAGENT_PREDICTOR_ID", cls.base_url_id),
            connect_timeout_s=_float("TOXAGENT_PREDICTOR_CONNECT_TIMEOUT", cls.connect_timeout_s),
            read_timeout_s=_float("TOXAGENT_PREDICTOR_READ_TIMEOUT", cls.read_timeout_s),
            attribution_read_timeout_s=_float(
                "TOXAGENT_PREDICTOR_ATTRIBUTION_TIMEOUT", cls.attribution_read_timeout_s
            ),
            max_batch_size=_int("TOXAGENT_PREDICTOR_MAX_BATCH", cls.max_batch_size),
        )


@dataclass(frozen=True)
class PredictSettings:
    """Abuse control for the stateless ``/v1/predict*`` routes.

    The session path's concurrent-run guard
    (``PolicySettings.max_concurrent_runs_per_session``) does not cover these
    routes; without a cap they are a direct amplification path onto ToxPred.
    This limiter is process-local — a multi-instance deployment gets
    ``N x max_inflight_per_principal`` — and a global limiter is deferred to the
    W9 abuse-controls work.
    """

    #: Per-principal in-flight cap across ``/v1/predict``, ``/v1/predict:batch``,
    #: ``/v1/predict/recognize`` and ``/v1/predict/explain``. A concurrent call
    #: over this returns ``429 provider_rate_limited``.
    max_inflight_per_principal: int = 4

    @classmethod
    def from_env(cls) -> "PredictSettings":
        return cls(
            max_inflight_per_principal=_int(
                "TOXAGENT_PREDICT_MAX_INFLIGHT", cls.max_inflight_per_principal
            ),
        )


@dataclass(frozen=True)
class PolicySettings:
    """Scientific and product policy. DEC-09: threshold overrides are off by
    default and, when enabled, are restricted to an explicit expert role."""

    default_endpoints: tuple[str, ...] = ("herg", "tox21")
    allow_threshold_overrides: bool = False
    threshold_override_roles: tuple[str, ...] = ("expert",)
    max_message_bytes: int = 32_768
    max_batch_size: int = 64
    #: Bounds the upload itself, independent of whether an OCR service is even
    #: configured (see config.OcrSettings) — an oversized image is refused the
    #: same way whether or not the capability behind it exists yet.
    max_image_bytes: int = 5_000_000
    max_concurrent_runs_per_session: int = 1
    #: Bound time spent behind another control-plane instance's short
    #: admission transaction. The caller gets a retryable conflict rather
    #: than accumulating an unbounded request queue on a hot session.
    admission_lock_timeout_ms: int = 1_000
    #: submit_grounded_answer is exempt from this (tools/runner.py); it bounds
    #: only read/search tool calls. A live evidence_research sweep (progress
    #: log §3.13, 2026-09-05) showed the old default of 12 was too small for
    #: its own honest workflow: one search can return up to
    #: ResearchSettings.max_results (10) accepted hits, and a model reading
    #: several of them via get_evidence_record before deciding what to cite —
    #: exactly what plan section 8.4 asks it to do rather than cite from a
    #: search snippet alone — used up the whole budget before it ever reached
    #: submit_grounded_answer, failing the run with no answer at all rather
    #: than a slower but honest one.
    max_tool_calls_per_run: int = 24
    max_answer_candidates_per_run: int = 2
    run_deadline_s: int = 300
    #: A CPU-bound OCSR forward pass (toxocr/, MolScribe) measured ~1-2s per
    #: image under normal load — comfortably inside run_deadline_s already.
    #: This separate, more generous deadline exists only as a safety margin
    #: against a cold model load or a contended host, not because the model
    #: itself is normally slow.
    structure_recognition_deadline_s: int = 1200

    @classmethod
    def from_env(cls) -> "PolicySettings":
        return cls(
            default_endpoints=_list("TOXAGENT_DEFAULT_ENDPOINTS", cls.default_endpoints),
            allow_threshold_overrides=_bool(
                "TOXAGENT_ALLOW_THRESHOLD_OVERRIDES", cls.allow_threshold_overrides
            ),
            threshold_override_roles=_list(
                "TOXAGENT_THRESHOLD_OVERRIDE_ROLES", cls.threshold_override_roles
            ),
            max_message_bytes=_int("TOXAGENT_MAX_MESSAGE_BYTES", cls.max_message_bytes),
            max_batch_size=_int("TOXAGENT_MAX_BATCH_SIZE", cls.max_batch_size),
            max_image_bytes=_int("TOXAGENT_MAX_IMAGE_BYTES", cls.max_image_bytes),
            max_concurrent_runs_per_session=_int(
                "TOXAGENT_MAX_CONCURRENT_RUNS", cls.max_concurrent_runs_per_session
            ),
            admission_lock_timeout_ms=_int(
                "TOXAGENT_ADMISSION_LOCK_TIMEOUT_MS", cls.admission_lock_timeout_ms
            ),
            max_tool_calls_per_run=_int("TOXAGENT_MAX_TOOL_CALLS", cls.max_tool_calls_per_run),
            max_answer_candidates_per_run=_int(
                "TOXAGENT_MAX_ANSWER_CANDIDATES", cls.max_answer_candidates_per_run
            ),
            run_deadline_s=_int("TOXAGENT_RUN_DEADLINE_S", cls.run_deadline_s),
            structure_recognition_deadline_s=_int(
                "TOXAGENT_STRUCTURE_RECOGNITION_DEADLINE_S", cls.structure_recognition_deadline_s
            ),
        )


@dataclass(frozen=True)
class RuntimeSettings:
    """Which agent runtime this deployment binds. ``scripted`` is the
    deterministic in-process runtime; it makes no provider request."""

    kind: str = "scripted"
    opencode_base_url: str = "http://127.0.0.1:4096"
    opencode_version: str = "1.17.11"
    #: Deployment-owned *base* directory for OpenCode runtime projects.  The
    #: V1 adapter derives a fresh child directory from the product run id; it
    #: never accepts a directory from a product request or from the model.
    opencode_directory: str = "/var/lib/toxagent/opencode-workspace"
    #: Connection timeout for the private OpenCode management API.  This is
    #: deliberately much shorter than a model turn; model progress arrives on
    #: the event stream after ``prompt_async`` has been accepted.
    opencode_request_timeout_s: float = 10.0
    #: The ``timeout`` OpenCode V1 puts on the run-scoped remote MCP server. V1
    #: applies it per JSON-RPC request, not only to connection setup, so it must
    #: clear the *longest* tool's hard timeout (``get_attribution``, 180 s in
    #: plan §8.6) or OpenCode aborts a legitimately slow call with
    #: ``-32001 Request timed out``. The old 5 s default did exactly that under
    #: control-plane load even for the ~20 ms ``get_analysis_slice``. The turn is
    #: still bounded by the run deadline in the gateway.
    opencode_mcp_timeout_ms: int = 180_000
    #: Local-only convenience: create and securely reap a child OpenCode
    #: workspace for each product run. Remote runtime hosts keep ownership of
    #: this lifecycle and leave this disabled.
    opencode_create_run_directories: bool = False
    #: Scaffolding for a runtime kind nothing implements yet — there is no
    #: harness/adapters/dsh.py, so TOXAGENT_RUNTIME_KIND=dsh fails to start a
    #: deployment today, by construction (see domain/runtime.py's
    #: RuntimeKind.DSH and ADR 0004's 2026-09-06 correction). `dsh_version`
    #: is pinned to the real, hash-verified spike in ADR 0007
    #: (deepseek-harness-sdk 0.1.2rc1) — not yet a claim that a working
    #: adapter exists at that version, only that this is what an adapter
    #: should target when one is written. `dsh_command` predates that spike
    #: and may not match how the real SDK actually launches its bundled
    #: binary (see ADR 0007) — do not treat it as verified.
    dsh_command: tuple[str, ...] = ("dsh", "sdk-server")
    dsh_version: str = "0.1.2rc1"
    dsh_home: str = ""
    provider_id: str = "scripted"
    model_id: str = "scripted-deterministic"
    agent_name: str = "toxagent"
    turn_deadline_s: int = 180
    #: Recorded in the run audit and, for a runtime whose protocol accepts a
    #: per-request step count, sent as the turn's budget. OpenCode V1 does not
    #: (its ``prompt_async`` has no step field — the checked-in agent profile's
    #: static ``maxSteps`` is the only cap it actually enforces, the same value
    #: for every intent; see agent_profiles/opencode/README.md and progress log
    #: §4.6). Both default to match that profile's ``maxSteps`` (currently 32,
    #: agent_profiles/opencode/toxagent.json) so the audit trail does not
    #: record an intended cap the runtime silently ignores or overrides.
    #:
    #: Live sweep 2026-09-06 (progress log §14.5): OpenCode counts one tool
    #: call as one step, and this was still 8 when max_tool_calls_per_run was
    #: raised to 24 (progress log §3.13) — an evidence_research turn doing
    #: legitimate, budget-compliant work (several searches plus several
    #: get_evidence_record reads, all under 24) hit OpenCode's own lower
    #: ceiling first and ended with no submit_grounded_answer call at all,
    #: never reaching the internal budget this exists to enforce. The model's
    #: own final text, captured by harness/gateway.py's diagnostic log, said
    #: so directly: "Maximum steps reached before submission completed."
    #: Raised to comfortably clear 24 reads plus up to 2 submit attempts.
    max_steps_qa: int = 32
    max_steps_research: int = 32
    #: A run's pre-flight health probe (``AgentRuntimeGateway.execute``, both
    #: a fresh run and a recovery run) retries this many times before giving
    #: up. A runtime that was just restarted after a crash can take a moment
    #: to become fully ready; a single point-in-time probe right after restart
    #: raced that window and failed a recovery run that would have succeeded
    #: a second later (progress log §3.8/§5.2 — observed downtime was ~2s).
    #: ``1`` restores the old no-retry behaviour.
    runtime_health_check_retries: int = 3
    #: Delay between health-probe attempts. Total added latency is bounded by
    #: ``(runtime_health_check_retries - 1) * runtime_health_check_retry_delay_s``,
    #: negligible against ``turn_deadline_s``.
    runtime_health_check_retry_delay_s: float = 1.0

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            kind=_env("TOXAGENT_RUNTIME_KIND", cls.kind),
            opencode_base_url=_env("TOXAGENT_OPENCODE_URL", cls.opencode_base_url).rstrip("/"),
            opencode_version=_env("TOXAGENT_OPENCODE_VERSION", cls.opencode_version),
            opencode_directory=_env("TOXAGENT_OPENCODE_DIRECTORY", cls.opencode_directory),
            opencode_request_timeout_s=_float(
                "TOXAGENT_OPENCODE_REQUEST_TIMEOUT", cls.opencode_request_timeout_s
            ),
            opencode_mcp_timeout_ms=_int(
                "TOXAGENT_OPENCODE_MCP_TIMEOUT_MS", cls.opencode_mcp_timeout_ms
            ),
            opencode_create_run_directories=_bool(
                "TOXAGENT_OPENCODE_CREATE_RUN_DIRECTORIES",
                cls.opencode_create_run_directories,
            ),
            dsh_command=_list("TOXAGENT_DSH_COMMAND", cls.dsh_command),
            dsh_version=_env("TOXAGENT_DSH_VERSION", cls.dsh_version),
            dsh_home=_env("TOXAGENT_DSH_HOME"),
            provider_id=_env("TOXAGENT_PROVIDER_ID", cls.provider_id),
            model_id=_env("TOXAGENT_MODEL_ID", cls.model_id),
            agent_name=_env("TOXAGENT_AGENT_NAME", cls.agent_name),
            turn_deadline_s=_int("TOXAGENT_TURN_DEADLINE_S", cls.turn_deadline_s),
            max_steps_qa=_int("TOXAGENT_MAX_STEPS_QA", cls.max_steps_qa),
            max_steps_research=_int("TOXAGENT_MAX_STEPS_RESEARCH", cls.max_steps_research),
            runtime_health_check_retries=_int(
                "TOXAGENT_RUNTIME_HEALTH_CHECK_RETRIES", cls.runtime_health_check_retries
            ),
            runtime_health_check_retry_delay_s=_float(
                "TOXAGENT_RUNTIME_HEALTH_CHECK_RETRY_DELAY_S",
                cls.runtime_health_check_retry_delay_s,
            ),
        )


@dataclass(frozen=True)
class ResearchSettings:
    provider: str = "europepmc"
    base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    #: Only these hosts may be reached for evidence. A model cannot add one.
    allowed_hosts: tuple[str, ...] = ("www.ebi.ac.uk", "europepmc.org")
    timeout_s: float = 20.0
    hard_timeout_s: float = 45.0
    max_results: int = 10
    contact_email: str = ""
    #: remaining-plan W3-06 ("provider circuit breaker/backoff"): after this
    #: many consecutive failed requests, the provider stops being called at
    #: all for `circuit_reset_after_s` and every call fails fast with
    #: EvidenceUnavailable instead of paying a full timeout each time. See
    #: research/circuit_breaker.py.
    circuit_failure_threshold: int = 5
    circuit_reset_after_s: float = 30.0

    @classmethod
    def from_env(cls) -> "ResearchSettings":
        return cls(
            provider=_env("TOXAGENT_RESEARCH_PROVIDER", cls.provider),
            base_url=_env("TOXAGENT_RESEARCH_URL", cls.base_url).rstrip("/"),
            allowed_hosts=_list("TOXAGENT_RESEARCH_ALLOWED_HOSTS", cls.allowed_hosts),
            timeout_s=_float("TOXAGENT_RESEARCH_TIMEOUT", cls.timeout_s),
            hard_timeout_s=_float("TOXAGENT_RESEARCH_HARD_TIMEOUT", cls.hard_timeout_s),
            max_results=_int("TOXAGENT_RESEARCH_MAX_RESULTS", cls.max_results),
            contact_email=_env("TOXAGENT_RESEARCH_CONTACT"),
            circuit_failure_threshold=_int(
                "TOXAGENT_RESEARCH_CIRCUIT_FAILURE_THRESHOLD", cls.circuit_failure_threshold
            ),
            circuit_reset_after_s=_float(
                "TOXAGENT_RESEARCH_CIRCUIT_RESET_AFTER_S", cls.circuit_reset_after_s
            ),
        )


@dataclass(frozen=True)
class OcrSettings:
    """Optical structure recognition (image -> SMILES). Pluggable like
    ``ResearchSettings``: an empty ``base_url`` means no OCR service is
    configured for this deployment, and ``api/app.py`` never builds a client
    for it — the same shape as an unconfigured research provider, not a
    special case."""

    base_url: str = ""
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "OcrSettings":
        return cls(
            base_url=_env("TOXAGENT_OCR_URL", cls.base_url).rstrip("/"),
            connect_timeout_s=_float("TOXAGENT_OCR_CONNECT_TIMEOUT", cls.connect_timeout_s),
            read_timeout_s=_float("TOXAGENT_OCR_READ_TIMEOUT", cls.read_timeout_s),
        )


@dataclass(frozen=True)
class SecuritySettings:
    """Capability-token signing and the development auth shim.

    ``static_tokens`` exists so the stack runs end to end without an identity
    provider; it is refused when ``environment`` is ``production``.
    """

    capability_secret: str = ""
    capability_ttl_s: int = 900
    capability_grace_s: int = 60
    environment: str = "development"
    static_tokens: tuple[str, ...] = ()
    mcp_path: str = "/internal/mcp"
    #: Absolute private URL reachable from the runtime host.  It intentionally
    #: differs from the public product API URL and is never supplied by a
    #: browser or model.  Required when the real OpenCode adapter is selected.
    mcp_runtime_url: str = ""
    #: Browser origins allowed to call the product API. Empty means no
    #: CORS middleware is mounted at all — a production deployment that
    #: forgets to set this fails closed (no browser client works) rather
    #: than open (every origin works).
    cors_allow_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        settings = cls(
            capability_secret=_env("TOXAGENT_CAPABILITY_SECRET"),
            capability_ttl_s=_int("TOXAGENT_CAPABILITY_TTL_S", cls.capability_ttl_s),
            capability_grace_s=_int("TOXAGENT_CAPABILITY_GRACE_S", cls.capability_grace_s),
            environment=_env("TOXAGENT_ENV", cls.environment),
            static_tokens=_list("TOXAGENT_STATIC_TOKENS", cls.static_tokens),
            mcp_path=_env("TOXAGENT_MCP_PATH", cls.mcp_path),
            mcp_runtime_url=_env("TOXAGENT_MCP_RUNTIME_URL").rstrip("/"),
            cors_allow_origins=_list("TOXAGENT_CORS_ALLOW_ORIGINS", cls.cors_allow_origins),
        )
        if settings.environment == "production":
            if not settings.capability_secret:
                raise ValueError("TOXAGENT_CAPABILITY_SECRET is required in production")
            if settings.static_tokens:
                raise ValueError(
                    "TOXAGENT_STATIC_TOKENS is a development shim and must be empty in production"
                )
        return settings


@dataclass(frozen=True)
class Settings:
    database_url: str
    predictor: PredictorSettings
    policy: PolicySettings
    predict: PredictSettings
    runtime: RuntimeSettings
    research: ResearchSettings
    ocr: OcrSettings
    security: SecuritySettings
    profiles_dir: Path = PROJECT_ROOT / "agent_profiles"
    #: remaining-plan W4-07: where a FilesystemObjectStore (the only
    #: implementation this deployment has today — see
    #: persistence/object_store.py's module docstring on why there is no GCS
    #: adapter yet) persists uploaded attachment bytes when no object_store
    #: is injected explicitly (tests always inject one; api/app.py's real
    #: default construction is what reads this).
    object_store_dir: Path = PROJECT_ROOT / ".data" / "attachments"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=_env("TOXAGENT_DATABASE_URL", "sqlite+aiosqlite:///./toxagent.db"),
            predictor=PredictorSettings.from_env(),
            policy=PolicySettings.from_env(),
            predict=PredictSettings.from_env(),
            runtime=RuntimeSettings.from_env(),
            research=ResearchSettings.from_env(),
            ocr=OcrSettings.from_env(),
            security=SecuritySettings.from_env(),
            profiles_dir=Path(_env("TOXAGENT_PROFILES_DIR") or PROJECT_ROOT / "agent_profiles"),
            object_store_dir=Path(
                _env("TOXAGENT_OBJECT_STORE_DIR") or PROJECT_ROOT / ".data" / "attachments"
            ),
        )
