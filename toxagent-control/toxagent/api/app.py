"""Application factory: the one place the object graph is assembled.

Composition happens here so that every other module receives its dependencies
and none of them reaches for a global. That is also what makes the tests able to
swap the predictor for a stub and the runtime for the scripted one without
touching a single line of workflow code.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..application.create_analysis import CreateAnalysis, CreateAnalysisBatch
from ..application.quick_predict import QuickPredict
from ..application.recognize_structure import RecognizeStructure
from ..application.run_scheduler import RunContext, RunScheduler
from ..application.sessions import SessionService
from ..application.startup_reconciliation import reconcile_orphaned_runs
from ..application.submit_message import SubmitMessage
from ..config import Settings
from ..domain.run import Intent
from ..harness.gateway import AgentRuntimeGateway
from ..harness.provider import AgentRuntimeProvider
from ..persistence.object_store import FilesystemObjectStore, ObjectStore
from ..persistence.sql.database import Database
from ..predictor.client import PredictorClient
from ..predictor.ocr_client import OcrClient
from ..research.interfaces import ResearchProvider
from ..research.providers import build_provider
from ..streaming.events import EventNotifier
from ..tools.bootstrap import build_registry
from ..tools.capability import CapabilityTokenService
from ..tools.mcp_server import mcp_asgi_app
from ..tools.runner import ToolRunner
from . import errors
from .auth import build_auth
from .predict_limits import PredictLimiter
from .routes import health, router

log = logging.getLogger("toxagent.startup")

DESCRIPTION = """\
ToxAgent control plane.

Evidence and decision support over the ToxPred predictor. Every number in an
accepted answer resolves to a stored observation and a field path; hERG, Tox21
and ClinTox stay three separate measurements; and no schema here can carry an
aggregate toxicity or safety verdict.
"""


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    predictor: PredictorClient | None = None,
    runtime_provider: AgentRuntimeProvider | None = None,
    research_provider: ResearchProvider | None = None,
    ocr_client: OcrClient | None = None,
    object_store: ObjectStore | None = None,
    create_schema: bool = False,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if research_provider is None:
        # A deployment fact, resolved once at startup: an unset or unknown
        # provider means the two evidence tools are simply never registered
        # below, not registered and always failing.
        research_provider = build_provider(settings.research)
    if runtime_provider is None and settings.runtime.kind == "opencode":
        # The real V1 adapter is composed explicitly from a pinned deployment
        # setting.  ``scripted`` intentionally remains injection-only so no
        # deterministic test harness can become a production model path.
        from ..harness.adapters.opencode_v1 import OpenCodeV1Provider

        runtime_provider = OpenCodeV1Provider(settings.runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        notifier = EventNotifier()
        db = database or Database(settings.database_url)
        db.set_commit_hook(lambda session_ids: notifier.notify(session_ids))
        if create_schema:
            await db.create_schema()
        # RunScheduler starts with no in-process tasks, so any run this
        # database still has non-terminal was left there by a process that no
        # longer exists (a clean shutdown drains every task first) — fail
        # those honestly before serving any request that could touch them.
        reconciled = await reconcile_orphaned_runs(db)
        if reconciled:
            log.warning("startup reconciliation failed %d orphaned run(s)", reconciled)
        client = predictor or PredictorClient(settings.predictor)
        # A deployment fact, like research_provider: an unset TOXAGENT_OCR_URL
        # means no OCR service exists here, and STRUCTURE_RECOGNITION never
        # gets a scheduler handler registered for it below.
        ocr = ocr_client or (
            OcrClient(
                settings.ocr.base_url,
                timeout_s=settings.ocr.read_timeout_s,
                connect_timeout_s=settings.ocr.connect_timeout_s,
            )
            if settings.ocr.base_url
            else None
        )
        # remaining-plan W4-07: only built when actually needed (OCR
        # configured) and only when the caller (almost always a test) hasn't
        # already injected one — a real deployment's only implementation
        # today is the filesystem adapter (persistence/object_store.py's
        # module docstring explains why there is no GCS adapter yet).
        objects = object_store or (FilesystemObjectStore(settings.object_store_dir) if ocr is not None else None)
        scheduler = RunScheduler(db)

        analysis = CreateAnalysis(db, client, settings.policy)
        batch = CreateAnalysisBatch(db, client, settings.policy)

        async def run_analysis(context: RunContext) -> None:
            await analysis.execute(
                actor=context.actor, session_id=context.session_id, run_id=context.run_id,
                smiles=context.smiles, endpoints=context.endpoints,
                threshold_overrides=context.threshold_overrides,
            )

        async def run_batch(context: RunContext) -> None:
            await batch.execute(
                actor=context.actor, session_id=context.session_id, run_id=context.run_id,
                smiles=list(context.batch_smiles), endpoints=context.endpoints,
                threshold_overrides=context.threshold_overrides,
            )

        scheduler.register(Intent.ANALYSIS, run_analysis)
        scheduler.register(Intent.ANALYSIS_BATCH, run_batch)

        if ocr is not None:
            assert objects is not None
            recognize_structure = RecognizeStructure(db, ocr, analysis, objects)

            async def run_recognize_structure(context: RunContext) -> None:
                assert context.attachment_id is not None
                await recognize_structure.execute(
                    actor=context.actor, session_id=context.session_id, run_id=context.run_id,
                    attachment_id=context.attachment_id,
                    endpoints=context.endpoints, threshold_overrides=context.threshold_overrides,
                )

            scheduler.register(Intent.STRUCTURE_RECOGNITION, run_recognize_structure)

        registry = build_registry(
            db, client, analysis, settings.policy,
            research_provider=research_provider, research_settings=settings.research,
        )
        runner = ToolRunner(registry, db, max_calls_per_run=settings.policy.max_tool_calls_per_run)

        app.state.settings = settings
        app.state.database = db
        app.state.predictor = client
        app.state.notifier = notifier
        app.state.scheduler = scheduler
        app.state.auth = build_auth(settings.security)
        app.state.sessions = SessionService(db)
        # Intent.EVIDENCE_RESEARCH only reaches a runtime turn when this
        # deployment actually has a way to fulfil it (Phase 5) — otherwise
        # submit_message answers capability_unavailable without spending a
        # runtime turn no tool exists for.
        app.state.submit_message = SubmitMessage(
            db, settings.policy, scheduler,
            evidence_research_available=research_provider is not None,
            structure_recognition_available=ocr is not None,
            object_store=objects,
        )
        app.state.create_analysis = analysis
        app.state.create_analysis_batch = batch
        # Stateless Quick Predict path (plan section 3): shares the predictor
        # client and policy with Lane D, shares no session machinery.
        app.state.quick_predict = QuickPredict(client, settings.policy)
        app.state.predict_limits = PredictLimiter(
            max_inflight_per_principal=settings.predict.max_inflight_per_principal,
            max_batch_size=settings.predictor.max_batch_size,
        )
        # Exposed for the stateless OCR proxy (Part B) and the capabilities
        # feature-detect; ``None`` when no OCR service is configured.
        app.state.ocr = ocr
        app.state.tool_registry = registry
        app.state.tool_runner = runner
        app.state.runtime_gateway = None

        # The tool plane needs a signing secret; a deployment with no runtime
        # bound yet (Phase 1) legitimately has none, and stays without an MCP
        # endpoint rather than mounting one nobody can authenticate to.
        if settings.security.capability_secret:
            capability_tokens = CapabilityTokenService(settings.security, db)
            app.state.capability_tokens = capability_tokens
            app.mount(
                settings.security.mcp_path,
                mcp_asgi_app(capability_tokens, registry, runner),
            )
        else:
            app.state.capability_tokens = None

        # The runtime remains an explicitly injected deployment dependency.
        # This prevents the development-only scripted adapter from silently
        # becoming a production model path merely because RuntimeSettings has
        # a convenient default kind.
        if runtime_provider is not None:
            if app.state.capability_tokens is None:
                raise ValueError("an agent runtime requires capability-token signing to be configured")
            if runtime_provider.kind == "opencode" and not settings.security.mcp_runtime_url:
                raise ValueError(
                    "TOXAGENT_MCP_RUNTIME_URL is required when the OpenCode runtime is enabled"
                )
            gateway = AgentRuntimeGateway(
                db,
                registry,
                app.state.capability_tokens,
                runtime_provider,
                settings.runtime,
                create_analysis=analysis,
                mcp_url=settings.security.mcp_runtime_url,
            )

            async def run_agentic(context: RunContext) -> None:
                await gateway.execute(context)

            for intent in (Intent.REPORT_QA, Intent.ATTRIBUTION, Intent.EVIDENCE_RESEARCH):
                scheduler.register(intent, run_agentic)
            app.state.runtime_gateway = gateway

        try:
            yield
        finally:
            await scheduler.drain()
            close_runtime = getattr(runtime_provider, "aclose", None)
            if close_runtime is not None:
                await close_runtime()
            close_research = getattr(research_provider, "aclose", None)
            if close_research is not None:
                await close_research()
            if ocr_client is None and ocr is not None:
                await ocr.aclose()
            if predictor is None:
                await client.aclose()
            if database is None:
                await db.dispose()

    app = FastAPI(
        title="ToxAgent control plane",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    errors.install(app)
    app.include_router(health)
    app.include_router(router)

    if settings.security.cors_allow_origins:
        # Bearer tokens, not cookies, carry auth here, so credentials stay
        # off — a stray "*" origin cannot ride along with a session cookie.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.security.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["authorization", "content-type", "last-event-id"],
            expose_headers=["last-event-id"],
        )

    return app
