"""Application factory.

Models load once at startup. A required model that cannot load fails startup
rather than letting the service answer with something else; an optional one
only removes its own endpoints.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import __version__
from ..application.attribution import AttributionService
from ..application.explain import ExplainService
from ..application.predictor import ToxicityPredictor
from ..domain.molecule import InvalidSmilesError
from ..scientific.artifacts import ArtifactError
from ..scientific.bootstrap import build_registry
from ..settings import Settings
from .errors import artifact_error_handler, invalid_smiles_handler, value_error_handler
from .routes import health_router, v1_router

DESCRIPTION = """\
Headless toxicity prediction.

Three endpoints with distinct meanings, never combined into one verdict:
hERG channel blockade, twelve Tox21 assay activities, and ClinTox clinical-trial
toxicity. Every probability carries the threshold that labelled it, where that
threshold came from, and the artifact that produced it.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.device not in {"cpu", "cuda"}:
            raise RuntimeError("TOXPRED_DEVICE must be exactly 'cpu' or 'cuda'")
        if settings.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("TOXPRED_DEVICE=cuda was requested but CUDA is unavailable")
        registry = build_registry(settings.manifest_path, eager_load=settings.eager_load, device=settings.device)
        app.state.settings = settings
        app.state.registry = registry
        app.state.predictor = ToxicityPredictor(
            registry, max_batch_size=settings.max_batch_size
        )
        app.state.attribution = AttributionService(registry)
        app.state.explain = ExplainService(app.state.attribution)
        yield

    app = FastAPI(
        title="ToxPred",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_exception_handler(InvalidSmilesError, invalid_smiles_handler)
    app.add_exception_handler(ArtifactError, artifact_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.include_router(health_router)
    app.include_router(v1_router)
    return app


app = create_app()
