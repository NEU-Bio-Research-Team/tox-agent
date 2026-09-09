"""Application factory. See toxocr/__init__.py for what this service is and
why it exists outside both toxpred and toxagent-control."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import __version__
from ..scientific.molscribe_predictor import (
    ImageDecodeError,
    MolScribePredictor,
    StructureNotDetected,
)
from ..settings import Settings
from .errors import image_decode_error_handler, structure_not_detected_handler
from .routes import health_router, v1_router

DESCRIPTION = """\
Optical chemical structure recognition: an uploaded image of a 2D structure in,
a SMILES string out. One endpoint, one job — this service knows nothing about
toxicity prediction and nothing about a chat session; toxagent-control decides
what to do with the SMILES it gets back.
"""


def create_app(settings: Settings | None = None, *, predictor: MolScribePredictor | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = predictor or MolScribePredictor(settings)
        if settings.eager_load:
            model.preload()
        app.state.settings = settings
        app.state.predictor = model
        yield

    app = FastAPI(title="ToxOCR", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    app.add_exception_handler(ImageDecodeError, image_decode_error_handler)
    app.add_exception_handler(StructureNotDetected, structure_not_detected_handler)
    app.include_router(health_router)
    app.include_router(v1_router)
    return app


app = create_app()
