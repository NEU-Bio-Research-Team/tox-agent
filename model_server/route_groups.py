from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter

from model_server.schemas import (
    AnalyzeResponse,
    AgentAnalyzeResponse,
    AgentChatResponse,
    BatchPredictResponse,
    ExplainResponse,
    PredictResponse,
    SmilesImageExtractionResponse,
    SmilesPreviewResponse,
)


RouteHandler = Callable[..., Any]


def build_system_router(
    *,
    health_handler: RouteHandler,
    health_alias: Optional[str] = None,
) -> APIRouter:
    router = APIRouter(tags=["system"])
    router.add_api_route("/health", health_handler, methods=["GET"])

    if health_alias and health_alias != "/health":
        router.add_api_route(health_alias, health_handler, methods=["GET"], include_in_schema=False)

    return router


def build_inference_api_router(
    *,
    image_extract_handler: RouteHandler,
    smiles_preview_handler: RouteHandler,
    predict_handler: RouteHandler,
    batch_predict_handler: RouteHandler,
    explain_handler: RouteHandler,
    analyze_handler: RouteHandler,
    agent_analyze_handler: RouteHandler,
    predict_alias: Optional[str] = None,
) -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/extract-smiles-from-image",
        image_extract_handler,
        methods=["POST"],
        response_model=SmilesImageExtractionResponse,
    )
    router.add_api_route(
        "/smiles/preview",
        smiles_preview_handler,
        methods=["POST"],
        response_model=SmilesPreviewResponse,
    )
    router.add_api_route(
        "/predict",
        predict_handler,
        methods=["POST"],
        response_model=PredictResponse,
    )

    if predict_alias and predict_alias != "/predict":
        router.add_api_route(
            predict_alias,
            predict_handler,
            methods=["POST"],
            response_model=PredictResponse,
            include_in_schema=False,
        )

    router.add_api_route(
        "/predict/batch",
        batch_predict_handler,
        methods=["POST"],
        response_model=BatchPredictResponse,
    )
    router.add_api_route(
        "/explain",
        explain_handler,
        methods=["POST"],
        response_model=ExplainResponse,
    )
    router.add_api_route(
        "/analyze",
        analyze_handler,
        methods=["POST"],
        response_model=AnalyzeResponse,
    )
    router.add_api_route(
        "/agent/analyze",
        agent_analyze_handler,
        methods=["POST"],
        response_model=AgentAnalyzeResponse,
    )
    return router


def build_report_chat_router(
    *,
    chat_handler: RouteHandler,
    chat_stream_handler: RouteHandler,
) -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/agent/chat",
        chat_handler,
        methods=["POST"],
        response_model=AgentChatResponse,
    )
    router.add_api_route("/agent/chat/stream", chat_stream_handler, methods=["POST"])
    return router