# model_server/main.py
"""
ToxAgent Model Server - FastAPI wrapper around SMILESGNN (XSmiles)

Endpoints: 
    GET /health         -> health check
    POST /predict       -> single molecule toxicity prediction
    POST /predict/batch -> batch prediction
    POST /explain       -> GNNExplainer atom/bond attribution
    POST /analyze       -> unified 3-output pipeline (clinical + mechanism + explainer)

Deployment: Docker container -> Google Cloud Run
"""

import sys
import asyncio
import base64
import io
import json
import logging
import os
import re
import uuid
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rdkit import Chem
from starlette.exceptions import HTTPException as StarletteHTTPException

# Important: Add project root to sys.path so project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    # Local runs can configure API keys and model options in repo-level env files.
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.local")

from agents import (
    ADK_AVAILABLE,
    build_final_report,
    researcher_agent,
    root_agent,
    run_orchestrator_flow,
    run_screening,
    screening_agent,
    writer_agent,
)
from agents.language import normalize_language

ADK_RUNTIME_IMPORT_ERROR: Optional[str] = None
try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
except Exception as exc:
    Runner = None
    InMemorySessionService = None
    Content = None
    Part = None
    ADK_RUNTIME_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from backend.inference import (
    aggregate_toxicity_verdict,
    load_model,
    load_tox21_gatv2_model,
    predict_batch,
    predict_clinical_toxicity,
    predict_toxicity_mechanism,
)
from backend.ood_guard import check_ood_risk
from backend.graph_data import smiles_to_pyg_data
from backend.gnn_explainer import explain_molecule, explain_tox21_task, visualize_explanation
from backend.workspace_mode import get_workspace_mode
from model_server.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    AgentEventRecord,
    ClinicalToxicityOutput,
    InferenceContextOutput,
    MechanismToxicityOutput,
    OodAssessmentOutput,
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ExplainRequest, ExplainResponse,
    AtomImportance, BondImportance,
    ToxicityExplanationOutput,
)

# Configuration
MODEL_DIR = PROJECT_ROOT / "models" / "smilesgnn_model"
CONFIG_PATH = PROJECT_ROOT / "config" / "smilesgnn_config.yaml"
TOX21_MODEL_DIR = PROJECT_ROOT / "models" / "tox21_gatv2_model"
TOX21_CONFIG_PATH = PROJECT_ROOT / "config" / "tox21_gatv2_config.yaml"
CLINICAL_METRICS_PATH = MODEL_DIR / "smilesgnn_model_metrics.txt"
CLINICAL_THRESHOLD_METRICS_PATH = MODEL_DIR / "clinical_threshold_metrics.json"
TOX21_THRESHOLDS_CANDIDATES = [
    TOX21_MODEL_DIR / "tox21_task_thresholds.json",
    TOX21_MODEL_DIR / "task_thresholds.json",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKSPACE_MODE = get_workspace_mode()
WORKSPACE_MODE_NAME = str(WORKSPACE_MODE.get("mode", "unknown"))
CLINTOX_ENABLED = bool(WORKSPACE_MODE.get("clintox_enabled", True))
TOX21_ENABLED = bool(WORKSPACE_MODE.get("tox21_enabled", True))
XSMILES_LOAD_REQUESTED = CLINTOX_ENABLED or MODEL_DIR.exists()

def _normalize_route(route: str, default: str) -> str:
    if not route:
        return default
    return route if route.startswith("/") else f"/{route}"

AIP_HEALTH_ROUTE = _normalize_route(os.getenv("AIP_HEALTH_ROUTE", "/health"), "/health")
AIP_PREDICT_ROUTE = _normalize_route(os.getenv("AIP_PREDICT_ROUTE", "/predict"), "/predict")
ADK_APP_NAME = os.getenv("AGENT_APP_NAME", "tox-agent")

adk_session_service = None
adk_runner = None


def _load_tox21_thresholds() -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Load per-task Tox21 thresholds from known calibration filenames."""
    for path in TOX21_THRESHOLDS_CANDIDATES:
        if not path.exists():
            continue

        with open(path, "r") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items()}, str(path)

        logger.warning("Ignoring threshold file with non-dict payload: %s", path)

    return None, None


def _load_clinical_reference_metrics(path: Path) -> Dict[str, float]:
    """Parse key:value metric lines from stored benchmark artifact."""
    if not path.exists():
        return {}

    parsed: Dict[str, float] = {}
    try:
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
    except Exception:
        return {}

    return parsed


def _load_optional_json_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    parsed: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_server")

# Lifespan: load model once at startup
model_state = {}


def _safe_model_dump(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            dumped = as_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {"value": str(value)}


def _strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _coerce_json_dict(value: Any, nested_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    parsed: Optional[Dict[str, Any]] = None

    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        candidate = _strip_markdown_code_fence(value)
        try:
            loaded = json.loads(candidate)
        except Exception:
            loaded = None
            match = re.search(r"\{[\s\S]*\}", candidate)
            if match:
                try:
                    loaded = json.loads(match.group(0))
                except Exception:
                    loaded = None
        if isinstance(loaded, dict):
            parsed = loaded

    if parsed is None:
        return None

    if nested_key and isinstance(parsed.get(nested_key), dict):
        return parsed[nested_key]

    return parsed


def _extract_state_payload(state: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Read state payloads that may be stored as nested JSON strings by ADK."""
    payload = _coerce_json_dict(state.get(key), nested_key=key)
    return payload if isinstance(payload, dict) else {}


def _recover_screening_payload_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    for _, raw_value in state.items():
        candidate = _coerce_json_dict(raw_value)
        if not isinstance(candidate, dict):
            continue

        nested = candidate.get("screening_result")
        if isinstance(nested, dict):
            return nested

        if isinstance(candidate.get("clinical"), dict) and isinstance(candidate.get("mechanism"), dict):
            return candidate

    return {}


def _recover_research_payload_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    for _, raw_value in state.items():
        candidate = _coerce_json_dict(raw_value)
        if not isinstance(candidate, dict):
            continue

        nested = candidate.get("research_result")
        if isinstance(nested, dict):
            return nested

        if isinstance(candidate.get("compound_info"), dict) or isinstance(candidate.get("literature"), dict):
            return candidate

    return {}


def _recover_final_report_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    for _, raw_value in state.items():
        candidate = _coerce_json_dict(raw_value)
        if not isinstance(candidate, dict):
            continue

        nested = candidate.get("final_report")
        if isinstance(nested, dict):
            return nested

        if isinstance(candidate.get("report_metadata"), dict) and isinstance(candidate.get("sections"), dict):
            return candidate

    return {}


def _is_final_report_schema_complete(final_report: Dict[str, Any]) -> bool:
    if not isinstance(final_report, dict) or not final_report:
        return False

    sections = final_report.get("sections")
    if not isinstance(sections, dict) or not sections:
        return False

    clinical = sections.get("clinical_toxicity")
    mechanism = sections.get("mechanism_toxicity")
    structural = sections.get("structural_explanation")
    recommendations = sections.get("recommendations")

    if not isinstance(clinical, dict) or not isinstance(mechanism, dict) or not isinstance(structural, dict):
        return False

    # Frontend and report consumers depend on these canonical keys.
    if "probability" not in clinical and "p_toxic" not in clinical:
        return False
    if "task_scores" not in mechanism:
        return False

    structural_payload: Dict[str, Any]
    if isinstance(structural.get("data"), dict):
        structural_payload = structural.get("data")
    else:
        structural_payload = structural

    if not any(
        key in structural_payload
        for key in ("molecule_png_base64", "heatmap_base64", "top_atoms", "top_bonds")
    ):
        return False

    if not isinstance(recommendations, list):
        return False

    return True


def _screening_payload_has_structural_data(screening_payload: Dict[str, Any]) -> bool:
    if not isinstance(screening_payload, dict):
        return False

    explanation = screening_payload.get("explanation")
    if not isinstance(explanation, dict):
        return False

    return bool(
        explanation.get("molecule_png_base64")
        or explanation.get("heatmap_base64")
        or explanation.get("top_atoms")
        or explanation.get("top_bonds")
    )


def _final_report_missing_structural_images(final_report: Dict[str, Any]) -> bool:
    if not isinstance(final_report, dict) or not final_report:
        return True

    sections = final_report.get("sections")
    if not isinstance(sections, dict):
        return True

    structural = sections.get("structural_explanation")
    if not isinstance(structural, dict):
        return True

    payload = structural.get("data") if isinstance(structural.get("data"), dict) else structural
    return not bool(payload.get("molecule_png_base64") or payload.get("heatmap_base64"))


def _is_vertex_model_not_found_error(exc: Exception) -> bool:
    message = str(exc or "")
    lowered = message.lower()
    return (
        "404" in lowered
        and "not_found" in lowered
        and "publishers/google/models" in lowered
    )


def _is_vertex_resource_exhausted_error(exc: Exception) -> bool:
    message = str(exc or "")
    lowered = message.lower()
    return (
        "429" in lowered
        or "resource_exhausted" in lowered
        or "rate exceeded" in lowered
        or "quota" in lowered
    )


def _resolve_fast_fallback_model() -> str:
    return (
        os.getenv("AGENT_MODEL_FAST")
        or os.getenv("GEMINI_MODEL")
        or "gemini-2.5-flash"
    ).strip()


def _resolve_pro_fallback_model() -> str:
    return (
        os.getenv("AGENT_MODEL_PRO")
        or "gemini-2.5-pro"
    ).strip()


def _resolve_quota_retry_model(current_model: Any) -> str:
    current = str(current_model or "").strip().lower()
    fast_model = _resolve_fast_fallback_model()
    pro_model = _resolve_pro_fallback_model()

    # Prefer cross-tier retry to escape per-model quota pressure.
    if "flash" in current:
        return pro_model
    if "pro" in current:
        return fast_model

    if current and current == fast_model.lower():
        return pro_model
    if current and current == pro_model.lower():
        return fast_model

    return fast_model


@contextmanager
def _temporary_agent_model(agent: Any, model_name: str):
    previous_model = getattr(agent, "model", None)
    changed = False

    requested = str(model_name or "").strip()
    current = str(previous_model or "").strip()

    if requested and requested.lower() != current.lower():
        try:
            setattr(agent, "model", requested)
            changed = True
        except Exception:
            changed = False

    try:
        yield changed, previous_model
    finally:
        if changed:
            try:
                setattr(agent, "model", previous_model)
            except Exception:
                pass


async def _read_adk_session_state_with_retry(
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    retries: int = 20,
    delay_seconds: float = 0.25,
) -> Dict[str, Any]:
    last_state: Dict[str, Any] = {}

    for attempt in range(retries):
        try:
            session = await adk_session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            state = getattr(session, "state", {}) if session is not None else {}
            if not isinstance(state, dict):
                state = {}
        except Exception:
            state = {}

        last_state = state
        final_report = _extract_state_payload(state, "final_report")
        screening_payload = _extract_state_payload(state, "screening_result")
        research_payload = _extract_state_payload(state, "research_result")

        # Wait for either final report or enough intermediate payloads to rebuild.
        if final_report or (screening_payload and research_payload) or attempt == retries - 1:
            return state

        await asyncio.sleep(delay_seconds)

    return last_state


async def _run_adk_agent_step(
    *,
    agent: Any,
    user_id: str,
    session_id: str,
    message: Any,
) -> None:
    if Runner is None or adk_session_service is None:
        return

    step_runner = Runner(
        agent=agent,
        session_service=adk_session_service,
        app_name=ADK_APP_NAME,
    )

    async def _run_with_runner(runner: Any) -> None:
        async for _ in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            pass

    try:
        await _run_with_runner(step_runner)
        return
    except Exception as exc:
        if "aclose(): asynchronous generator is already running" in str(exc):
            logger.warning("ADK step stream-close race for %s: %s", getattr(agent, "name", "agent"), exc)
            return

        if _is_vertex_model_not_found_error(exc):
            current_location = (
                os.getenv("GEMINI_LOCATION")
                or os.getenv("GOOGLE_CLOUD_LOCATION")
                or ""
            ).strip().lower()

            if current_location != "global":
                logger.warning(
                    "ADK step model unavailable in location=%s for %s; retrying with global",
                    current_location or "unset",
                    getattr(agent, "name", "agent"),
                )
                os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
                os.environ["GEMINI_LOCATION"] = "global"

                retry_runner = Runner(
                    agent=agent,
                    session_service=adk_session_service,
                    app_name=ADK_APP_NAME,
                )
                try:
                    await _run_with_runner(retry_runner)
                    return
                except Exception as retry_exc:
                    if "aclose(): asynchronous generator is already running" in str(retry_exc):
                        logger.warning(
                            "ADK step stream-close race for %s after global retry: %s",
                            getattr(agent, "name", "agent"),
                            retry_exc,
                        )
                    else:
                        logger.warning(
                            "ADK step failed for %s after global retry: %s",
                            getattr(agent, "name", "agent"),
                            retry_exc,
                        )
                    return

        if _is_vertex_resource_exhausted_error(exc):
            fallback_model = _resolve_quota_retry_model(getattr(agent, "model", None))
            with _temporary_agent_model(agent, fallback_model) as (changed, previous_model):
                if changed:
                    logger.warning(
                        "ADK step quota exhausted for %s with model=%s; retrying with model=%s",
                        getattr(agent, "name", "agent"),
                        previous_model,
                        fallback_model,
                    )
                    retry_runner = Runner(
                        agent=agent,
                        session_service=adk_session_service,
                        app_name=ADK_APP_NAME,
                    )
                    try:
                        await _run_with_runner(retry_runner)
                        return
                    except Exception as retry_exc:
                        if "aclose(): asynchronous generator is already running" in str(retry_exc):
                            logger.warning(
                                "ADK step stream-close race for %s after model retry: %s",
                                getattr(agent, "name", "agent"),
                                retry_exc,
                            )
                        else:
                            logger.warning(
                                "ADK step failed for %s after model retry: %s",
                                getattr(agent, "name", "agent"),
                                retry_exc,
                            )
                        return

        logger.warning("ADK step failed for %s: %s", getattr(agent, "name", "agent"), exc)


def _initialize_adk_runtime() -> None:
    global adk_session_service, adk_runner

    adk_session_service = None
    adk_runner = None

    if not ADK_AVAILABLE:
        logger.info("ADK runtime disabled (ADK_AVAILABLE=False)")
        return

    if Runner is None or InMemorySessionService is None:
        logger.warning("ADK runtime import unavailable: %s", ADK_RUNTIME_IMPORT_ERROR)
        return

    try:
        adk_session_service = InMemorySessionService()
        adk_runner = Runner(
            agent=root_agent,
            session_service=adk_session_service,
            app_name=ADK_APP_NAME,
        )
        logger.info("ADK runtime initialized (app_name=%s)", ADK_APP_NAME)
    except Exception as exc:
        adk_session_service = None
        adk_runner = None
        err = f"{type(exc).__name__}: {exc}"
        startup_errors = model_state.setdefault("startup_errors", {})
        startup_errors["adk_runtime"] = err
        logger.warning("ADK runtime initialization FAILED: %s", err)


def _extract_event_function_calls(event: Any) -> List[Dict[str, Any]]:
    getter = getattr(event, "get_function_calls", None)
    if not callable(getter):
        return []

    try:
        calls = getter() or []
    except Exception:
        return []

    payload: List[Dict[str, Any]] = []
    for call in calls:
        name = getattr(call, "name", None)
        if name is None and isinstance(call, dict):
            name = call.get("name")

        args = getattr(call, "args", None)
        if args is None and isinstance(call, dict):
            args = call.get("args")

        payload.append(
            {
                "name": name,
                "args": args if isinstance(args, dict) else _safe_model_dump(args),
            }
        )

    return payload


def _extract_event_function_responses(event: Any) -> List[Dict[str, Any]]:
    getter = getattr(event, "get_function_responses", None)
    if not callable(getter):
        return []

    try:
        responses = getter() or []
    except Exception:
        return []

    payload: List[Dict[str, Any]] = []
    for response in responses:
        name = getattr(response, "name", None)
        if name is None and isinstance(response, dict):
            name = response.get("name")
        payload.append(
            {
                "name": name,
                "response": _safe_model_dump(response),
            }
        )

    return payload


def _extract_event_text_preview(event: Any) -> Optional[str]:
    content = getattr(event, "content", None)
    if content is None:
        return None
    parts = getattr(content, "parts", None)
    if not isinstance(parts, list):
        return None

    chunks: List[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(str(text))

    if not chunks:
        return None
    return " ".join(chunks)[:500]


def _is_final_event_response(event: Any) -> bool:
    checker = getattr(event, "is_final_response", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass

    event_type = getattr(event, "type", None)
    return isinstance(event_type, str) and "final" in event_type.lower()


def _agent_runtime_unavailable_detail() -> Dict[str, str]:
    startup_error = _startup_errors().get("adk_runtime")
    if startup_error:
        return {
            "error": "adk_runtime_unavailable",
            "message": startup_error,
        }
    if not ADK_AVAILABLE:
        return {
            "error": "adk_unavailable",
            "message": "ADK compatibility layer is active (google-adk import failed in agents).",
        }
    if ADK_RUNTIME_IMPORT_ERROR:
        return {
            "error": "adk_import_error",
            "message": ADK_RUNTIME_IMPORT_ERROR,
        }
    return {
        "error": "adk_runner_not_initialized",
        "message": "ADK runner was not initialized.",
    }


def _startup_errors() -> Dict[str, str]:
    return model_state.get("startup_errors", {})


def _xsmiles_ready() -> bool:
    return all(k in model_state for k in ("model", "tokenizer", "wrapped"))


def _tox21_ready() -> bool:
    return all(k in model_state for k in ("tox21_model", "tox21_tasks"))


def _required_models_ready() -> bool:
    required = []
    if CLINTOX_ENABLED:
        required.append(_xsmiles_ready())
    if TOX21_ENABLED:
        required.append(_tox21_ready())
    return bool(required) and all(required)


def _feature_disabled_error(feature: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "error": "feature_disabled",
            "feature": feature,
            "workspace_mode": WORKSPACE_MODE_NAME,
            "message": (
                f"{feature} is disabled in workspace mode '{WORKSPACE_MODE_NAME}'."
            ),
        },
    )


def _feature_not_ready_error(feature: str) -> HTTPException:
    detail = {
        "error": "model_not_ready",
        "feature": feature,
        "workspace_mode": WORKSPACE_MODE_NAME,
        "message": f"{feature} model is not loaded.",
    }
    startup_error = _startup_errors().get(feature)
    if startup_error:
        detail["startup_error"] = startup_error
    return HTTPException(status_code=503, detail=detail)


def _ensure_xsmiles_available() -> None:
    if _xsmiles_ready():
        return
    if not CLINTOX_ENABLED and not XSMILES_LOAD_REQUESTED:
        raise _feature_disabled_error("xsmiles")
    raise _feature_not_ready_error("xsmiles")


def _ensure_tox21_available() -> None:
    if not TOX21_ENABLED:
        raise _feature_disabled_error("tox21")
    if not _tox21_ready():
        raise _feature_not_ready_error("tox21")


def _fallback_mechanism_result(threshold: float) -> Dict[str, object]:
    return {
        "task_scores": {},
        "active_tasks": [],
        "highest_risk_task": "UNAVAILABLE_NO_TOX21_MODEL",
        "highest_risk_score": 0.0,
        "assay_hits": 0,
        "threshold_used": float(threshold),
        "task_thresholds": {},
    }


def _fallback_explanation(
    target_task: Optional[str] = None,
    mol: Optional[Chem.Mol] = None,
) -> ToxicityExplanationOutput:
    task = target_task or "UNAVAILABLE_NO_TOX21_MODEL"
    molecule_png_base64 = _render_molecule_png(mol)
    return ToxicityExplanationOutput(
        target_task=task,
        target_task_score=0.0,
        top_atoms=[],
        top_bonds=[],
        heatmap_base64=None,
        molecule_png_base64=molecule_png_base64,
        explainer_note=(
            "Tox21 model unavailable. Returning placeholder explanation payload for API contract testing."
        ),
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Loading production models on %s (workspace_mode=%s, clintox_enabled=%s, tox21_enabled=%s)...",
        DEVICE,
        WORKSPACE_MODE_NAME,
        CLINTOX_ENABLED,
        TOX21_ENABLED,
    )

    model_state["model_lock"] = asyncio.Lock()
    model_state["startup_errors"] = {}

    if XSMILES_LOAD_REQUESTED:
        try:
            model, tokenizer, wrapped_model = load_model(
                MODEL_DIR,
                CONFIG_PATH,
                DEVICE,
                enforce_workspace_mode=False,
            )
            model_state["model"] = model
            model_state["tokenizer"] = tokenizer
            model_state["wrapped"] = wrapped_model
            clinical_reference_metrics = _load_clinical_reference_metrics(CLINICAL_METRICS_PATH)
            clinical_reference_metrics.update(
                _load_optional_json_metrics(CLINICAL_THRESHOLD_METRICS_PATH)
            )
            model_state["clinical_reference_metrics"] = clinical_reference_metrics
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            model_state["startup_errors"]["xsmiles"] = msg
            logger.warning("XSmiles model load FAILED: %s", msg)
    else:
        logger.info(
            "Skipping XSmiles model load because clintox_enabled=false and model dir is absent"
        )

    if TOX21_ENABLED:
        try:
            tox21_model, tox21_tasks = load_tox21_gatv2_model(
                model_dir=TOX21_MODEL_DIR,
                config_path=TOX21_CONFIG_PATH,
                device=DEVICE,
            )
            task_thresholds, task_threshold_source = _load_tox21_thresholds()
            model_state["tox21_model"] = tox21_model
            model_state["tox21_tasks"] = tox21_tasks
            model_state["tox21_thresholds"] = task_thresholds
            model_state["tox21_thresholds_source"] = task_threshold_source
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            model_state["startup_errors"]["tox21"] = msg
            logger.warning("Tox21 model load FAILED: %s", msg)
    else:
        logger.info("Skipping Tox21 model load because tox21_enabled=false")

    _initialize_adk_runtime()

    if _required_models_ready():
        logger.info("Production models loaded successfully.")
    else:
        logger.warning(
            "Model server started in degraded mode. startup_errors=%s",
            _startup_errors(),
        )
    yield
    model_state.clear()

# FastAPI App
app = FastAPI(
    title="ToxAgent Model Server",
    description="SMILESGNN toxicity prediction API for ToxAgent agentic system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return structured API errors when details provide explicit error payloads."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def _invalid_smiles_error(smiles: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": "invalid_smiles",
            "message": f"Cannot parse SMILES: {smiles}",
            "smiles": smiles,
        },
    )


def _model_lock() -> asyncio.Lock:
    lock = model_state.get("model_lock")
    if lock is None:
        lock = asyncio.Lock()
        model_state["model_lock"] = lock
    return lock


def _explainer_timeout_error(smiles: str, timeout_ms: int) -> HTTPException:
    return HTTPException(
        status_code=504,
        detail={
            "error": "explainer_timeout",
            "message": f"Explainer exceeded timeout of {timeout_ms} ms",
            "smiles": smiles,
            "timeout_ms": int(timeout_ms),
        },
    )

# Health Check
async def health():
    xsmiles_ready = _xsmiles_ready()
    tox21_ready = _tox21_ready()
    model_ready = _required_models_ready()
    startup_errors = _startup_errors()

    if model_ready:
        status = "healthy"
    elif startup_errors:
        status = "degraded"
    else:
        status = "starting"

    payload = {
        "status": status,
        "model_loaded": model_ready,
        "xsmiles_loaded": xsmiles_ready,
        "tox21_loaded": tox21_ready,
        "clintox_enabled": CLINTOX_ENABLED,
        "tox21_enabled": TOX21_ENABLED,
        "workspace_mode": WORKSPACE_MODE_NAME,
        "startup_errors": startup_errors,
        "adk_available": ADK_AVAILABLE,
        "adk_runtime_ready": adk_runner is not None and adk_session_service is not None,
        "adk_runtime_import_error": ADK_RUNTIME_IMPORT_ERROR,
        "tox21_thresholds_loaded": model_state.get("tox21_thresholds") is not None,
        "tox21_thresholds_source": model_state.get("tox21_thresholds_source"),
        "tox21_threshold_count": len(model_state.get("tox21_thresholds") or {}),
        "inference_lock_initialized": "model_lock" in model_state,
        "device": DEVICE,
        "model_dir_exists": MODEL_DIR.exists(),
        "tox21_model_dir_exists": TOX21_MODEL_DIR.exists(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
    }
    status_code = 200 if model_ready else 503
    return JSONResponse(content=payload, status_code=status_code)


def _render_explanation_heatmap(result: dict) -> str:
    """Render explanation panel to base64 PNG."""
    buf = io.BytesIO()
    # Match scripts/explain_smilesgnn.py default rendering.
    visualize_explanation(result, save_path=buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _render_molecule_png(
    mol: Chem.Mol,
    size: Tuple[int, int] = (800, 420),
) -> Optional[str]:
    """Render plain molecule structure as base64 PNG for UI fallback."""
    try:
        # Import lazily so missing optional shared libs do not break server startup.
        from rdkit.Chem import Draw

        image = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None

app.add_api_route("/health", health, methods=["GET"], tags=["system"])
if AIP_HEALTH_ROUTE != "/health":
    app.add_api_route(AIP_HEALTH_ROUTE, health, methods=["GET"], include_in_schema=False)

# Single Prediction
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict clinical toxicity for a single SMILES molecule"""
    _ensure_xsmiles_available()
    
    smiles = req.smiles.strip()

    # Validate and canonicalize SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _invalid_smiles_error(smiles)
    canonical = Chem.MolToSmiles(mol) if mol else None

    try: 
        async with _model_lock():
            results_df = predict_batch(
                smiles_list=[smiles],
                tokenizer=model_state["tokenizer"],
                wrapped_model=model_state["wrapped"],
                device=DEVICE,
                threshold=req.threshold,
                enforce_workspace_mode=False,
            )
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

    row = results_df.iloc[0]
    p_toxic = float(row["P(toxic)"]) if row["P(toxic)"] is not None else 0.0
    predicted = str(row["Predicted"])

    # Map predicted label to uppercase for consistency
    label_map = {"Toxic": "TOXIC", "Non-toxic": "NON_TOXIC", "Parse error": "PARSE_ERROR"}
    label = label_map.get(predicted, "UNKNOWN")

    # Confidence: distance from threshold
    confidence = abs(p_toxic - req.threshold) / max(req.threshold, 1 - req.threshold)

    return PredictResponse(
        smiles=smiles,
        canonical_smiles=canonical,
        p_toxic=p_toxic,
        label=label,
        confidence=min(confidence, 1.0),
        threshold_used=req.threshold,
    )

if AIP_PREDICT_ROUTE != "/predict":
    app.add_api_route(
        AIP_PREDICT_ROUTE,
        predict,
        methods=["POST"],
        response_model=PredictResponse,
        include_in_schema=False,
    )

# Batch Prediction
@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(req: BatchPredictRequest):
    """Predict clinical toxicity for a list of SMILES molecules."""
    _ensure_xsmiles_available()
    
    if len(req.smiles_list) > 500:
        raise HTTPException(400, "Batch size limited to 500 molecules")
    
    try:
        async with _model_lock():
            results_df = predict_batch(
                smiles_list=req.smiles_list,
                tokenizer=model_state["tokenizer"],
                wrapped_model=model_state["wrapped"],
                device=DEVICE,
                threshold=req.threshold,
                enforce_workspace_mode=False,
            )

    except Exception as e:
        raise HTTPException(500, f"Batch inference error: {str(e)}")
    
    results = []
    for _, row in results_df.iterrows():
        p = float(row["P(toxic)"]) if row["P(toxic)"] is not None else 0.0
        predicted = str(row["Predicted"])
        label_map = {"Toxic": "TOXIC", "Non-toxic": "NON_TOXIC", "Parse error": "PARSE_ERROR"}
        label = label_map.get(predicted, "UNKNOWN")
        confidence = abs(p - req.threshold) / max(req.threshold, 1 - req.threshold)
        results.append(PredictResponse(
            smiles=row.get("SMILES", ""),
            canonical_smiles=None,
            p_toxic=p,
            label=label,
            confidence=min(confidence, 1.0),
            threshold_used=req.threshold,
        ))

    n_toxic = sum(1 for r in results if r.label == "TOXIC")
    n_non_toxic = sum(1 for r in results if r.label == "NON_TOXIC")
    n_errors = sum(1 for r in results if r.label == "PARSE_ERROR")

    return BatchPredictResponse(
        results=results,
        total=len(results),
        n_toxic=n_toxic,
        n_non_toxic=n_non_toxic,
        n_errors=n_errors,
    )

# GNNExplainer
@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """
    Generate GNNExplainer atom/bond attribution for a molecule.

    NOTE: GNNExplainer atom/bond attribution for a molecule.
    (P(toxic) from explainer ≠ P(toxic) from /predict). 
    This is a known limitation documented in XSmiles README. 
    Use /predict for final label; use /explain only for structural attribution.
    """
    _ensure_xsmiles_available()
    
    smiles = req.smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _invalid_smiles_error(smiles)

    # First get real prediction for targert_class determination
    async with _model_lock():
        pred_df = predict_batch(
            smiles_list=[smiles],
            tokenizer=model_state["tokenizer"],
            wrapped_model=model_state["wrapped"],
            device=DEVICE,
            enforce_workspace_mode=False,
        )
    p_toxic = float(pred_df.iloc[0]["P(toxic)"])
    label_str = str(pred_df.iloc[0]["Predicted"])
    target_class = req.target_class if req.target_class is not None else (1 if p_toxic >= 0.5 else 0)

    # Run GNNExplainer
    try:
        pyg = smiles_to_pyg_data(smiles, label=target_class)
        async with _model_lock():
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    explain_molecule,
                    smiles,
                    model_state["model"],
                    model_state["tokenizer"],
                    pyg,
                    DEVICE,
                    req.epochs,
                    target_class,
                ),
                timeout=float(req.explainer_timeout_ms) / 1000.0,
            )
    except asyncio.TimeoutError:
        raise _explainer_timeout_error(smiles=smiles, timeout_ms=req.explainer_timeout_ms)
    except Exception as e:
        raise HTTPException(500, f"GNNExplainer error: {str(e)}")
    
    heatmap_b64 = _render_explanation_heatmap(result)
    molecule_b64 = _render_molecule_png(mol)

    # Build atom/bond lists
    atom_importance = result["atom_importance"]
    bond_importance = result["bond_importance"]

    top_atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        imp = float(atom_importance[idx])
        top_atoms.append(AtomImportance(
            atom_idx=idx,
            element=atom.GetSymbol(),
            importance=round(imp, 4),
            is_in_ring=atom.IsInRing(),
            is_aromatic=atom.GetIsAromatic(),
        ))
    top_atoms.sort(key=lambda x: x.importance, reverse=True)

    top_bonds = []
    for bond in mol.GetBonds():
        k = bond.GetIdx()
        imp = float(bond_importance[k]) if k < len(bond_importance) else 0.0
        a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
        a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
        top_bonds.append(BondImportance(
            bond_idx=k,
            atom_pair=f"{a1}({bond.GetBeginAtomIdx()}) - {a2}({bond.GetEndAtomIdx()})",
            bond_type=str(bond.GetBondTypeAsDouble()),
            importance=round(imp, 4),
        ))
    top_bonds.sort(key=lambda x: x.importance, reverse=True)
    top_bonds = top_bonds[:10]
    
    label_map = {"Toxic": "TOXIC", "Non-toxic": "NON_TOXIC"}
    return ExplainResponse(
        smiles=smiles,
        p_toxic=p_toxic,
        label=label_map.get(label_str, "UNKNOWN"),
        top_atoms=top_atoms[:10],
        top_bonds=top_bonds,
        heatmap_base64=heatmap_b64,
        molecule_png_base64=molecule_b64,
        chemical_interpretation=f"Top contributing atom: {top_atoms[0].element}",
        explainer_note="GNNExplainer optimizes only the GATv2 graph pathway.",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Unified production endpoint with three outputs for one SMILES:
      1) Toxic / Non-toxic from XSmiles (clinical)
      2) Type of toxicity from GATv2 (Tox21 tasks)
      3) Toxicity explainer from GNNExplainer (task-specific)
    """
    _ensure_xsmiles_available()
    tox21_available = TOX21_ENABLED and _tox21_ready()

    smiles = req.smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _invalid_smiles_error(smiles)
    canonical = Chem.MolToSmiles(mol)
    ood_assessment_raw = check_ood_risk(canonical)

    try:
        async with _model_lock():
            clinical_raw = predict_clinical_toxicity(
                smiles=smiles,
                tokenizer=model_state["tokenizer"],
                wrapped_model=model_state["wrapped"],
                device=DEVICE,
                threshold=req.clinical_threshold,
                enforce_workspace_mode=False,
            )

            if tox21_available:
                mechanism_raw = predict_toxicity_mechanism(
                    smiles=smiles,
                    model=model_state["tox21_model"],
                    task_names=model_state["tox21_tasks"],
                    device=DEVICE,
                    threshold=req.mechanism_threshold,
                    task_thresholds=model_state.get("tox21_thresholds"),
                    batch_size=64,
                )
            else:
                mechanism_raw = _fallback_mechanism_result(req.mechanism_threshold)
    except Exception as e:
        raise HTTPException(500, f"Unified inference error: {str(e)}")

    final_verdict = aggregate_toxicity_verdict(
        clinical_is_toxic=bool(clinical_raw["is_toxic"]),
        assay_hits=int(mechanism_raw["assay_hits"]),
    )

    explanation_payload = None
    should_explain = True
    if req.explain_only_if_alert and int(mechanism_raw["assay_hits"]) <= 0 and req.target_task is None:
        should_explain = False

    if should_explain:
        # Priority 1: clinical heatmap from the SMILESGNN branch (matches script-style plot).
        try:
            clinical_target_class = 1 if bool(clinical_raw["is_toxic"]) else 0
            clinical_pyg_data = smiles_to_pyg_data(smiles, label=clinical_target_class)
            if clinical_pyg_data is None:
                raise ValueError(f"Unable to featurize SMILES for explanation: {smiles}")

            async with _model_lock():
                explain_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        explain_molecule,
                        smiles,
                        model_state["model"],
                        model_state["tokenizer"],
                        clinical_pyg_data,
                        "cpu",
                        req.explainer_epochs,
                        clinical_target_class,
                    ),
                    timeout=float(req.explainer_timeout_ms) / 1000.0,
                )

            heatmap_b64 = _render_explanation_heatmap(explain_result)
            molecule_b64 = _render_molecule_png(mol) or heatmap_b64
            atom_importance = explain_result["atom_importance"]
            bond_importance = explain_result["bond_importance"]

            top_atoms = []
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                imp = float(atom_importance[idx])
                top_atoms.append(AtomImportance(
                    atom_idx=idx,
                    element=atom.GetSymbol(),
                    importance=round(imp, 4),
                    is_in_ring=atom.IsInRing(),
                    is_aromatic=atom.GetIsAromatic(),
                ))
            top_atoms.sort(key=lambda x: x.importance, reverse=True)

            top_bonds = []
            for bond in mol.GetBonds():
                bond_idx = bond.GetIdx()
                imp = float(bond_importance[bond_idx]) if bond_idx < len(bond_importance) else 0.0
                a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
                a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
                top_bonds.append(BondImportance(
                    bond_idx=bond_idx,
                    atom_pair=f"{a1}({bond.GetBeginAtomIdx()}) - {a2}({bond.GetEndAtomIdx()})",
                    bond_type=str(bond.GetBondTypeAsDouble()),
                    importance=round(imp, 4),
                ))
            top_bonds.sort(key=lambda x: x.importance, reverse=True)

            explanation_payload = ToxicityExplanationOutput(
                target_task="CLINICAL_TOXICITY",
                target_task_score=float(clinical_raw["p_toxic"]),
                top_atoms=top_atoms[:10],
                top_bonds=top_bonds[:10],
                heatmap_base64=heatmap_b64,
                molecule_png_base64=molecule_b64,
                explainer_note=(
                    "Clinical GNNExplainer heatmap for SMILESGNN graph pathway; "
                    "red regions contribute more to the predicted clinical toxicity probability."
                ),
            )
        except asyncio.TimeoutError:
            raise _explainer_timeout_error(
                smiles=smiles,
                timeout_ms=req.explainer_timeout_ms,
            )
        except Exception as clinical_exc:
            logger.warning(
                "Clinical explanation failed; falling back to tox21 task explainer: %s",
                clinical_exc,
            )

        # Priority 2 fallback: tox21 task-specific explanation if clinical path fails.
        if explanation_payload is None:
            target_task = req.target_task or str(mechanism_raw["highest_risk_task"])
            if target_task == "—":
                tasks = model_state.get("tox21_tasks", [])
                target_task = tasks[0] if tasks else ""

            if target_task:
                if not tox21_available:
                    explanation_payload = _fallback_explanation(target_task=target_task, mol=mol)
                else:
                    try:
                        async with _model_lock():
                            explain_result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    explain_tox21_task,
                                    smiles,
                                    model_state["tox21_model"],
                                    model_state["tox21_tasks"],
                                    target_task,
                                    "cpu",
                                    req.explainer_epochs,
                                    req.mechanism_threshold,
                                ),
                                timeout=float(req.explainer_timeout_ms) / 1000.0,
                            )
                        heatmap_b64 = _render_explanation_heatmap(explain_result)

                        atom_importance = explain_result["atom_importance"]
                        bond_importance = explain_result["bond_importance"]

                        top_atoms = []
                        for atom in mol.GetAtoms():
                            idx = atom.GetIdx()
                            imp = float(atom_importance[idx])
                            top_atoms.append(AtomImportance(
                                atom_idx=idx,
                                element=atom.GetSymbol(),
                                importance=round(imp, 4),
                                is_in_ring=atom.IsInRing(),
                                is_aromatic=atom.GetIsAromatic(),
                            ))
                        top_atoms.sort(key=lambda x: x.importance, reverse=True)

                        top_bonds = []
                        for bond in mol.GetBonds():
                            bond_idx = bond.GetIdx()
                            imp = float(bond_importance[bond_idx]) if bond_idx < len(bond_importance) else 0.0
                            a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
                            a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
                            top_bonds.append(BondImportance(
                                bond_idx=bond_idx,
                                atom_pair=f"{a1}({bond.GetBeginAtomIdx()}) - {a2}({bond.GetEndAtomIdx()})",
                                bond_type=str(bond.GetBondTypeAsDouble()),
                                importance=round(imp, 4),
                            ))
                        top_bonds.sort(key=lambda x: x.importance, reverse=True)

                        explanation_payload = ToxicityExplanationOutput(
                            target_task=target_task,
                            target_task_score=float(explain_result["prediction_prob"]),
                            top_atoms=top_atoms[:10],
                            top_bonds=top_bonds[:10],
                            heatmap_base64=heatmap_b64,
                            molecule_png_base64=_render_molecule_png(mol) or heatmap_b64,
                            explainer_note=(
                                "Fallback Tox21 task-specific GNNExplainer because "
                                "clinical explanation was unavailable for this request."
                            ),
                        )
                    except asyncio.TimeoutError:
                        raise _explainer_timeout_error(
                            smiles=smiles,
                            timeout_ms=req.explainer_timeout_ms,
                        )
                    except Exception as e:
                        raise HTTPException(500, f"Task-level explanation error: {str(e)}")

    # Always return the full 12-task map to keep API outputs stable and complete.
    mechanism_scores = dict(mechanism_raw["task_scores"])

    clinical_output = ClinicalToxicityOutput(
        label=str(clinical_raw["label"]),
        is_toxic=bool(clinical_raw["is_toxic"]),
        confidence=float(clinical_raw["confidence"]),
        p_toxic=float(clinical_raw["p_toxic"]),
        threshold_used=float(clinical_raw["threshold_used"]),
    )

    mechanism_output = MechanismToxicityOutput(
        task_scores=mechanism_scores,
        active_tasks=list(mechanism_raw["active_tasks"]),
        highest_risk_task=str(mechanism_raw["highest_risk_task"]),
        highest_risk_score=float(mechanism_raw["highest_risk_score"]),
        assay_hits=int(mechanism_raw["assay_hits"]),
        threshold_used=float(mechanism_raw["threshold_used"]),
        task_thresholds=dict(mechanism_raw["task_thresholds"]),
    )

    ood_assessment = OodAssessmentOutput(
        ood_risk=str(ood_assessment_raw.get("ood_risk", "LOW")),
        flag=bool(ood_assessment_raw.get("flag", False)),
        reason=str(ood_assessment_raw.get("reason", "")),
        rare_elements=list(ood_assessment_raw.get("rare_elements") or []),
        high_risk_elements=list(ood_assessment_raw.get("high_risk_elements") or []),
        recommendation=ood_assessment_raw.get("recommendation"),
    )

    reliability_warning: Optional[str] = None
    if ood_assessment.flag:
        reliability_warning = ood_assessment.reason
    elif not tox21_available:
        reliability_warning = "Mechanism model is unavailable; only clinical signal is used."

    inference_context = InferenceContextOutput(
        workspace_mode=WORKSPACE_MODE_NAME,
        threshold_policy=str(WORKSPACE_MODE.get("threshold_policy", "balanced")),
        clinical_threshold_applied=float(req.clinical_threshold),
        clinical_model_loaded=_xsmiles_ready(),
        tox21_model_loaded=tox21_available,
        explainer_used=bool(should_explain),
        explanation_available=explanation_payload is not None,
        tox21_threshold_source=model_state.get("tox21_thresholds_source"),
        clinical_reference_metrics=dict(model_state.get("clinical_reference_metrics") or {}),
    )

    return AnalyzeResponse(
        smiles=smiles,
        canonical_smiles=canonical,
        clinical=clinical_output,
        mechanism=mechanism_output,
        explanation=explanation_payload,
        ood_assessment=ood_assessment,
        reliability_warning=reliability_warning,
        inference_context=inference_context,
        final_verdict=final_verdict,
    )


@app.post("/agent/analyze", response_model=AgentAnalyzeResponse)
async def agent_analyze(req: AgentAnalyzeRequest):
    """Execute the ADK agent runtime and return final report + tool-calling traces."""
    smiles = req.smiles.strip()
    language = normalize_language(req.language)
    if not smiles:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "smiles_empty",
                "message": "SMILES is required for agent analysis.",
            },
        )

    session_id = req.session_id or f"session_{uuid.uuid4().hex[:12]}"

    async def _build_fallback_response(cause: str) -> AgentAnalyzeResponse:
        """Fallback to deterministic orchestrator flow when ADK runtime is unavailable."""
        fallback_state = await asyncio.to_thread(
            run_orchestrator_flow,
            smiles,
            int(req.max_literature_results),
            language,
            float(req.clinical_threshold),
            float(req.mechanism_threshold),
        )

        final_report_payload = _coerce_json_dict(
            fallback_state.get("final_report"),
            nested_key="final_report",
        ) or {}
        validation_status_payload = fallback_state.get("validation_status")
        summary = final_report_payload.get("executive_summary") if isinstance(final_report_payload, dict) else None

        fallback_events: List[AgentEventRecord] = []
        if req.include_agent_events:
            fallback_preview = (
                f"Kich hoat fallback ADK: {cause}"
                if language == "vi"
                else f"ADK fallback activated: {cause}"
            )
            fallback_events.append(
                AgentEventRecord(
                    type="fallback",
                    author="System",
                    function_calls=[],
                    function_responses=[],
                    is_final=True,
                    text_preview=fallback_preview,
                )
            )

        return AgentAnalyzeResponse(
            session_id=session_id,
            adk_available=bool(ADK_AVAILABLE and adk_runner is not None and adk_session_service is not None),
            runtime_mode="deterministic_fallback",
            runtime_note=cause,
            validation_status=validation_status_payload,
            final_report=final_report_payload,
            final_text=summary if isinstance(summary, str) else None,
            agent_events=fallback_events,
            state_keys=sorted(fallback_state.keys()),
        )

    if adk_runner is None or adk_session_service is None or Content is None or Part is None:
        return await _build_fallback_response("adk_runtime_unavailable")

    user_id = req.user_id.strip() or "default_user"
    initial_state = {
        "smiles_input": smiles,
        "max_literature_results": int(req.max_literature_results),
        "language": language,
        "clinical_threshold": float(req.clinical_threshold),
        "mechanism_threshold": float(req.mechanism_threshold),
    }

    try:
        await adk_session_service.create_session(
            app_name=ADK_APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state=initial_state,
        )
    except Exception:
        # If session already exists, update its state and continue.
        try:
            existing = await adk_session_service.get_session(
                app_name=ADK_APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            if existing is None:
                raise RuntimeError("session_not_found")
            state = getattr(existing, "state", None)
            if not isinstance(state, dict):
                state = {}
                setattr(existing, "state", state)
            state.update(initial_state)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "agent_session_init_failed",
                    "message": str(exc),
                },
            ) from exc

    message = Content(
        role="user",
        parts=[Part(text=f"Analyze toxicity for SMILES: {smiles}. Respond in language={language}.")],
    )

    events: List[AgentEventRecord] = []
    final_text: Optional[str] = None
    stream_close_race = False
    force_adk_continuation = False

    async def _consume_adk_events(runner: Any) -> None:
        nonlocal final_text
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            function_calls = _extract_event_function_calls(event)
            function_responses = _extract_event_function_responses(event)
            is_final = _is_final_event_response(event)
            text_preview = _extract_event_text_preview(event)

            if req.include_agent_events:
                events.append(
                    AgentEventRecord(
                        type=getattr(event, "type", None),
                        author=getattr(event, "author", None),
                        function_calls=function_calls,
                        function_responses=function_responses,
                        is_final=is_final,
                        text_preview=text_preview,
                    )
                )

            if is_final and final_text is None and text_preview:
                final_text = text_preview

    async def _retry_root_with_fallback_model(reason: str) -> bool:
        nonlocal final_text, stream_close_race

        fallback_model = _resolve_quota_retry_model(getattr(root_agent, "model", None))
        with _temporary_agent_model(root_agent, fallback_model) as (changed, previous_model):
            if not changed:
                return False

            logger.warning(
                "ADK root %s with model=%s; retrying with model=%s",
                reason,
                previous_model,
                fallback_model,
            )

            retry_root_runner = Runner(
                agent=root_agent,
                session_service=adk_session_service,
                app_name=ADK_APP_NAME,
            )

            events.clear()
            final_text = None

            try:
                await _consume_adk_events(retry_root_runner)
                return True
            except Exception as retry_exc:
                if "aclose(): asynchronous generator is already running" in str(retry_exc):
                    logger.warning(
                        "ADK root stream close race detected after model retry; proceeding: %s",
                        retry_exc,
                    )
                    stream_close_race = True
                    return True

                logger.warning("ADK root failed after model retry: %s", retry_exc)
                return False

    try:
        await _consume_adk_events(adk_runner)
    except Exception as exc:
        # ADK occasionally raises a stream-close race after finishing events.
        # In that case we continue and read session state instead of forcing fallback.
        if "aclose(): asynchronous generator is already running" in str(exc):
            logger.warning(
                "ADK stream close race detected; proceeding with session state read: %s",
                exc,
            )
            stream_close_race = True
        elif _is_vertex_model_not_found_error(exc):
            current_location = (
                os.getenv("GEMINI_LOCATION")
                or os.getenv("GOOGLE_CLOUD_LOCATION")
                or ""
            ).strip().lower()

            if current_location != "global":
                logger.warning(
                    "ADK root model unavailable in location=%s; retrying root runner with global",
                    current_location or "unset",
                )
                os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
                os.environ["GEMINI_LOCATION"] = "global"

                retry_root_runner = Runner(
                    agent=root_agent,
                    session_service=adk_session_service,
                    app_name=ADK_APP_NAME,
                )

                events.clear()
                final_text = None

                try:
                    await _consume_adk_events(retry_root_runner)
                except Exception as retry_exc:
                    if "aclose(): asynchronous generator is already running" in str(retry_exc):
                        logger.warning(
                            "ADK root stream close race detected after global retry; proceeding: %s",
                            retry_exc,
                        )
                        stream_close_race = True
                    else:
                        if _is_vertex_resource_exhausted_error(retry_exc):
                            model_retry_ok = await _retry_root_with_fallback_model(
                                "quota exhausted after global retry"
                            )
                            if not model_retry_ok:
                                logger.warning(
                                    "ADK runtime failed after global+model retries, using deterministic fallback: %s",
                                    retry_exc,
                                )
                                return await _build_fallback_response(f"agent_runtime_error: {retry_exc}")
                        else:
                            logger.warning("ADK runtime failed after global retry, using deterministic fallback: %s", retry_exc)
                            return await _build_fallback_response(f"agent_runtime_error: {retry_exc}")
            else:
                logger.warning("ADK runtime failed with model unavailable in global location, using deterministic fallback: %s", exc)
                return await _build_fallback_response(f"agent_runtime_error: {exc}")
        elif _is_vertex_resource_exhausted_error(exc):
            model_retry_ok = await _retry_root_with_fallback_model("quota exhausted")
            if not model_retry_ok:
                logger.warning(
                    "ADK root quota-exhausted and model retry unavailable; forcing ADK step continuation path: %s",
                    exc,
                )
                force_adk_continuation = True
        else:
            logger.warning("ADK runtime failed, using deterministic fallback: %s", exc)
            return await _build_fallback_response(f"agent_runtime_error: {exc}")

    if stream_close_race:
        # Give ADK session state time to flush after async stream-close race.
        await asyncio.sleep(4.0)

    try:
        state = await _read_adk_session_state_with_retry(
            app_name=ADK_APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "agent_session_read_failed",
                "message": str(exc),
            },
        ) from exc

    final_report = _extract_state_payload(state, "final_report")

    validation_status = state.get("validation_status")
    if validation_status is None:
        validation_payload = _extract_state_payload(state, "validation_result")
        if isinstance(validation_payload, dict):
            validation_status = validation_payload.get("validation_status")

    if not final_report:
        # Some model outputs place final report under a raw parsed object without nested key.
        parsed_report = _coerce_json_dict(state.get("final_report"))
        if isinstance(parsed_report, dict):
            final_report = parsed_report

    if not final_report:
        final_report = _recover_final_report_from_state(state)

    screening_payload = _extract_state_payload(state, "screening_result")
    research_payload = _extract_state_payload(state, "research_result")

    if not screening_payload:
        screening_payload = _recover_screening_payload_from_state(state)
    if not research_payload:
        research_payload = _recover_research_payload_from_state(state)

    if not screening_payload:
        logger.warning("ADK screening payload missing from session state; hydrating via deterministic screening")
        try:
            hydrated = await asyncio.to_thread(
                run_screening,
                smiles,
                language,
                float(req.clinical_threshold),
                float(req.mechanism_threshold),
            )
        except Exception as exc:
            logger.warning("Deterministic screening hydration failed (missing payload): %s", exc)
            hydrated = None

        if isinstance(hydrated, dict):
            hydrated_payload = _coerce_json_dict(
                hydrated.get("screening_result"),
                nested_key="screening_result",
            ) or {}
            if isinstance(hydrated_payload, dict) and hydrated_payload:
                screening_payload = hydrated_payload

    if screening_payload and not _screening_payload_has_structural_data(screening_payload):
        logger.warning("ADK screening payload missing structural explanation; hydrating via deterministic screening")
        try:
            hydrated = await asyncio.to_thread(
                run_screening,
                smiles,
                language,
                float(req.clinical_threshold),
                float(req.mechanism_threshold),
            )
        except Exception as exc:
            logger.warning("Deterministic screening hydration failed: %s", exc)
            hydrated = None

        if isinstance(hydrated, dict):
            hydrated_payload = _coerce_json_dict(
                hydrated.get("screening_result"),
                nested_key="screening_result",
            ) or {}
            if isinstance(hydrated_payload, dict) and hydrated_payload:
                screening_payload = hydrated_payload

    # ADK root orchestrator occasionally stops after validator with incomplete state.
    # Continue remaining steps explicitly with ADK sub-agents to preserve LLM flow.
    needs_adk_continuation = (
        (str(validation_status or "").upper() == "VALID" or force_adk_continuation)
        and (not screening_payload or not research_payload or not final_report)
        and Runner is not None
        and adk_session_service is not None
    )
    if needs_adk_continuation:
        logger.warning("ADK continuation triggered due incomplete state after root orchestrator")
        await _run_adk_agent_step(
            agent=screening_agent,
            user_id=user_id,
            session_id=session_id,
            message=message,
        )
        await _run_adk_agent_step(
            agent=researcher_agent,
            user_id=user_id,
            session_id=session_id,
            message=message,
        )
        await _run_adk_agent_step(
            agent=writer_agent,
            user_id=user_id,
            session_id=session_id,
            message=message,
        )

        state = await _read_adk_session_state_with_retry(
            app_name=ADK_APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        final_report = _extract_state_payload(state, "final_report")
        if not final_report:
            final_report = _recover_final_report_from_state(state)

        screening_payload = _extract_state_payload(state, "screening_result")
        research_payload = _extract_state_payload(state, "research_result")
        if not screening_payload:
            screening_payload = _recover_screening_payload_from_state(state)
        if not research_payload:
            research_payload = _recover_research_payload_from_state(state)

    report_schema_complete = _is_final_report_schema_complete(final_report)
    if (not final_report or not report_schema_complete) and screening_payload:
        if not final_report:
            logger.warning("ADK writer output missing/invalid; rebuilding final report from session state")
        else:
            logger.warning("ADK writer report schema mismatch; rebuilding final report from session state")

        final_report = build_final_report(
            smiles_input=smiles,
            screening_result=screening_payload,
            research_result=research_payload,
            language=language,
        )
        if final_text is None:
            rebuilt_summary = final_report.get("executive_summary")
            if isinstance(rebuilt_summary, str) and rebuilt_summary:
                final_text = rebuilt_summary

    if final_report and screening_payload and _final_report_missing_structural_images(final_report):
        logger.warning("ADK final report missing structural images; rebuilding from hydrated screening payload")
        final_report = build_final_report(
            smiles_input=smiles,
            screening_result=screening_payload,
            research_result=research_payload,
            language=language,
        )
        if final_text is None:
            rebuilt_summary = final_report.get("executive_summary")
            if isinstance(rebuilt_summary, str) and rebuilt_summary:
                final_text = rebuilt_summary

    if final_text is None and isinstance(final_report, dict):
        summary = final_report.get("executive_summary")
        if isinstance(summary, str) and summary:
            final_text = summary

    report_has_payload = isinstance(final_report, dict) and bool(final_report)
    if not report_has_payload:
        logger.warning(
            "ADK session state incomplete; switching to deterministic full-pipeline fallback "
            "(validation_status=%s, has_payload=%s, state_keys=%s)",
            validation_status,
            report_has_payload,
            sorted(state.keys()),
        )
        return await _build_fallback_response(
            f"adk_state_incomplete: validation_status={validation_status}, has_payload={report_has_payload}"
        )

    return AgentAnalyzeResponse(
        session_id=session_id,
        adk_available=ADK_AVAILABLE,
        runtime_mode="adk",
        runtime_note=None,
        validation_status=validation_status,
        final_report=final_report,
        final_text=final_text,
        agent_events=events if req.include_agent_events else [],
        state_keys=sorted(state.keys()),
    )