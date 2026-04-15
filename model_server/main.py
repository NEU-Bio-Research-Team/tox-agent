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
import pickle
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch 
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
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
    chat_with_report,
    create_chat_session,
    get_session as get_report_chat_session,
    researcher_agent,
    root_agent,
    run_evidence_qa,
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

try:
    from google import genai as google_genai
except Exception:
    google_genai = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

MOLSCRIBE_IMPORT_ERROR: Optional[str] = None
try:
    from molscribe import MolScribe
except Exception as exc:
    MolScribe = None
    MOLSCRIBE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from backend.inference import (
    aggregate_toxicity_verdict,
    load_clinical_head_model,
    load_model,
    load_pretrained_dual_head_bundle,
    load_pretrained_gin_tox21_bundle,
    load_tox21_gatv2_model,
    predict_batch,
    predict_clinical_head_from_tox21_task_scores,
    predict_clinical_toxicity,
    predict_clinical_proxy_from_tox21,
    predict_pretrained_dual_head_outputs,
    predict_pretrained_gin_tox21_outputs,
    predict_toxicity_mechanism,
)
from backend.ood_guard import check_ood_risk
from backend.graph_data import get_feature_dims, smiles_to_pyg_data
from backend.attentivefp_model import create_attentivefp_model
from backend.gps_model import create_gps_model
from backend.featurization import featurize_fingerprint
from backend.gnn_explainer import (
    explain_molecule,
    explain_tox21_task,
    explain_tox21_task_gradient,
    explain_tox21_task_pretrained_dual_head_gradient,
    explain_tox21_task_pretrained_gin,
    visualize_explanation,
)
from backend.workspace_mode import get_workspace_mode
from model_server.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    AgentChatRequest,
    AgentChatResponse,
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
    SmilesImageExtractionResponse,
	SmilesPreviewRequest,
	SmilesPreviewResponse,
)

# Configuration
MODELS_ROOT = PROJECT_ROOT / "models"
MODEL_DIR = MODELS_ROOT / "smilesgnn_model"
CONFIG_PATH = PROJECT_ROOT / "config" / "smilesgnn_config.yaml"
TOX21_MODEL_DIR = MODELS_ROOT / "tox21_gatv2_model"
TOX21_CONFIG_PATH = PROJECT_ROOT / "config" / "tox21_gatv2_config.yaml"
TOX21_PRETRAINED_GIN_MODEL_DIR = MODELS_ROOT / "tox21_pretrained_gin_model"
TOX21_ATTENTIVEFP_MODEL_DIR = MODELS_ROOT / "tox21_attentivefp_model"
TOX21_GPS_MODEL_DIR = MODELS_ROOT / "tox21_gps_model"
TOX21_FINGERPRINT_MODEL_DIR = MODELS_ROOT / "tox21_fingerprint_model"
TOX21_FINGERPRINT_CONFIG_PATH = PROJECT_ROOT / "config" / "tox21_fingerprint_config.yaml"
CLINICAL_HEAD_MODEL_DIR = MODELS_ROOT / "clinical_head_model"
CLINICAL_METRICS_PATH = MODEL_DIR / "smilesgnn_model_metrics.txt"
CLINICAL_THRESHOLD_METRICS_PATH = MODEL_DIR / "clinical_threshold_metrics.json"
TOX21_THRESHOLDS_CANDIDATES = [
    TOX21_MODEL_DIR / "tox21_task_thresholds.json",
    TOX21_MODEL_DIR / "task_thresholds.json",
]
PRETRAINED_DUAL_HEAD_BACKENDS: Dict[str, Dict[str, Any]] = {
    "chemberta": {
        "display_name": "ChemBERTa",
        "model_dir": MODELS_ROOT / "pretrained_2head_herg_chemberta_model",
    },
    "pubchem": {
        "display_name": "PubChem",
        "model_dir": MODELS_ROOT / "pretrained_2head_herg_pubchem_model",
    },
    "molformer": {
        "display_name": "MolFormer",
        "model_dir": MODELS_ROOT / "pretrained_2head_herg_molformer_model",
    },
}

DEFAULT_BINARY_TOX_MODEL_KEY = "pretrained_2head_herg_chemberta_model"
DEFAULT_TOX_TYPE_MODEL_KEY = "tox21_ensemble_3_best"

TOX21_PRETRAINED_GIN_MODEL_KEY = "tox21_pretrained_gin_model"
TOX21_ATTENTIVEFP_MODEL_KEY = "tox21_attentivefp_model"
TOX21_GPS_MODEL_KEY = "tox21_gps_model"
TOX21_FINGERPRINT_MODEL_KEY = "tox21_fingerprint_model"

TOX21_ENSEMBLE_3_BEST_MODEL_KEY = "tox21_ensemble_3_best"
DUALHEAD_ENSEMBLE3_SIMPLE_MODEL_KEY = "dualhead_ensemble3_simple"
DUALHEAD_ENSEMBLE3_WEIGHTED_MODEL_KEY = "dualhead_ensemble3_weighted"
DUALHEAD_ENSEMBLE5_SIMPLE_MODEL_KEY = "dualhead_ensemble5_simple"
DUALHEAD_ENSEMBLE6_SIMPLE_MODEL_KEY = "dualhead_ensemble6_simple"

TOX21_TASK_NAMES_FALLBACK: List[str] = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

DUALHEAD_ENSEMBLE_HERG_MEMBERS_DEFAULT: List[str] = [
    "pretrained_2head_herg_chemberta_model",
    "pretrained_2head_herg_molformer_model",
]
DUALHEAD_ENSEMBLE3_TOX_MEMBERS: List[str] = [
    "pretrained_2head_herg_chemberta_model",
    "pretrained_2head_herg_molformer_model",
    TOX21_PRETRAINED_GIN_MODEL_KEY,
]
DUALHEAD_ENSEMBLE5_TOX_MEMBERS: List[str] = [
    "pretrained_2head_herg_chemberta_model",
    "pretrained_2head_herg_molformer_model",
    TOX21_ATTENTIVEFP_MODEL_KEY,
    TOX21_FINGERPRINT_MODEL_KEY,
    TOX21_GPS_MODEL_KEY,
]
DUALHEAD_ENSEMBLE6_TOX_MEMBERS: List[str] = [
    "pretrained_2head_herg_chemberta_model",
    "pretrained_2head_herg_molformer_model",
    TOX21_ATTENTIVEFP_MODEL_KEY,
    TOX21_FINGERPRINT_MODEL_KEY,
    TOX21_GPS_MODEL_KEY,
    TOX21_PRETRAINED_GIN_MODEL_KEY,
]

_DUALHEAD_ENSEMBLE_3_SIMPLE_SPEC: Dict[str, Any] = {
    "model_dir": MODELS_ROOT / "dualhead_ensemble3",
    "clinical_mode": "simple",
    "mechanism_mode": "simple",
    "clinical_members": list(DUALHEAD_ENSEMBLE_HERG_MEMBERS_DEFAULT),
    "mechanism_members": list(DUALHEAD_ENSEMBLE3_TOX_MEMBERS),
    "explainer_engine": TOX21_PRETRAINED_GIN_MODEL_KEY,
}

DUALHEAD_ENSEMBLE_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    DUALHEAD_ENSEMBLE3_SIMPLE_MODEL_KEY: dict(_DUALHEAD_ENSEMBLE_3_SIMPLE_SPEC),
    TOX21_ENSEMBLE_3_BEST_MODEL_KEY: dict(_DUALHEAD_ENSEMBLE_3_SIMPLE_SPEC),
    DUALHEAD_ENSEMBLE3_WEIGHTED_MODEL_KEY: {
        "model_dir": MODELS_ROOT / "dualhead_weighted_ensemble3",
        "clinical_mode": "weighted",
        "mechanism_mode": "taskwise_weighted",
        "clinical_members": list(DUALHEAD_ENSEMBLE_HERG_MEMBERS_DEFAULT),
        "mechanism_members": list(DUALHEAD_ENSEMBLE3_TOX_MEMBERS),
        "weights_path": MODELS_ROOT / "dualhead_weighted_ensemble3" / "dualhead_metrics.json",
        "explainer_engine": TOX21_PRETRAINED_GIN_MODEL_KEY,
    },
    DUALHEAD_ENSEMBLE5_SIMPLE_MODEL_KEY: {
        "model_dir": MODELS_ROOT / "dualhead_ensemble5",
        "clinical_mode": "simple",
        "mechanism_mode": "simple",
        "clinical_members": list(DUALHEAD_ENSEMBLE_HERG_MEMBERS_DEFAULT),
        "mechanism_members": list(DUALHEAD_ENSEMBLE5_TOX_MEMBERS),
        "explainer_engine": "pretrained_2head_herg_molformer_model",
    },
    DUALHEAD_ENSEMBLE6_SIMPLE_MODEL_KEY: {
        "model_dir": MODELS_ROOT / "dualhead_ensemble6",
        "clinical_mode": "simple",
        "mechanism_mode": "simple",
        "clinical_members": list(DUALHEAD_ENSEMBLE_HERG_MEMBERS_DEFAULT),
        "mechanism_members": list(DUALHEAD_ENSEMBLE6_TOX_MEMBERS),
        "explainer_engine": TOX21_PRETRAINED_GIN_MODEL_KEY,
    },
}

ENSEMBLE_BINARY_MODEL_KEYS = set(DUALHEAD_ENSEMBLE_MODEL_SPECS.keys())
ENSEMBLE_TOX_TYPE_MODEL_KEYS = set(DUALHEAD_ENSEMBLE_MODEL_SPECS.keys())

AUX_TOX21_MEMBER_MODEL_DIRS: Dict[str, Path] = {
    TOX21_ATTENTIVEFP_MODEL_KEY: TOX21_ATTENTIVEFP_MODEL_DIR,
    TOX21_GPS_MODEL_KEY: TOX21_GPS_MODEL_DIR,
    TOX21_FINGERPRINT_MODEL_KEY: TOX21_FINGERPRINT_MODEL_DIR,
}

TOX21_ENSEMBLE_3_BEST_EXPLAINER_KEY = TOX21_PRETRAINED_GIN_MODEL_KEY
BINARY_ENSEMBLE_TOX_MODEL_KEY = TOX21_ENSEMBLE_3_BEST_MODEL_KEY
BINARY_ENSEMBLE_3_BEST_MEMBERS: List[str] = list(DUALHEAD_ENSEMBLE_HERG_MEMBERS_DEFAULT)

TOX21_ENSEMBLE_3_BEST_MEMBERS: List[str] = list(DUALHEAD_ENSEMBLE3_TOX_MEMBERS)

DUAL_HEAD_MODEL_DIRS: Dict[str, Path] = {
    "pretrained_2head_herg_chemberta_model": MODELS_ROOT / "pretrained_2head_herg_chemberta_model",
    "pretrained_2head_herg_chemberta_quick": MODELS_ROOT / "pretrained_2head_herg_chemberta_quick",
    "pretrained_2head_herg_molformer_model": MODELS_ROOT / "pretrained_2head_herg_molformer_model",
    "pretrained_2head_herg_molformer_quick": MODELS_ROOT / "pretrained_2head_herg_molformer_quick",
    "pretrained_2head_herg_pubchem_model": MODELS_ROOT / "pretrained_2head_herg_pubchem_model",
    "pretrained_2head_herg_pubchem_quick": MODELS_ROOT / "pretrained_2head_herg_pubchem_quick",
}

TOX_TYPE_MODEL_DIRS: Dict[str, Path] = {
    "tox21_gatv2_model": TOX21_MODEL_DIR,
    TOX21_PRETRAINED_GIN_MODEL_KEY: TOX21_PRETRAINED_GIN_MODEL_DIR,
    TOX21_ENSEMBLE_3_BEST_MODEL_KEY: MODELS_ROOT / "dualhead_ensemble3",
    DUALHEAD_ENSEMBLE3_SIMPLE_MODEL_KEY: MODELS_ROOT / "dualhead_ensemble3",
    DUALHEAD_ENSEMBLE3_WEIGHTED_MODEL_KEY: MODELS_ROOT / "dualhead_weighted_ensemble3",
    DUALHEAD_ENSEMBLE5_SIMPLE_MODEL_KEY: MODELS_ROOT / "dualhead_ensemble5",
    DUALHEAD_ENSEMBLE6_SIMPLE_MODEL_KEY: MODELS_ROOT / "dualhead_ensemble6",
    "pretrained_2head_herg_chemberta_model": MODELS_ROOT / "pretrained_2head_herg_chemberta_model",
    "pretrained_2head_herg_chemberta_quick": MODELS_ROOT / "pretrained_2head_herg_chemberta_quick",
    "pretrained_2head_herg_molformer_model": MODELS_ROOT / "pretrained_2head_herg_molformer_model",
    "pretrained_2head_herg_molformer_quick": MODELS_ROOT / "pretrained_2head_herg_molformer_quick",
    "pretrained_2head_herg_pubchem_model": MODELS_ROOT / "pretrained_2head_herg_pubchem_model",
    "pretrained_2head_herg_pubchem_quick": MODELS_ROOT / "pretrained_2head_herg_pubchem_quick",
}
INFERENCE_BACKEND_ALIASES = {
    "xsmiles": "xsmiles",
    "default": "xsmiles",
    "auto": "xsmiles",
    "chembert": "chemberta",
    "chemberta": "chemberta",
    "pubchem": "pubchem",
    "molformer": "molformer",
}
DEFAULT_INFERENCE_BACKEND = os.getenv("DEFAULT_INFERENCE_BACKEND", "xsmiles").strip().lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKSPACE_MODE = get_workspace_mode()
WORKSPACE_MODE_NAME = str(WORKSPACE_MODE.get("mode", "unknown"))
CLINTOX_ENABLED = bool(WORKSPACE_MODE.get("clintox_enabled", True))
TOX21_ENABLED = bool(WORKSPACE_MODE.get("tox21_enabled", True))
XSMILES_LOAD_REQUESTED = CLINTOX_ENABLED or MODEL_DIR.exists()
CLINICAL_HEAD_LOAD_REQUESTED = CLINICAL_HEAD_MODEL_DIR.exists()
FAST_EXPLAINER_EPOCH_CUTOFF = max(1, int(os.getenv("FAST_EXPLAINER_EPOCH_CUTOFF", "80")))
CLINICAL_SIGNAL_STRATEGY = os.getenv("CLINICAL_SIGNAL_STRATEGY", "auto").strip().lower()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


SMILES_IMAGE_SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpg",
    "image/jpeg",
    "image/webp",
}
SMILES_IMAGE_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SMILES_IMAGE_MAX_BYTES = max(1024, _env_int("SMILES_IMAGE_MAX_BYTES", 5 * 1024 * 1024))
MOLSCRIBE_REPO_ID = os.getenv("MOLSCRIBE_REPO_ID", "yujieq/MolScribe").strip()
MOLSCRIBE_CHECKPOINT_NAME = os.getenv(
    "MOLSCRIBE_CHECKPOINT_NAME",
    "swin_base_char_aux_1m.pth",
).strip()
MOLSCRIBE_MODEL_PATH = os.getenv("MOLSCRIBE_MODEL_PATH", "").strip()
MOLSCRIBE_AUTO_DOWNLOAD = _env_flag("MOLSCRIBE_AUTO_DOWNLOAD", True)
MOLSCRIBE_PRELOAD_ON_STARTUP = _env_flag("MOLSCRIBE_PRELOAD_ON_STARTUP", True)
MOLSCRIBE_DEVICE = os.getenv("MOLSCRIBE_DEVICE", DEVICE).strip().lower() or DEVICE

def _normalize_route(route: str, default: str) -> str:
    if not route:
        return default
    return route if route.startswith("/") else f"/{route}"

AIP_HEALTH_ROUTE = _normalize_route(os.getenv("AIP_HEALTH_ROUTE", "/health"), "/health")
AIP_PREDICT_ROUTE = _normalize_route(os.getenv("AIP_PREDICT_ROUTE", "/predict"), "/predict")
ADK_APP_NAME = os.getenv("AGENT_APP_NAME", "tox-agent")

adk_session_service = None
adk_runner = None
_report_chat_lock = threading.Lock()
_report_chat_by_analysis_session: Dict[str, str] = {}


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

# Runtime state (models are loaded lazily on first real request/health probe).
model_state: Dict[str, Any] = {}
_model_init_lock = asyncio.Lock()
_model_init_lock_sync = threading.Lock()


def _clear_loaded_models() -> None:
    for key in (
        "model",
        "tokenizer",
        "wrapped",
        "tox21_model",
        "tox21_tasks",
        "tox21_thresholds",
        "tox21_thresholds_source",
        "clinical_head_model",
        "clinical_head_meta",
        "clinical_reference_metrics",
        "pretrained_dual_head_bundles",
        "tox21_aux_member_bundles",
        "ensemble_weight_payloads",
        "molscribe_predictor",
        "molscribe_checkpoint_path",
    ):
        model_state.pop(key, None)


def _initialize_runtime_state() -> None:
    if "model_lock" not in model_state:
        model_state["model_lock"] = asyncio.Lock()
    if "model_lock_sync" not in model_state:
        model_state["model_lock_sync"] = threading.Lock()
    if "ocr_lock_sync" not in model_state:
        model_state["ocr_lock_sync"] = threading.Lock()

    model_state.setdefault("startup_errors", {})
    model_state.setdefault("clinical_reference_metrics", {})
    model_state.setdefault("pretrained_dual_head_bundles", {})
    model_state.setdefault("tox21_aux_member_bundles", {})
    model_state.setdefault("ensemble_weight_payloads", {})
    model_state.setdefault("models_loaded", False)


def _load_all_models_sync() -> None:
    """Load all required models synchronously; guarded by outer init locks."""
    _initialize_runtime_state()
    _clear_loaded_models()

    startup_errors = model_state.setdefault("startup_errors", {})
    startup_errors.clear()

    logger.info(
        "Lazy-loading production models on %s (workspace_mode=%s, clintox_enabled=%s, tox21_enabled=%s)...",
        DEVICE,
        WORKSPACE_MODE_NAME,
        CLINTOX_ENABLED,
        TOX21_ENABLED,
    )

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
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            startup_errors["xsmiles"] = msg
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
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            startup_errors["tox21"] = msg
            logger.warning("Tox21 model load FAILED: %s", msg)
    else:
        logger.info("Skipping Tox21 model load because tox21_enabled=false")

    if CLINICAL_HEAD_LOAD_REQUESTED:
        if not TOX21_ENABLED:
            logger.warning(
                "Clinical head directory exists, but tox21 is disabled; skipping clinical head load"
            )
        else:
            try:
                clinical_head_model, clinical_head_meta = load_clinical_head_model(
                    model_dir=CLINICAL_HEAD_MODEL_DIR,
                    device=DEVICE,
                )
                model_state["clinical_head_model"] = clinical_head_model
                model_state["clinical_head_meta"] = clinical_head_meta
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                startup_errors["clinical_head"] = msg
                logger.warning("Clinical head model load FAILED: %s", msg)

    model_state["models_loaded"] = _required_models_ready()
    if model_state["models_loaded"]:
        logger.info("Production models loaded successfully.")
    else:
        logger.warning(
            "Model server running in degraded mode after lazy load. startup_errors=%s",
            _startup_errors(),
        )


def _ensure_models_loaded_sync() -> None:
    _initialize_runtime_state()
    if model_state.get("models_loaded"):
        return

    with _model_init_lock_sync:
        if model_state.get("models_loaded"):
            return

        try:
            _load_all_models_sync()
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            model_state.setdefault("startup_errors", {})["model_bootstrap"] = msg
            model_state["models_loaded"] = False
            logger.exception("Unexpected model bootstrap failure: %s", msg)


async def _ensure_models_loaded() -> None:
    _initialize_runtime_state()
    if model_state.get("models_loaded"):
        return

    async with _model_init_lock:
        if model_state.get("models_loaded"):
            return
        await asyncio.to_thread(_ensure_models_loaded_sync)


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


def _resolve_report_chat_model() -> str:
    return os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()


def _build_report_chat_client(location_override: Optional[str] = None) -> Tuple[Optional[Any], str]:
    if google_genai is None:
        return None, "google_genai_not_available"

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            return google_genai.Client(api_key=api_key), "api_key"
        except Exception as exc:
            return None, f"api_key_client_error:{type(exc).__name__}"

    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
    )
    if not project:
        return None, "missing_project_for_vertexai"

    configured_location = (
        location_override
        or os.getenv("GEMINI_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or "global"
    )
    try:
        return (
            google_genai.Client(
                vertexai=True,
                project=project,
                location=configured_location,
            ),
            f"vertex_adc:{configured_location}",
        )
    except Exception as exc:
        return None, f"vertex_client_error:{type(exc).__name__}"


def _build_report_chat_prompt(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    lines = [system_prompt.strip(), "", "=== CONVERSATION ==="]
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _report_chat_available_sections_text(session_id: str) -> str:
    session = get_report_chat_session(session_id)
    if session is None:
        return (
            "EXECUTIVE SUMMARY, CLINICAL TOXICITY PREDICTIONS, TOXICITY MECHANISMS, "
            "RECOMMENDATIONS, CURATED LITERATURE EVIDENCE, QUALITY FLAGS"
        )

    final_report = _coerce_json_dict(session.report_state.get("final_report")) or {}
    sections = _coerce_json_dict(final_report.get("sections")) or {}
    evidence = _coerce_json_dict(session.report_state.get("evidence_qa_result")) or {}

    ordered_entries: List[Tuple[str, Tuple[str, ...], str]] = [
        ("executive_summary", ("executive_summary",), "EXECUTIVE SUMMARY"),
        ("clinical_toxicity", ("clinical_toxicity", "toxicity_predictions"), "CLINICAL TOXICITY PREDICTIONS"),
        ("mechanism_toxicity", ("mechanism_toxicity", "toxicity_mechanisms"), "TOXICITY MECHANISMS"),
        ("molrag_evidence", ("molrag_evidence", "molrag"), "MOLRAG EVIDENCE"),
        ("fusion_result", ("fusion_result",), "BASELINE/MOLRAG FUSION"),
        ("literature_context", ("literature_context",), "LITERATURE CONTEXT"),
        ("ood_assessment", ("ood_assessment",), "OOD ASSESSMENT"),
        ("inference_context", ("inference_context",), "INFERENCE CONTEXT"),
        ("recommendations", ("recommendations",), "RECOMMENDATIONS"),
        ("risk_level", ("risk_level",), "RISK LEVEL"),
    ]

    available: List[str] = []
    for _, candidate_keys, label in ordered_entries:
        if any(key in final_report or key in sections for key in candidate_keys):
            available.append(label)

    curated_articles = evidence.get("curated_articles")
    if isinstance(curated_articles, list) and curated_articles:
        available.append("CURATED LITERATURE EVIDENCE")

    if "research_quality_flags" in evidence:
        available.append("QUALITY FLAGS")

    if not available:
        available = [
            "EXECUTIVE SUMMARY",
            "CLINICAL TOXICITY PREDICTIONS",
            "TOXICITY MECHANISMS",
            "RECOMMENDATIONS",
            "CURATED LITERATURE EVIDENCE",
            "QUALITY FLAGS",
        ]

    deduped = list(dict.fromkeys(available))
    return ", ".join(deduped)


def _normalize_report_chat_out_of_scope_reply(session_id: str, text: str) -> str:
    response_text = str(text or "").strip()
    refusal_prefix = "Thông tin này không có trong report hiện tại."
    if refusal_prefix not in response_text:
        return response_text

    available_sections_text = _report_chat_available_sections_text(session_id)
    return (
        "Thông tin này không có trong report hiện tại. "
        f"Report chỉ bao gồm: {available_sections_text}."
    )


def _report_chat_confidence_level(session_id: str) -> str:
    session = get_report_chat_session(session_id)
    if session is None:
        return "LOW"

    evidence = _coerce_json_dict(session.report_state.get("evidence_qa_result")) or {}
    return _normalize_evidence_confidence(evidence.get("evidence_confidence"))


def _normalize_report_chat_citation_confidence(session_id: str, text: str) -> str:
    response_text = str(text or "").strip()
    if not response_text:
        return response_text

    if response_text.startswith("Thông tin này không có trong report hiện tại."):
        return response_text

    confidence = _report_chat_confidence_level(session_id)

    response_text = re.sub(
        r"(\|\s*Evidence\s*:\s*)(HIGH|MEDIUM|LOW|UNKNOWN)",
        lambda match: f"{match.group(1)}{confidence}",
        response_text,
        flags=re.IGNORECASE,
    )

    has_evidence_tag = bool(re.search(r"\|\s*Evidence\s*:", response_text, flags=re.IGNORECASE))
    has_source_block = "[Source:" in response_text or "[Nguồn:" in response_text

    if has_source_block and not has_evidence_tag:
        response_text = re.sub(
            r"\[Source:\s*([^\]]+)\]",
            lambda match: f"[Source: {match.group(1)} | Evidence: {confidence}]",
            response_text,
            count=1,
        )
        response_text = re.sub(
            r"\[Nguồn:\s*([^\]]+)\]",
            lambda match: f"[Nguồn: {match.group(1)} | Evidence: {confidence}]",
            response_text,
            count=1,
        )

    if not has_source_block and not has_evidence_tag:
        response_text = f"{response_text} [Source: EXECUTIVE SUMMARY | Evidence: {confidence}]"

    return response_text


def _extract_relevant_papers_from_final_report(final_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections = _coerce_json_dict(final_report.get("sections")) or {}
    literature_context = _coerce_json_dict(final_report.get("literature_context")) or _coerce_json_dict(
        sections.get("literature_context")
    )
    papers = literature_context.get("relevant_papers")
    if not isinstance(papers, list):
        return []

    sanitized: List[Dict[str, Any]] = []
    for paper in papers:
        if isinstance(paper, dict):
            sanitized.append(paper)
    return sanitized


def _infer_evidence_qa_result_from_report(final_report: Dict[str, Any]) -> Dict[str, Any]:
    papers = _extract_relevant_papers_from_final_report(final_report)
    total_curated = len(papers)
    confidence = "MEDIUM" if total_curated >= 3 else "LOW"

    flags: List[str] = ["rehydrated_from_final_report_only"]
    if total_curated == 0:
        flags.append("no_articles_found")

    return {
        "research_result_sanitized": {},
        "curated_articles": papers,
        "total_articles_in": total_curated,
        "total_articles_curated": total_curated,
        "high_relevance_count": 0,
        "evidence_confidence": confidence,
        "research_quality_flags": flags,
    }


def _format_report_chat_fallback_reply(session_id: str, user_message: str) -> str:
    session = get_report_chat_session(session_id)
    if session is None:
        return "Session expired or not found. Please restart the analysis."

    final_report = _coerce_json_dict(session.report_state.get("final_report")) or {}
    confidence = _report_chat_confidence_level(session_id)
    summary = str(final_report.get("executive_summary") or "").strip()

    normalized_message = str(user_message or "").lower()
    is_vietnamese = bool(re.search(r"[ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", normalized_message))
    if not is_vietnamese:
        vi_hints = ("cơ chế", "độc tính", "thuốc", "báo cáo", "khuyến nghị", "rủi ro")
        is_vietnamese = any(token in normalized_message for token in vi_hints)

    if confidence == "LOW":
        prefix_en = "Based on limited evidence, "
        prefix_vi = "Dựa trên bằng chứng còn hạn chế, "
    elif confidence == "MEDIUM":
        prefix_en = "Evidence suggests "
        prefix_vi = "Bằng chứng hiện có cho thấy "
    else:
        prefix_en = ""
        prefix_vi = ""

    if is_vietnamese:
        body = (
            f"{prefix_vi}{summary}"
            if summary
            else f"{prefix_vi}thông tin chi tiết chưa sẵn sàng trong phiên chat hiện tại."
        )
        return _normalize_report_chat_citation_confidence(
            session_id,
            f"{body} [Source: EXECUTIVE SUMMARY | Evidence: {confidence}]",
        )

    body = (
        f"{prefix_en}{summary}"
        if summary
        else f"{prefix_en}detailed report chat content is currently unavailable in this runtime."
    )
    return _normalize_report_chat_citation_confidence(
        session_id,
        f"{body} [Source: EXECUTIVE SUMMARY | Evidence: {confidence}]",
    )


def _make_report_chat_llm_caller(session_id: str) -> Callable[..., str]:
    def _llm_caller(system_prompt: str, messages: List[Dict[str, str]], max_tool_calls: int = 3) -> str:
        prompt = _build_report_chat_prompt(system_prompt, messages)
        client, auth_mode = _build_report_chat_client()
        if client is None:
            logger.warning("Report chat LLM unavailable (%s); using deterministic fallback", auth_mode)
            return _format_report_chat_fallback_reply(
                session_id,
                messages[-1].get("content", "") if messages else "",
            )

        model_name = _resolve_report_chat_model()
        _ = max_tool_calls  # Reserved for future native tool-calling loop.

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.2},
            )
            text = str(getattr(response, "text", "") or "").strip()
            if text:
                normalized = _normalize_report_chat_out_of_scope_reply(session_id, text)
                return _normalize_report_chat_citation_confidence(session_id, normalized)
        except Exception as exc:
            logger.warning("Report chat generation failed (%s): %s", auth_mode, exc)

        return _format_report_chat_fallback_reply(
            session_id,
            messages[-1].get("content", "") if messages else "",
        )

    return _llm_caller


def _build_evidence_qa_result(
    research_payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _normalize_evidence_qa_result(payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload) if isinstance(payload, dict) else {}

        curated_articles = normalized.get("curated_articles")
        if not isinstance(curated_articles, list):
            curated_articles = []
        curated_articles = [item for item in curated_articles if isinstance(item, dict)]

        research_flags = normalized.get("research_quality_flags")
        if not isinstance(research_flags, list):
            research_flags = []
        research_flags = [str(flag) for flag in research_flags if str(flag).strip()]

        total_curated_raw = normalized.get("total_articles_curated", len(curated_articles))
        total_in_raw = normalized.get("total_articles_in", total_curated_raw)
        high_relevance_raw = normalized.get("high_relevance_count", 0)

        try:
            total_curated = int(total_curated_raw)
        except (TypeError, ValueError):
            total_curated = len(curated_articles)

        try:
            total_in = int(total_in_raw)
        except (TypeError, ValueError):
            total_in = total_curated

        try:
            high_relevance_count = int(high_relevance_raw)
        except (TypeError, ValueError):
            high_relevance_count = 0

        normalized["curated_articles"] = curated_articles
        normalized["research_quality_flags"] = research_flags
        normalized["total_articles_curated"] = max(total_curated, 0)
        normalized["total_articles_in"] = max(total_in, 0)
        normalized["high_relevance_count"] = max(high_relevance_count, 0)
        normalized["evidence_confidence"] = _normalize_evidence_confidence(
            normalized.get("evidence_confidence")
        )

        return normalized

    if isinstance(state, dict):
        qa_from_state = _extract_state_payload(state, "evidence_qa_result")
        if qa_from_state:
            return _normalize_evidence_qa_result(qa_from_state)

    qa_wrapped = run_evidence_qa(research_payload if isinstance(research_payload, dict) else {})
    if isinstance(qa_wrapped, dict):
        qa_result = qa_wrapped.get("evidence_qa_result")
        if isinstance(qa_result, dict):
            return _normalize_evidence_qa_result(qa_result)
    return _normalize_evidence_qa_result({})


def _upsert_report_chat_session(
    *,
    analysis_session_id: str,
    smiles: str,
    final_report: Dict[str, Any],
    research_payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    evidence_qa_result: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not isinstance(final_report, dict) or not final_report:
        return None

    evidence_payload = (
        _build_evidence_qa_result(research_payload, state)
        if not isinstance(evidence_qa_result, dict)
        else _build_evidence_qa_result(research_payload={}, state={"evidence_qa_result": evidence_qa_result})
    )

    report_state = {
        "smiles_input": smiles,
        "final_report": final_report,
        "evidence_qa_result": evidence_payload,
    }
    chat_session_id = create_chat_session(report_state)
    with _report_chat_lock:
        _report_chat_by_analysis_session[analysis_session_id] = chat_session_id
    return chat_session_id


def _resolve_report_chat_session_id(
    *,
    chat_session_id: Optional[str],
    analysis_session_id: Optional[str],
) -> Optional[str]:
    if isinstance(chat_session_id, str) and chat_session_id.strip():
        candidate = chat_session_id.strip()
        if get_report_chat_session(candidate) is not None:
            return candidate

    if isinstance(analysis_session_id, str) and analysis_session_id.strip():
        with _report_chat_lock:
            mapped = _report_chat_by_analysis_session.get(analysis_session_id.strip())
        if mapped and get_report_chat_session(mapped) is not None:
            return mapped

    return None


def _rehydrate_report_chat_session_from_payload(
    *,
    requested_chat_session_id: Optional[str],
    analysis_session_id: Optional[str],
    report_state_payload: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not isinstance(report_state_payload, dict) or not report_state_payload:
        return None

    final_report = _coerce_json_dict(report_state_payload.get("final_report")) or {}
    if not final_report:
        return None

    report_metadata = final_report.get("report_metadata")
    smiles_input = report_state_payload.get("smiles_input")
    if not isinstance(smiles_input, str) or not smiles_input.strip():
        if isinstance(report_metadata, dict):
            smiles_input = str(report_metadata.get("smiles") or "").strip()
        else:
            smiles_input = ""

    reconstructed_report_state = {
        "smiles_input": str(smiles_input or ""),
        "final_report": final_report,
        "evidence_qa_result": _build_evidence_qa_result(
            research_payload={},
            state={
                "evidence_qa_result": (
                    _coerce_json_dict(report_state_payload.get("evidence_qa_result"))
                    or _infer_evidence_qa_result_from_report(final_report)
                )
            },
        ),
    }
    rehydrated_chat_session_id = create_chat_session(reconstructed_report_state)

    if isinstance(analysis_session_id, str) and analysis_session_id.strip():
        with _report_chat_lock:
            _report_chat_by_analysis_session[analysis_session_id.strip()] = rehydrated_chat_session_id

    logger.info(
        "Rehydrated report chat session from payload "
        "(requested_chat_session_id=%s, analysis_session_id=%s, rehydrated_chat_session_id=%s)",
        requested_chat_session_id,
        analysis_session_id,
        rehydrated_chat_session_id,
    )
    return rehydrated_chat_session_id


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

    has_visuals = bool(
        structural_payload.get("molecule_png_base64")
        or structural_payload.get("heatmap_base64")
    )
    top_atoms = structural_payload.get("top_atoms")
    top_bonds = structural_payload.get("top_bonds")
    has_ranked_features = (
        (isinstance(top_atoms, list) and len(top_atoms) > 0)
        or (isinstance(top_bonds, list) and len(top_bonds) > 0)
    )

    if not (has_visuals or has_ranked_features):
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

    has_visuals = bool(explanation.get("molecule_png_base64") or explanation.get("heatmap_base64"))
    top_atoms = explanation.get("top_atoms")
    top_bonds = explanation.get("top_bonds")
    has_ranked_features = (
        (isinstance(top_atoms, list) and len(top_atoms) > 0)
        or (isinstance(top_bonds, list) and len(top_bonds) > 0)
    )
    return has_visuals or has_ranked_features


def _screening_payload_has_molrag_data(screening_payload: Dict[str, Any]) -> bool:
    if not isinstance(screening_payload, dict):
        return False

    molrag = screening_payload.get("molrag")
    if not isinstance(molrag, dict):
        return False

    return bool(
        molrag.get("enabled")
        or "retrieved_examples" in molrag
        or "reasoning_summary" in molrag
        or "suggested_label" in molrag
    )


def _extract_explanation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    explanation = payload.get("explanation")
    if isinstance(explanation, dict):
        return explanation

    if (
        payload.get("heatmap_base64")
        or payload.get("molecule_png_base64")
        or isinstance(payload.get("top_atoms"), list)
        or isinstance(payload.get("top_bonds"), list)
    ):
        return payload

    return {}


def _merge_explanation_into_screening(
    screening_payload: Dict[str, Any],
    explanation_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(screening_payload, dict):
        return screening_payload
    if not isinstance(explanation_payload, dict) or not explanation_payload:
        return screening_payload

    merged = dict(screening_payload)
    merged["explanation"] = explanation_payload
    return merged


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


def _compose_inference_signature(
    inference_backend: str,
    binary_tox_model: str,
    tox_type_model: str,
) -> str:
    return f"{inference_backend}|binary={binary_tox_model}|tox_type={tox_type_model}"


def _parse_inference_signature(raw_signature: str) -> Dict[str, str]:
    parsed = {"backend": "", "binary": "", "tox_type": ""}
    signature = str(raw_signature or "").strip()
    if not signature:
        return parsed

    segments = [segment.strip() for segment in signature.split("|") if segment.strip()]
    if not segments:
        return parsed

    parsed["backend"] = segments[0].lower()
    for segment in segments[1:]:
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key_norm = key.strip().lower()
        if key_norm in parsed:
            parsed[key_norm] = value.strip()

    return parsed


def _screening_payload_matches_requested_models(
    screening_payload: Dict[str, Any],
    inference_backend: str,
    binary_tox_model: str,
    tox_type_model: str,
) -> bool:
    if not isinstance(screening_payload, dict):
        return False

    inference_context = screening_payload.get("inference_context")
    if not isinstance(inference_context, dict):
        return False

    signature = str(inference_context.get("inference_backend") or "").strip()
    parsed = _parse_inference_signature(signature)

    return (
        parsed["backend"] == str(inference_backend or "").strip().lower()
        and parsed["binary"].lower() == str(binary_tox_model or "").strip().lower()
        and parsed["tox_type"].lower() == str(tox_type_model or "").strip().lower()
    )


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


def _normalize_evidence_confidence(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
        return normalized
    return "LOW"


def _flatten_exception_messages(exc: BaseException, max_messages: int = 12) -> List[str]:
    pending: List[BaseException] = [exc]
    seen_ids: set[int] = set()
    messages: List[str] = []

    while pending and len(messages) < max_messages:
        current = pending.pop(0)
        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)

        text = str(current or "").strip()
        if text and text not in messages:
            messages.append(text)

        nested = getattr(current, "exceptions", None)
        if isinstance(nested, (list, tuple)):
            for child in nested:
                if isinstance(child, BaseException):
                    pending.append(child)

        for linked in (getattr(current, "__cause__", None), getattr(current, "__context__", None)):
            if isinstance(linked, BaseException):
                pending.append(linked)

    return messages


def _is_adk_taskgroup_runtime_error(exc: Exception) -> bool:
    lowered_messages = [message.lower() for message in _flatten_exception_messages(exc)]
    if not lowered_messages:
        lowered_messages = [str(exc or "").lower()]

    combined = " | ".join(lowered_messages)
    return (
        "taskgroup" in combined
        or "exceptiongroup" in combined
        or (
            "unhandled errors" in combined
            and "sub-exception" in combined
        )
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
    include_events: bool = False,
    event_sink: Optional[List[AgentEventRecord]] = None,
) -> None:
    if Runner is None or adk_session_service is None:
        return

    step_runner = Runner(
        agent=agent,
        session_service=adk_session_service,
        app_name=ADK_APP_NAME,
    )

    async def _run_with_runner(runner: Any) -> None:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if include_events and event_sink is not None:
                event_sink.append(
                    AgentEventRecord(
                        type=getattr(event, "type", None),
                        author=getattr(event, "author", None),
                        function_calls=_extract_event_function_calls(event),
                        function_responses=_extract_event_function_responses(event),
                        is_final=_is_final_event_response(event),
                        text_preview=_extract_event_text_preview(event),
                    )
                )

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


def _clinical_head_ready() -> bool:
    return all(k in model_state for k in ("clinical_head_model", "clinical_head_meta"))


def _required_models_ready() -> bool:
    required = []
    if CLINTOX_ENABLED:
        required.append(_xsmiles_ready())
    if TOX21_ENABLED:
        required.append(_tox21_ready())

    if CLINICAL_SIGNAL_STRATEGY == "xsmiles":
        required.append(_xsmiles_ready())
    elif CLINICAL_SIGNAL_STRATEGY == "clinical_head" and TOX21_ENABLED:
        required.append(_clinical_head_ready())

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

    top_atoms: List[AtomImportance] = []
    top_bonds: List[BondImportance] = []
    if mol is not None:
        for atom in list(mol.GetAtoms())[:10]:
            top_atoms.append(
                AtomImportance(
                    atom_idx=int(atom.GetIdx()),
                    element=str(atom.GetSymbol()),
                    importance=0.0,
                    is_in_ring=bool(atom.IsInRing()),
                    is_aromatic=bool(atom.GetIsAromatic()),
                )
            )

        for bond in list(mol.GetBonds())[:10]:
            begin_idx = int(bond.GetBeginAtomIdx())
            end_idx = int(bond.GetEndAtomIdx())
            begin_atom = mol.GetAtomWithIdx(begin_idx)
            end_atom = mol.GetAtomWithIdx(end_idx)
            top_bonds.append(
                BondImportance(
                    bond_idx=int(bond.GetIdx()),
                    atom_pair=f"{begin_atom.GetSymbol()}({begin_idx}) - {end_atom.GetSymbol()}({end_idx})",
                    bond_type=str(bond.GetBondTypeAsDouble()),
                    importance=0.0,
                )
            )

    return ToxicityExplanationOutput(
        target_task=task,
        target_task_score=0.0,
        top_atoms=top_atoms,
        top_bonds=top_bonds,
        heatmap_base64=None,
        molecule_png_base64=molecule_png_base64,
        explainer_note=(
            "Explainer attribution is unavailable in this runtime. "
            "Returning deterministic structural placeholders for API contract stability."
        ),
    )


def _resolve_clinical_signal_source(
    clinical_model_available: bool,
    clinical_head_available: bool,
    tox21_available: bool,
) -> str:
    """Resolve which engine provides clinical signal for this request."""
    strategy = CLINICAL_SIGNAL_STRATEGY
    if strategy not in {"auto", "xsmiles", "clinical_head", "tox21_proxy"}:
        strategy = "auto"

    if strategy == "xsmiles":
        if clinical_model_available:
            return "xsmiles"
        return "unavailable"

    if strategy == "clinical_head":
        if clinical_head_available and tox21_available:
            return "clinical_head"
        return "unavailable"

    if strategy == "tox21_proxy":
        if tox21_available:
            return "tox21_proxy"
        return "unavailable"

    # strategy == auto
    if clinical_model_available:
        return "xsmiles"
    if clinical_head_available and tox21_available:
        return "clinical_head"
    if tox21_available:
        return "tox21_proxy"
    return "unavailable"


def _resolve_inference_backend(raw_backend: Optional[str]) -> str:
    candidate = str(raw_backend or DEFAULT_INFERENCE_BACKEND or "xsmiles").strip().lower()
    resolved = INFERENCE_BACKEND_ALIASES.get(candidate)
    if resolved:
        return resolved

    supported = sorted(set(["xsmiles", *PRETRAINED_DUAL_HEAD_BACKENDS.keys()]))
    raise HTTPException(
        status_code=400,
        detail={
            "error": "invalid_inference_backend",
            "message": f"Unsupported inference_backend='{raw_backend}'.",
            "supported_backends": supported,
        },
    )


def _is_pretrained_backend(inference_backend: str) -> bool:
    return inference_backend in PRETRAINED_DUAL_HEAD_BACKENDS


def _pretrained_backend_source_name(inference_backend: str) -> str:
    return f"pretrained_dual_head:{inference_backend}"


def _load_pretrained_bundle_sync(inference_backend: str) -> Dict[str, Any]:
    backend_cfg = PRETRAINED_DUAL_HEAD_BACKENDS.get(inference_backend)
    if backend_cfg is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_inference_backend",
                "message": f"Unsupported pretrained backend: {inference_backend}",
            },
        )

    bundles = model_state.setdefault("pretrained_dual_head_bundles", {})
    if inference_backend in bundles:
        return bundles[inference_backend]

    model_dir = Path(backend_cfg.get("model_dir", ""))
    startup_key = f"pretrained_{inference_backend}"

    if not model_dir.exists():
        detail = f"Model directory not found: {model_dir}"
        model_state.setdefault("startup_errors", {})[startup_key] = detail
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": detail,
            },
        )

    try:
        bundle = load_pretrained_dual_head_bundle(model_dir=model_dir, device=DEVICE)
    except HTTPException:
        raise
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        model_state.setdefault("startup_errors", {})[startup_key] = msg
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": f"Failed loading pretrained backend '{inference_backend}'.",
                "startup_error": msg,
            },
        )

    bundle["display_name"] = backend_cfg.get("display_name", inference_backend)
    bundle["inference_backend"] = inference_backend
    bundles[inference_backend] = bundle
    model_state.get("startup_errors", {}).pop(startup_key, None)
    return bundle


def _pretrained_model_source_name(model_key: str) -> str:
    return f"pretrained_dual_head_model:{model_key}"


def _pretrained_gin_model_source_name(model_key: str) -> str:
    return f"pretrained_gin_model:{model_key}"


def _is_dual_head_model_key(model_key: str) -> bool:
    return model_key in DUAL_HEAD_MODEL_DIRS


def _is_pretrained_gin_model_key(model_key: str) -> bool:
    return model_key == TOX21_PRETRAINED_GIN_MODEL_KEY


def _is_ensemble_tox_type_model_key(model_key: str) -> bool:
    return model_key in ENSEMBLE_TOX_TYPE_MODEL_KEYS


def _is_ensemble_binary_model_key(model_key: str) -> bool:
    return model_key in ENSEMBLE_BINARY_MODEL_KEYS


def _get_tox21_task_names() -> List[str]:
    task_names = list(model_state.get("tox21_tasks") or [])
    if task_names:
        return task_names
    return list(TOX21_TASK_NAMES_FALLBACK)


def _get_ensemble_spec(model_key: str) -> Optional[Dict[str, Any]]:
    spec = DUALHEAD_ENSEMBLE_MODEL_SPECS.get(str(model_key or "").strip())
    if spec is None:
        return None
    return dict(spec)


def _load_ensemble_weight_payload(model_key: str) -> Dict[str, Any]:
    payload_cache = model_state.setdefault("ensemble_weight_payloads", {})
    cache_key = str(model_key or "").strip()
    if cache_key in payload_cache:
        cached = payload_cache.get(cache_key)
        return dict(cached) if isinstance(cached, dict) else {}

    spec = _get_ensemble_spec(cache_key)
    if not spec:
        payload_cache[cache_key] = {}
        return {}

    weights_path = spec.get("weights_path")
    if not isinstance(weights_path, Path) or not weights_path.exists():
        payload_cache[cache_key] = {}
        return {}

    try:
        with open(weights_path, "r") as f:
            raw = json.load(f)
    except Exception:
        payload_cache[cache_key] = {}
        return {}

    weights_payload = dict(raw.get("weights") or {}) if isinstance(raw, dict) else {}
    payload_cache[cache_key] = weights_payload
    return dict(weights_payload)


def _load_aux_tox21_member_bundle_sync(model_key: str) -> Dict[str, Any]:
    resolved_key = str(model_key or "").strip()
    if resolved_key not in AUX_TOX21_MEMBER_MODEL_DIRS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_tox21_member_model",
                "message": f"Unsupported auxiliary tox21 member model: {resolved_key}",
            },
        )

    bundles = model_state.setdefault("tox21_aux_member_bundles", {})
    cache_key = f"aux_tox21::{resolved_key}"
    if cache_key in bundles:
        return bundles[cache_key]

    model_dir = AUX_TOX21_MEMBER_MODEL_DIRS[resolved_key]
    startup_key = f"aux_tox21_{resolved_key}"

    if not model_dir.exists():
        detail = f"Model directory not found: {model_dir}"
        model_state.setdefault("startup_errors", {})[startup_key] = detail
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": detail,
            },
        )

    try:
        if resolved_key == TOX21_FINGERPRINT_MODEL_KEY:
            n_bits = 2048
            radius = 2
            if TOX21_FINGERPRINT_CONFIG_PATH.exists():
                with open(TOX21_FINGERPRINT_CONFIG_PATH, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                mc = dict(cfg.get("model") or {})
                n_bits = int(mc.get("nbits", n_bits))
                radius = int(mc.get("radius", radius))

            task_names = list(_get_tox21_task_names())
            estimators: Dict[str, Any] = {}
            model_subdir = model_dir / "models"
            for task in task_names:
                model_path = model_subdir / f"{task}.pkl"
                if not model_path.exists():
                    continue
                with open(model_path, "rb") as f:
                    estimators[task] = pickle.load(f)

            if not estimators:
                raise FileNotFoundError(f"No fingerprint estimators found in {model_subdir}")

            bundle = {
                "kind": "fingerprint",
                "model_key": resolved_key,
                "task_names": task_names,
                "estimators": estimators,
                "n_bits": int(n_bits),
                "radius": int(radius),
                "model_dir": str(model_dir),
            }
        else:
            config_path = model_dir / "config.yaml"
            ckpt_path = model_dir / "best_model.pt"
            if not config_path.exists() or not ckpt_path.exists():
                missing = []
                if not config_path.exists():
                    missing.append(str(config_path))
                if not ckpt_path.exists():
                    missing.append(str(ckpt_path))
                raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            mc = dict(cfg.get("model") or {})

            node_feat_dim, edge_feat_dim = get_feature_dims()
            num_tasks = int(mc.get("num_tasks", len(_get_tox21_task_names())))

            if resolved_key == TOX21_ATTENTIVEFP_MODEL_KEY:
                model = create_attentivefp_model(
                    node_feat_dim=node_feat_dim,
                    edge_feat_dim=edge_feat_dim,
                    hidden_channels=int(mc.get("hidden_channels", 200)),
                    num_layers=int(mc.get("num_layers", 2)),
                    num_timesteps=int(mc.get("num_timesteps", 2)),
                    dropout=float(mc.get("dropout", 0.2)),
                    num_tasks=int(num_tasks),
                )
            elif resolved_key == TOX21_GPS_MODEL_KEY:
                model = create_gps_model(
                    node_feat_dim=node_feat_dim,
                    edge_feat_dim=edge_feat_dim,
                    hidden_channels=int(mc.get("hidden_channels", 128)),
                    num_layers=int(mc.get("num_layers", 4)),
                    heads=int(mc.get("heads", 4)),
                    dropout=float(mc.get("dropout", 0.2)),
                    num_tasks=int(num_tasks),
                )
            else:
                raise ValueError(f"Unsupported auxiliary graph member: {resolved_key}")

            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()

            task_names = list(_get_tox21_task_names())
            if num_tasks != len(task_names):
                if num_tasks <= len(task_names):
                    task_names = task_names[:num_tasks]
                else:
                    task_names = [f"task_{idx}" for idx in range(num_tasks)]

            bundle = {
                "kind": "graph",
                "model_key": resolved_key,
                "task_names": task_names,
                "model": model,
                "model_dir": str(model_dir),
            }
    except HTTPException:
        raise
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        model_state.setdefault("startup_errors", {})[startup_key] = msg
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": f"Failed loading auxiliary tox21 member '{resolved_key}'.",
                "startup_error": msg,
            },
        )

    bundles[cache_key] = bundle
    model_state.get("startup_errors", {}).pop(startup_key, None)
    return bundle


def _member_score_from_estimator(estimator: Any, features: np.ndarray) -> float:
    if estimator is None:
        return float("nan")

    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(features)
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            return float(arr[1])
        if arr.size == 1:
            return float(arr[0])

    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(features), dtype=np.float32).reshape(-1)
        if raw.size:
            return float(1.0 / (1.0 + np.exp(-raw[0])))

    pred = np.asarray(estimator.predict(features), dtype=np.float32).reshape(-1)
    if pred.size:
        return float(pred[0])

    return float("nan")


def _predict_aux_tox21_member_mechanism_sync(
    smiles: str,
    model_key: str,
    mechanism_threshold: float,
) -> Dict[str, Any]:
    bundle = _load_aux_tox21_member_bundle_sync(model_key)
    task_names = list(bundle.get("task_names") or _get_tox21_task_names())
    threshold_map = {task: float(mechanism_threshold) for task in task_names}

    if bundle.get("kind") == "fingerprint":
        n_bits = int(bundle.get("n_bits", 2048))
        radius = int(bundle.get("radius", 2))
        fp = featurize_fingerprint(smiles=smiles, radius=radius, n_bits=n_bits).reshape(1, -1)

        estimators = dict(bundle.get("estimators") or {})
        probs = np.array(
            [_member_score_from_estimator(estimators.get(task), fp) for task in task_names],
            dtype=np.float32,
        )
    else:
        data = smiles_to_pyg_data(smiles, label=None)
        if data is None:
            probs = np.full((len(task_names),), np.nan, dtype=np.float32)
        else:
            model = bundle["model"]
            data = data.to(DEVICE)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                logits = model(data.x, data.edge_index, data.edge_attr, batch)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)

            if probs.shape[0] != len(task_names):
                if probs.shape[0] < len(task_names):
                    padded = np.full((len(task_names),), np.nan, dtype=np.float32)
                    padded[: probs.shape[0]] = probs
                    probs = padded
                else:
                    probs = probs[: len(task_names)]

    task_scores = {
        task: float(probs[idx]) if np.isfinite(probs[idx]) else float("nan")
        for idx, task in enumerate(task_names)
    }
    active_tasks = [
        task
        for idx, task in enumerate(task_names)
        if np.isfinite(probs[idx]) and float(probs[idx]) >= float(mechanism_threshold)
    ]

    if np.any(np.isfinite(probs)):
        safe_probs = np.where(np.isfinite(probs), probs, -np.inf)
        top_idx = int(np.argmax(safe_probs))
        highest_risk_task = str(task_names[top_idx])
        highest_risk_score = float(probs[top_idx]) if np.isfinite(probs[top_idx]) else 0.0
    else:
        highest_risk_task = "—"
        highest_risk_score = 0.0

    return {
        "task_scores": task_scores,
        "active_tasks": active_tasks,
        "highest_risk_task": highest_risk_task,
        "highest_risk_score": highest_risk_score,
        "assay_hits": int(len(active_tasks)),
        "mechanistic_alert": bool(len(active_tasks) > 0),
        "threshold_used": float(mechanism_threshold),
        "task_thresholds": threshold_map,
        "source": str(model_key),
    }


def _predict_tox21_member_mechanism_sync(
    smiles: str,
    member_key: str,
    mechanism_threshold: float,
) -> Dict[str, Any]:
    if _is_pretrained_gin_model_key(member_key):
        gin_bundle = _load_pretrained_gin_bundle_sync(member_key)
        outputs = predict_pretrained_gin_tox21_outputs(
            smiles_list=[smiles],
            model=gin_bundle["model"],
            task_names=list(gin_bundle.get("task_names") or _get_tox21_task_names()),
            device=DEVICE,
            mechanism_threshold=float(mechanism_threshold),
            task_thresholds=gin_bundle.get("tox21_thresholds"),
            batch_size=32,
            source_name=_pretrained_gin_model_source_name(member_key),
        )
        return dict(outputs[0].get("mechanism") or {})

    if _is_dual_head_model_key(member_key):
        member_bundle = _load_dual_head_bundle_sync(member_key)
        outputs = predict_pretrained_dual_head_outputs(
            smiles_list=[smiles],
            model=member_bundle["model"],
            tokenizer=member_bundle["tokenizer"],
            task_names=list(member_bundle.get("task_names") or _get_tox21_task_names()),
            device=DEVICE,
            clinical_threshold=0.5,
            mechanism_threshold=float(mechanism_threshold),
            task_thresholds=member_bundle.get("tox21_thresholds"),
            max_length=int(member_bundle.get("max_length", 128)),
            batch_size=32,
            source_name=_pretrained_model_source_name(member_key),
        )
        return dict(outputs[0].get("mechanism") or {})

    if member_key in AUX_TOX21_MEMBER_MODEL_DIRS:
        try:
            return _predict_aux_tox21_member_mechanism_sync(
                smiles=smiles,
                model_key=member_key,
                mechanism_threshold=float(mechanism_threshold),
            )
        except HTTPException as exc:
            logger.warning(
                "Skipping auxiliary tox21 ensemble member '%s' due to load/inference error: %s",
                member_key,
                exc.detail,
            )
            task_names = _get_tox21_task_names()
            return {
                "task_scores": {task: float("nan") for task in task_names},
                "active_tasks": [],
                "highest_risk_task": "—",
                "highest_risk_score": 0.0,
                "assay_hits": 0,
                "mechanistic_alert": False,
                "threshold_used": float(mechanism_threshold),
                "task_thresholds": {
                    task: float(mechanism_threshold)
                    for task in task_names
                },
                "source": f"{member_key}:unavailable",
            }

    if member_key == "tox21_gatv2_model":
        if TOX21_ENABLED and _tox21_ready():
            return predict_toxicity_mechanism(
                smiles=smiles,
                model=model_state["tox21_model"],
                task_names=model_state["tox21_tasks"],
                device=DEVICE,
                threshold=float(mechanism_threshold),
                task_thresholds=model_state.get("tox21_thresholds"),
                batch_size=64,
            )
        return _fallback_mechanism_result(float(mechanism_threshold))

    return _fallback_mechanism_result(float(mechanism_threshold))


def _blend_member_scores_simple(member_scores: List[np.ndarray]) -> np.ndarray:
    stacked = np.stack(member_scores, axis=0)
    valid_counts = np.sum(np.isfinite(stacked), axis=0)
    sums = np.nansum(stacked, axis=0)
    out = (sums / np.maximum(valid_counts, 1)).astype(np.float32)
    out = np.where(valid_counts > 0, out, 0.5)
    return out


def _blend_member_scores_weighted(
    member_scores: List[np.ndarray],
    member_weights: np.ndarray,
) -> np.ndarray:
    stacked = np.stack(member_scores, axis=0)
    weights = np.asarray(member_weights, dtype=np.float32).reshape(-1)
    if weights.size != stacked.shape[0]:
        return _blend_member_scores_simple(member_scores)

    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    out = np.zeros((stacked.shape[1],), dtype=np.float32)
    for col in range(stacked.shape[1]):
        vals = stacked[:, col]
        mask = np.isfinite(vals)
        if not np.any(mask):
            out[col] = 0.5
            continue
        w = weights[mask]
        if float(w.sum()) <= 0.0:
            out[col] = float(np.mean(vals[mask]))
            continue
        out[col] = float(np.sum(vals[mask] * w) / np.sum(w))
    return out


def _blend_member_scores_taskwise(
    member_scores: List[np.ndarray],
    taskwise_weights: np.ndarray,
) -> np.ndarray:
    stacked = np.stack(member_scores, axis=0)
    weights = np.asarray(taskwise_weights, dtype=np.float32)
    if weights.ndim != 2:
        return _blend_member_scores_simple(member_scores)

    if weights.shape[0] != stacked.shape[1] or weights.shape[1] != stacked.shape[0]:
        return _blend_member_scores_simple(member_scores)

    out = np.zeros((stacked.shape[1],), dtype=np.float32)
    for task_idx in range(stacked.shape[1]):
        vals = stacked[:, task_idx]
        mask = np.isfinite(vals)
        if not np.any(mask):
            out[task_idx] = 0.5
            continue
        w = np.where(np.isfinite(weights[task_idx]) & (weights[task_idx] > 0.0), weights[task_idx], 0.0)
        w = w[mask]
        if float(w.sum()) <= 0.0:
            out[task_idx] = float(np.mean(vals[mask]))
            continue
        out[task_idx] = float(np.sum(vals[mask] * w) / np.sum(w))
    return out


def _predict_ensemble_mechanism_sync(
    model_key: str,
    smiles: str,
    mechanism_threshold: float,
) -> Dict[str, Any]:
    spec = _get_ensemble_spec(model_key)
    if not spec:
        return _fallback_mechanism_result(float(mechanism_threshold))

    task_names = list(_get_tox21_task_names())
    member_keys = list(spec.get("mechanism_members") or [])

    member_scores: List[np.ndarray] = []
    member_sources: List[str] = []

    for member_key in member_keys:
        mechanism = _predict_tox21_member_mechanism_sync(
            smiles=smiles,
            member_key=str(member_key),
            mechanism_threshold=float(mechanism_threshold),
        )
        scores = dict(mechanism.get("task_scores") or {})
        score_vec = np.array(
            [float(scores.get(task, np.nan)) for task in task_names],
            dtype=np.float32,
        )
        member_scores.append(score_vec)
        member_sources.append(str(mechanism.get("source") or member_key))

    if not member_scores:
        return _fallback_mechanism_result(threshold=float(mechanism_threshold))

    mechanism_mode = str(spec.get("mechanism_mode", "simple"))
    if mechanism_mode == "taskwise_weighted":
        weights_payload = _load_ensemble_weight_payload(model_key)
        raw_taskwise_weights = np.asarray(weights_payload.get("tox21_taskwise_weights") or [], dtype=np.float32)
        raw_members = list(weights_payload.get("tox21_members") or [])

        if raw_taskwise_weights.ndim == 2 and raw_members and raw_taskwise_weights.shape[1] == len(raw_members):
            aligned = np.zeros((raw_taskwise_weights.shape[0], len(member_keys)), dtype=np.float32)
            for idx, member in enumerate(member_keys):
                if member in raw_members:
                    aligned[:, idx] = raw_taskwise_weights[:, raw_members.index(member)]
            raw_taskwise_weights = aligned

        mean_scores = _blend_member_scores_taskwise(member_scores, raw_taskwise_weights)
    else:
        mean_scores = _blend_member_scores_simple(member_scores)

    threshold_map = {
        task_name: float(mechanism_threshold)
        for task_name in task_names
    }
    active_tasks = [
        task_name
        for idx, task_name in enumerate(task_names)
        if float(mean_scores[idx]) >= threshold_map[task_name]
    ]
    top_idx = int(np.argmax(mean_scores))

    return {
        "task_scores": {
            task_name: float(mean_scores[idx])
            for idx, task_name in enumerate(task_names)
        },
        "active_tasks": active_tasks,
        "highest_risk_task": str(task_names[top_idx]),
        "highest_risk_score": float(mean_scores[top_idx]),
        "assay_hits": int(len(active_tasks)),
        "mechanistic_alert": bool(len(active_tasks) > 0),
        "threshold_used": float(mechanism_threshold),
        "task_thresholds": threshold_map,
        "source": str(model_key),
        "ensemble_members": member_keys,
        "ensemble_member_sources": member_sources,
    }


def _predict_ensemble_clinical_sync(
    model_key: str,
    smiles: str,
    clinical_threshold: float,
    mechanism_threshold: float,
) -> Dict[str, Any]:
    spec = _get_ensemble_spec(model_key)
    if not spec:
        return {
            "label": "NON_TOXIC",
            "is_toxic": False,
            "confidence": 0.0,
            "p_toxic": 0.0,
            "threshold_used": float(clinical_threshold),
            "source": str(model_key),
            "ensemble_members": [],
            "ensemble_member_sources": [],
            "ensemble_member_probs": {},
        }

    member_keys = list(spec.get("clinical_members") or [])
    member_probs: List[float] = []
    member_sources: List[str] = []
    member_prob_map: Dict[str, float] = {}

    for member_key in member_keys:
        if not _is_dual_head_model_key(member_key):
            continue
        member_bundle = _load_dual_head_bundle_sync(member_key)
        outputs = predict_pretrained_dual_head_outputs(
            smiles_list=[smiles],
            model=member_bundle["model"],
            tokenizer=member_bundle["tokenizer"],
            task_names=list(member_bundle.get("task_names") or []),
            device=DEVICE,
            clinical_threshold=float(clinical_threshold),
            mechanism_threshold=float(mechanism_threshold),
            task_thresholds=member_bundle.get("tox21_thresholds"),
            max_length=int(member_bundle.get("max_length", 128)),
            batch_size=32,
            source_name=_pretrained_model_source_name(member_key),
        )
        clinical = dict(outputs[0].get("clinical") or {})
        p_toxic = float(clinical.get("p_toxic", np.nan))
        if np.isfinite(p_toxic):
            member_probs.append(p_toxic)
            member_sources.append(str(clinical.get("source") or member_key))
            member_prob_map[str(member_key)] = p_toxic

    if not member_probs:
        return {
            "label": "NON_TOXIC",
            "is_toxic": False,
            "confidence": 0.0,
            "p_toxic": 0.0,
            "threshold_used": float(clinical_threshold),
            "source": str(model_key),
            "ensemble_members": member_keys,
            "ensemble_member_sources": member_sources,
            "ensemble_member_probs": member_prob_map,
        }

    clinical_mode = str(spec.get("clinical_mode", "simple"))
    if clinical_mode == "weighted":
        weights_payload = _load_ensemble_weight_payload(model_key)
        raw_weights = np.asarray(weights_payload.get("herg_weights") or [], dtype=np.float32).reshape(-1)
        raw_members = list(weights_payload.get("herg_members") or [])
        if raw_weights.size and raw_members:
            aligned_weights = np.zeros((len(member_keys),), dtype=np.float32)
            for idx, member in enumerate(member_keys):
                if member in raw_members:
                    aligned_weights[idx] = float(raw_weights[raw_members.index(member)])

            probs_arr = np.array([member_prob_map.get(member, np.nan) for member in member_keys], dtype=np.float32)
            valid = np.isfinite(probs_arr)
            if np.any(valid):
                w = aligned_weights[valid]
                if float(w.sum()) <= 0.0:
                    mean_p_toxic = float(np.mean(probs_arr[valid]))
                else:
                    mean_p_toxic = float(np.sum(probs_arr[valid] * w) / np.sum(w))
            else:
                mean_p_toxic = 0.0
        else:
            mean_p_toxic = float(np.mean(np.asarray(member_probs, dtype=np.float32)))
    else:
        mean_p_toxic = float(np.mean(np.asarray(member_probs, dtype=np.float32)))

    is_toxic = bool(mean_p_toxic >= float(clinical_threshold))
    confidence = abs(mean_p_toxic - float(clinical_threshold)) / max(
        float(clinical_threshold),
        1.0 - float(clinical_threshold),
    )

    return {
        "label": "TOXIC" if is_toxic else "NON_TOXIC",
        "is_toxic": is_toxic,
        "confidence": float(min(confidence, 1.0)),
        "p_toxic": mean_p_toxic,
        "threshold_used": float(clinical_threshold),
        "source": str(model_key),
        "ensemble_members": member_keys,
        "ensemble_member_sources": member_sources,
        "ensemble_member_probs": member_prob_map,
    }


def _resolve_explainer_engine_model_key(tox_type_model_key: str) -> str:
    """Pick explainer engine model based on selected tox-type model key."""
    if _is_ensemble_tox_type_model_key(tox_type_model_key):
        spec = _get_ensemble_spec(tox_type_model_key) or {}
        return str(spec.get("explainer_engine") or TOX21_ENSEMBLE_3_BEST_EXPLAINER_KEY)
    if tox_type_model_key in TOX_TYPE_MODEL_DIRS:
        return tox_type_model_key
    return "tox21_gatv2_model"


def _resolve_binary_tox_model_key(raw_model: Optional[str]) -> str:
    candidate = str(raw_model or "").strip()
    if not candidate:
        return DEFAULT_BINARY_TOX_MODEL_KEY
    if candidate in DUAL_HEAD_MODEL_DIRS or _is_ensemble_binary_model_key(candidate):
        return candidate

    supported_models = sorted(set([*DUAL_HEAD_MODEL_DIRS.keys(), *ENSEMBLE_BINARY_MODEL_KEYS]))
    raise HTTPException(
        status_code=400,
        detail={
            "error": "invalid_binary_tox_model",
            "message": f"Unsupported binary_tox_model='{raw_model}'.",
            "requested_model": raw_model,
            "supported_models": supported_models,
        },
    )


def _resolve_tox_type_model_key(raw_model: Optional[str]) -> str:
    candidate = str(raw_model or "").strip()
    if not candidate:
        return DEFAULT_TOX_TYPE_MODEL_KEY
    if candidate in TOX_TYPE_MODEL_DIRS or _is_ensemble_tox_type_model_key(candidate):
        return candidate

    supported_models = sorted(set([*TOX_TYPE_MODEL_DIRS.keys(), *ENSEMBLE_TOX_TYPE_MODEL_KEYS]))
    raise HTTPException(
        status_code=400,
        detail={
            "error": "invalid_tox_type_model",
            "message": f"Unsupported tox_type_model='{raw_model}'.",
            "requested_model": raw_model,
            "supported_models": supported_models,
        },
    )


def _load_dual_head_bundle_sync(model_key: str) -> Dict[str, Any]:
    resolved_key = str(model_key or "").strip()
    if resolved_key not in DUAL_HEAD_MODEL_DIRS:
        supported_models = sorted(DUAL_HEAD_MODEL_DIRS.keys())
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_dual_head_model_key",
                "message": f"Unsupported dual-head model key='{model_key}'.",
                "requested_model": model_key,
                "supported_models": supported_models,
            },
        )
    model_dir = DUAL_HEAD_MODEL_DIRS[resolved_key]

    bundles = model_state.setdefault("pretrained_dual_head_bundles", {})
    cache_key = f"dual_head::{resolved_key}"
    if cache_key in bundles:
        return bundles[cache_key]

    startup_key = f"dual_head_{resolved_key}"

    if not model_dir.exists():
        detail = f"Model directory not found: {model_dir}"
        model_state.setdefault("startup_errors", {})[startup_key] = detail
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": detail,
            },
        )

    try:
        bundle = load_pretrained_dual_head_bundle(model_dir=model_dir, device=DEVICE)
    except HTTPException:
        raise
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        model_state.setdefault("startup_errors", {})[startup_key] = msg
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": f"Failed loading dual-head model '{resolved_key}'.",
                "startup_error": msg,
            },
        )

    bundle["display_name"] = resolved_key
    bundle["inference_backend"] = _pretrained_model_source_name(resolved_key)
    bundle["model_key"] = resolved_key
    bundle["model_dir"] = str(model_dir)
    bundles[cache_key] = bundle
    model_state.get("startup_errors", {}).pop(startup_key, None)
    return bundle


def _load_pretrained_gin_bundle_sync(model_key: str = TOX21_PRETRAINED_GIN_MODEL_KEY) -> Dict[str, Any]:
    resolved_key = TOX21_PRETRAINED_GIN_MODEL_KEY
    model_dir = TOX21_PRETRAINED_GIN_MODEL_DIR

    bundles = model_state.setdefault("pretrained_dual_head_bundles", {})
    cache_key = f"pretrained_gin::{resolved_key}"
    if cache_key in bundles:
        return bundles[cache_key]

    startup_key = f"pretrained_gin_{resolved_key}"

    if not model_dir.exists():
        detail = f"Model directory not found: {model_dir}"
        model_state.setdefault("startup_errors", {})[startup_key] = detail
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": detail,
            },
        )

    try:
        bundle = load_pretrained_gin_tox21_bundle(model_dir=model_dir, device=DEVICE)
    except HTTPException:
        raise
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        model_state.setdefault("startup_errors", {})[startup_key] = msg
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "feature": startup_key,
                "workspace_mode": WORKSPACE_MODE_NAME,
                "message": f"Failed loading pretrained-gin model '{resolved_key}'.",
                "startup_error": msg,
            },
        )

    bundle["display_name"] = resolved_key
    bundle["inference_backend"] = _pretrained_gin_model_source_name(resolved_key)
    bundle["model_key"] = resolved_key
    bundle["model_dir"] = str(model_dir)
    bundles[cache_key] = bundle
    model_state.get("startup_errors", {}).pop(startup_key, None)
    return bundle


def _predict_ensemble_3_best_mechanism_sync(
    smiles: str,
    mechanism_threshold: float,
) -> Dict[str, Any]:
    return _predict_ensemble_mechanism_sync(
        model_key=TOX21_ENSEMBLE_3_BEST_MODEL_KEY,
        smiles=smiles,
        mechanism_threshold=float(mechanism_threshold),
    )


def _predict_ensemble_3_best_clinical_sync(
    smiles: str,
    clinical_threshold: float,
    mechanism_threshold: float,
) -> Dict[str, Any]:
    return _predict_ensemble_clinical_sync(
        model_key=TOX21_ENSEMBLE_3_BEST_MODEL_KEY,
        smiles=smiles,
        clinical_threshold=float(clinical_threshold),
        mechanism_threshold=float(mechanism_threshold),
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Initializing model server runtime on %s (workspace_mode=%s, clintox_enabled=%s, tox21_enabled=%s). Lazy model loading is enabled.",
        DEVICE,
        WORKSPACE_MODE_NAME,
        CLINTOX_ENABLED,
        TOX21_ENABLED,
    )

    _initialize_runtime_state()
    _initialize_adk_runtime()
    await asyncio.to_thread(_preload_smiles_image_runtime_sync)

    yield
    model_state.clear()

# FastAPI App
app = FastAPI(
    title="ToxAgent Model Server",
    description="SMILESGNN toxicity prediction API for ToxAgent agentic system",
    version="0.0.6",
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


def _model_lock_sync() -> threading.Lock:
    lock = model_state.get("model_lock_sync")
    if lock is None:
        lock = threading.Lock()
        model_state["model_lock_sync"] = lock
    return lock


def _ocr_lock_sync() -> threading.Lock:
    lock = model_state.get("ocr_lock_sync")
    if lock is None:
        lock = threading.Lock()
        model_state["ocr_lock_sync"] = lock
    return lock


def _image_extraction_http_error(
    *,
    status_code: int,
    error_code: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> HTTPException:
    detail: Dict[str, Any] = {
        "error": error_code,
        "message": message,
    }
    if isinstance(extra, dict) and extra:
        detail.update(extra)
    return HTTPException(status_code=status_code, detail=detail)


def _ocr_ready() -> bool:
    return model_state.get("molscribe_predictor") is not None


def _resolve_molscribe_checkpoint_path_sync() -> str:
    cached = model_state.get("molscribe_checkpoint_path")
    if isinstance(cached, str) and cached.strip() and Path(cached).exists():
        return cached

    if MOLSCRIBE_MODEL_PATH:
        explicit = Path(MOLSCRIBE_MODEL_PATH)
        if not explicit.exists():
            raise RuntimeError(f"MOLSCRIBE_MODEL_PATH does not exist: {explicit}")
        resolved = str(explicit)
        model_state["molscribe_checkpoint_path"] = resolved
        return resolved

    if not MOLSCRIBE_AUTO_DOWNLOAD:
        raise RuntimeError(
            "MolScribe checkpoint is missing. Set MOLSCRIBE_MODEL_PATH or enable MOLSCRIBE_AUTO_DOWNLOAD."
        )

    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is unavailable; cannot auto-download MolScribe checkpoint."
        )

    ckpt_path = hf_hub_download(repo_id=MOLSCRIBE_REPO_ID, filename=MOLSCRIBE_CHECKPOINT_NAME)
    model_state["molscribe_checkpoint_path"] = str(ckpt_path)
    return str(ckpt_path)


def _load_molscribe_predictor_sync() -> Any:
    predictor = model_state.get("molscribe_predictor")
    if predictor is not None:
        return predictor

    with _ocr_lock_sync():
        predictor = model_state.get("molscribe_predictor")
        if predictor is not None:
            return predictor

        if MolScribe is None:
            raise RuntimeError(
                f"MolScribe import failed: {MOLSCRIBE_IMPORT_ERROR or 'unknown_import_error'}"
            )

        checkpoint_path = _resolve_molscribe_checkpoint_path_sync()
        requested_device = (MOLSCRIBE_DEVICE or "cpu").lower()
        if requested_device == "cuda" and torch.cuda.is_available():
            ocr_device = torch.device("cuda")
        else:
            ocr_device = torch.device("cpu")

        predictor = MolScribe(checkpoint_path, device=ocr_device)
        model_state["molscribe_predictor"] = predictor
        model_state.setdefault("startup_errors", {}).pop("smiles_image_extractor", None)
        return predictor


def _preload_smiles_image_runtime_sync() -> None:
    if not MOLSCRIBE_PRELOAD_ON_STARTUP:
        logger.info("Skipping MolScribe preload because MOLSCRIBE_PRELOAD_ON_STARTUP=false")
        return

    try:
        _load_molscribe_predictor_sync()
        logger.info("MolScribe OCR runtime preloaded successfully.")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        model_state.setdefault("startup_errors", {})["smiles_image_extractor"] = msg
        logger.warning("MolScribe preload failed: %s", msg)


def _preprocess_ocr_image(raw_image: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw_image)) as image:
            normalized = ImageOps.autocontrast(image.convert("RGB"))
            return np.asarray(normalized, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise _image_extraction_http_error(
            status_code=415,
            error_code="unsupported_image_format",
            message="Image content could not be decoded. Supported formats: PNG/JPG/JPEG/WebP.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise _image_extraction_http_error(
            status_code=415,
            error_code="unsupported_image_format",
            message="Failed to process uploaded image.",
            extra={"startup_error": f"{type(exc).__name__}: {exc}"},
        ) from exc


def _extract_smiles_from_image_sync(image_rgb: np.ndarray) -> Dict[str, Any]:
    predictor = _load_molscribe_predictor_sync()

    output = predictor.predict_image(image_rgb, return_confidence=True)
    if not isinstance(output, dict):
        raise RuntimeError("MolScribe returned malformed output.")

    smiles = str(output.get("smiles") or "").strip()
    if not smiles:
        raise _image_extraction_http_error(
            status_code=422,
            error_code="smiles_not_detected",
            message="No SMILES sequence was detected from the uploaded image.",
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _image_extraction_http_error(
            status_code=422,
            error_code="invalid_smiles_from_image",
            message="Extracted text is not a valid SMILES sequence.",
            extra={"smiles": smiles},
        )

    canonical_smiles = Chem.MolToSmiles(mol)

    confidence_raw = output.get("confidence")
    confidence: Optional[float] = None
    if isinstance(confidence_raw, (int, float)):
        confidence = float(max(0.0, min(1.0, confidence_raw)))

    warnings: List[str] = []
    if canonical_smiles != smiles:
        warnings.append("smiles_canonicalized")

    return {
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "confidence": confidence,
        "warnings": warnings,
    }


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


def _run_with_timeout_sync(fn: Any, timeout_ms: int, *args: Any, **kwargs: Any) -> Any:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=float(timeout_ms) / 1000.0)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError("explainer_timeout") from exc

# Health Check
async def health():
    try:
        await _ensure_models_loaded()
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        model_state.setdefault("startup_errors", {})["model_bootstrap"] = msg
        logger.warning("Lazy model loading failed in /health: %s", msg)

    xsmiles_ready = _xsmiles_ready()
    tox21_ready = _tox21_ready()
    clinical_head_ready = _clinical_head_ready()
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
        "clinical_head_loaded": clinical_head_ready,
        "clintox_enabled": CLINTOX_ENABLED,
        "tox21_enabled": TOX21_ENABLED,
        "clinical_signal_strategy": CLINICAL_SIGNAL_STRATEGY,
        "workspace_mode": WORKSPACE_MODE_NAME,
        "startup_errors": startup_errors,
        "adk_available": ADK_AVAILABLE,
        "adk_runtime_ready": adk_runner is not None and adk_session_service is not None,
        "adk_runtime_import_error": ADK_RUNTIME_IMPORT_ERROR,
        "molscribe_import_error": MOLSCRIBE_IMPORT_ERROR,
        "smiles_image_extraction": {
            "enabled": MolScribe is not None,
            "preload_on_startup": MOLSCRIBE_PRELOAD_ON_STARTUP,
            "ready": _ocr_ready(),
            "checkpoint_path": model_state.get("molscribe_checkpoint_path"),
            "max_upload_bytes": SMILES_IMAGE_MAX_BYTES,
            "supported_mime_types": sorted(SMILES_IMAGE_SUPPORTED_MIME_TYPES),
            "startup_error": startup_errors.get("smiles_image_extractor"),
        },
        "tox21_thresholds_loaded": model_state.get("tox21_thresholds") is not None,
        "tox21_thresholds_source": model_state.get("tox21_thresholds_source"),
        "tox21_threshold_count": len(model_state.get("tox21_thresholds") or {}),
        "inference_lock_initialized": "model_lock" in model_state,
        "device": DEVICE,
        "model_dir_exists": MODEL_DIR.exists(),
        "tox21_model_dir_exists": TOX21_MODEL_DIR.exists(),
        "tox21_pretrained_gin_model_dir_exists": TOX21_PRETRAINED_GIN_MODEL_DIR.exists(),
        "clinical_head_model_dir_exists": CLINICAL_HEAD_MODEL_DIR.exists(),
        "pretrained_backends": {
            name: {
                "display_name": cfg.get("display_name", name),
                "model_dir_exists": Path(cfg.get("model_dir", "")).exists(),
                "loaded": name in (model_state.get("pretrained_dual_head_bundles") or {}),
            }
            for name, cfg in PRETRAINED_DUAL_HEAD_BACKENDS.items()
        },
        "tox_type_models": {
            key: {
                "model_dir_exists": model_dir.exists(),
                "is_ensemble": _is_ensemble_tox_type_model_key(key),
            }
            for key, model_dir in TOX_TYPE_MODEL_DIRS.items()
        },
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
    """Render plain molecule structure as base64 PNG.

    This should remain independent from attribution heatmap rendering so UI can
    always distinguish plain structure vs explainer visualization.
    """
    try:
        # Import lazily so missing optional shared libs do not break server startup.
        from rdkit.Chem import Draw

        image = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        pass

    # Fallback path that does not rely on PIL-backed MolToImage.
    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        width, height = int(size[0]), int(size[1])
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return base64.b64encode(drawer.GetDrawingText()).decode("utf-8")
    except Exception:
        return None

app.add_api_route("/health", health, methods=["GET"], tags=["system"])
if AIP_HEALTH_ROUTE != "/health":
    app.add_api_route(AIP_HEALTH_ROUTE, health, methods=["GET"], include_in_schema=False)


@app.post("/extract-smiles-from-image", response_model=SmilesImageExtractionResponse)
async def extract_smiles_from_image(file: UploadFile = File(...)):
    filename = file.filename or "uploaded-image"
    extension = Path(filename).suffix.lower()
    content_type = str(file.content_type or "").strip().lower()

    if content_type not in SMILES_IMAGE_SUPPORTED_MIME_TYPES and extension not in SMILES_IMAGE_SUPPORTED_EXTENSIONS:
        raise _image_extraction_http_error(
            status_code=415,
            error_code="unsupported_image_format",
            message="Unsupported image format. Use PNG/JPG/JPEG/WebP.",
            extra={
                "filename": filename,
                "content_type": content_type or None,
            },
        )

    started = time.perf_counter()
    try:
        payload = await file.read()
        if not payload:
            raise _image_extraction_http_error(
                status_code=422,
                error_code="smiles_not_detected",
                message="Uploaded image is empty.",
            )

        if len(payload) > SMILES_IMAGE_MAX_BYTES:
            raise _image_extraction_http_error(
                status_code=413,
                error_code="image_too_large",
                message="Uploaded image exceeds the 5MB limit.",
                extra={
                    "max_bytes": SMILES_IMAGE_MAX_BYTES,
                    "received_bytes": len(payload),
                },
            )

        preprocessed = _preprocess_ocr_image(payload)

        try:
            extracted = await asyncio.to_thread(_extract_smiles_from_image_sync, preprocessed)
        except HTTPException:
            raise
        except Exception as exc:
            raise _image_extraction_http_error(
                status_code=503,
                error_code="extraction_service_unavailable",
                message="Image extraction service is unavailable.",
                extra={"startup_error": f"{type(exc).__name__}: {exc}"},
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        warnings = list(extracted.get("warnings") or [])

        return SmilesImageExtractionResponse(
            smiles=str(extracted.get("smiles") or ""),
            canonical_smiles=str(extracted.get("canonical_smiles") or ""),
            confidence=extracted.get("confidence"),
            warnings=warnings,
            error_code=None,
            message=f"SMILES extracted successfully in {elapsed_ms:.0f} ms.",
        )
    finally:
        await file.close()


@app.post("/smiles/preview", response_model=SmilesPreviewResponse)
async def preview_smiles(req: SmilesPreviewRequest):
    smiles = (req.smiles or "").strip()
    if not smiles:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_smiles",
                "message": "SMILES input is empty.",
            },
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _invalid_smiles_error(smiles)

    canonical = Chem.MolToSmiles(mol)
    size = (int(req.width), int(req.height))
    depiction_b64 = _render_molecule_png(mol, size=size)

    if not depiction_b64:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "render_failed",
                "message": "Failed to render molecule depiction.",
            },
        )

    return SmilesPreviewResponse(
        input_smiles=smiles,
        canonical_smiles=canonical,
        molecule_png_base64=depiction_b64,
        error_code=None,
        message="OK",
    )

# Single Prediction
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict clinical toxicity for a single SMILES molecule"""
    await _ensure_models_loaded()
    inference_backend = _resolve_inference_backend(req.inference_backend)
    
    smiles = req.smiles.strip()

    # Validate and canonicalize SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _invalid_smiles_error(smiles)
    canonical = Chem.MolToSmiles(mol) if mol else None

    try:
        async with _model_lock():
            if _is_pretrained_backend(inference_backend):
                bundle = _load_pretrained_bundle_sync(inference_backend)
                outputs = predict_pretrained_dual_head_outputs(
                    smiles_list=[smiles],
                    model=bundle["model"],
                    tokenizer=bundle["tokenizer"],
                    task_names=list(bundle.get("task_names") or []),
                    device=DEVICE,
                    clinical_threshold=float(req.threshold),
                    mechanism_threshold=float(req.threshold),
                    task_thresholds=bundle.get("tox21_thresholds"),
                    max_length=int(bundle.get("max_length", 128)),
                    source_name=_pretrained_backend_source_name(inference_backend),
                )
                clinical_raw = dict(outputs[0].get("clinical") or {})
                p_toxic = float(clinical_raw.get("p_toxic", 0.0))
                label = str(clinical_raw.get("label", "UNKNOWN"))
                confidence = float(clinical_raw.get("confidence", 0.0))
            else:
                tox21_available = TOX21_ENABLED and _tox21_ready()
                clinical_model_available = CLINTOX_ENABLED and _xsmiles_ready()
                clinical_head_available = _clinical_head_ready()

                clinical_source = _resolve_clinical_signal_source(
                    clinical_model_available=clinical_model_available,
                    clinical_head_available=clinical_head_available,
                    tox21_available=tox21_available,
                )
                if clinical_source == "unavailable":
                    if CLINICAL_SIGNAL_STRATEGY == "clinical_head" and tox21_available and not clinical_head_available:
                        raise _feature_not_ready_error("clinical_head")
                    if TOX21_ENABLED and not tox21_available:
                        raise _feature_not_ready_error("tox21")
                    raise _feature_not_ready_error("xsmiles")

                if clinical_source == "xsmiles":
                    results_df = predict_batch(
                        smiles_list=[smiles],
                        tokenizer=model_state["tokenizer"],
                        wrapped_model=model_state["wrapped"],
                        device=DEVICE,
                        threshold=req.threshold,
                        enforce_workspace_mode=False,
                    )

                    row = results_df.iloc[0]
                    p_toxic = float(row["P(toxic)"]) if row["P(toxic)"] is not None else 0.0
                    predicted = str(row["Predicted"])

                    # Map predicted label to uppercase for consistency
                    label_map = {"Toxic": "TOXIC", "Non-toxic": "NON_TOXIC", "Parse error": "PARSE_ERROR"}
                    label = label_map.get(predicted, "UNKNOWN")

                    # Confidence: distance from threshold
                    confidence = abs(p_toxic - req.threshold) / max(req.threshold, 1 - req.threshold)
                elif clinical_source == "clinical_head":
                    mechanism_raw = predict_toxicity_mechanism(
                        smiles=smiles,
                        model=model_state["tox21_model"],
                        task_names=model_state["tox21_tasks"],
                        device=DEVICE,
                        threshold=float(req.threshold),
                        task_thresholds=model_state.get("tox21_thresholds"),
                        batch_size=64,
                    )
                    clinical_head_meta = dict(model_state.get("clinical_head_meta") or {})
                    clinical_task_names = list(clinical_head_meta.get("task_names") or model_state["tox21_tasks"])
                    clinical_raw = predict_clinical_head_from_tox21_task_scores(
                        task_scores=dict(mechanism_raw.get("task_scores") or {}),
                        task_names=clinical_task_names,
                        clinical_head_model=model_state["clinical_head_model"],
                        threshold=float(req.threshold),
                        device=DEVICE,
                        smiles=smiles,
                        feature_spec=dict(clinical_head_meta.get("feature_spec") or {}),
                    )
                    p_toxic = float(clinical_raw["p_toxic"])
                    label = str(clinical_raw["label"])
                    confidence = float(clinical_raw["confidence"])
                else:
                    mechanism_raw = predict_toxicity_mechanism(
                        smiles=smiles,
                        model=model_state["tox21_model"],
                        task_names=model_state["tox21_tasks"],
                        device=DEVICE,
                        threshold=float(req.threshold),
                        task_thresholds=model_state.get("tox21_thresholds"),
                        batch_size=64,
                    )
                    clinical_raw = predict_clinical_proxy_from_tox21(
                        mechanism_result=mechanism_raw,
                        threshold=float(req.threshold),
                    )
                    p_toxic = float(clinical_raw["p_toxic"])
                    label = str(clinical_raw["label"])
                    confidence = float(clinical_raw["confidence"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

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
    await _ensure_models_loaded()
    inference_backend = _resolve_inference_backend(req.inference_backend)
    
    if len(req.smiles_list) > 500:
        raise HTTPException(400, "Batch size limited to 500 molecules")
    
    results = []
    try:
        async with _model_lock():
            if _is_pretrained_backend(inference_backend):
                bundle = _load_pretrained_bundle_sync(inference_backend)
                outputs = predict_pretrained_dual_head_outputs(
                    smiles_list=list(req.smiles_list),
                    model=bundle["model"],
                    tokenizer=bundle["tokenizer"],
                    task_names=list(bundle.get("task_names") or []),
                    device=DEVICE,
                    clinical_threshold=float(req.threshold),
                    mechanism_threshold=float(req.threshold),
                    task_thresholds=bundle.get("tox21_thresholds"),
                    max_length=int(bundle.get("max_length", 128)),
                    batch_size=64,
                    source_name=_pretrained_backend_source_name(inference_backend),
                )

                for item in outputs:
                    clinical_raw = dict(item.get("clinical") or {})
                    results.append(
                        PredictResponse(
                            smiles=str(item.get("smiles", "")),
                            canonical_smiles=item.get("canonical_smiles"),
                            p_toxic=float(clinical_raw.get("p_toxic", 0.0)),
                            label=str(clinical_raw.get("label", "UNKNOWN")),
                            confidence=float(clinical_raw.get("confidence", 0.0)),
                            threshold_used=req.threshold,
                        )
                    )
            else:
                tox21_available = TOX21_ENABLED and _tox21_ready()
                clinical_model_available = CLINTOX_ENABLED and _xsmiles_ready()
                clinical_head_available = _clinical_head_ready()

                clinical_source = _resolve_clinical_signal_source(
                    clinical_model_available=clinical_model_available,
                    clinical_head_available=clinical_head_available,
                    tox21_available=tox21_available,
                )
                if clinical_source == "unavailable":
                    if CLINICAL_SIGNAL_STRATEGY == "clinical_head" and tox21_available and not clinical_head_available:
                        raise _feature_not_ready_error("clinical_head")
                    if TOX21_ENABLED and not tox21_available:
                        raise _feature_not_ready_error("tox21")
                    raise _feature_not_ready_error("xsmiles")

                if clinical_source == "xsmiles":
                    results_df = predict_batch(
                        smiles_list=req.smiles_list,
                        tokenizer=model_state["tokenizer"],
                        wrapped_model=model_state["wrapped"],
                        device=DEVICE,
                        threshold=req.threshold,
                        enforce_workspace_mode=False,
                    )

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
                elif clinical_source in {"clinical_head", "tox21_proxy"}:
                    for raw_smiles in req.smiles_list:
                        smiles = str(raw_smiles).strip()
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            results.append(
                                PredictResponse(
                                    smiles=smiles,
                                    canonical_smiles=None,
                                    p_toxic=0.0,
                                    label="PARSE_ERROR",
                                    confidence=0.0,
                                    threshold_used=req.threshold,
                                )
                            )
                            continue

                        mechanism_raw = predict_toxicity_mechanism(
                            smiles=smiles,
                            model=model_state["tox21_model"],
                            task_names=model_state["tox21_tasks"],
                            device=DEVICE,
                            threshold=float(req.threshold),
                            task_thresholds=model_state.get("tox21_thresholds"),
                            batch_size=64,
                        )

                        if clinical_source == "clinical_head":
                            clinical_head_meta = dict(model_state.get("clinical_head_meta") or {})
                            clinical_task_names = list(clinical_head_meta.get("task_names") or model_state["tox21_tasks"])
                            clinical_raw = predict_clinical_head_from_tox21_task_scores(
                                task_scores=dict(mechanism_raw.get("task_scores") or {}),
                                task_names=clinical_task_names,
                                clinical_head_model=model_state["clinical_head_model"],
                                threshold=float(req.threshold),
                                device=DEVICE,
                                smiles=smiles,
                                feature_spec=dict(clinical_head_meta.get("feature_spec") or {}),
                            )
                        else:
                            clinical_raw = predict_clinical_proxy_from_tox21(
                                mechanism_result=mechanism_raw,
                                threshold=float(req.threshold),
                            )

                        results.append(
                            PredictResponse(
                                smiles=smiles,
                                canonical_smiles=Chem.MolToSmiles(mol),
                                p_toxic=float(clinical_raw["p_toxic"]),
                                label=str(clinical_raw["label"]),
                                confidence=float(clinical_raw["confidence"]),
                                threshold_used=req.threshold,
                            )
                        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Batch inference error: {str(e)}")

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
    await _ensure_models_loaded()
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


def _analyze_request_sync(req: AnalyzeRequest) -> AnalyzeResponse:
    """Synchronous core analyze implementation used by both API and in-process tools."""
    _ensure_models_loaded_sync()
    inference_backend = _resolve_inference_backend(req.inference_backend)
    binary_tox_model_key = _resolve_binary_tox_model_key(req.binary_tox_model)
    tox_type_model_key = _resolve_tox_type_model_key(req.tox_type_model)

    tox21_available = TOX21_ENABLED and _tox21_ready()
    clinical_model_available = CLINTOX_ENABLED and _xsmiles_ready()
    clinical_head_available = _clinical_head_ready()
    pretrained_backend_loaded = False
    clinical_source = _pretrained_model_source_name(binary_tox_model_key)
    tox_type_model_loaded_for_context = False
    selected_explainer_model_key = _resolve_explainer_engine_model_key(tox_type_model_key)

    smiles = req.smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _invalid_smiles_error(smiles)
    canonical = Chem.MolToSmiles(mol)
    ood_assessment_raw = check_ood_risk(canonical)

    def _build_ranked_importance(
        atom_importance: np.ndarray,
        bond_importance: np.ndarray,
    ) -> Tuple[List[AtomImportance], List[BondImportance]]:
        top_atoms: List[AtomImportance] = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            imp = float(atom_importance[idx]) if idx < len(atom_importance) else 0.0
            top_atoms.append(
                AtomImportance(
                    atom_idx=idx,
                    element=atom.GetSymbol(),
                    importance=round(imp, 4),
                    is_in_ring=atom.IsInRing(),
                    is_aromatic=atom.GetIsAromatic(),
                )
            )
        top_atoms.sort(key=lambda x: x.importance, reverse=True)

        top_bonds: List[BondImportance] = []
        for bond in mol.GetBonds():
            bond_idx = bond.GetIdx()
            imp = float(bond_importance[bond_idx]) if bond_idx < len(bond_importance) else 0.0
            a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
            a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
            top_bonds.append(
                BondImportance(
                    bond_idx=bond_idx,
                    atom_pair=f"{a1}({bond.GetBeginAtomIdx()}) - {a2}({bond.GetEndAtomIdx()})",
                    bond_type=str(bond.GetBondTypeAsDouble()),
                    importance=round(imp, 4),
                )
            )
        top_bonds.sort(key=lambda x: x.importance, reverse=True)
        return top_atoms[:10], top_bonds[:10]

    try:
        with _model_lock_sync():
            clinical_bundle: Optional[Dict[str, Any]] = None
            mechanism_failure: Optional[Exception] = None
            try:
                if tox_type_model_key == "tox21_gatv2_model":
                    if tox21_available:
                        tox_type_model_loaded_for_context = True
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
                elif _is_pretrained_gin_model_key(tox_type_model_key):
                    gin_bundle = _load_pretrained_gin_bundle_sync(tox_type_model_key)
                    tox_type_model_loaded_for_context = True
                    mechanism_outputs = predict_pretrained_gin_tox21_outputs(
                        smiles_list=[smiles],
                        model=gin_bundle["model"],
                        task_names=list(gin_bundle.get("task_names") or []),
                        device=DEVICE,
                        mechanism_threshold=float(req.mechanism_threshold),
                        task_thresholds=gin_bundle.get("tox21_thresholds"),
                        batch_size=32,
                        source_name=_pretrained_gin_model_source_name(tox_type_model_key),
                    )
                    mechanism_raw = dict(mechanism_outputs[0].get("mechanism") or {})
                elif _is_ensemble_tox_type_model_key(tox_type_model_key):
                    mechanism_raw = _predict_ensemble_mechanism_sync(
                        model_key=tox_type_model_key,
                        smiles=smiles,
                        mechanism_threshold=float(req.mechanism_threshold),
                    )
                    tox_type_model_loaded_for_context = True
                else:
                    mechanism_bundle = (
                        clinical_bundle
                        if tox_type_model_key == binary_tox_model_key
                        else _load_dual_head_bundle_sync(tox_type_model_key)
                    )
                    tox_type_model_loaded_for_context = True
                    mechanism_outputs = predict_pretrained_dual_head_outputs(
                        smiles_list=[smiles],
                        model=mechanism_bundle["model"],
                        tokenizer=mechanism_bundle["tokenizer"],
                        task_names=list(mechanism_bundle.get("task_names") or []),
                        device=DEVICE,
                        clinical_threshold=float(req.clinical_threshold),
                        mechanism_threshold=float(req.mechanism_threshold),
                        task_thresholds=mechanism_bundle.get("tox21_thresholds"),
                        max_length=int(mechanism_bundle.get("max_length", 128)),
                        source_name=_pretrained_model_source_name(tox_type_model_key),
                    )
                    mechanism_raw = dict(mechanism_outputs[0].get("mechanism") or {})
            except HTTPException as exc:
                mechanism_failure = exc
            except Exception as exc:
                mechanism_failure = exc

            if mechanism_failure is not None:
                if tox21_available:
                    logger.warning(
                        "Requested tox_type_model '%s' unavailable; falling back to tox21_gatv2: %s",
                        tox_type_model_key,
                        mechanism_failure,
                    )
                    mechanism_raw = predict_toxicity_mechanism(
                        smiles=smiles,
                        model=model_state["tox21_model"],
                        task_names=model_state["tox21_tasks"],
                        device=DEVICE,
                        threshold=req.mechanism_threshold,
                        task_thresholds=model_state.get("tox21_thresholds"),
                        batch_size=64,
                    )
                    tox_type_model_loaded_for_context = True
                elif isinstance(mechanism_failure, HTTPException):
                    raise mechanism_failure
                else:
                    raise HTTPException(500, f"Unified inference error: {str(mechanism_failure)}")

            clinical_failure: Optional[Exception] = None
            try:
                if _is_ensemble_binary_model_key(binary_tox_model_key):
                    clinical_raw = _predict_ensemble_clinical_sync(
                        model_key=binary_tox_model_key,
                        smiles=smiles,
                        clinical_threshold=float(req.clinical_threshold),
                        mechanism_threshold=float(req.mechanism_threshold),
                    )
                    pretrained_backend_loaded = True
                else:
                    clinical_bundle = _load_dual_head_bundle_sync(binary_tox_model_key)
                    pretrained_backend_loaded = True

                    clinical_outputs = predict_pretrained_dual_head_outputs(
                        smiles_list=[smiles],
                        model=clinical_bundle["model"],
                        tokenizer=clinical_bundle["tokenizer"],
                        task_names=list(clinical_bundle.get("task_names") or []),
                        device=DEVICE,
                        clinical_threshold=float(req.clinical_threshold),
                        mechanism_threshold=float(req.mechanism_threshold),
                        task_thresholds=clinical_bundle.get("tox21_thresholds"),
                        max_length=int(clinical_bundle.get("max_length", 128)),
                        source_name=_pretrained_model_source_name(binary_tox_model_key),
                    )
                    clinical_raw = dict(clinical_outputs[0].get("clinical") or {})
                    clinical_raw["source"] = _pretrained_model_source_name(binary_tox_model_key)
            except HTTPException as exc:
                clinical_failure = exc
            except Exception as exc:
                clinical_failure = exc

            if clinical_failure is not None:
                if tox21_available:
                    logger.warning(
                        "Requested binary_tox_model '%s' unavailable; using tox21 proxy clinical signal: %s",
                        binary_tox_model_key,
                        clinical_failure,
                    )
                    clinical_raw = predict_clinical_proxy_from_tox21(
                        mechanism_result=mechanism_raw,
                        threshold=float(req.clinical_threshold),
                    )
                    clinical_raw["source"] = "tox21_proxy"
                    clinical_raw["fallback_reason"] = f"{type(clinical_failure).__name__}: {clinical_failure}"
                    pretrained_backend_loaded = False
                elif isinstance(clinical_failure, HTTPException):
                    raise clinical_failure
                else:
                    raise HTTPException(500, f"Unified inference error: {str(clinical_failure)}")

            mechanism_raw.setdefault("task_scores", {})
            mechanism_raw.setdefault("active_tasks", [])
            mechanism_raw.setdefault("highest_risk_task", "—")
            mechanism_raw.setdefault("highest_risk_score", 0.0)
            mechanism_raw.setdefault("assay_hits", int(len(mechanism_raw.get("active_tasks") or [])))
            mechanism_raw.setdefault("threshold_used", float(req.mechanism_threshold))
            mechanism_raw.setdefault("task_thresholds", {})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Unified inference error: {str(exc)}")

    clinical_signal_source = str(clinical_raw.get("source") or "xsmiles")

    final_verdict = aggregate_toxicity_verdict(
        clinical_is_toxic=bool(clinical_raw["is_toxic"]),
        assay_hits=int(mechanism_raw["assay_hits"]),
    )

    explanation_payload = None
    should_explain = True
    if req.explain_only_if_alert and int(mechanism_raw["assay_hits"]) <= 0 and req.target_task is None:
        should_explain = False

    if should_explain:
        if clinical_signal_source == "xsmiles" and clinical_model_available:
            try:
                clinical_target_class = 1 if bool(clinical_raw["is_toxic"]) else 0
                clinical_pyg_data = smiles_to_pyg_data(smiles, label=clinical_target_class)
                if clinical_pyg_data is None:
                    raise ValueError(f"Unable to featurize SMILES for explanation: {smiles}")

                with _model_lock_sync():
                    explain_result = _run_with_timeout_sync(
                        explain_molecule,
                        req.explainer_timeout_ms,
                        smiles,
                        model_state["model"],
                        model_state["tokenizer"],
                        clinical_pyg_data,
                        "cpu",
                        req.explainer_epochs,
                        clinical_target_class,
                    )

                heatmap_b64 = _render_explanation_heatmap(explain_result)
                atom_importance = explain_result["atom_importance"]
                bond_importance = explain_result["bond_importance"]
                top_atoms, top_bonds = _build_ranked_importance(
                    atom_importance=atom_importance,
                    bond_importance=bond_importance,
                )

                explanation_payload = ToxicityExplanationOutput(
                    target_task="CLINICAL_TOXICITY",
                    target_task_score=float(clinical_raw["p_toxic"]),
                    top_atoms=top_atoms,
                    top_bonds=top_bonds,
                    heatmap_base64=heatmap_b64,
                    molecule_png_base64=_render_molecule_png(mol),
                    explainer_note=(
                        "Clinical GNNExplainer heatmap for SMILESGNN graph pathway; "
                        "red regions contribute more to the predicted clinical toxicity probability."
                    ),
                )
            except TimeoutError:
                logger.warning(
                    "Clinical explanation timed out after %s ms; returning fallback explanation payload.",
                    req.explainer_timeout_ms,
                )
                explanation_payload = _fallback_explanation(
                    target_task="CLINICAL_TOXICITY",
                    mol=mol,
                )
            except Exception as clinical_exc:
                logger.warning(
                    "Clinical explanation failed; falling back to tox21 task explainer: %s",
                    clinical_exc,
                )

        if explanation_payload is None:
            if selected_explainer_model_key == TOX21_PRETRAINED_GIN_MODEL_KEY:
                engine_tasks = [
                    "NR-AR",
                    "NR-AR-LBD",
                    "NR-AhR",
                    "NR-Aromatase",
                    "NR-ER",
                    "NR-ER-LBD",
                    "NR-PPAR-gamma",
                    "SR-ARE",
                    "SR-ATAD5",
                    "SR-HSE",
                    "SR-MMP",
                    "SR-p53",
                ]
            else:
                engine_tasks = list(model_state.get("tox21_tasks") or [])

            target_task = req.target_task or str(mechanism_raw["highest_risk_task"])
            if target_task not in engine_tasks:
                target_task = engine_tasks[0] if engine_tasks else ""

            if target_task:
                try:
                    explain_result = None
                    use_fast_explainer = int(req.explainer_epochs) <= FAST_EXPLAINER_EPOCH_CUTOFF

                    with _model_lock_sync():
                        if selected_explainer_model_key == "tox21_gatv2_model":
                            if not tox21_available:
                                explanation_payload = _fallback_explanation(target_task=target_task, mol=mol)
                            elif use_fast_explainer:
                                explain_result = _run_with_timeout_sync(
                                    explain_tox21_task_gradient,
                                    req.explainer_timeout_ms,
                                    smiles,
                                    model_state["tox21_model"],
                                    model_state["tox21_tasks"],
                                    target_task,
                                    "cpu",
                                    req.mechanism_threshold,
                                )
                            else:
                                explain_result = _run_with_timeout_sync(
                                    explain_tox21_task,
                                    req.explainer_timeout_ms,
                                    smiles,
                                    model_state["tox21_model"],
                                    model_state["tox21_tasks"],
                                    target_task,
                                    "cpu",
                                    req.explainer_epochs,
                                    req.mechanism_threshold,
                                )
                        elif selected_explainer_model_key == TOX21_PRETRAINED_GIN_MODEL_KEY:
                            gin_bundle = _load_pretrained_gin_bundle_sync(TOX21_PRETRAINED_GIN_MODEL_KEY)
                            explain_result = _run_with_timeout_sync(
                                explain_tox21_task_pretrained_gin,
                                req.explainer_timeout_ms,
                                smiles,
                                gin_bundle["model"],
                                list(gin_bundle.get("task_names") or []),
                                target_task,
                                "cpu",
                                req.explainer_epochs,
                                req.mechanism_threshold,
                            )
                        elif _is_dual_head_model_key(selected_explainer_model_key):
                            dual_bundle = _load_dual_head_bundle_sync(selected_explainer_model_key)
                            explain_result = _run_with_timeout_sync(
                                explain_tox21_task_pretrained_dual_head_gradient,
                                req.explainer_timeout_ms,
                                smiles,
                                dual_bundle["model"],
                                dual_bundle["tokenizer"],
                                list(dual_bundle.get("task_names") or []),
                                target_task,
                                "cpu",
                                int(dual_bundle.get("max_length", 128)),
                                req.mechanism_threshold,
                            )
                        else:
                            explanation_payload = _fallback_explanation(target_task=target_task, mol=mol)

                    if explanation_payload is None and explain_result is not None:
                        heatmap_b64 = _render_explanation_heatmap(explain_result)

                        atom_importance = explain_result["atom_importance"]
                        bond_importance = explain_result["bond_importance"]
                        top_atoms, top_bonds = _build_ranked_importance(
                            atom_importance=atom_importance,
                            bond_importance=bond_importance,
                        )

                        explainer_method = str(
                            explain_result.get("explainer_method") or "gnnexplainer"
                        )
                        if selected_explainer_model_key == TOX21_PRETRAINED_GIN_MODEL_KEY:
                            explainer_note = (
                                "Task-level GNNExplainer on Pretrained-GIN engine selected from tox_type_model; "
                                "for ensemble mode this uses the designated best-member explainer."
                            )
                        elif _is_dual_head_model_key(selected_explainer_model_key):
                            explainer_note = (
                                "Task-level dual-head gradient saliency aligned with selected tox_type_model "
                                "(transformer backbone)."
                            )
                        elif explainer_method == "gradient_saliency":
                            explainer_note = (
                                "Fast gradient-saliency explainer for Tox21 task; "
                                "selected automatically because explainer_epochs is below the stability cutoff."
                            )
                        else:
                            explainer_note = (
                                "Task-level GNNExplainer aligned with selected tox_type_model routing."
                            )

                        explanation_payload = ToxicityExplanationOutput(
                            target_task=target_task,
                            target_task_score=float(explain_result["prediction_prob"]),
                            top_atoms=top_atoms,
                            top_bonds=top_bonds,
                            heatmap_base64=heatmap_b64,
                            molecule_png_base64=_render_molecule_png(mol),
                            explainer_note=explainer_note,
                        )
                except TimeoutError:
                    logger.warning(
                        "Task-level explanation timed out after %s ms for task '%s'; returning fallback explanation payload.",
                        req.explainer_timeout_ms,
                        target_task,
                    )
                    explanation_payload = _fallback_explanation(target_task=target_task, mol=mol)
                except Exception as exc:
                    logger.warning(
                        "Task-level explanation failed for task '%s'; returning fallback explanation payload: %s",
                        target_task,
                        exc,
                    )
                    explanation_payload = _fallback_explanation(target_task=target_task, mol=mol)

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
    elif clinical_signal_source.startswith("pretrained_dual_head"):
        explainer_route_note = (
            f"explainer_engine={selected_explainer_model_key}"
            if selected_explainer_model_key == tox_type_model_key
            else (
                "explainer_engine="
                f"{selected_explainer_model_key} (fallback from tox_type={tox_type_model_key})"
            )
        )
        reliability_warning = (
            "Clinical and mechanism outputs use selected model keys "
            f"(binary={binary_tox_model_key}, tox_type={tox_type_model_key}); "
            f"structural explanation routed by selected tox-type model ({explainer_route_note})."
        )
    elif clinical_signal_source == "clinical_head":
        reliability_warning = (
            "Clinical verdict is generated by lightweight transfer head "
            "on top of Tox21 task probabilities."
        )
    elif clinical_signal_source == "tox21_proxy":
        reliability_warning = (
            "Clinical verdict is derived from weighted Tox21 proxy tasks "
            "(SR-p53, SR-MMP, SR-ARE, NR-AhR, SR-HSE)."
        )
    elif not tox21_available:
        reliability_warning = "Mechanism model is unavailable; only clinical signal is used."

    clinical_reference_metrics = {}

    if clinical_signal_source == "tox21_proxy":
        clinical_reference_metrics["clinical_proxy_coverage"] = float(
            clinical_raw.get("proxy_coverage", 0.0)
        )

    inference_backend_loaded = bool(pretrained_backend_loaded)
    clinical_loaded_for_context = bool(pretrained_backend_loaded)

    inference_context = InferenceContextOutput(
        workspace_mode=WORKSPACE_MODE_NAME,
        inference_backend=f"{inference_backend}|binary={binary_tox_model_key}|tox_type={tox_type_model_key}",
        inference_backend_loaded=inference_backend_loaded,
        threshold_policy=str(WORKSPACE_MODE.get("threshold_policy", "balanced")),
        clinical_threshold_applied=float(req.clinical_threshold),
        clinical_model_loaded=clinical_loaded_for_context,
        tox21_model_loaded=tox_type_model_loaded_for_context,
        explainer_used=bool(should_explain),
        explanation_available=explanation_payload is not None,
        tox21_threshold_source=model_state.get("tox21_thresholds_source"),
        clinical_reference_metrics=clinical_reference_metrics,
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


def analyze_molecule_sync(
    smiles: str,
    clinical_threshold: float = 0.35,
    mechanism_threshold: float = 0.5,
    inference_backend: str = "xsmiles",
    binary_tox_model: str = DEFAULT_BINARY_TOX_MODEL_KEY,
    tox_type_model: str = DEFAULT_TOX_TYPE_MODEL_KEY,
    explain_only_if_alert: bool = False,
    explainer_epochs: int = 200,
    explainer_timeout_ms: int = 30000,
    target_task: Optional[str] = None,
) -> Dict[str, Any]:
    """Sync in-process entrypoint used by tool calls to avoid asyncio event-loop crossovers."""
    try:
        req = AnalyzeRequest(
            smiles=smiles,
            clinical_threshold=float(clinical_threshold),
            mechanism_threshold=float(mechanism_threshold),
            inference_backend=str(inference_backend),
            binary_tox_model=str(binary_tox_model),
            tox_type_model=str(tox_type_model),
            explain_only_if_alert=bool(explain_only_if_alert),
            explainer_epochs=int(explainer_epochs),
            explainer_timeout_ms=int(explainer_timeout_ms),
            target_task=target_task,
            return_all_scores=True,
        )
        response = _analyze_request_sync(req)
        return response.model_dump()
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            err = detail.get("error") or detail.get("message") or str(detail)
        else:
            err = str(detail)
        return {
            "error": str(err),
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }


def _deterministic_screening_payload(
    smiles: str,
    language: str,
    clinical_threshold: float,
    mechanism_threshold: float,
    inference_backend: str,
    binary_tox_model: str,
    tox_type_model: str,
    molrag_enabled: bool = False,
    molrag_top_k: int = 5,
    molrag_min_similarity: float = 0.15,
) -> Dict[str, Any]:
    """Build screening payload directly from deterministic ScreeningAgent flow."""
    def _ensure_structural_explanation(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return payload

        explanation_candidate = payload.get("explanation")
        if isinstance(explanation_candidate, dict) and _screening_payload_has_structural_data({"explanation": explanation_candidate}):
            return payload

        canonical_smiles = str(payload.get("canonical_smiles") or payload.get("smiles") or smiles).strip()
        mol = Chem.MolFromSmiles(canonical_smiles) if canonical_smiles else None
        mechanism_payload = payload.get("mechanism") if isinstance(payload.get("mechanism"), dict) else {}
        target_task = str(mechanism_payload.get("highest_risk_task") or "").strip() or None

        fallback_explanation = _fallback_explanation(target_task=target_task, mol=mol).model_dump()
        merged_payload = dict(payload)
        merged_payload["explanation"] = fallback_explanation
        return merged_payload

    screening_wrapped = run_screening(
        smiles_input=smiles,
        language=normalize_language(language),
        clinical_threshold=float(clinical_threshold),
        mechanism_threshold=float(mechanism_threshold),
        inference_backend=str(inference_backend),
        binary_tox_model=str(binary_tox_model),
        tox_type_model=str(tox_type_model),
        molrag_enabled=bool(molrag_enabled),
        molrag_top_k=int(molrag_top_k),
        molrag_min_similarity=float(molrag_min_similarity),
    )

    if isinstance(screening_wrapped, dict) and not screening_wrapped.get("screening_error"):
        screening_payload = screening_wrapped.get("screening_result")
        if isinstance(screening_payload, dict) and screening_payload:
            return _ensure_structural_explanation(screening_payload)

    # Defensive fallback: keep a minimal screening payload if deterministic
    # ScreeningAgent flow returns malformed output.
    analysis = analyze_molecule_sync(
        smiles=smiles,
        clinical_threshold=float(clinical_threshold),
        mechanism_threshold=float(mechanism_threshold),
        inference_backend=str(inference_backend),
        binary_tox_model=str(binary_tox_model),
        tox_type_model=str(tox_type_model),
        explain_only_if_alert=False,
    )
    if not isinstance(analysis, dict) or analysis.get("error"):
        return {}

    fallback_payload = {
        "summary": None,
        "smiles": analysis.get("smiles", smiles),
        "canonical_smiles": analysis.get("canonical_smiles", smiles),
        "clinical": analysis.get("clinical") or {},
        "mechanism": analysis.get("mechanism") or {},
        "explanation": analysis.get("explanation") or {},
        "ood_assessment": analysis.get("ood_assessment") or {},
        "reliability_warning": analysis.get("reliability_warning"),
        "inference_context": analysis.get("inference_context") or {},
        "final_verdict": analysis.get("final_verdict", "UNKNOWN"),
        "error": None,
    }
    return _ensure_structural_explanation(fallback_payload)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Unified production endpoint with three outputs for one SMILES:
      1) Toxic / Non-toxic from XSmiles (clinical)
      2) Type of toxicity from GATv2 (Tox21 tasks)
      3) Toxicity explainer from GNNExplainer (task-specific)
    """
    await _ensure_models_loaded()
    return await asyncio.to_thread(_analyze_request_sync, req)


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

    resolved_inference_backend = _resolve_inference_backend(req.inference_backend)
    resolved_binary_tox_model = _resolve_binary_tox_model_key(req.binary_tox_model)
    resolved_tox_type_model = _resolve_tox_type_model_key(req.tox_type_model)
    expected_inference_signature = _compose_inference_signature(
        resolved_inference_backend,
        resolved_binary_tox_model,
        resolved_tox_type_model,
    )

    recovery_notes: List[str] = []

    def _note_recovery(note: str) -> None:
        if note and note not in recovery_notes:
            recovery_notes.append(note)

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
            resolved_inference_backend,
            resolved_binary_tox_model,
            resolved_tox_type_model,
            bool(req.molrag_enabled),
            int(req.molrag_top_k),
            float(req.molrag_min_similarity),
        )

        final_report_payload = _coerce_json_dict(
            fallback_state.get("final_report"),
            nested_key="final_report",
        ) or {}
        validation_status_payload = fallback_state.get("validation_status")
        summary = final_report_payload.get("executive_summary") if isinstance(final_report_payload, dict) else None
        research_payload = _coerce_json_dict(
            fallback_state.get("research_result"),
            nested_key="research_result",
        ) or {}
        evidence_qa_result_payload = _build_evidence_qa_result(
            research_payload,
            fallback_state if isinstance(fallback_state, dict) else None,
        )
        chat_session_id = _upsert_report_chat_session(
            analysis_session_id=session_id,
            smiles=smiles,
            final_report=final_report_payload,
            research_payload=research_payload,
            state=fallback_state if isinstance(fallback_state, dict) else None,
            evidence_qa_result=evidence_qa_result_payload,
        )

        fallback_events: List[AgentEventRecord] = []
        if req.include_agent_events:
            fallback_preview = f"ADK fallback activated: {cause}"
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
            chat_session_id=chat_session_id,
            adk_available=bool(ADK_AVAILABLE and adk_runner is not None and adk_session_service is not None),
            runtime_mode="deterministic_fallback",
            runtime_note=cause,
            validation_status=validation_status_payload,
            final_report=final_report_payload,
            evidence_qa_result=evidence_qa_result_payload,
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
        "inference_backend": resolved_inference_backend,
        "binary_tox_model": resolved_binary_tox_model,
        "tox_type_model": resolved_tox_type_model,
        "molrag_enabled": bool(req.molrag_enabled),
        "molrag_top_k": int(req.molrag_top_k),
        "molrag_min_similarity": float(req.molrag_min_similarity),
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
    force_rebuild_report_from_screening = False

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
                    _note_recovery("adk_stream_close_race")
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
            _note_recovery("adk_stream_close_race")
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
                        _note_recovery("adk_stream_close_race")
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
                        elif _is_adk_taskgroup_runtime_error(retry_exc):
                            logger.warning(
                                "ADK root global retry ended with TaskGroup runtime; forcing ADK step continuation: %s",
                                retry_exc,
                            )
                            force_adk_continuation = True
                            _note_recovery("adk_taskgroup_step_continuation")
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
                _note_recovery("adk_forced_step_continuation")
        elif _is_adk_taskgroup_runtime_error(exc):
            logger.warning(
                "ADK root raised TaskGroup runtime error; forcing ADK step continuation path: %s",
                exc,
            )
            force_adk_continuation = True
            _note_recovery("adk_taskgroup_step_continuation")
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

    explanation_raw_payload = _extract_state_payload(state, "explanation_raw")
    if not explanation_raw_payload:
        explanation_raw_payload = _extract_explanation_payload(screening_payload)
    if screening_payload and explanation_raw_payload:
        screening_payload = _merge_explanation_into_screening(screening_payload, explanation_raw_payload)

    if not screening_payload:
        logger.warning("ADK screening payload missing from session state; hydrating via deterministic screening")
        try:
            hydrated = await asyncio.to_thread(
                _deterministic_screening_payload,
                smiles,
                language,
                float(req.clinical_threshold),
                float(req.mechanism_threshold),
                resolved_inference_backend,
                resolved_binary_tox_model,
                resolved_tox_type_model,
                bool(req.molrag_enabled),
                int(req.molrag_top_k),
                float(req.molrag_min_similarity),
            )
        except Exception as exc:
            logger.warning("Deterministic screening hydration failed (missing payload): %s", exc)
            hydrated = None

        if isinstance(hydrated, dict) and hydrated:
            screening_payload = hydrated
            explanation_raw_payload = _extract_explanation_payload(hydrated)
            _note_recovery("screening_hydrated_missing")

    if screening_payload and not _screening_payload_has_structural_data(screening_payload):
        logger.warning("ADK screening payload missing structural explanation; hydrating via deterministic screening")
        try:
            hydrated = await asyncio.to_thread(
                _deterministic_screening_payload,
                smiles,
                language,
                float(req.clinical_threshold),
                float(req.mechanism_threshold),
                resolved_inference_backend,
                resolved_binary_tox_model,
                resolved_tox_type_model,
                bool(req.molrag_enabled),
                int(req.molrag_top_k),
                float(req.molrag_min_similarity),
            )
        except Exception as exc:
            logger.warning("Deterministic screening hydration failed: %s", exc)
            hydrated = None

        if isinstance(hydrated, dict) and hydrated:
            screening_payload = hydrated
            explanation_raw_payload = _extract_explanation_payload(hydrated)
            _note_recovery("screening_hydrated_no_structural")

    if screening_payload and explanation_raw_payload:
        screening_payload = _merge_explanation_into_screening(screening_payload, explanation_raw_payload)

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
        _note_recovery("adk_step_continuation")
        await _run_adk_agent_step(
            agent=screening_agent,
            user_id=user_id,
            session_id=session_id,
            message=message,
            include_events=req.include_agent_events,
            event_sink=events if req.include_agent_events else None,
        )
        await _run_adk_agent_step(
            agent=researcher_agent,
            user_id=user_id,
            session_id=session_id,
            message=message,
            include_events=req.include_agent_events,
            event_sink=events if req.include_agent_events else None,
        )
        await _run_adk_agent_step(
            agent=writer_agent,
            user_id=user_id,
            session_id=session_id,
            message=message,
            include_events=req.include_agent_events,
            event_sink=events if req.include_agent_events else None,
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

        if not explanation_raw_payload:
            explanation_raw_payload = _extract_state_payload(state, "explanation_raw")
        if not explanation_raw_payload:
            explanation_raw_payload = _extract_explanation_payload(screening_payload)
        if screening_payload and explanation_raw_payload:
            screening_payload = _merge_explanation_into_screening(screening_payload, explanation_raw_payload)

    if bool(req.molrag_enabled) and not _screening_payload_has_molrag_data(screening_payload):
        logger.warning("MolRAG requested but missing from screening payload; hydrating deterministic screening")
        try:
            hydrated = await asyncio.to_thread(
                _deterministic_screening_payload,
                smiles,
                language,
                float(req.clinical_threshold),
                float(req.mechanism_threshold),
                resolved_inference_backend,
                resolved_binary_tox_model,
                resolved_tox_type_model,
                True,
                int(req.molrag_top_k),
                float(req.molrag_min_similarity),
            )
        except Exception as exc:
            logger.warning("Deterministic screening hydration failed for MolRAG recovery: %s", exc)
            hydrated = None

        if isinstance(hydrated, dict) and hydrated:
            screening_payload = hydrated
            explanation_raw_payload = _extract_explanation_payload(hydrated)
            screening_payload = _merge_explanation_into_screening(
                screening_payload,
                explanation_raw_payload,
            )
            force_rebuild_report_from_screening = True
            _note_recovery("screening_hydrated_molrag")

    if screening_payload and not _screening_payload_matches_requested_models(
        screening_payload,
        inference_backend=resolved_inference_backend,
        binary_tox_model=resolved_binary_tox_model,
        tox_type_model=resolved_tox_type_model,
    ):
        observed_signature = ""
        if isinstance(screening_payload.get("inference_context"), dict):
            observed_signature = str(
                screening_payload.get("inference_context", {}).get("inference_backend") or ""
            )

        logger.warning(
            "ADK screening model mismatch detected; expected=%s, observed=%s. Rehydrating deterministic screening.",
            expected_inference_signature,
            observed_signature,
        )

        try:
            hydrated = await asyncio.to_thread(
                _deterministic_screening_payload,
                smiles,
                language,
                float(req.clinical_threshold),
                float(req.mechanism_threshold),
                resolved_inference_backend,
                resolved_binary_tox_model,
                resolved_tox_type_model,
                bool(req.molrag_enabled),
                int(req.molrag_top_k),
                float(req.molrag_min_similarity),
            )
        except Exception as exc:
            logger.warning("Deterministic screening hydration failed after model mismatch: %s", exc)
            hydrated = None

        if isinstance(hydrated, dict) and hydrated:
            screening_payload = hydrated
            explanation_raw_payload = _extract_explanation_payload(hydrated)
            screening_payload = _merge_explanation_into_screening(
                screening_payload,
                explanation_raw_payload,
            )
            force_rebuild_report_from_screening = True
            _note_recovery("screening_hydrated_model_mismatch")
        else:
            logger.warning(
                "Unable to enforce selected model usage from ADK screening payload; switching to deterministic fallback"
            )
            return await _build_fallback_response("model_selection_enforcement_failed")

    report_schema_complete = _is_final_report_schema_complete(final_report)
    if (not final_report or not report_schema_complete) and not screening_payload:
        logger.warning(
            "Final report is incomplete and screening payload is missing; hydrating deterministic screening before rebuild"
        )
        try:
            hydrated = await asyncio.to_thread(
                _deterministic_screening_payload,
                smiles,
                language,
                float(req.clinical_threshold),
                float(req.mechanism_threshold),
                resolved_inference_backend,
                resolved_binary_tox_model,
                resolved_tox_type_model,
                bool(req.molrag_enabled),
                int(req.molrag_top_k),
                float(req.molrag_min_similarity),
            )
        except Exception as exc:
            logger.warning("Deterministic screening hydration failed during final report recovery: %s", exc)
            hydrated = None

        if isinstance(hydrated, dict) and hydrated:
            screening_payload = hydrated
            explanation_raw_payload = _extract_explanation_payload(hydrated)
            screening_payload = _merge_explanation_into_screening(
                screening_payload,
                explanation_raw_payload,
            )
            _note_recovery("screening_hydrated_report_recovery")

    if (force_rebuild_report_from_screening or not final_report or not report_schema_complete) and screening_payload:
        if force_rebuild_report_from_screening:
            logger.warning("Rebuilding final report to align with requested model selection")
            _note_recovery("final_report_rebuilt_model_alignment")
        elif not final_report:
            logger.warning("ADK writer output missing/invalid; rebuilding final report from session state")
            _note_recovery("final_report_rebuilt_missing")
        else:
            logger.warning("ADK writer report schema mismatch; rebuilding final report from session state")
            _note_recovery("final_report_rebuilt_schema")

        final_report = build_final_report(
            smiles_input=smiles,
            screening_result=screening_payload,
            research_result=research_payload,
            explanation_raw=explanation_raw_payload,
            language=language,
        )
        if final_text is None:
            rebuilt_summary = final_report.get("executive_summary")
            if isinstance(rebuilt_summary, str) and rebuilt_summary:
                final_text = rebuilt_summary

    if final_report and screening_payload and _final_report_missing_structural_images(final_report):
        logger.warning("ADK final report missing structural images; rebuilding from hydrated screening payload")
        _note_recovery("final_report_rebuilt_structural_images")
        final_report = build_final_report(
            smiles_input=smiles,
            screening_result=screening_payload,
            research_result=research_payload,
            explanation_raw=explanation_raw_payload,
            language=language,
        )
        if final_text is None:
            rebuilt_summary = final_report.get("executive_summary")
            if isinstance(rebuilt_summary, str) and rebuilt_summary:
                final_text = rebuilt_summary

    if isinstance(final_report, dict):
        summary = final_report.get("executive_summary")
        if isinstance(summary, str) and summary:
            # Always prioritize the final executive summary over early validator text.
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

    runtime_note = "adk_runtime_ok"
    if recovery_notes:
        runtime_note = f"adk_runtime_ok_with_recovery:{','.join(recovery_notes)}"
        if req.include_agent_events:
            recovery_preview = (
                f"ADK recovery ap dung: {', '.join(recovery_notes)}"
                if language == "vi"
                else f"ADK recovery applied: {', '.join(recovery_notes)}"
            )
            events.append(
                AgentEventRecord(
                    type="recovery",
                    author="System",
                    function_calls=[],
                    function_responses=[],
                    is_final=False,
                    text_preview=recovery_preview,
                )
            )

    evidence_qa_result_payload = _build_evidence_qa_result(
        research_payload if isinstance(research_payload, dict) else {},
        state if isinstance(state, dict) else None,
    )

    chat_session_id = _upsert_report_chat_session(
        analysis_session_id=session_id,
        smiles=smiles,
        final_report=final_report,
        research_payload=research_payload if isinstance(research_payload, dict) else {},
        state=state if isinstance(state, dict) else None,
        evidence_qa_result=evidence_qa_result_payload,
    )

    return AgentAnalyzeResponse(
        session_id=session_id,
        chat_session_id=chat_session_id,
        adk_available=ADK_AVAILABLE,
        runtime_mode="adk",
        runtime_note=runtime_note,
        validation_status=validation_status,
        final_report=final_report,
        evidence_qa_result=evidence_qa_result_payload,
        final_text=final_text,
        agent_events=events if req.include_agent_events else [],
        state_keys=sorted(state.keys()),
    )


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest):
    """Handle follow-up QA against a frozen per-report chat session."""
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "chat_message_empty",
                "message": "message is required for report chat.",
            },
        )

    chat_session_id = _resolve_report_chat_session_id(
        chat_session_id=req.chat_session_id,
        analysis_session_id=req.analysis_session_id,
    )
    if not chat_session_id:
        chat_session_id = _rehydrate_report_chat_session_from_payload(
            requested_chat_session_id=req.chat_session_id,
            analysis_session_id=req.analysis_session_id,
            report_state_payload=req.report_state,
        )
    if not chat_session_id:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "chat_session_not_found",
                "message": "Report chat session not found. Run /agent/analyze again to refresh session context.",
            },
        )

    llm_caller = _make_report_chat_llm_caller(chat_session_id)

    try:
        answer, session = await asyncio.to_thread(
            chat_with_report,
            chat_session_id,
            user_message,
            llm_caller,
            3,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "chat_runtime_error",
                "message": str(exc),
            },
        ) from exc

    if session is None:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "chat_session_expired",
                "message": "Chat session expired. Please rerun analysis to restart report chat.",
            },
        )

    return AgentChatResponse(chat_session_id=chat_session_id, response=answer)
