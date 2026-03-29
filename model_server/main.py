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
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import torch 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rdkit import Chem
from starlette.exceptions import HTTPException as StarletteHTTPException

# Important: Add project root to sys.path so project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import (
    aggregate_toxicity_verdict,
    load_model,
    load_tox21_gatv2_model,
    predict_batch,
    predict_clinical_toxicity,
    predict_toxicity_mechanism,
)
from backend.graph_data import smiles_to_pyg_data
from backend.gnn_explainer import explain_molecule, explain_tox21_task, visualize_explanation
from backend.workspace_mode import get_workspace_mode
from model_server.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClinicalToxicityOutput,
    MechanismToxicityOutput,
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_server")

# Lifespan: load model once at startup
model_state = {}


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


def _fallback_explanation(target_task: Optional[str] = None) -> ToxicityExplanationOutput:
    task = target_task or "UNAVAILABLE_NO_TOX21_MODEL"
    return ToxicityExplanationOutput(
        target_task=task,
        target_task_score=0.0,
        top_atoms=[],
        top_bonds=[],
        heatmap_base64=None,
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
    visualize_explanation(result, figsize=(13, 5), save_path=buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

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
    
    if len(req.smile_list) > 500:
        raise HTTPException(400, "Batch size limited to 500 molecules")
    
    try:
        async with _model_lock():
            results_df = predict_batch(
                smiles_list=req.smile_list,
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
        target_task = req.target_task or str(mechanism_raw["highest_risk_task"])
        if target_task == "—":
            tasks = model_state.get("tox21_tasks", [])
            target_task = tasks[0] if tasks else ""

        if target_task:
            if not tox21_available:
                explanation_payload = _fallback_explanation(target_task=target_task)
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
                        explainer_note="GNNExplainer is task-specific and explains one selected Tox21 head.",
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

    return AnalyzeResponse(
        smiles=smiles,
        canonical_smiles=canonical,
        clinical=clinical_output,
        mechanism=mechanism_output,
        explanation=explanation_payload,
        final_verdict=final_verdict,
    )