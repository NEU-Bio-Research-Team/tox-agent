# ToxAgent Function-Calling Flow (Current Production State)

## 1) Scope

This document describes the current production inference flow for one-input analysis in ToxAgent after:
- Tox21 GATv2 retraining with focal loss
- Per-task threshold calibration
- Unified API output contract stabilization

Primary endpoint for function-calling integration:
- `POST /analyze`


## 2) Runtime Entry and Model Loading

Server entry:
- `model_server/main.py`

At startup (`lifespan`):
1. Loads clinical branch (XSmiles/SMILESGNN):
   - model from `models/smilesgnn_model/best_model.pt`
   - tokenizer from `models/smilesgnn_model/tokenizer.pkl`
2. Loads mechanistic branch (Tox21 GATv2):
   - model from `models/tox21_gatv2_model/best_model.pt`
   - task names from checkpoint metadata or fallback task list
3. Loads per-task threshold file if available (first match wins):
   - `models/tox21_gatv2_model/tox21_task_thresholds.json`
   - `models/tox21_gatv2_model/task_thresholds.json`
4. Exposes readiness in `/health`:
   - `tox21_thresholds_loaded`
   - `tox21_thresholds_source`
   - `tox21_threshold_count`
   - `inference_lock_initialized`


## 3) Unified Call Graph for POST /analyze

Request path:
1. Parse and validate input SMILES
2. Clinical prediction (`predict_clinical_toxicity`):
   - calls `predict_batch` (XSmiles branch)
   - returns: `label`, `is_toxic`, `p_toxic`, `confidence`, `threshold_used`
3. Mechanism prediction (`predict_toxicity_mechanism`):
   - calls `predict_tox21_batch` (Tox21 branch)
   - computes per-task hit set against loaded calibrated thresholds
   - returns full task score map and hit summary
4. Verdict aggregation (`aggregate_toxicity_verdict`):
   - `CONFIRMED_TOXIC`: clinical toxic + assay_hits > 0
   - `MECHANISTIC_ALERT`: clinical non-toxic + assay_hits > 0
   - `CLINICAL_CONCERN`: clinical toxic + assay_hits == 0
   - `LIKELY_SAFE`: clinical non-toxic + assay_hits == 0
5. Explanation gating:
   - if `explain_only_if_alert=true` and no mechanism alert and no explicit `target_task`, explanation is skipped
   - otherwise runs task-specific explainer (`explain_tox21_task`)
6. Build and return `AnalyzeResponse`


## 4) API Contract (Function-Calling Facing)

### AnalyzeRequest
- `smiles` (required)
- `clinical_threshold` (default 0.5)
- `mechanism_threshold` (default 0.5)
- `return_all_scores` (accepted for compatibility)
- `explain_only_if_alert` (default true)
- `explainer_epochs` (default 200)
- `explainer_timeout_ms` (default 30000)
- `target_task` (optional)

### ExplainRequest
- `smiles` (required)
- `epochs` (default 200)
- `explainer_timeout_ms` (default 30000)
- `target_class` (optional)

### AnalyzeResponse
- `smiles`
- `canonical_smiles`
- `clinical`:
  - `label`, `is_toxic`, `confidence`, `p_toxic`, `threshold_used`
- `mechanism`:
  - `task_scores` (full 12-task map)
  - `active_tasks`
  - `highest_risk_task`
  - `highest_risk_score`
  - `assay_hits`
  - `threshold_used`
  - `task_thresholds` (per-task threshold map)
- `explanation`:
  - null when gated off
  - otherwise includes `target_task`, `target_task_score`, top atoms/bonds, and optional heatmap
- `final_verdict`


## 5) Important Current Behavior

1. `task_scores` always returns full 12 tasks.
   - This was intentionally stabilized for predictable downstream function-calling.

2. `return_all_scores` is currently accepted but does not truncate `task_scores`.
   - Kept for backward compatibility in request payloads.

3. Explanation is conditional by default.
   - `explain_only_if_alert=true` means no explainer run for likely-safe/no-hit cases.
   - Set `explain_only_if_alert=false` or provide `target_task` to force explanation.

4. Inference is serialized with an async model lock.
   - Prevents concurrent model forward calls on shared model objects (CPU/GPU safety first).

5. Explainer timeout is enforced.
   - If explainer exceeds `explainer_timeout_ms`, API returns `504` with structured payload.


## 6) Retraining and Calibration Artifacts (Now Generated)

Produced under `models/tox21_gatv2_model/`:
- `best_model.pt`
- `tox21_gatv2_metrics.txt`
- `tox21_task_metrics.csv`
- `training_curves.png`
- `tox21_task_thresholds.json` (primary calibrated thresholds)
- `task_thresholds.json` (compatibility alias)
- `tox21_threshold_calibration.csv` (per-task threshold search report)


## 7) Current Training Configuration (Tox21)

From `config/tox21_gatv2_config.yaml` training section:
- `loss_type: focal`
- `focal_alpha: 0.25`
- `focal_gamma: 2.0`
- threshold calibration enabled with grid search:
  - `threshold_min: 0.05`
  - `threshold_max: 0.95`
  - `threshold_step: 0.01`
  - `fallback_threshold: 0.5`


## 8) Function-Calling Integration Recommendation

Use `POST /analyze` as the single callable operation per molecule.

Suggested policy in caller/orchestrator:
1. Call `/analyze` with `explain_only_if_alert=true` for low-cost default.
2. If caller needs rationale regardless of alert state, call with:
   - `explain_only_if_alert=false`
   - optional `target_task` to lock explanation target.
3. Trust `final_verdict` as top-level routing signal, and preserve `clinical` + `mechanism` blocks for auditability.
4. Set timeout based on explanation mode:
    - when `explanation == null`: typically low latency
    - when explanation runs: use larger timeout budget


## 9) Error Contract (Structured)

The API returns structured JSON for key production errors.

### Invalid SMILES
- HTTP status: `400`
- Payload:
```json
{
   "error": "invalid_smiles",
   "message": "Cannot parse SMILES: ...",
   "smiles": "..."
}
```

### Explainer timeout
- HTTP status: `504`
- Payload:
```json
{
   "error": "explainer_timeout",
   "message": "Explainer exceeded timeout of 30000 ms",
   "smiles": "...",
   "timeout_ms": 30000
}
```


## 10) Fast Troubleshooting

- Why explanation is null?
  - Usually because `explain_only_if_alert=true` and `assay_hits==0`.

- Why thresholds look all 0.5?
  - Threshold file not loaded. Check `/health` fields:
    - `tox21_thresholds_loaded`
    - `tox21_thresholds_source`
    - `tox21_threshold_count`

- Why verdict changed after retrain?
  - Clinical and mechanism heads are independent; calibrated thresholds can change hit counts and final verdict category.

- Why API seems slower on some calls?
   - Explanation path is much heavier than plain predict/analyze-without-explainer.
   - Tune `explainer_epochs` and `explainer_timeout_ms` in caller.


## 11) Source Files for This Flow

- `model_server/main.py`
- `model_server/schemas.py`
- `src/inference.py`
- `src/gnn_explainer.py`
- `src/graph_train.py`
- `scripts/train_tox21_gatv2.py`
- `config/tox21_gatv2_config.yaml`
