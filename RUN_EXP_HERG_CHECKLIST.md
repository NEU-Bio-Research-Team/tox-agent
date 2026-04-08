# RUN EXP hERG DEBUG CHECKLIST

## 1) Environment + preflight
- [ ] Activate env
  - conda activate drug-tox-env
- [ ] Verify PyG binary deps are healthy
  - python -c "import torch, torch_scatter, torch_geometric; print(torch.__version__, torch.version.cuda, torch_scatter.__version__, torch_geometric.__version__)"
  - If torch_scatter import fails, reinstall with matching wheel:
    - pip install --no-cache-dir --force-reinstall torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
- [ ] Confirm script/config
  - ls scripts/train_smilesgnn_herg_exp.py
  - ls config/xsmiles_herg_exp_config.yaml
- [ ] Config path format check
  - Use .yaml suffix: config/xsmiles_herg_exp_config.yaml
  - Do not use /yaml (this will fail path resolution on older script versions)
- [ ] Confirm key config knobs are loaded
  - phase1.loss_type = weighted_bce
  - phase2.backbone_learning_rate <= 1e-4
  - phase2.freeze_graph_layers = [0, 1]
  - phase2.selection_metric = joint_auc

## 2) Phase-1 sanity run (detect F1 collapse early)
- [ ] Run phase1-only
  - python scripts/train_smilesgnn_herg_exp.py --config config/xsmiles_herg_exp_config.yaml --phase1-only --device cuda
- [ ] Verify logs show class balance summary
  - [ClassBalance] Tox21 train macro positive rate
  - pos_weight(min/median/max)
- [ ] Check phase1 output JSON (from latest run folder)
  - phase1.val_metrics.macro_auc_roc should be reasonable (>0.65 typically)
  - phase1.val_metrics.macro_f1 should not stay at exact 0.0 across all epochs
  - task confusion matrices should include some predicted positives (not all [TN,0;FN,0])

## 3) Full 2-phase run
- [ ] Run full experiment with full logs
  - mkdir -p logs/exp_v3
  - python scripts/train_smilesgnn_herg_exp.py --config config/xsmiles_herg_exp_config.yaml --device cuda 2>&1 | tee logs/exp_v3/herg_exp_debug_run.log
- [ ] Confirm phase2 log lines include
  - joint_score_selected
  - joint_score_weighted
  - joint_score_unweighted

## 4) Artifact checks
- [ ] Verify output artifacts
  - ls models/smilesgnn_herg_exp_model
- [ ] Required files
  - best_model.pt
  - tokenizer.pkl
  - herg_threshold_metrics.json
  - tox21_task_thresholds.json
  - xsmiles_herg_exp_metrics.json

## 5) Joint-score formula validation (critical)
- [ ] Confirm formula metadata exists in JSON
  - phase2.beta_herg_effective
  - phase2.selection_metric
  - phase2.selection_formula
- [ ] Confirm weighted history exists
  - phase2.history.val_joint_score_weighted
  - phase2.history.val_tox21_macro_auc
  - phase2.history.val_herg_auc
- [ ] Recompute and compare one epoch (example with Python)
  - python - <<'PY'
import json
from pathlib import Path
p = Path('models/smilesgnn_herg_exp_model/xsmiles_herg_exp_metrics.json')
d = json.loads(p.read_text())
h = d['phase2']['history']
beta = float(d['phase2']['beta_herg_effective'])
i = int(max(range(len(h['val_joint_score_weighted'])), key=lambda k: h['val_joint_score_weighted'][k]))
tox = float(h['val_tox21_macro_auc'][i])
herg = float(h['val_herg_auc'][i])
calc = (tox + beta * herg) / (1.0 + beta)
saved = float(h['val_joint_score_weighted'][i])
print({'epoch': i + 1, 'calc': round(calc, 6), 'saved': round(saved, 6), 'abs_diff': abs(calc - saved)})
PY

## 6) Phase-2 freezing and forgetting check
- [ ] Verify freeze report in JSON
  - phase2.freeze_report.freeze_graph_layers_applied should include [0, 1]
  - phase2.freeze_report.num_frozen_params > 0
- [ ] Check catastrophic forgetting signals
  - Compare tox21 val/test macro_auc_roc before vs after phase2
  - Large drop (>0.01-0.02) suggests phase2 LR still too aggressive

## 7) hERG threshold and test metrics check
- [ ] Confirm both threshold modes are saved
  - metrics.herg.threshold_default_0p5.test
  - metrics.herg.threshold_calibrated.test
- [ ] Confirm calibration payload
  - threshold.threshold
  - threshold.sensitivity
  - threshold.specificity
- [ ] Prefer calibrated threshold for deployment report, but keep 0.5 as baseline comparator

## 8) Fast ablation matrix (debug root cause)
- [ ] A1: freeze_graph_layers [0,1], beta_herg 3.0 (default)
- [ ] A2: freeze_graph_layers [], beta_herg 3.0
- [ ] A3: freeze_graph_layers [0,1], beta_herg 2.0
- [ ] A4: freeze_graph_layers [0,1], beta_herg 5.0
- [ ] Keep seed fixed (42) for fair comparison

## 9) Reproducibility capture
- [ ] Save git commit hash
  - git rev-parse --short HEAD
- [ ] Save config snapshot used for run
  - cp config/xsmiles_herg_exp_config.yaml logs/exp_v3/xsmiles_herg_exp_config.used.yaml
- [ ] Record CUDA and torch
  - python - <<'PY'
import torch
print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'is_cuda_available': torch.cuda.is_available()})
PY
