# RUN EXP hERG CHECKLIST

## 1) Environment + preflight
- [ ] Activate env
  - conda activate drug-tox-env
- [ ] Confirm config and script exist
  - ls scripts/train_smilesgnn_herg_exp.py
  - ls config/xsmiles_herg_exp_config.yaml
- [ ] Ensure pytdc is installed
  - pip show pytdc

## 2) Smoke run (phase1 only)
- [ ] Run quick smoke to verify data/tokenizer/model build path
  - python scripts/train_smilesgnn_herg_exp.py --config config/xsmiles_herg_exp_config.yaml --phase1-only --device cuda
- [ ] Verify logs show
  - hERG sizes train/val/test
  - Tox21 task list
  - No label-column error (label)

## 3) Full experiment run (main)
- [ ] Run full 2-phase training
  - python scripts/train_smilesgnn_herg_exp.py --config config/xsmiles_herg_exp_config.yaml --device cuda
- [ ] Confirm phase2 log prints effective_beta_herg and epoch metrics

## 4) Artifact checks
- [ ] Verify model output folder exists
  - ls models/smilesgnn_herg_exp_model
- [ ] Required artifacts
  - best_model.pt
  - tokenizer.pkl
  - smilesgnn_model_metrics.txt
  - herg_threshold_metrics.json
  - tox21_task_thresholds.json
  - xsmiles_herg_exp_metrics.json

## 5) Metrics checks
- [ ] Open xsmiles_herg_exp_metrics.json and validate keys
  - metrics.tox21.threshold_default_0p5
  - metrics.tox21.threshold_calibrated
  - metrics.herg.train
  - metrics.herg.val
  - metrics.herg.test
  - phase2.beta_herg_effective
- [ ] Primary monitor
  - hERG: auc_roc, pr_auc, f1
  - Tox21: macro_auc_roc, macro_pr_auc, macro_f1

## 6) Beta ablation (recommended)
Run the same script with fixed beta values and compare hERG-vs-Tox21 tradeoff.

- [ ] beta_herg = 1.0
- [ ] beta_herg = 2.0
- [ ] beta_herg = 3.0 (default)
- [ ] beta_herg = 5.0

Suggested process:
- Copy config to 4 variants (or edit beta_herg each run)
- Keep all other hyperparameters identical
- Use same seed for fair comparison

## 7) Selection rule
- [ ] If deployment target prioritizes hERG, choose run with highest hERG val/test auc_roc and stable pr_auc
- [ ] If preserving Tox21 transfer is important, choose best Pareto point between:
  - hERG auc_roc/pr_auc
  - Tox21 macro_auc_roc

## 8) Reproducibility notes
- [ ] Record git commit hash
- [ ] Record exact config used
- [ ] Record device and CUDA version
- [ ] Save console log to file
  - python scripts/train_smilesgnn_herg_exp.py --config config/xsmiles_herg_exp_config.yaml --device cuda | tee logs/exp_v3/herg_exp_run.log
