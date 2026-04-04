# Run Exp V3 Checklist

Muc tieu:
- D1: learned proxy co ECFP4 (khong con bi gioi han boi 12 tox21 probs).
- D3: train head voi regularization ro rang + threshold fit train+val.
- Tao bang so sanh v3.

## 1) Chay lai exp v3 (copy nguyen block)

    set -euo pipefail
    cd /home/minhquang/tox-agent
    conda activate drug-tox-env

    mkdir -p logs/exp_v3
    mkdir -p models/tox21_clinical_proxy_learned_ecfp_v3
    mkdir -p models/tox21_clinical_proxy_svm_ecfp_v3
    mkdir -p models/clinical_head_model_ecfp4_v3

    echo "[1/4] D1 learned LR-CV + ECFP4"
    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode learned_lr_cv \
      --use-ecfp4-features \
      --ecfp-radius 2 \
      --ecfp-bits 256 \
      --lr-max-iter 500 \
      --lr-regularization l2 \
      --lr-c-grid "0.01,0.03,0.1,0.3,1.0" \
      --learned-min-val-auc 0.55 \
      --threshold-calibration cv \
      --threshold-fit-split train_val \
      --threshold-cv-folds 3 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_learned_ecfp_v3 \
      2>&1 | tee logs/exp_v3/d1_learned_ecfp_v3.log

    echo "[2/4] D1 optional SVM-RBF + ECFP4"
    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode learned_svm_rbf \
      --use-ecfp4-features \
      --ecfp-radius 2 \
      --ecfp-bits 256 \
      --svm-c 1.0 \
      --svm-gamma scale \
      --learned-min-val-auc 0.55 \
      --threshold-calibration cv \
      --threshold-fit-split train_val \
      --threshold-cv-folds 3 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_svm_ecfp_v3 \
      2>&1 | tee logs/exp_v3/d1_svm_ecfp_v3.log

    echo "[3/4] D3 clinical head + ECFP4 (regularized)"
    python scripts/train_clinical_head.py \
      --use-ecfp4-features \
      --ecfp-radius 2 \
      --ecfp-bits 256 \
      --dropout 0.4 \
      --weight-decay 1e-3 \
      --hidden-dim 64 \
      --early-stopping-patience 40 \
      --threshold-calibration cv \
      --threshold-fit-split train_val \
      --threshold-cv-folds 3 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/clinical_head_model_ecfp4_v3 \
      2>&1 | tee logs/exp_v3/d3_head_ecfp4_v3.log

    echo "[4/4] Compare"
    python scripts/compare_direction_metrics.py \
      --direction1-json models/tox21_clinical_proxy_learned_ecfp_v3/clinical_proxy_metrics.json \
      --direction3-json models/clinical_head_model_ecfp4_v3/clinical_head_metrics.json \
      --output results/direction_comparison_v3.csv \
      2>&1 | tee logs/exp_v3/compare_v3.log

    cat results/direction_comparison_v3.csv

## 2) Check nhanh

    grep -E "Proxy mode|Threshold used|Train AUC/F1|Val   AUC/F1|Test  AUC/F1|fallback" logs/exp_v3/d1_learned_ecfp_v3.log | tail -n 20
    grep -E "Best epoch|Threshold used|Train AUC/F1|Val   AUC/F1|Test  AUC/F1" logs/exp_v3/d3_head_ecfp4_v3.log | tail -n 20

## 3) Ghi chu

- D1 co fallback gate. Neu khong muon fallback, them --disable-learned-fallback.
- D3 threshold calibration da fit tren train+val bang --threshold-fit-split train_val.
