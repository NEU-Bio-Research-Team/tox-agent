# Run Exp V2 Checklist (Updated)

Muc tieu:
- Fix overfitting and threshold instability cho Direction 3.
- Fix low-discriminative learned proxy behavior cho Direction 1.
- Tao output de so sanh trong 1 bang CSV.

## Why command bi vo dong truoc do

- Neu viet command nhieu dong, phai co dau \\ o cuoi dong.
- Neu khong co \\, shell se chay tung dong rieng va bao command not found.
- Nho tao truoc thu muc logs/exp_v2 truoc khi dung tee.

## Copy khoi lenh duoi day va chay nguyen block

    set -euo pipefail
    cd /home/minhquang/tox-agent
    conda activate drug-tox-env

    mkdir -p logs/exp_v2
    mkdir -p models/tox21_clinical_proxy_learned_v2
    mkdir -p models/tox21_clinical_proxy_weighted_v2
    mkdir -p models/clinical_head_model_ecfp4_v2

    echo "[1/6] Direction 1 - Learned LR-CV (regularized + fallback guard)"
    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode learned_lr_cv \
      --lr-max-iter 500 \
      --lr-regularization l2 \
      --lr-c-grid "0.01,0.03,0.1,0.3,1.0" \
      --learned-min-val-auc 0.55 \
      --threshold-calibration cv \
      --threshold-fit-split train_val \
      --threshold-cv-folds 3 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_learned_v2 \
      2>&1 | tee logs/exp_v2/d1_learned_v2.log

    echo "[2/6] Direction 1 - Weighted baseline"
    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode weighted \
      --threshold-calibration cv \
      --threshold-fit-split train_val \
      --threshold-cv-folds 3 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_weighted_v2 \
      2>&1 | tee logs/exp_v2/d1_weighted_v2.log

    echo "[3/6] Direction 1 - Optional SVM-RBF proxy"
    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode learned_svm_rbf \
      --svm-c 1.0 \
      --svm-gamma scale \
      --learned-min-val-auc 0.55 \
      --threshold-calibration cv \
      --threshold-fit-split train_val \
      --threshold-cv-folds 3 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_svm_v2 \
      2>&1 | tee logs/exp_v2/d1_svm_v2.log

    echo "[4/6] Direction 3 - Clinical head + ECFP4 (regularized)"
    python scripts/train_clinical_head.py \
      --use-ecfp4-features \
      --ecfp-radius 2 \
      --ecfp-bits 256 \
      --dropout 0.4 \
      --weight-decay 1e-3 \
      --early-stopping-patience 40 \
      --tox21-missing-impute 0.5 \
      --threshold-calibration cv \
      --threshold-fit-split train_val \
      --threshold-cv-folds 3 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/clinical_head_model_ecfp4_v2 \
      2>&1 | tee logs/exp_v2/d3_head_ecfp4_v2.log

    echo "[5/6] Aggregate comparison table"
    python scripts/compare_direction_metrics.py \
      --direction1-json models/tox21_clinical_proxy_learned_v2/clinical_proxy_metrics.json \
      --direction3-json models/clinical_head_model_ecfp4_v2/clinical_head_metrics.json \
      --output results/direction_comparison_v2.csv \
      2>&1 | tee logs/exp_v2/compare_v2.log

    echo "[6/6] Show final output"
    ls -lh results/direction_comparison_v2.csv
    cat results/direction_comparison_v2.csv

## Quick sanity checks

    echo "[Check] Learned mode + fallback status"
    grep -E "Proxy mode|fallback|Threshold used|Saved metrics" logs/exp_v2/d1_learned_v2.log | tail -n 20

    echo "[Check] D3 regularization params actually applied"
    grep -E "Feature rows|Threshold used|Saved checkpoint|Saved metrics" logs/exp_v2/d3_head_ecfp4_v2.log | tail -n 20

    echo "[Check] Comparison rows"
    grep -E "direction_1|direction_3" logs/exp_v2/compare_v2.log | tail -n 20

## Notes

- Neu muon ep learned proxy khong fallback ve weighted, them option --disable-learned-fallback.
- Neu muon threshold chi fit tren val, doi --threshold-fit-split val.
- Default moi trong code da duoc cap nhat theo huong regularization manh hon cho ca D1 va D3.
