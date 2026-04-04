# Run Exp V2 Checklist

Muc tieu: chay lai Direction 1 va Direction 3 voi pipeline moi, sau do tong hop bang so sanh.

## Nguyen nhan loi ban vua gap

- Script khong bi viet nham option.
- Loi command not found xay ra vi shell da coi moi dong --option la mot lenh rieng.
- Can them dau \ o cuoi dong neu viet lenh nhieu dong, hoac viet 1 dong duy nhat.
- Ban go nham lst thay vi ls.
- Thu muc logs/exp_v2 chua ton tai nen tee bi fail.

## Copy block nay va chay nguyen khoi

    set -euo pipefail
    cd /home/minhquang/tox-agent
    conda activate drug-tox-env

    mkdir -p logs/exp_v2
    mkdir -p models/tox21_clinical_proxy_learned
    mkdir -p models/tox21_clinical_proxy_weighted
    mkdir -p models/clinical_head_model_ecfp4

    echo "[1/5] Direction 1 - Learned proxy (LogisticRegressionCV)"
    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode learned_lr_cv \
      --threshold-calibration cv \
      --lr-cv-folds 5 \
      --threshold-cv-folds 5 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_learned \
      2>&1 | tee logs/exp_v2/d1_learned.log

    echo "[2/5] Direction 1 - Weighted proxy (fixed missing handling, no renorm)"
    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode weighted \
      --threshold-calibration cv \
      --threshold-cv-folds 5 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_weighted \
      2>&1 | tee logs/exp_v2/d1_weighted.log

    echo "[3/5] Direction 3 - Clinical head + ECFP4"
    python scripts/train_clinical_head.py \
      --use-ecfp4-features \
      --ecfp-radius 2 \
      --ecfp-bits 1024 \
      --tox21-missing-impute 0.5 \
      --threshold-calibration cv \
      --threshold-cv-folds 5 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/clinical_head_model_ecfp4 \
      2>&1 | tee logs/exp_v2/d3_head_ecfp4.log

    echo "[4/5] Aggregate comparison table"
    python scripts/compare_direction_metrics.py \
      --direction1-json models/tox21_clinical_proxy_learned/clinical_proxy_metrics.json \
      --direction3-json models/clinical_head_model_ecfp4/clinical_head_metrics.json \
      --output results/direction_comparison_v2.csv \
      2>&1 | tee logs/exp_v2/compare.log

    echo "[5/5] Show final table"
    ls -lh results/direction_comparison_v2.csv
    cat results/direction_comparison_v2.csv

## Kiem tra nhanh de chac chan da dung config moi

    echo "[Check] Direction 1 mode"
    grep -E "Proxy mode|Threshold used|Saved metrics" logs/exp_v2/d1_learned.log | tail -n 5

    echo "[Check] Direction 3 input dim va ECFP"
    grep -E "input_dim|Feature signal check|Saved checkpoint" logs/exp_v2/d3_head_ecfp4.log | tail -n 10

    echo "[Check] Compare output"
    grep -E "direction_1|direction_3" logs/exp_v2/compare.log | tail -n 10

## Neu muon benchmark weighted legacy (renormalize missing)

    python scripts/eval_tox21_clinical_proxy.py \
      --proxy-mode weighted \
      --renormalize-missing \
      --threshold-calibration cv \
      --threshold-cv-folds 5 \
      --split-type scaffold \
      --seed 42 \
      --output-dir models/tox21_clinical_proxy_weighted_legacy \
      2>&1 | tee logs/exp_v2/d1_weighted_legacy.log
