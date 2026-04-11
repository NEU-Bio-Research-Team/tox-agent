# Experiment Runbook (Tox21-only mode)

Tai lieu nay huong dan reproducibility cho workspace hien tai theo huong Tox21.
ClinTox code KHONG bi xoa, nhung da duoc disable boi workspace mode.

## 1. Workspace mode

Workspace mode nam o:
- `config/workspace_mode.yaml`

Mac dinh hien tai:
- `mode: tox21_only`
- `clintox_enabled: false`
- `tox21_enabled: true`

Neu ban chay script ClinTox, script se dung voi thong bao `[DISABLED:CLINTOX]`.

## 2. Yeu cau

- OS: Linux
- Conda env: `drug-tox-env`
- CUDA (neu co): dung `--device cuda`, neu khong thi `--device cpu`

## 3. Setup moi truong

```bash
cd /home/mluser/BRT-FDA/MinhQuang/tox-agent
conda activate drug-tox-env

# Kiem tra nhanh
which python
python -V
```

Neu chua co env:

```bash
conda env create -f environment.yml
conda activate drug-tox-env
```

## 4. Quickstart (Tox21)

```bash
cd /home/mluser/BRT-FDA/MinhQuang/tox-agent
conda activate drug-tox-env
mkdir -p logs

python scripts/train_tox21_gatv2.py --device cuda --config config/tox21_gatv2_config.yaml 2>&1 | tee logs/03_train_tox21_gatv2.log
```

Output mac dinh:
- `models/tox21_gatv2_model/best_model.pt`
- `models/tox21_gatv2_model/tox21_gatv2_metrics.txt`
- `models/tox21_gatv2_model/tox21_task_metrics.csv`
- `models/tox21_gatv2_model/training_curves.png`

## 5. Tox21 prediction workflow

### 5.1 Single SMILES

```bash
python scripts/predict_tox21.py --smiles "CCO" --device cuda
```

### 5.2 Batch file prediction

```bash
python scripts/predict_tox21.py --input-file test_data/smiles_only.csv --device cuda
```

Mac dinh ket qua se duoc luu tai:
- `results/tox21_predictions.csv`

## 6. CPU fallback

Doi `--device cuda` thanh `--device cpu`:

```bash
python scripts/train_tox21_gatv2.py --device cpu --config config/tox21_gatv2_config.yaml
```

## 7. Cac metric can doi chieu

- `models/tox21_gatv2_model/tox21_gatv2_metrics.txt`
- `models/tox21_gatv2_model/tox21_task_metrics.csv`
- `results/tox21_predictions.csv`

Chi so khuyen nghi:
- Macro AUC-ROC
- Macro PR-AUC
- Macro F1
- Micro AUC-ROC
- Micro PR-AUC

## 8. Full command block (copy-run)

```bash
cd /home/mluser/BRT-FDA/MinhQuang/tox-agent
conda activate drug-tox-env
mkdir -p logs

python scripts/train_tox21_gatv2.py --device cuda --config config/tox21_gatv2_config.yaml 2>&1 | tee logs/03_train_tox21_gatv2.log
python scripts/predict_tox21.py --input-file test_data/smiles_only.csv --device cuda --output results/tox21_predictions.csv
```

## 9. Troubleshooting nhanh

- Loi khong tim thay torch-geometric/rdkit:
  - Xac nhan env: `conda activate drug-tox-env`
  - Kiem tra `python -V` va `which python`
- CUDA khong available:
  - Chuyen `--device cpu`
- Chay script ClinTox bi dung ngay:
  - Day la hanh vi dung theo mode tox21_only
  - Kiem tra `config/workspace_mode.yaml`

## 10. ClinTox scripts (archived, disabled)

Cac script duoi day duoc giu lai de tham khao, nhung khong duoc phep chay trong mode hien tai:

- `scripts/train_hybrid.py`
- `scripts/train_gatv2.py`
- `scripts/train_gin.py`
- `scripts/train_gatv2_transfer.py`
- `scripts/explain_smilesgnn.py`
- `scripts/generate_curves.py`
- `scripts/consolidate_results.py`
