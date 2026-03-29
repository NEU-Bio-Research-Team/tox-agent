# Experiment Runbook (Workspace hien tai)

Tai lieu nay huong dan reproducibility cho workspace hien tai theo 3 muc tieu:
- Reproduce model chinh SMILESGNN tren ClinTox.
- Chay baseline GATv2 tren ClinTox de doi chieu.
- Chay pipeline chat luong cao: Tox21 pretrain -> transfer fine-tune sang ClinTox.

## 1. Yeu cau

- OS: Linux
- Conda env: `drug-tox-env`
- CUDA (neu co): chay voi `--device cuda`, neu khong thi `--device cpu`

## 2. Setup moi truong

```bash
cd /home/minhquang/projects/tox-agent
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

## 3. Quickstart (chi reproduce model chinh)

```bash
cd /home/minhquang/projects/tox-agent
conda activate drug-tox-env
python scripts/train_hybrid.py --device cuda
```

Output mac dinh:
- `models/smilesgnn_model/best_model.pt`
- `models/smilesgnn_model/tokenizer.pkl`
- `models/smilesgnn_model/smilesgnn_model_metrics.txt`
- `models/smilesgnn_model/training_curves.png`

## 4. Reproduce day du cac experiment

### 4.1 Tao thu muc log

```bash
cd /home/minhquang/projects/tox-agent
mkdir -p logs
conda activate drug-tox-env
```

### 4.2 ClinTox baseline: SMILESGNN (model chinh)

```bash
python scripts/train_hybrid.py --device cuda | tee logs/01_train_hybrid.log
```

Config mac dinh: `config/smilesgnn_config.yaml`

### 4.3 ClinTox baseline: GATv2

```bash
python scripts/train_gatv2.py --device cuda | tee logs/02_train_gatv2.log
```

Config mac dinh: `config/gatv2_config.yaml`

Output mac dinh:
- `models/gatv2_model/best_model.pt`
- `models/gatv2_model/gatv2_model_metrics.txt`
- `models/gatv2_model/training_curves.png`

### 4.4 Tox21 multi-task pretrain (GATv2)

```bash
python scripts/train_tox21_gatv2.py --device cuda \
  --config config/tox21_gatv2_config.yaml \
  | tee logs/03_train_tox21_gatv2.log
```

Output mac dinh:
- `models/tox21_gatv2_model/best_model.pt`
- `models/tox21_gatv2_model/tox21_gatv2_metrics.txt`
- `models/tox21_gatv2_model/tox21_task_metrics.csv`
- `models/tox21_gatv2_model/training_curves.png`

### 4.5 Transfer: Tox21 -> ClinTox (GATv2)

```bash
python scripts/train_gatv2_transfer.py --device cuda \
  --config config/gatv2_transfer_config.yaml \
  | tee logs/04_train_gatv2_transfer.log
```

Output mac dinh:
- `models/gatv2_transfer_model/best_model.pt`
- `models/gatv2_transfer_model/gatv2_transfer_metrics.txt`
- `models/gatv2_transfer_model/training_curves.png`

Luu y: `config/gatv2_transfer_config.yaml` dang tro den checkpoint pretrain:
- `./models/tox21_gatv2_model/best_model.pt`

Vi vay phai chay xong buoc 4.4 truoc buoc 4.5.

## 5. Chay tren CPU (neu khong co CUDA)

Doi `--device cuda` thanh `--device cpu` trong cac lenh tren.

Vi du:

```bash
python scripts/train_hybrid.py --device cpu
```

## 6. Thu tu khuyen nghi de bao toan reproducibility

1. Chay baseline ClinTox truoc (SMILESGNN, GATv2).
2. Chay Tox21 multi-task pretrain.
3. Chay transfer Tox21 -> ClinTox.
4. Tong hop metric sau moi run (khong doi config giua cac lan so sanh).

## 7. Cac file metric can doi chieu

- SMILESGNN: `models/smilesgnn_model/smilesgnn_model_metrics.txt`
- GATv2 baseline: `models/gatv2_model/gatv2_model_metrics.txt`
- Tox21 pretrain: `models/tox21_gatv2_model/tox21_gatv2_metrics.txt`
- Transfer ClinTox: `models/gatv2_transfer_model/gatv2_transfer_metrics.txt`

Chi so khuyen nghi de so sanh chat luong ClinTox:
- AUC-ROC
- PR-AUC
- F1
- Recall toxic class (neu can bo sung them script evaluate chi tiet)

## 8. Full command block (copy-run)

```bash
cd /home/minhquang/projects/tox-agent
conda activate drug-tox-env
mkdir -p logs

python scripts/train_hybrid.py --device cuda | tee logs/01_train_hybrid.log
python scripts/train_gatv2.py --device cuda | tee logs/02_train_gatv2.log
python scripts/train_tox21_gatv2.py --device cuda --config config/tox21_gatv2_config.yaml | tee logs/03_train_tox21_gatv2.log
python scripts/train_gatv2_transfer.py --device cuda --config config/gatv2_transfer_config.yaml | tee logs/04_train_gatv2_transfer.log
```

## 9. Troubleshooting nhanh

- Loi khong tim thay torch-geometric/rdkit:
  - Xac nhan dang dung dung env: `conda activate drug-tox-env`
  - Kiem tra `python -V` va `which python`
- Loi checkpoint transfer khong ton tai:
  - Chua chay xong Tox21 pretrain (buoc 4.4)
- CUDA khong available:
  - Chuyen sang `--device cpu`

## 10. Ghi chu quan trong

- Cac config mac dinh dang dung `seed: 42`.
- Neu ban chay nhieu lan de bao cao khoa hoc, nen:
  - Giu nguyen config,
  - Luu log moi lan,
  - Ghi ro thoi gian va git commit hash.
