# Báo cáo: Nhánh hoạt tính (bioactivity) — kết quả P0–P2 và danh sách file mới

**Ngày:** 15/09/2026
**Phạm vi:** Triển khai P0–P2 của `TOXAGENT_BIOACTIVITY_BENCHMARK_PLAN_VI.md` (data contract, panel, 4 split view, baseline B0–B2b) và bổ sung B3.5 (CheMeleon) sau khi rà soát SOTA.
**Trạng thái:** tất cả code + manifest đã `git add`, **chưa commit**.

---

## 1. Kết quả benchmark

### 1.1 Panel và dữ liệu

- Nguồn: ChEMBL 37 qua REST API (không tải dump — dump cần ~25GB, ổ đĩa lúc đó chỉ còn 9.3GB trống). API xác nhận đang serve đúng `ChEMBL_37`, 8.760 request, **0 lỗi/retry**.
- Panel: **16 target / 27 task**, trải trên **8 protein family** đã annotate (enzyme, membrane receptor, epigenetic regulator, transcription factor, ion channel, transporter, cytosolic other, nuclear other), chọn bằng rule cố định trước (không dùng model score).
- Dataset HQ-Exact: 142.805 record thô → **120.240 dòng đã aggregate**, **96.374 hợp chất duy nhất**.
- 4 split view đã freeze: temporal (**primary**), cluster, scaffold, random (diagnostic). Tỷ lệ 60/10/10/20 (train/val/calibration/test) — lệch so với đề xuất gốc 70/10/5/15 trong plan vì 5% calibration chỉ đủ ngưỡng ≥150 dòng/task cho 9/27 task, còn 10% đủ cho 15/27 task.

### 1.2 Baseline B0–B2b (view temporal = primary, macro MAE, đơn vị pChEMBL)

| model | temporal | cluster | scaffold | random | cliff dir-acc (temporal) |
|---|---:|---:|---:|---:|---:|
| B0 median | 1.060 | 1.025 | 0.982 | 0.940 | 0.000 |
| B1 ECFP4 kNN | 0.991 | 0.770 | 0.595 | 0.511 | 0.504 |
| B2a RF | 0.945 | 0.725 | 0.582 | 0.509 | 0.584 |
| B2b LightGBM | **0.943** | 0.706 | 0.551 | 0.470 | 0.527 |

**Nhận xét quan trọng:** model tốt nhất chỉ hơn dummy median ~11% trên view primary (so với ~50% trên view random) — đúng khó khăn thật của bài toán prospective. Cliff direction accuracy 0.50–0.58, gần như random — lặp lại đúng phát hiện của MoleculeACE trên chính panel này.

### 1.3 Chẩn đoán OOD (đo trực tiếp bằng Tanimoto, không suy đoán)

| view | median max-Tanimoto tới train | % có neighbor gần (>0.7) trong train |
|---|---:|---:|
| temporal | 0.416 | 5.6% |
| cluster | 0.526 | 17.2% |
| scaffold | 0.714 | **53.7%** |
| random | 0.790 | 79.4% |

**Phát hiện:** view scaffold — thứ vẫn thường được coi là chuẩn generalization trong QSAR — thực ra **quá nửa** compound test có bản gần giống trong train ở panel này (Bemis-Murcko scaffold đổi ở lõi vòng, không đổi ở nhánh thế, nên hai phân tử rất giống nhau vẫn bị tách khác scaffold). Chỉ cluster mới thực sự làm việc OOD ngoài temporal.

### 1.4 B3.5 — CheMeleon (Chemprop fine-tune từ checkpoint pretrain)

Sau khi rà soát literature hiện hành, phát hiện CheMeleon (Burns, 2026) công bố 97% win-rate trên MoleculeACE (đúng điểm yếu cliff ở trên) — đã cài đặt và benchmark thật (không chỉ đề xuất):

| view | b2b LightGBM | CheMeleon | cliff dir-acc: b2b→CheMeleon |
|---|---:|---:|---|
| temporal (**PRIMARY**) | 0.943 | 0.971 | 0.527 → **0.560** |
| cluster | 0.706 | 0.706 | 0.655 → **0.703** |
| scaffold | 0.551 | 0.548 | 0.690 → **0.718** |
| random | 0.470 | 0.478 | 0.766 → **0.824** |

**Kết luận:** macro MAE gần như hòa với LightGBM (không view nào CheMeleon thắng rõ), nhưng cliff direction accuracy tăng **nhất quán +3 đến +6 điểm %** ở cả 4 view — hiệu ứng thật, đúng hướng nhưng khiêm tốn hơn nhiều so với 97% công bố (vì MoleculeACE đánh giá từng assay đơn lẻ đã curate, còn ở đây là multi-task 27 task không đồng nhất). Nên giữ CheMeleon trong model tournament cho use case nhạy cliff, không thay LightGBM làm model MAE tốt nhất.

### 1.5 Chưa làm

B4 (GATv2), B5 (ChemBERTa), B6 (KPGT/Uni-Mol2), ToxAct-TAC-MoE (N1), toàn bộ track V2 (D0–D5, N2, cần hạ tầng compound–target chưa có), track `Censored-Extended`.

---

## 2. Danh sách file mới

### 2.1 Đã `git add` (sẽ vào git khi commit) — 33 file

```
TOXAGENT_BIOACTIVITY_BENCHMARK_PLAN_VI.md          # plan doc, bump lên v1.1

backend/predictor/research/bioactivity/
├── README.md                          # runbook đầy đủ + kết quả
├── ingest/
│   ├── __init__.py
│   ├── chembl_client.py               # client API ChEMBL, pin release, HQ-Exact filters
│   ├── target_annotation.py           # resolve protein family + gene symbol
│   ├── profile_targets.py             # P0: profile toàn bộ target universe
│   ├── select_panel.py                # chọn panel bằng rule cố định
│   ├── standardize.py                 # chuẩn hóa cấu trúc, connectivity key
│   ├── build_dataset.py               # extract + aggregate + manifest
│   ├── split.py                       # 4 split view + eligibility gate
│   └── task_keys.py                   # phân biệt task_key (aggregation) vs task_unit_key (modelling)
└── models/
    ├── __init__.py
    ├── featurize.py                   # ECFP4 + descriptor + context one-hot encoder
    ├── ecfp_baselines.py              # B0/B1/B2a/B2b
    └── chemeleon_chemprop.py          # B3.5 adapter (chạy dưới Python 3.11 riêng)

backend/predictor/evals/bioactivity/
├── metrics.py                         # regression/ranking/cliff/calibration metrics
├── run_benchmark.py                   # runner B0-B2b, đọc frozen split, verify hash
├── bench_chemeleon.py                 # runner riêng cho B3.5 (Python 3.11 env)
├── stress/
│   ├── __init__.py
│   ├── activity_cliffs.py             # xây cliff pair set (held-out)
│   └── ood.py                         # đo Tanimoto similarity tới train
└── manifests/                         # ← các artifact "đông cứng" (frozen), nhỏ, cần track
    ├── panel-v1.json
    ├── qualifying_targets.csv
    ├── dataset_manifest.json
    ├── benchmark_report.json          # kết quả B0-B2b đầy đủ
    ├── benchmark_report_chemeleon.json# kết quả B3.5 đầy đủ
    ├── ood_similarity.json
    └── splits/
        ├── split_manifest.json
        ├── split-temporal.json
        ├── split-cluster.json
        ├── split-scaffold.json
        └── split-random.json

backend/predictor/tests/unit/
└── test_bioactivity_data_contract.py  # 22 test, cover leakage/standardize/metrics invariant
```

**Ghi chú kỹ thuật quan trọng:** thư mục gốc dự kiến trong plan là `research/bioactivity/data/`, nhưng đã đổi tên thành `ingest/` vì `.gitignore` của repo có rule `data/` không neo (unanchored) — khớp với **bất kỳ** thư mục tên `data` ở **bất kỳ** cấp nào, không chỉ thư mục cache gốc repo. Nếu giữ tên `data/`, toàn bộ code pipeline sẽ bị git bỏ qua âm thầm (đã xảy ra thật trong lúc làm, phát hiện và sửa ngay).

### 2.2 Bị `.gitignore` — không vào git (2 nhóm)

**Nhóm A — cache dữ liệu thô, tái tạo được từ ChEMBL API (`data/bioactivity/`, ~18MB):**

```
data/bioactivity/
├── profile/
│   ├── target_profile.csv             # profile 5.869 target (P0 pass 1+2)
│   └── profile_summary.json
└── hq-v1/
    ├── raw_activities.json.gz         # cache 142.805 record thô từ API
    ├── hq_exact.csv.gz                # dataset đã aggregate (120.240 dòng)
    ├── dataset_manifest.json          # (bản làm việc — bản đông cứng đã copy sang manifests/)
    └── splits/                        # (bản làm việc — bản đông cứng đã copy sang manifests/)
```

**Nhóm B — output benchmark, tái tạo được bằng cách chạy lại runner (`backend/predictor/evals/bioactivity/results/`, ~1.1GB):**

```
backend/predictor/evals/bioactivity/results/
├── benchmark_report.json              # (bản làm việc — đã đông cứng vào manifests/)
├── benchmark_report_chemeleon.json    # (bản làm việc — đã đông cứng vào manifests/)
├── ood_similarity.json                # (bản làm việc — đã đông cứng vào manifests/)
├── preds-<view>-<model>.npy           # 20 file — dự đoán từng dòng của từng model/view
└── chemeleon_workdir_<view>/          # 4 thư mục, ~250MB/thư mục
    ├── train.csv, query.csv, preds.csv
    └── model/model_0/
        ├── best.pt                    # checkpoint CheMeleon đã fine-tune
        ├── checkpoints/*.ckpt         # checkpoint Lightning (bao gồm epoch tốt nhất)
        └── trainer_logs/              # log loss theo epoch
```

Cả hai nhóm này **cố ý** không vào git — đây là cache/output có thể tái tạo lại từ code + manifest đã đông cứng, không phải nguồn thật (source of truth). Muốn giữ lại kết quả nào lâu dài, copy file JSON nhỏ liên quan sang `evals/bioactivity/manifests/` (đã làm cho 5 file: panel, dataset_manifest, benchmark_report, benchmark_report_chemeleon, ood_similarity).

### 2.3 Thay đổi môi trường (không phải file trong repo, nhưng cần biết)

- **`drug-tox-env`**: cài thêm `chemprop` (bản pip mới nhất tương thích — nhưng phát hiện bị conflict với 1 bản cài kiểu editable cũ trỏ tới code ngoài repo tại `/home/mluser/Tox_pred/ref_source/FRAIL/...`, đã force-reinstall để sửa — bản cũ đó vốn đã hỏng sẵn dưới Python 3.10 nên không mất gì).
- **`comosa_phase1`** (env của project khác, Python 3.11): cài thêm `rdkit` + `chemprop>=2.2` để chạy CheMeleon — đã kiểm tra bằng `pip install --dry-run` trước khi cài, xác nhận không đụng tới torch/numpy/pandas/sklearn sẵn có của project đó.

---

## 3. Việc cần quyết định tiếp theo

1. **Commit hay không?** Toàn bộ đã stage, chưa commit — đang chờ xác nhận.
2. **B4–B6, N1 (ToxAct-TAC-MoE)** chưa làm — nên làm tiếp hay dừng ở đây để đánh giá thêm dữ liệu (panel có 12/27 task dưới ngưỡng eligibility, đáng cân nhắc trước khi đầu tư kiến trúc mới)?
3. **V2 (compound–target)** cần hạ tầng hoàn toàn mới (protein embedding, DTI dataset) — chưa bắt đầu.
