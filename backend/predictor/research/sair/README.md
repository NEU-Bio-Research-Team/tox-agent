# SAIR Phase-1 research pipeline

Thư mục này chứa pipeline nghiên cứu activity trên SAIR. Nó không phải đường
serving production và không đưa dữ liệu hoặc model artifact vào image predictor.

## Cấu trúc

- `inspect_schema.py`: xác minh schema, định danh file và ánh xạ cột logic.
- `prepare_phase1.py`: loại measurement lặp theo structure, aggregate theo
  protein–ligand pair, áp dụng label-quality gate và tạo exact-sequence
  cold-target split.
- `contracts.py`: contract và leakage audit dùng chung cho các split strategy.
- `baseline.py`: baseline B0 có thể tái lập, prediction interval và
  applicability sơ bộ.

Luồng dữ liệu:

```text
data/sair/raw/sair.parquet
  -> inspect_schema
  -> prepare_phase1
  -> data/sair/processed/phase1_pairs.parquet + manifest
  -> baseline
  -> research-only model artifact + metric report
```

## Môi trường

Từ repository root:

```bash
python -m pip install -e 'backend/predictor[sair]'
export PYTHONPATH=backend/predictor/research
```

Các dependency SAIR là optional để production predictor không bắt buộc mang
theo stack xử lý Parquet và huấn luyện.

## Tái lập pipeline hiện tại

Dataset pin:

- revision: `8ab398166696ea8387fe099e925bfc76da319b60`
- SHA-256: `10f76e8c9baf6ebd43d9a1bbab97c07a8e6da16311f8990db3e9635bb928ab28`

```bash
python -m sair.inspect_schema \
  --dataset data/sair/raw/sair.parquet \
  --out data/sair/metadata/sair-schema-report.json \
  --dataset-version 8ab398166696ea8387fe099e925bfc76da319b60

python -m sair.prepare_phase1 \
  --source data/sair/raw/sair.parquet \
  --output data/sair/processed/phase1_pairs.parquet \
  --manifest data/sair/metadata/phase1-manifest.json \
  --dataset-revision 8ab398166696ea8387fe099e925bfc76da319b60 \
  --dataset-sha256 10f76e8c9baf6ebd43d9a1bbab97c07a8e6da16311f8990db3e9635bb928ab28 \
  --max-label-span 1.0 \
  --split-salt toxagent-sair-exact-sequence-v1

python -m sair.baseline \
  --phase1 data/sair/processed/phase1_pairs.parquet \
  --phase1-manifest data/sair/metadata/phase1-manifest.json \
  --artifact data/sair/artifacts/sair-descriptor-hgb-b0.joblib \
  --report data/sair/metadata/sair-descriptor-hgb-b0-report.json
```

Tất cả output nằm dưới `data/`, đã được Git ignore. Manifest và report chứa
checksum để phát hiện input hoặc artifact không khớp.

## Ranh giới an toàn

Baseline B0 luôn ghi `admitted: false`. Không được nối artifact này vào HTTP
route hoặc dùng kết quả prioritisation như kết luận “safe”. Muốn production
admission cần vượt scientific evaluation gate, hoàn thiện OOD evaluation,
model registry và serving integration được duyệt.
