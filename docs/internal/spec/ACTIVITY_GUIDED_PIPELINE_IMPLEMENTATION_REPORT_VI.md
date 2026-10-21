# Báo cáo triển khai Activity-Guided Toxicity Screening

**Ngày cập nhật:** 2026-10-20
**Nhánh:** `feat/bioactivity-first-pipeline-redesign`
**Trạng thái:** Research pipeline đã chạy trên SAIR thật; chưa được admission cho production

## 1. Kết luận hiện tại

Đã triển khai và kiểm thử được đường nghiên cứu:

```text
SAIR parquet
  -> schema/identity validation
  -> loại lặp pose và measurement trùng
  -> aggregate protein-ligand pair
  -> label-quality gate
  -> exact-sequence cold-target split
  -> baseline training
  -> uncertainty interval + applicability
  -> activity-guided domain decision
  -> toxicity stage (mới ở contract/domain, chưa nối production)
```

Pipeline hiện đủ để tái lập việc chuẩn bị dữ liệu, huấn luyện baseline và thử
inference offline. Pipeline chưa được bật qua HTTP/API production vì baseline
chưa đạt chất lượng khoa học cần thiết.

## 2. Dữ liệu SAIR đã tải

Chỉ tải file metadata/activity, không tải corpus cấu trúc 3D:

- file: `data/sair/raw/sair.parquet`;
- Hugging Face revision:
  `8ab398166696ea8387fe099e925bfc76da319b60`;
- kích thước: 634.889.174 bytes;
- SHA-256:
  `10f76e8c9baf6ebd43d9a1bbab97c07a8e6da16311f8990db3e9635bb928ab28`;
- license metadata: `CC-BY-4.0`;
- file nằm dưới `data/` và đã được Git ignore.

Không tải `structures_compressed`, confidence archives hoặc pLDDT archives.

## 3. Kết quả kiểm tra schema thật

Schema inspector xác nhận file tương thích Phase-1 contract:

| Logical field | Cột SAIR |
|---|---|
| Entry identity | `entry_id` |
| Ligand | `SMILES` |
| Protein identity | `protein` |
| Protein sequence | `sequence` |
| Activity label | `pIC50` |
| Assay context | `assay` |
| Protein family | `family` |

Thống kê chính:

- 8.803.710 source rows;
- 9 Parquet row groups;
- 1.048.857 protein-ligand pairs;
- 1.760.742 measurement duy nhất sau khi loại exact duplicate;
- 5.203 protein identifiers;
- 5.149 protein sequences;
- 734.359 ligand SMILES;
- không có null ở protein, sequence, SMILES, source, assay hoặc pIC50;
- `description` là metadata tùy chọn và null ở 350.425 source rows;
- pIC50 nằm trong khoảng 4,0 đến 12,0.

File triển khai:

- `backend/predictor/research/sair/inspect_schema.py`;
- báo cáo cục bộ: `data/sair/metadata/sair-schema-report.json`.

## 4. ETL Phase-1 đã triển khai

File chính:

- `backend/predictor/research/sair/prepare_phase1.py`.

ETL thực hiện:

1. Xác minh checksum của source parquet.
2. Kiểm tra đầy đủ các cột bắt buộc.
3. Từ chối null hoặc pIC50 không hữu hạn ở trường bắt buộc.
4. Kiểm tra một `entry_id` không ánh xạ tới nhiều ligand/protein identity.
5. Loại exact duplicate do cùng measurement được lặp theo pose/structure.
6. Aggregate measurement về protein-ligand pair.
7. Lưu median, mean, standard deviation, min, max và span của pIC50.
8. Giữ measurement count, source count và assay count.
9. Đánh dấu label disagreement thay vì âm thầm loại thông tin.
10. Sinh split deterministic theo protein sequence.
11. Kiểm tra protein sequence không xuất hiện xuyên partition.
12. Ghi processed parquet và manifest có checksum.

### Policy label tạm thời

- nhãn học máy: median pIC50 sau khi loại exact duplicate;
- cặp có `pIC50 span > 1.0` được gắn
  `high_measurement_disagreement`;
- các cặp này không được đưa vào training **và không được đưa vào calibration**;
  chúng **vẫn nằm trong test**, vì bỏ chúng khỏi test sẽ biến kết quả thành số
  trên phần dễ của SAIR (xem `LABEL_FILTER_SPLITS`, issue #36);
- ETL hiện group theo `entry_id` và lưu `assay_primary=first`; nghĩa là
  **chưa** group theo assay context, nên median có thể gộp hai phép đo khác
  nhau. Đây là blocker khoa học, chưa được giải quyết (issue #39);
- policy được ghi trong manifest và vẫn cần scientific owner phê duyệt. Ngưỡng
  `1.0` là policy tạm, không phải một hằng số khoa học.

Kết quả:

- tổng pair rows: 1.048.857;
- training-eligible: 1.012.874;
- bị loại do label disagreement: 35.983;
- processed parquet: khoảng 27,6 MB;
- processed SHA-256:
  `a1e7ae9543c44a5f6ac4b17e2d236ae332bff3de48e1dc7c92eb21c573498f0a`.

Output cục bộ:

- `data/sair/processed/phase1_pairs.parquet`;
- `data/sair/metadata/phase1-manifest.json`.

## 5. Split và leakage guard

Đã triển khai generic split contract trong:

- `backend/predictor/research/sair/contracts.py`.

Các guard hiện có:

- duplicate `entry_id` không được xuất hiện nhiều lần trong assignment;
- ligand group không đi xuyên partition khi dùng ligand holdout;
- target group không đi xuyên partition khi dùng target holdout;
- ligand-target pair group không đi xuyên partition khi dùng pair holdout;
- replicate group không được đi xuyên partition với mọi strategy;
- manifest chỉ được sinh khi leakage audit đạt.

Split thật hiện dùng `exact_sequence_cold_target_hash_v1`:

| Partition | Pair rows đủ điều kiện | Protein sequences |
|---|---:|---:|
| Train | 828.976 | 4.115 |
| Validation | 92.842 | 506 |
| Test | 91.056 | 519 |

Số protein sequence đi xuyên partition: `0`.

Giới hạn: đây là exact-sequence holdout, chưa phải sequence-cluster hoặc
protein-family holdout. Các protein tương đồng cao vẫn có thể nằm ở nhiều
partition khác nhau.

## 6. Baseline `ACT-D2` đã huấn luyện

> **Tên gọi.** Tên `B0` từng được dùng cho hai thứ khác nhau: bảng roadmap
> trong plan gọi `B0` là *global/target median*, còn code lại gọi `B0` là
> *descriptor + amino-acid composition + HGB*. Hai mô hình này khác hẳn nhau
> về sức mạnh, nên câu "tốt hơn B0" trước đây không xác định được nghĩa. Từ
> nay dùng ID cố định (issue #38):
>
> | ID | Model |
> |---|---|
> | `ACT-D0` | Global median; target/context median khi target đã có trong train |
> | `ACT-D1` | Ligand-only ECFP + kNN/RF/boosting |
> | `ACT-D2` | Descriptor ligand + amino-acid composition + HGB — **chính là phần dưới đây** |
> | `ACT-D3` | ECFP + frozen protein embedding + boosted trees/MLP |
> | `ACT-M` | Ligand encoder + frozen protein encoder + regression head |
>
> `model_id` trong artifact vẫn là `sair-descriptor-hgb-b0` vì nó đã được ghi
> vào artifact/report; đổi nó là một thay đổi artifact riêng, không phải việc
> sửa tài liệu.

File triển khai:

- `backend/predictor/research/sair/baseline.py`.

Baseline sử dụng:

- ligand 2D descriptors từ RDKit;
- protein amino-acid composition và sequence length;
- `HistGradientBoostingRegressor`;
- deterministic sampling;
- validation residual để tạo prediction interval 90%;
- descriptor bounds 0,5–99,5 percentile cho applicability sơ bộ.

Tập benchmark:

- 100.000 train pairs;
- 25.000 validation pairs;
- 25.000 test pairs.

### Metric

> **Các số dưới đây đã hết hạn — chờ chạy lại.** Chúng được tạo khi
> `baseline.py` áp bộ lọc `training_eligible` cho **cả ba** split, nên hàng
> `Test` đo trên phần SAIR đã bỏ các pair có measurement bất đồng
> (`pic50_span > max_label_span`), không phải trên phân phối test đầy đủ.
> Bộ lọc giờ chỉ áp cho train/validation (xem `LABEL_FILTER_SPLITS`), và
> `train_baseline` báo hai view cạnh nhau: `metrics.test` trên toàn bộ test
> và `metrics.test_training_eligible_only` tương ứng với các số cũ. Cần chạy
> lại B0 rồi thay bảng này bằng cả hai view. Issue #36.

| Partition | MAE | RMSE | R² | Pearson |
|---|---:|---:|---:|---:|
| Train | 0,802 | 1,004 | 0,446 | 0,676 |
| Validation | 1,043 | 1,273 | 0,081 | 0,321 |
| Test (đã lọc span) | 1,024 | 1,250 | 0,143 | 0,406 |

Prediction interval 90%:

- half-width: khoảng 2,040 pIC50 — tức full width khoảng 4,08 pIC50;
- test coverage: 0,902, trên tập test **đã lọc** nói trên.

Half-width là **một** quantile của absolute residual trên validation, dùng
chung cho mọi cặp, nên coverage 0,902 không chứng minh uncertainty thích ứng
theo từng target hay candidate (issue #41).

Artifact:

- `data/sair/artifacts/sair-descriptor-hgb-b0.joblib`;
- kích thước khoảng 434 KB;
- model ID: `sair-descriptor-hgb-b0`;
- admission: `false`;
- lý do: descriptor B0 chỉ là benchmark, chưa phải production activity model.

Báo cáo metric cục bộ:

- `data/sair/metadata/sair-descriptor-hgb-b0-report.json`.

## 7. Activity Predictor contract

Đã định nghĩa wire contract nghiêm ngặt tại:

- `backend/control/src/toxagent/predictor/activity_schemas.py`.

Request gồm:

- ligand SMILES;
- target ID;
- optional UniProt ID;
- **assay context (bắt buộc)** — response luôn có trường này, nên request
  không có nó đồng nghĩa server tự chọn một context mà caller không thấy và
  không kiểm được (issue #34, ADR 0014);
- optional model selection.

Response gồm:

- input và canonical SMILES;
- resolved target identity;
- assay context;
- predicted pIC50;
- uncertainty;
- prediction interval;
- ligand, target và joint applicability;
- model ID, dataset ID, split ID và artifact hashes.

Contract từ chối:

- field không khai báo;
- NaN hoặc infinity;
- interval không chứa prediction;
- model ID giữa prediction và provenance không khớp.

Contract này chưa được công bố thành route vì chưa có admitted model.

## 8. Domain workflow đã triển khai

Các file domain được tách theo trách nhiệm:

- `backend/control/src/toxagent/domain/activity.py`: value object, estimate,
  provenance và immutable activity snapshot;
- `backend/control/src/toxagent/domain/prioritisation.py`: target-specific
  policy và prioritisation decision;
- `backend/control/src/toxagent/domain/screening.py`: state machine điều phối;
  đồng thời re-export public API cũ để caller hiện tại không bị vỡ sau refactor.

Các domain object:

- `TargetIdentity`;
- `ActivitySubject` — ligand chuẩn hóa + target identity + assay context; là
  identity mà **mọi** artifact của screening phải khớp (ADR 0014);
- `TargetRole` — vì sao target này được sàng lọc;
- `ActivityEstimate`;
- `ActivityModelProvenance`;
- `ActivitySnapshot`;
- `PrioritisationDecision`;
- `TargetPriorityPolicy` — khóa theo `(target_id, assay_context)`;
- `ContinuationAuthorisation` — quyền đi qua gate, do server cấp;
- `ActivityGuidedScreening`.

State machine:

```text
CREATED
  -> ACTIVITY_RUNNING
  -> ACTIVITY_COMPLETED
  -> PRIORITISED | REVIEW_REQUIRED | LOW_PRIORITY
  -> TOXICITY_RUNNING
  -> COMPLETED | TOXICITY_FAILED
```

Các trạng thái kết thúc riêng:

- `ACTIVITY_FAILED`;
- `TOXICITY_FAILED`;
- `COMPLETED`;
- `COMPLETED_WITH_TOXICITY_NOT_RUN`.

Quy tắc prioritisation hiện có:

- ngoài applicability domain: `REVIEW`;
- thiếu uncertainty hoặc interval: `REVIEW`;
- uncertainty vượt policy: `REVIEW`;
- toàn bộ interval trên target-specific threshold: `PRIORITISE`;
- toàn bộ interval dưới threshold: `LOW_PRIORITY`;
- interval cắt threshold: `REVIEW`.

Nếu `REVIEW` hoặc `LOW_PRIORITY` vẫn tiếp tục toxicity thì `start_toxicity`
đòi một `ContinuationAuthorisation` do server cấp, **không** phải hai chuỗi
actor/reason. Người phê duyệt lấy từ authenticated principal
(`application.policy.authorise_toxicity_continuation` kiểm role theo
`PolicySettings.screening_continuation_roles`); lý do được lưu làm dữ liệu
audit và tự nó không tạo ra quyền. Nghĩa là không có tham số nào để caller
hay một model tự khai người phê duyệt của chính nó (issue #35, ADR 0014).

Việc không chạy toxicity có trạng thái riêng và không bao giờ được biểu diễn
như kết quả “negative” hoặc “safe”.

ID prefix mới:

- `act_*`: activity snapshot;
- `pri_*`: prioritisation decision;
- `scr_*`: activity-guided screening.

## 9. ADR và tài liệu kiến trúc

Đã thêm:

- `docs/adr/0013-activity-guided-screening-boundary.md`;
- `docs/internal/spec/ACTIVITY_GUIDED_TOXICITY_SCREENING_IMPLEMENTATION_PLAN_VI.md`.

Quyết định chính:

- activity và toxicity là hai phép đo khác nhau;
- không đưa pIC50 vào toxicity endpoint hiện tại;
- không tạo activity-toxicity fusion score;
- control plane sở hữu workflow tuần tự;
- activity snapshot và toxicity analysis snapshot được lưu riêng rồi liên kết;
- chỉ artifact vượt admission gate mới được phục vụ.

## 10. Dependency và reproducibility

Đã bổ sung optional dependency group `sair` trong
`backend/predictor/pyproject.toml`:

- `pyarrow>=17`;
- `pandas`;
- `scikit-learn`.

Đã sửa control dependency thành `pyjwt[crypto]>=2.8` vì OIDC RSA thực sự cần
crypto backend.

Môi trường nghiên cứu hiện dùng:

- PyArrow 25.0.1;
- scikit-learn 1.7.2;
- SciPy 1.15.3;
- joblib 1.5.3;
- Ruff 0.13.3.

## 11. Test đã thêm

Các file test mới:

- `backend/control/tests/unit/test_activity_guided_screening.py`;
- `backend/predictor/tests/unit/test_sair_schema_contract.py`;
- `backend/predictor/tests/unit/test_sair_split_contract.py`;
- `backend/predictor/tests/unit/test_sair_phase1_etl.py`;
- `backend/predictor/tests/unit/test_sair_baseline.py`.

Các trường hợp được kiểm thử:

- activity regression semantics;
- uncertainty và interval validation;
- target-specific prioritisation;
- review/override audit;
- toxicity skipped không bị hiểu thành negative;
- state transition không hợp lệ;
- schema alias ambiguity;
- required SAIR fields;
- exact duplicate removal;
- label disagreement gate;
- identity conflict;
- deterministic cold-target split;
- ligand/target/pair/replicate leakage;
- deterministic baseline sampling;
- invalid SMILES/protein sequence;
- baseline inference interval và non-admission.

Kết quả verification **do tác giả báo cáo tại commit `681a1c1`**, trong môi
trường nghiên cứu nêu ở §10. Đây không phải số được một lần rà soát độc lập
xác nhận lại:

- control unit suite: `1901 passed`;
- predictor unit suite: `233 passed, 5 skipped`;
- SAIR regression cuối: `19 passed`;
- Ruff: `All checks passed`;
- documentation check: `documentation OK`;
- `git diff --check`: đạt.

Các số này đã thay đổi từ khi sửa #33/#34/#35/#36 — xem CHANGELOG hoặc chạy
lại suite. Khi cập nhật, ghi rõ đó là số đã chạy hay số được báo lại.

`pip check` trên môi trường dùng chung vẫn báo một số package ảnh y tế bên
ngoài ToxAgent thiếu optional dependencies. Các lỗi này không thuộc dependency
graph của control/predictor và không làm các test ToxAgent thất bại.

## 12. Những phần chưa hoàn tất

### 12.1. Scientific gates

- phê duyệt cách deduplicate và aggregate measurement;
- phê duyệt ngưỡng `pIC50 span > 1.0`;
- xác định assay context nào được phép trộn;
- tạo protein sequence clusters hoặc family holdout;
- định nghĩa primary metric và admission threshold;
- huấn luyện ligand-protein model mạnh hơn B0;
- calibration uncertainty/applicability theo target domain.

### 12.2. Production engineering

- activity model registry entry và artifact verification;
- production provider/load path;
- Activity Predictor HTTP route;
- control-plane activity client;
- database migration cho activity snapshot, decision và screening workflow;
- repository implementation;
- idempotent application service/worker orchestration;
- API cho activity-guided screening;
- frontend activity panel và prioritisation explanation;
- report sections tách activity khỏi toxicity;
- integration, contract và end-to-end tests;
- deployment configuration và observability.

## 13. Vì sao chưa nối production

Baseline có test R² khoảng 0,143 và prediction interval khoảng ±2,04 pIC50.
Việc dùng model này để quyết định có chạy toxicity hay không có nguy cơ loại
sai candidate. Vì vậy code cố ý giữ artifact ở trạng thái `admitted=false` và
không tạo production route.

Đây là một quality gate, không phải thiếu wiring đơn thuần. Hướng đúng tiếp
theo là huấn luyện/evaluate ligand-protein model tốt hơn, đóng băng split khoa
học, đạt admission criteria rồi mới nối persistence, API và UI.

## 14. Trạng thái Git

Thay đổi được mô tả trong báo cáo này đã commit tại
`681a1c1b92377071411a832762db4203294931f2` trên nhánh
`feat/bioactivity-first-pipeline-redesign`. Các sửa chữa sau rà soát
(#33/#34/#35/#36) nằm trên nhánh `fix/activity-identity-guards` dựng từ commit
đó.

Dataset, processed parquet, model artifact và metadata runtime nằm dưới
`data/` và được Git ignore. Vì vậy các số trong §6 **chưa tái lập được** từ
riêng repo: cần tải lại SAIR và chạy lại pipeline.
