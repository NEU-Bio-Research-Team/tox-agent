# Kế hoạch triển khai Activity-Guided Toxicity Screening cho ToxAgent

**Trạng thái:** Contract/domain và research Phase-1 đã chạy; chưa production
**Nhánh:** `feat/bioactivity-first-pipeline-redesign`
**Ngày cập nhật:** 2026-10-20
**Phạm vi:** SAIR activity screening trước hERG/Tox21 toxicity screening

## 1. Tóm tắt quyết định

ToxAgent sẽ chuyển từ cách chạy activity và toxicity song song rồi cố gắng
late-fusion hai output khác nghĩa sang một workflow nối tiếp:

```text
Input molecule + target
          │
          ▼
Canonical structure + target resolution
          │
          ▼
Activity Predictor (SAIR)
pIC50 + uncertainty + applicability
          │
          ▼
Auditable Prioritisation Policy
LOW_PRIORITY / REVIEW / PRIORITISE
          │
          ├── LOW_PRIORITY ──► Không chạy toxicity; ghi rõ lý do
          │
          ├── REVIEW ────────► Người dùng hoặc policy quyết định
          │
          └── PRIORITISE ────► Current ToxPred
                                      │
                               hERG + 12 Tox21
                                      │
                                      ▼
                         Evidence + Activity–Safety Report
```

Activity không được dùng làm feature để suy ra toxicity. Activity chỉ điều
khiển candidate có được chuyển tiếp sang toxicity screening hay không.

Không tạo fusion score giữa pIC50 và xác suất hERG/Tox21. Không tạo kết luận
toàn cục `TOXIC/NON-TOXIC`.

## 2. Hiện trạng repository

Repository đã có hai phần có thể tái sử dụng nhưng chưa được nối thành workflow
activity-first.

### 2.1. Bioactivity research hiện có

`backend/predictor/research/bioactivity/` và
`backend/predictor/evals/bioactivity/` đã có:

- thu thập và chuẩn hóa dữ liệu ChEMBL;
- ECFP4 và descriptor featurization;
- B0 median, B1 kNN, B2a Random Forest, B2b LightGBM;
- temporal, cluster, scaffold và random split;
- MAE, RMSE, Pearson, Spearman, enrichment và activity-cliff metrics;
- frozen split manifest và kiểm tra dataset SHA-256;
- OOD/similarity analysis;
- unit test cho data contract và leakage invariant.

Phần này là research artifact, chưa được admit để serving. Model hiện tại cũng
là fixed-target benchmark, chưa có contract `ligand + protein -> pIC50` như SAIR.

### 2.2. Toxicity production hiện có

Predictor production hiện phục vụ:

- hERG blockade probability;
- 12 Tox21 assay activity probabilities;
- endpoint-specific thresholds;
- chemical applicability guard;
- attribution và provenance;
- immutable analysis snapshot trong control plane.

Các contract hiện tại giả định output classification là xác suất trong `[0, 1]`.
Do đó pIC50 regression không được thêm trực tiếp như một toxicity endpoint.

### 2.3. SAIR hiện chưa có trong repository

Repository hiện không có `sair.parquet`, SAIR manifest hoặc SAIR-specific
loader. Trước khi viết model cần xác minh dataset thật, license và schema thực
tế. Không được xây loader chỉ dựa trên tên field được mô tả trong report.

## 3. Các nguyên tắc bất biến

1. **Activity và toxicity là hai phép đo khác nhau.**
2. **pIC50 không phải xác suất toxicity.**
3. **Activity không được dùng để suy ra safety.**
4. **hERG và từng Tox21 assay phải giữ độc lập.**
5. **Không đếm Tox21 hit thành toxicity severity.**
6. **Không chạy toxicity không có nghĩa là không có toxicity.**
7. **Prioritisation là policy, không phải output thô của model.**
8. **Mọi threshold phải có target/task context, version và provenance.**
9. **Attribution giải thích model, không chứng minh mechanism hoặc causality.**
10. **Stage lỗi hoặc bị skip phải được biểu diễn rõ, không được thay bằng giá
    trị mặc định hoặc kết luận an toàn.**

## 4. Boundary kiến trúc

### 4.1. Không mở rộng `ToxicityPredictor` bằng pIC50

Activity và toxicity có request/response, dependency, artifact và scaling khác
nhau. Cần một bounded context `ActivityPredictor` riêng.

Trong MVP, activity code có thể chạy cùng deployment với predictor để giảm chi
phí vận hành, nhưng phải tách:

- domain types;
- API route;
- model registry;
- artifact admission;
- readiness;
- metrics và evaluation contract.

Boundary này cho phép tách activity thành service riêng sau này mà không thay
đổi contract của control plane.

### 4.2. Toxicity predictor được giữ nguyên

Không retrain toxicity trong MVP. Workflow gọi predictor hiện có bằng canonical
SMILES và nhận hERG/Tox21 như hiện tại.

### 4.3. Control plane sở hữu orchestration

Control plane, không phải activity model và không phải toxicity model, sở hữu:

- thứ tự các stage;
- prioritisation policy;
- retry/idempotency;
- manual override;
- persistence và audit trail;
- quyết định gọi hoặc không gọi toxicity;
- composite Activity–Safety Report.

## 5. SAIR data contract

### 5.1. Gate trước khi triển khai

Trước khi viết training pipeline phải xác minh:

- quyền truy cập và license/redistribution;
- schema thực tế của file;
- checksum và dataset version;
- số row, ligand, target, protein và pose duy nhất;
- pIC50 có phải experimental ground truth hay có label suy diễn;
- cách xác định protein identity;
- assay/source context;
- quan hệ giữa một activity measurement và nhiều predicted pose;
- missingness của structural score và confidence field;
- khả năng tái tạo train/evaluation split.

Output của bước này là một tracked manifest, không phải commit toàn bộ dataset.

### 5.2. Phase 1 canonical row

Phase 1 không dùng 3D pose làm input. Một canonical activity row dự kiến gồm:

```text
measurement_id
entry_id
ligand_key
canonical_smiles
target_id
uniprot_id
protein_sequence
protein_sequence_sha256
protein_family
assay_type
assay_context_key
data_source
pIC50
provenance
```

Nếu nhiều pose tham chiếu cùng một activity measurement, các pose không tạo
thêm label mới. Chúng chỉ được liên kết để dùng ở phase structure-aware sau này.

### 5.3. Chuẩn hóa ligand

Áp dụng cùng nguyên tắc đang dùng ở bioactivity research:

- parse SMILES bằng RDKit;
- chuẩn hóa parent structure;
- xử lý salt/counterion;
- giữ stereochemistry trong structure representation;
- dùng connectivity key cho leakage grouping;
- không tự ý sửa coordination complex thành ligand fragment;
- reject hoặc flag cấu trúc không hợp lệ với reason code.

### 5.4. Chuẩn hóa protein

- Resolve target sang stable identifier, ưu tiên UniProt accession.
- Normalize sequence alphabet và loại whitespace.
- Lưu sequence SHA-256.
- Không tin sequence do client gửi mà không có provenance.
- Unknown hoặc ambiguous target phải trả typed error.
- Nếu cho phép explicit sequence ở research mode, phải lưu nguyên sequence hash,
  source và resolution status.

### 5.5. Assay context

pIC50 chỉ có thể so sánh đúng khi target và assay context phù hợp. Không được
pool tất cả record của cùng protein mà bỏ qua:

- assay type;
- source dataset;
- construct/variant;
- experimental context nếu SAIR cung cấp;
- measurement definition.

Context phải được đưa vào model hoặc dữ liệu phải được filter thành một contract
đồng nhất.

### 5.6. Missing structural fields

Ở phase structure-aware, missing value không được biến thành zero. Mỗi feature
cần:

```text
value + missing_indicator
```

Pose không đạt quality gate phải được exclude hoặc down-weight theo policy có
version; không được coi chất lượng ngang pose hợp lệ.

## 6. Split contract và leakage prevention

Chỉ group theo `entry_id` là chưa đủ. Cùng ligand hoặc cùng ligand–protein system
có thể xuất hiện dưới nhiều entry.

Các key chuẩn:

```text
ligand_key = standardized parent connectivity key
target_key = canonical protein identity
system_key = ligand_key + target_key + assay_context_key
```

### 6.1. Evaluation views

| View | Mục tiêu |
|---|---|
| Grouped system | Không tách pose/measurement của cùng system |
| Scaffold | Đánh giá ligand scaffold mới |
| Cold target | Đánh giá protein chưa xuất hiện trong train |
| Protein cluster | Ngăn homolog protein bị rải qua train/test |
| Temporal | Dùng nếu provenance thời gian đáng tin cậy |
| Random | Chỉ diagnostic, không dùng quyết định release |

Protein split nên dựa thêm trên sequence cluster để tránh trường hợp hai protein
gần như giống nhau nằm ở train và test nhưng vẫn được gọi là cold-target.

### 6.2. Split invariants

- Không pose nào của cùng measurement nằm ở nhiều split.
- Không `system_key` nằm ở nhiều split.
- Scaffold view không rò ligand scaffold theo rule đã khai báo.
- Cold-target view không rò target identity.
- Protein-cluster view không rò sequence cluster.
- Split manifest chứa dataset hash, rule version, seed và counts.
- Benchmark chỉ đọc frozen split; tuyệt đối không tự re-split.

## 7. Model tournament Phase 1

Không bắt đầu bằng model sâu trước khi có baseline.

| ID | Model | Mục tiêu |
|---|---|---|
| B0 | Median pIC50 theo target/assay | Dummy floor |
| B1 | ECFP4 ligand + target identity | Known-target classical baseline |
| B2 | ECFP4 + frozen protein embedding + LightGBM/MLP | Cross-target baseline |
| B3 | Ligand encoder + frozen protein encoder + regression head | Neural pair model |
| B4 | Fine-tuned ligand/protein encoders | Chỉ sau khi B3 đạt gate |

Protein embedding cần precompute và cache theo sequence SHA-256. Không encode lại
cùng một protein cho từng ligand row.

Không bắt buộc activity và toxicity dùng chung encoder. Shared encoder chỉ được
nghiên cứu nếu benchmark sau này chứng minh lợi ích.

## 8. Evaluation contract

### 8.1. Metrics chính

- MAE;
- RMSE;
- Pearson;
- Spearman;
- activity-cliff direction accuracy;
- per-target performance;
- per-protein-family performance;
- per-assay-context performance;
- scaffold/cold-target/protein-cluster performance;
- uncertainty coverage và interval width;
- applicability-stratified performance.

Không chọn model bằng random split. Primary release view phải được pre-register
sau khi xác minh metadata SAIR; ưu tiên temporal hoặc cold-target/protein-cluster
tùy product claim.

### 8.2. Artifact admission gates

Một activity checkpoint chỉ được serving khi có:

- frozen dataset và split manifest;
- performance tốt hơn B0/B1 trên primary view;
- không suy giảm nghiêm trọng trên OOD view;
- uncertainty được đánh giá;
- model/config/tokenizer/encoder hashes;
- ligand feature schema version;
- protein encoder revision;
- target catalog version;
- golden inference parity;
- documented limitations;
- reproducible model card.

## 9. Serving contract

### 9.1. Activity request

Ưu tiên target catalog thay vì nhận raw protein sequence không định danh:

```json
{
  "smiles": "...",
  "target": {
    "target_id": "EGFR",
    "uniprot_id": "P00533"
  },
  "model_id": "sair-ligand-protein-v1"
}
```

Research mode có thể cho explicit sequence, nhưng sequence hash và provenance là
bắt buộc.

### 9.2. Activity response

```json
{
  "input_smiles": "...",
  "canonical_smiles": "...",
  "target": {
    "target_id": "EGFR",
    "uniprot_id": "P00533",
    "sequence_sha256": "..."
  },
  "prediction": {
    "predicted_pic50": 8.1,
    "uncertainty": 0.42,
    "prediction_interval": [7.3, 8.9],
    "model_id": "sair-ligand-protein-v1"
  },
  "applicability": {
    "ligand": "in_domain",
    "target": "in_domain",
    "joint": "in_domain",
    "reasons": []
  },
  "provenance": {}
}
```

Response này chưa chứa `PRIORITISE`; model không sở hữu quyết định workflow.

### 9.3. API dự kiến

Không thay đổi `/v1/predict` hiện tại. Thêm:

```text
POST /v1/activity/predictions
POST /v1/screenings
GET  /v1/screenings/{screening_id}
POST /v1/screenings/{screening_id}/continue-to-toxicity
```

## 10. Prioritisation policy

### 10.1. Output

```text
PRIORITISE
REVIEW
LOW_PRIORITY
```

### 10.2. Input của policy

- predicted pIC50;
- uncertainty/prediction interval;
- ligand applicability;
- target applicability;
- joint applicability;
- target/task-specific threshold;
- assay context;
- policy version;
- optional expert override.

### 10.3. Nguyên tắc threshold

Không dùng một threshold toàn cục như `pIC50 >= 7`. Threshold phải là:

- per target;
- per task/assay context; hoặc
- percentile của training distribution với provenance rõ.

Threshold là policy artifact, không phải constant rải trong application code.

### 10.4. Quy tắc an toàn workflow

- `PRIORITISE`: tự động chuyển sang toxicity theo policy.
- `REVIEW`: dừng chờ quyết định hoặc chạy toxicity theo explicit review policy.
- `LOW_PRIORITY`: mặc định skip toxicity trong batch; hiển thị rõ `not_run`.
- `force_toxicity=true`: chỉ role được phép; phải audit actor và reason.
- Activity prediction lỗi/OOD không được tự biến thành `LOW_PRIORITY`; mặc định
  phải là `REVIEW` hoặc failed stage.

## 11. Domain và persistence

### 11.1. Activity snapshot

```text
ActivitySnapshot
├── input_smiles
├── canonical_smiles
├── target_id
├── target_sequence_hash
├── assay_context
├── predicted_pic50
├── uncertainty
├── prediction_interval
├── ligand_applicability
├── target_applicability
├── joint_applicability
├── model_id
├── artifact hashes
└── dataset/split provenance
```

### 11.2. Prioritisation decision

```text
PrioritisationDecision
├── activity_snapshot_id
├── status
├── policy_id
├── threshold/value/source
├── reason_codes
├── overridden
├── override_actor
└── created_at
```

### 11.3. Workflow aggregate

```text
ActivityGuidedScreening
├── molecule identity
├── target identity
├── activity_snapshot_id
├── prioritisation_decision_id
├── toxicity_analysis_id (nullable)
├── stage statuses
└── failure/skip reasons
```

Không nhét activity vào `AnalysisSnapshot` toxicity hiện tại. Composite workflow
tham chiếu activity snapshot và toxicity analysis snapshot độc lập.

Các table dự kiến:

- `activity_snapshots`;
- `prioritisation_decisions`;
- `activity_guided_screenings`.

Không backfill analysis cũ. Feature được đưa vào sau flag.

## 12. Workflow state machine

```text
CREATED
  │
  ▼
ACTIVITY_RUNNING
  │
  ├──► ACTIVITY_FAILED
  │
  ▼
ACTIVITY_COMPLETED
  │
  ├──► LOW_PRIORITY ──► COMPLETED_WITH_TOXICITY_NOT_RUN
  │
  ├──► REVIEW_REQUIRED
  │         │
  │         └──► TOXICITY_RUNNING (explicit continuation)
  │
  └──► PRIORITISED ──► TOXICITY_RUNNING
                              │
                              ├──► TOXICITY_FAILED
                              │
                              ▼
                           COMPLETED
```

Mỗi stage phải:

- có idempotency key riêng;
- retry độc lập;
- checkpoint ngay khi thành công;
- không chạy lại stage đã hoàn thành;
- có timeout/budget riêng;
- lưu typed failure/skip reason;
- phát event cho UI timeline.

## 13. Frontend

Frontend hiển thị stage theo thứ tự:

```text
Molecule + Target
      ↓
Activity Result
      ↓
Prioritisation Decision
      ↓
Toxicity Screening
      ↓
Evidence / Report
```

### 13.1. Activity card

- target identity;
- predicted pIC50;
- prediction interval;
- applicability;
- model/version;
- activity explanation khi có;
- limitations.

### 13.2. Prioritisation card

- `PRIORITISE`, `REVIEW` hoặc `LOW_PRIORITY`;
- policy ID;
- reason codes;
- threshold source;
- override status;
- nút tiếp tục toxicity khi được phép.

### 13.3. Toxicity stage

- dùng UI hERG/Tox21 hiện tại;
- nếu chưa chạy, hiển thị `Not run` và lý do;
- không render placeholder như một prediction âm tính;
- không tạo activity–toxicity fusion gauge.

## 14. Activity–Safety Report

Report mới gồm:

1. Molecule profile;
2. Target/protein profile;
3. Activity prediction;
4. Activity uncertainty và applicability;
5. Activity explanation;
6. Candidate prioritisation;
7. Toxicity stage status;
8. hERG và từng Tox21 assay nếu đã chạy;
9. Toxicity explanation;
10. External evidence;
11. Activity–safety interpretation;
12. Limitations và provenance.

Report validator phải chặn:

- “activity predicts toxicity”;
- “low activity means safe”;
- cộng/trung bình pIC50 với toxicity probabilities;
- đếm Tox21 hit thành severity;
- suy diễn `TOXIC/NON-TOXIC` toàn cục;
- coi stage chưa chạy là negative result;
- diễn giải attribution thành causal mechanism.

## 15. XAI và structure-aware phase

### 15.1. Activity XAI

Chỉ thêm sau khi activity model vượt benchmark gate. Các lớp giải thích phải tách:

- ligand atom/substructure attribution;
- protein residue attribution;
- protein–ligand contact attribution ở phase 3D.

XAI phải được benchmark với random hoặc counterfactual control. Không ship heatmap
chỉ vì model có gradient.

### 15.2. 3D augmentation

Sau MVP mới thêm:

- predicted complex representation;
- Boltz confidence;
- docking scores;
- PoseBusters quality flags;
- pocket/contact features.

Ablation bắt buộc:

```text
SMILES only
vs
SMILES + protein
vs
SMILES + protein + 3D
```

3D chỉ được giữ nếu cải thiện primary/OOD view, không chỉ random split.

## 16. Lộ trình triển khai

### Milestone 0 — SAIR dataset validation

- lấy dataset thật;
- xác minh license và schema;
- profile cardinality/missingness/duplicates;
- xác minh semantics pIC50;
- tạo dataset manifest/checksum;
- chọn canonical row definition.

**Exit gate:** Có schema và manifest được review; chưa cần model.

### Milestone 1 — Data contract và frozen splits

- ligand/protein standardization;
- assay-context normalization;
- pose-to-measurement collapse;
- grouped/scaffold/cold-target/protein-cluster split;
- leakage tests;
- tracked split manifest.

**Exit gate:** Không có leakage theo declared invariants.

### Milestone 2 — Baseline benchmark

- B0/B1/B2;
- metric suite;
- activity-cliff/OOD analysis;
- per-target/per-family report;
- frozen benchmark result.

**Exit gate:** Có baseline đáng tin để model sâu phải vượt qua.

### Milestone 3 — Ligand–protein model

- B3 architecture;
- precomputed protein embeddings;
- uncertainty method;
- model tournament;
- artifact/model card.

**Exit gate:** Model vượt admission gates trên primary/OOD view.

### Milestone 4 — Activity serving

- typed API;
- separate model registry;
- artifact verification;
- readiness;
- golden parity;
- batch inference.

**Exit gate:** Stable activity service contract.

### Milestone 5 — Sequential orchestration

- activity client trong control plane;
- activity snapshot persistence;
- versioned prioritisation policy;
- screening state machine;
- reuse current toxicity analysis;
- retry/idempotency/events.

**Exit gate:** Một candidate chạy end-to-end activity -> gate -> toxicity.

### Milestone 6 — Frontend và report

- stage timeline;
- activity/prioritisation cards;
- explicit toxicity-not-run state;
- composite report;
- validation rules.

**Exit gate:** UI và report không làm sai semantics khoa học.

### Milestone 7 — XAI và 3D

- ligand/protein attribution;
- XAI benchmark;
- 3D feature pipeline;
- ablation;
- pose quality policy.

**Exit gate:** Chỉ ship feature có bằng chứng cải thiện và explanation validation.

## 17. Cấu trúc code dự kiến

```text
backend/activity/
├── pyproject.toml
├── research/
│   └── sair/
│       ├── inspect_schema.py
│       ├── standardize.py
│       ├── build_dataset.py
│       ├── split.py
│       └── models/
├── evals/
│   └── activity/
│       ├── metrics.py
│       ├── run_benchmark.py
│       ├── stress/
│       └── manifests/
├── registry/
│   └── models/
└── src/activitypred/
    ├── api/
    ├── application/
    ├── domain/
    └── scientific/

backend/control/src/toxagent/
├── activity/
│   ├── client.py
│   └── schemas.py
├── application/screening/
│   ├── create_activity_screening.py
│   ├── prioritise_candidate.py
│   ├── continue_to_toxicity.py
│   └── project_screening.py
└── domain/
    └── screening.py

frontend/src/features/screening/
├── ActivityStageCard.tsx
├── PrioritisationCard.tsx
├── ToxicityStageCard.tsx
└── ScreeningTimeline.tsx
```

Vị trí process/deployment cuối cùng của `backend/activity` có thể thay đổi sau
ADR. Điều bất biến là activity phải có bounded context và contract riêng.

## 18. Test matrix bắt buộc

### Data tests

- schema mismatch fail closed;
- dataset hash mismatch fail closed;
- invalid SMILES có reason code;
- pose của cùng measurement không tách split;
- cùng system không tách split;
- cùng ligand không leak qua entry khác;
- protein homolog không leak qua protein-cluster view;
- missing structural value không biến thành zero;
- assay context không bị collapse âm thầm.

### Model tests

- output pIC50 hữu hạn;
- batch giữ đúng order;
- unknown target trả typed error;
- artifact/model/encoder revision khớp registry;
- golden inference parity;
- uncertainty interval có coverage report;
- primary metric tốt hơn baseline theo gate đã đăng ký.

### Policy tests

- threshold target-specific;
- OOD/uncertain không tự `PRIORITISE`;
- activity failure không biến thành `LOW_PRIORITY`;
- override cần role và reason;
- policy snapshot tái lập cùng quyết định.

### Orchestration tests

- `PRIORITISE` gọi toxicity đúng một lần;
- `REVIEW` không tự tiếp tục khi policy không cho phép;
- `LOW_PRIORITY` tạo explicit `not_run`;
- retry không chạy lại stage đã hoàn thành;
- toxicity failure không xóa activity result;
- stage event đúng thứ tự;
- legacy `/v1/predict` không thay đổi.

### Report/UI tests

- toxicity chưa chạy không hiển thị negative;
- không có fusion score;
- không có global toxic/non-toxic verdict;
- activity và toxicity XAI hiển thị riêng;
- report claim có đúng observation/provenance;
- validator chặn wording sai semantics.

## 19. Ngoài phạm vi MVP

- full 3D complex encoder;
- pocket/contact XAI;
- ToxCast;
- shared activity–toxicity encoder;
- causal learning;
- matched molecular pair optimisation;
- global toxic/non-toxic classifier;
- activity–toxicity fusion score;
- retrain toxicity model.

## 20. Rủi ro và câu hỏi mở

1. SAIR dataset có được phép tải, lưu và phát hành artifact dẫn xuất không?
2. pIC50 trong SAIR đồng nhất đến mức nào giữa ChEMBL và BindingDB?
3. Assay context có đủ để tạo task đáng tin không?
4. Protein split nào phù hợp product claim: exact target, sequence cluster hay
   family holdout?
5. Có metadata thời gian đủ tốt để temporal split làm primary không?
6. Target catalog sẽ lấy từ đâu và version thế nào?
7. Uncertainty method nào phù hợp: ensemble, conformal hay calibrated residual?
8. Policy prioritisation do product owner hay scientific owner phê duyệt?
9. `REVIEW` mặc định dừng hay vẫn chạy toxicity trong single-molecule mode?
10. Activity service dùng cùng deployment hay tách service ngay từ đầu?

Các câu hỏi này phải được đóng bằng ADR hoặc manifest; không giải quyết bằng
constant ẩn trong code.

## 21. Definition of Done cho MVP

MVP hoàn tất khi:

- SAIR data contract và split manifest được đóng băng;
- activity model vượt registered baseline trên primary/OOD view;
- activity inference có uncertainty, applicability và provenance;
- prioritisation policy target-aware và có version;
- workflow activity -> prioritisation -> toxicity chạy idempotent;
- current hERG/Tox21 predictor được tái sử dụng không đổi semantics;
- toxicity skipped/failed được biểu diễn rõ;
- UI và report tách activity khỏi toxicity;
- không có fusion score hoặc global toxicity verdict;
- contract, integration, report và frontend tests đều pass;
- legacy prediction flow vẫn pass regression suite.

## 22. Bước code đầu tiên được đề xuất

Không bắt đầu bằng neural model hoặc frontend. Thứ tự đầu tiên:

1. Viết ADR xác nhận boundary activity/toxicity và vị trí deployment.
2. Nhận file SAIR thật và tạo schema profiler read-only.
3. Tạo dataset manifest với checksum và license metadata.
4. Định nghĩa canonical activity row và target identity.
5. Viết leakage-safe split contract cùng unit tests.
6. Chạy B0 baseline trước khi triển khai ligand–protein model.

Chỉ sau khi các bước trên được review mới bắt đầu model serving và control-plane
orchestration.

## 23. Trạng thái triển khai ban đầu — 2026-10-20

Đã triển khai lớp contract/domain ban đầu:

- ADR 0013 ở trạng thái `Proposed`, xác nhận activity và toxicity là hai phép đo
  tách biệt trong một workflow tuần tự;
- contract request/response nghiêm ngặt cho Activity Predictor tương lai, gồm
  target identity, pIC50 regression, uncertainty, prediction interval,
  applicability và provenance;
- domain types bất biến cho activity snapshot, policy prioritisation theo target
  và state machine `activity -> prioritisation -> toxicity`;
- trạng thái kết thúc riêng cho trường hợp toxicity không chạy, không diễn giải
  nhầm thành kết quả âm tính/an toàn;
- manual review/override muốn tiếp tục toxicity phải lưu actor và lý do;
- SAIR parquet schema inspector read-only: kiểm tra metadata, map alias theo
  fail-closed, ghi checksum/source/version/license và không tải corpus 3D;
- canonical Phase-1 activity-row contract và split-manifest builder fail-closed;
  leakage audit chặn duplicate entry, group ligand/target/pair và replicate
  xuất hiện xuyên partition;
- unit tests bằng synthetic fixtures cho policy, workflow, wire contract và
  schema mapping. Các fixture này không phải dữ liệu huấn luyện và không chứng
  minh model SAIR đã tồn tại.

Chưa triển khai hoặc chưa được phê duyệt cho production:

- review khoa học cho aggregation, assay semantics và label-disagreement policy;
- ligand–protein model đủ mạnh, uncertainty/applicability calibration theo
  target domain và model registry/admission;
- database migration/repository, predictor client/route, orchestration worker,
  UI và report tích hợp;
- production enablement. Hiện tại không có route nào quảng bá activity model.

## 24. Kết quả chạy SAIR thật — 2026-10-20

Đã tải duy nhất `sair.parquet` tại revision
`8ab398166696ea8387fe099e925bfc76da319b60`; checksum SHA-256 là
`10f76e8c9baf6ebd43d9a1bbab97c07a8e6da16311f8990db3e9635bb928ab28`.
Không tải corpus cấu trúc 3D.

Schema thật tương thích contract Phase-1 và có 8.803.710 dòng. Các dòng này
không phải 8,8 triệu cặp độc lập: chúng rút về 1.048.857 protein–ligand pairs
và 1.760.742 measurement duy nhất sau khi loại lặp do pose/structure. Có
35.983 pairs có biên độ measurement pIC50 lớn hơn 1,0; ETL tạm loại các pairs
này khỏi training thay vì lấy trung bình im lặng.

Phase-1 artifact hiện có:

- 1.048.857 pair rows;
- 1.012.874 rows đủ điều kiện theo policy tạm `pIC50 span <= 1.0`;
- exact-sequence cold-target split: 828.976 train, 92.842 validation và
  91.056 test rows;
- không có protein sequence đi xuyên partition;
- processed parquet khoảng 27,6 MB, có checksum và manifest.

Baseline B0 dùng 100.000 train, 25.000 validation và 25.000 test rows. Kết quả
test: MAE 1,024 pIC50, RMSE 1,250, R² 0,143, Pearson 0,406; prediction interval
90% có coverage 0,902. Đây chỉ là descriptor benchmark để kiểm tra pipeline.
Artifact được gắn `admitted=false` vì chất lượng chưa đủ để phục vụ production.

Các gate tiếp theo trước khi nối API production:

- scientific owner phê duyệt aggregation và ngưỡng label disagreement;
- thay exact-sequence hash bằng sequence-cluster/family holdout được đóng băng;
- huấn luyện ligand–protein model mạnh hơn và so sánh với B0;
- calibration/applicability theo từng miền target;
- admission criteria được phê duyệt. Chỉ artifact vượt các gate này mới được
  đăng ký vào serving và control-plane orchestration.
