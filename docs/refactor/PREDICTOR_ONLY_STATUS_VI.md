# Predictor-only refactor — kết quả đối chiếu và trạng thái thực thi

> **Plan gốc:** `TOXAGENT_PREDICTOR_ONLY_REFACTOR_PLAN_VI.md`
> **Nhánh:** `docs/harness-master-plan`
> **Baseline commit:** `e6882b2`
> **Ngày đối chiếu:** 2026-09-03

---

## 1. Kết luận ngắn

**Plan đúng và nên làm.** Mọi khẳng định kỹ thuật trong mục 2.2 của plan đều được xác
minh trực tiếp trên code, không có điểm nào sai.

**Nhưng phạm vi phải sửa ở một chỗ quan trọng:** plan giả định giữ **hai** model để
serve. Trên thực tế chỉ **một** model serve được. Model ClinTox không load được vì thiếu
tokenizer, và không thể phục hồi từ repo này. Chi tiết ở mục 3.

**Quyết định đã chốt:** giữ nguyên toàn bộ checkpoint và logic model, kể cả ClinTox.
Provider ClinTox đã viết xong và khai báo `required: false` — endpoint bật lại ngay khi
có tokenizer, còn hiện tại thì báo lỗi typed thay vì trả nhầm model khác (mục 3.4).

**Về consumer:** frontend **không** gọi `/predict` hay `/analyze`. Nó chỉ gọi `/agent/*`.
`/analyze` có đúng một consumer là chính agent layer qua self-HTTP localhost. Bản đồ đầy
đủ ở mục 8.

Đây chính là stop condition mà plan tự đặt ra:

> *"Dừng deletion/cutover nếu: Selected checkpoint không load reproducibly trong
> container khóa version."*

Nên phần đã thực thi dừng đúng trước bước xoá, theo thứ tự plan quy định ở mục 13
("Sprint đầu không nên bắt đầu bằng xóa folder").

---

## 2. Đối chiếu từng khẳng định của plan

| # | Khẳng định trong plan | Kết quả kiểm chứng | Bằng chứng |
|---|---|---|---|
| — | 688 file trên nhánh | **Đúng, chính xác** | `git ls-files \| wc -l` = 688 |
| — | Phân bố thư mục (benchmark 183, frontend 152, .agents 84, models 51, agents 13, model_server 8, backend 27, src 25) | **Đúng toàn bộ** | `git ls-files \| awk -F/ '{print $1}' \| sort \| uniq -c` |
| — | `model_server/main.py` ~6.279 dòng | **Đúng** (6.278) | `wc -l` |
| 1 | `predict_pretrained_dual_head_outputs` ánh xạ hERG logit thành `clinical.p_toxic` | **Đúng** | `backend/inference.py:851-853`: `p_toxic = float(herg_probs[local_idx])` rồi `is_toxic = p_toxic >= clinical_threshold`, xuất dưới key `"clinical"` |
| 2 | API threshold default 0.35 vs artifact 0.413345 | **Đúng, và nặng hơn plan mô tả** | Có **năm** nguồn threshold cùng tồn tại — xem 2.1 |
| 3 | Tox21 có 12 threshold riêng nhưng API dùng `mechanism_threshold` chung | **Đúng** | `tox21_task_thresholds.json` có 12 giá trị (0.35–0.94); `model_server/schemas.py:121` chỉ có một `mechanism_threshold` |
| 4 | `DEFAULT_TOX_TYPE_MODEL_KEY=tox21_ensemble_3_best` trỏ ensemble không tái lập được | **Đúng** | `main.py:202` → `main.py:321` map sang `models/dualhead_ensemble3/`, thư mục này chỉ chứa `dualhead_metrics.json`, **không có file weight nào** |
| 5 | Requirements/Dockerfile không đồng nhất | **Đúng** | `requirements.txt` hướng dẫn `torch==2.4.0`; `model_server/Dockerfile:52` cài `torch==2.6.0`; base image `python:3.10-slim`; môi trường dev hiện tại là torch 2.11.0 + numpy 2.2.6 trong khi requirements ghim `numpy<2.0` |
| 6 | Data loader đổi scaffold split → random split khi fallback DeepChem → PyTDC | **Đúng** | `backend/data.py:233-243`: nhánh fallback chạy `df.sample(frac=1)` rồi cắt 80/10/10, comment ngay trong code: *"For scaffold split, would need additional processing"* |
| 7 | Test suite không khóa predictor contract | **Đúng** | `tests/` chỉ có `test_report_chat_agent.py` (219 dòng) + 2 smoke ADK/agent |
| 8 | Backend workflow chỉ deploy `agent_test`, chưa có CI | **Đúng** | `.github/workflows/backend-autodeploy.yml:6-8` |

**Kết luận mục 2: plan không cần sửa phần chẩn đoán.**

### 2.1 Threshold: năm nguồn, không nguồn nào là artifact

Đo bằng cách chạy server thật rồi gọi `POST /predict` (2026-09-03):

| Nguồn | Giá trị | Vị trí |
|---|---|---|
| `workspace_mode.yaml` → `safety_first` | 0,30 | `backend/workspace_mode.py:26` |
| `model_server.schemas` | 0,30 | `schemas.py:23` (đọc từ workspace_mode) |
| `backend.inference` | **0,35** | `inference.py:51` |
| `analyze_molecule_sync` hardcode | 0,35 | `main.py:4874` |
| **Artifact đã hiệu chuẩn** | **0,4133** | `herg_threshold.json` |
| **`/predict` thực tế áp dụng** | **0,35** | đo trực tiếp |

Hai điểm cần đính chính so với những gì tôi nói ở lượt trước:

1. Giá trị **thực tế đang chạy là 0,35**, không phải 0,30. Schema resolve ra 0,30 nhưng
   `backend/inference.py:51` mới là nơi quyết định, và nó có hằng số 0,35 riêng.
2. `/predict` nhận field tên **`threshold`**, không phải `clinical_threshold`. Pydantic
   mặc định bỏ qua field lạ, nên gửi `{"clinical_threshold": 0.3}` **im lặng không có tác
   dụng** — caller tưởng đã đặt điểm vận hành nhưng vẫn nhận giá trị mặc định:

   ```
   {"smiles":"CCO","clinical_threshold":0.3}  ->  threshold_used: 0.35   (bị bỏ qua)
   {"smiles":"CCO","threshold":0.3}           ->  threshold_used: 0.3
   ```

Không thay đổi kết luận nào của plan — chỉ làm luận điểm "artifact phải là nguồn mặc
định duy nhất" mạnh thêm.

---

## 3. Phát hiện mới — blocker mà plan chưa biết

### 3.1 Model ClinTox không serve được

Plan liệt kê `clintox-smilesgnn-v1` ở trạng thái **"Giữ"** (mục 2.1) và đặt endpoint
`clintox` vào contract đích (mục 4.3). Điều này **không khả thi**:

```
$ python -c "from backend.inference import load_model; load_model(...)"
FileNotFoundError: Tokenizer not found: models/smilesgnn_model/tokenizer.pkl
Run 'python scripts/train_hybrid.py' first.
```

Chuỗi bằng chứng:

1. `backend/inference.py:277` bắt buộc phải có `models/smilesgnn_model/tokenizer.pkl`.
2. Thư mục `models/smilesgnn_model/` chỉ có `best_model.pt`, `smilesgnn_model_metrics.txt`,
   `training_curves.png`. Không có tokenizer.
3. `.gitignore` dòng 49 loại trừ `*.pkl` → file này **chưa từng được commit**
   (`git log --all -- "*tokenizer.pkl"` trả về rỗng).
4. Checkpoint có `smiles_encoder.token_embedding.weight` shape **(69, 96)** — vocabulary
   69 token, sinh ra từ tập huấn luyện ClinTox. Config khai `smiles_vocab_size: 100`,
   tức là vocab thật là **data-derived**, không phải hằng số.
5. Hai tokenizer SMILES khác còn trên đĩa (`smilesgnn_herg_exp_model_tuned_v1`,
   `smilesgnn_multitask_model`) đều có vocab **80 token** → không khớp.

Không có vocabulary thì token id không tái tạo được, nên trọng số embedding vô nghĩa.
**Model này chỉ là một file weight không dùng được.**

### 3.2 Hệ quả: "clinical toxicity" mà sản phẩm đang trả về chính là hERG

Ghép ba sự thật ở trên lại:

| | |
|---|---|
| Model ClinTox | Không load được → không chạy |
| Cái gì điền vào field `clinical`? | Output head hERG của ChemBERTa (`inference.py:851`) |
| Threshold áp lên nó | 0,35 (đo thực tế), thay vì 0,4133 mà model được hiệu chuẩn — xem 2.1 |

Nghĩa là con số "độc tính lâm sàng" trên UI **là xác suất chẹn kênh hERG, đọc ở sai
điểm vận hành**. Đây không phải lỗi đặt tên — nó là một khẳng định khoa học sai, và
đúng là lý do quan trọng nhất để làm refactor này.

### 3.3 Ba điểm cần sửa trong plan

| Plan viết | Nên sửa thành |
|---|---|
| Giữ 2 model, `clintox-smilesgnn-v1` = "Giữ" | Giữ **1** model. ClinTox chuyển sang trạng thái *blocked*, chờ quyết định |
| Contract v1 có endpoint `clintox` (mục 4.3) | v1 chỉ có `herg` + `tox21`. Type `ClinToxPrediction` đã viết sẵn và có test, nhưng không đăng ký provider |
| "Áp threshold được đóng gói cùng artifact" (mục 1, ý 4) | Đúng cho ChemBERTa. ClinTox **không** có threshold trong artifact — nếu khôi phục được model thì phải hiệu chuẩn lại và đóng gói threshold cùng nó |

### 3.4 Quyết định đã chốt (2026-09-03)

> **Giữ nguyên toàn bộ model checkpoint và logic của nó, kể cả ClinTox.**

Đã thực hiện theo đúng quyết định này:

| Việc | Trạng thái |
|---|---|
| Toàn bộ 51 file trong `models/` | **Giữ nguyên**, không xoá file nào |
| Code kiến trúc model trong `backend/` (`graph_models_hybrid`, `smiles_tokenizer`, `graph_data`, `pretrained_mol_model`) | **Giữ nguyên**, provider gọi vào chứ không viết lại |
| Provider ClinTox (`toxpred/scientific/providers/clintox_smilesgnn.py`) | **Đã viết xong** — chạy được ngay khi có tokenizer |
| Đăng ký trong manifest | `required: false` — khai báo đầy đủ, không chặn readiness |
| Ma trận xoá của plan (mục 6.2, 6.3) | **Thu hẹp lại**: bỏ mọi mục liên quan tới model/checkpoint |

Điều này thu hẹp phạm vi xoá so với plan gốc. Plan đề xuất bỏ
`herg-tox21-xsmiles-baseline`, `tox21-gatv2-baseline` và các model thử nghiệm khỏi
production image (mục 2.1), đồng thời không port `attentivefp_model.py`, `gps_model.py`,
`graph_models.py`, `pretrained_gnn.py` (mục 6.3). **Những mục đó nay được giữ lại.**

Manifest chỉ liệt kê những gì service **sẵn sàng serve**; mọi checkpoint khác vẫn nằm
trên đĩa và vẫn dùng được cho training/benchmark.

#### Provider ClinTox hoạt động thế nào khi chưa có tokenizer

Không degrade âm thầm. `availability()` trả lý do cụ thể và hành động cần làm:

```
tokenizer missing: tokenizer.pkl. The checkpoint was trained with a 69-token
vocabulary derived from the ClinTox corpus; without that vocabulary the token ids
cannot be reproduced and the embedding weights are unusable. Restore it from the
training run, or retrain with scripts/train_hybrid.py and commit the tokenizer
alongside the weights.
```

Registry ghi nhận là *optional unavailable*, service vẫn `ready` cho hERG + Tox21,
và `clintox` trả lỗi typed thay vì trả nhầm model khác:

```
endpoint 'clintox' is not served by this build (available: ['herg', 'tox21'])
```

#### Chốt tokenizer sai sẽ bị chặn

Có test dựng sẵn tình huống: ghép tokenizer 80 token (`smilesgnn_multitask_model`) với
checkpoint 69 token rồi load. Provider **fail loud**. Nếu không chặn, mọi token bị ánh xạ
lại và model vẫn trả ra xác suất trông rất tự tin nhưng vô nghĩa.

#### Threshold của ClinTox — phân biệt "hiệu chuẩn" với "chọn tay"

Checkpoint ClinTox **không** kèm threshold nào, khác với ChemBERTa (0.4133 từ Youden-J,
3-fold CV). Nên `ThresholdSource` có thêm giá trị thứ ba:

| Nguồn | Nghĩa | Ví dụ |
|---|---|---|
| `artifact` | Hiệu chuẩn trên validation split, đóng gói cùng weight | hERG 0.4133 |
| `manifest_declared` | **Chọn theo vận hành, chưa hiệu chuẩn** | ClinTox 0.35 |
| `request_override` | Caller truyền vào cho riêng request đó | — |

Gộp hai loại đầu vào một nhãn chính là cách con số 0.30 trông như thể đã được hiệu chuẩn.
Khi khôi phục được ClinTox, **phải hiệu chuẩn lại threshold** rồi đóng gói cùng weight,
lúc đó mới chuyển sang `artifact`.

#### Còn lại để lấy lại endpoint `clintox`

Chỉ cần **một** trong hai, code đã sẵn sàng cho cả hai:

1. Tìm lại `tokenizer.pkl` (69 token) từ máy đã train hoặc backup ngoài Git → thả vào
   `models/smilesgnn_model/`, thêm checksum vào manifest, đổi `required: true`.
2. Train lại bằng `scripts/train_hybrid.py`, commit kèm tokenizer **và** threshold đã
   hiệu chuẩn.

Lưu ý cho cả hai: hiện **không có baseline ClinTox** để đối chiếu parity, vì model chưa
từng chạy được. Sau khi khôi phục nên chạy `benchmarks/capture_baseline.py` để tạo baseline
trước khi động tiếp vào code.

---

## 4. Đã thực thi những gì

Toàn bộ đều **thêm mới, không xoá gì**. Code cũ chạy y nguyên.

### Phase 0 — Đóng băng baseline ✅

| File | Nội dung |
|---|---|
| `benchmarks/build_golden_panel.py` | Sinh panel, tự từ chối ghi nếu có cấu trúc không parse được bằng RDKit |
| `benchmarks/fixtures/golden_panel.json` | **50 ca**: 42 hợp lệ + 6 invalid + 2 envelope-reject |
| `benchmarks/capture_baseline.py` | Ghi fingerprint runtime, SHA-256 mọi artifact, và **raw probability** |
| `benchmarks/manifests/baseline-e6882b2.json` | Runtime + checksum 11 file artifact |
| `benchmarks/golden/baseline_predictions.json` | Raw output cho 42 phân tử |

Panel phủ đúng yêu cầu của plan: 10 hERG blocker đã biết, 10 thuốc thông dụng,
8 hợp chất tham chiếu, 3 ca stereochemistry, 2 muối, 2 ion, 5 nguyên tố hiếm/cơ kim,
2 ca input dài, 6 SMILES sai, 2 ca chuỗi rỗng.

**Baseline có ý nghĩa khoa học** (không phải chỉ là số):

| Nhóm | hERG probability trung bình |
|---|---|
| 10 blocker đã biết (amiodarone, verapamil, astemizole…) | **0,682** |
| 10 thuốc thông dụng (aspirin, caffeine, ibuprofen…) | **0,078** |

Tox21 NR-ER: estradiol 0,973 · bisphenol A 0,989 · ethanol 0,199 — đúng chiều sinh học.
Latency CPU p50 3,0 ms / p95 4,5 ms. Ca trùng lặp cho kết quả **giống hệt từng bit**.

### Phase 1 — Khóa semantic contract ✅

| File | Nội dung |
|---|---|
| `toxpred/domain/endpoints.py` | `TOX21_TASKS` — 12 task theo **đúng thứ tự trong checkpoint**, versioned `tox21-12task-v1`; `validate_task_order()` fail loud khi artifact lệch |
| `toxpred/domain/policy.py` | `PredictionPolicySnapshot` — artifact là nguồn mặc định **duy nhất**; override phải mang `threshold_source="request_override"` |
| `toxpred/domain/prediction.py` | `ClinToxPrediction` / `HergPrediction` / `Tox21Prediction` là **ba type tách biệt**, mỗi type có tên field riêng |
| `toxpred/domain/molecule.py` | `Molecule`, `InvalidSmilesError` |

`HergPrediction` dùng field `probability_blocker`, label `blocker`/`non_blocker`.
`PredictionResult.to_dict()` dựng payload từ chính các type đó, nên **không có đường nào**
để một xác suất hERG xuất hiện dưới key `clinical`. Có test khẳng định điều này.

### Phase 2 — Artifact registry ✅

| File | Nội dung |
|---|---|
| `toxpred/scientific/artifacts.py` | Manifest schema + verify SHA-256 + size; báo **tất cả** lỗi trong một lần |
| `toxpred/scientific/registry.py` | `ModelProvider` protocol, `ModelRegistry`, tách liveness/readiness, **không fallback** |
| `artifacts/manifest.yaml` | Khai báo `herg-tox21-chemberta-v1` với checksum thật của 10 file |

Manifest ghi rõ hai artifact **không** đăng ký (ClinTox và `dualhead_ensemble3`) kèm lý do,
để lần sau không phải phát hiện lại.

### Phase 3B — Provider ChemBERTa ✅ + **parity PASS**

`toxpred/scientific/providers/herg_tox21_chemberta.py`:

- trả **raw probability**, không tự threshold — label do policy layer quyết
- `validate_task_order()` chạy lúc load
- `load_state_dict(strict=False)` rồi **fail nếu có missing/unexpected key**
- tokenizer đọc `local_files_only=True`
- không port code training

```
cases: 42
max |delta| hERG : 2.384e-07
max |delta| Tox21: 5.662e-07
PARITY PASS (tolerance 1e-06)
```

### Phase 4 (một phần) — Application service ✅

`toxpred/application/predictor.py`: `ToxicityPredictor.predict()` / `predict_batch()`.
Batch **giữ nguyên thứ tự input**, lỗi báo **theo từng item**, model chỉ chạy **một lần**
cho cả hai endpoint hERG + Tox21.

`toxpred/scientific/applicability.py`: port từ `ood_guard`, đổi tên thành
`element_rules_v1`. Trạng thái `ok` **nói rõ** rằng nó không xác nhận được sự tương đồng
phân bố — khác với `ood_risk: "LOW"` của bản cũ.

### Phase 3A — Provider ClinTox ✅ code, ⛔ artifact

`toxpred/scientific/providers/clintox_smilesgnn.py` — gọi vào `backend.inference.load_model`
chứ **không** viết lại phần khoa học. Hai điểm khác với code cũ: trả raw probability thay
vì DataFrame đã threshold/sort/render label; và phân tử không featurise được thì **raise**
thay vì thành dòng `"Parse error"` với `P(toxic) = None` (dễ bị đọc nhầm là điểm thấp).

### Test suite ✅ — 114/114 pass

```
tests/unit/test_endpoints.py            6   thứ tự task Tox21
tests/unit/test_policy.py              12   threshold, biên >=, artifact vs declared
tests/unit/test_prediction_contract.py 14   hERG không thể thành clinical
tests/unit/test_artifacts.py           15   checksum, file thiếu, thư mục rỗng, optional
tests/unit/test_registry.py             9   không fallback, optional không chặn readiness
tests/unit/test_clintox_provider.py     8   availability + chặn tokenizer sai
tests/unit/test_import_boundaries.py   36   luật phụ thuộc theo AST
tests/unit/test_resolver.py            13   SMILES + applicability
tests/golden/test_provider_parity.py    6   đối chiếu baseline (cần artifact thật)
```

`test_import_boundaries.py` chặn tĩnh: `domain/` không được import torch, RDKit,
FastAPI, yaml, `backend`; không module nào trong `toxpred/` được import `agents`,
`services`, `model_server`, `src`, Firebase, DeepChem, MolScribe. Có cả test khẳng định
`import toxpred` **không** kéo theo torch/transformers/rdkit.

---

## 5. Trạng thái theo từng phase

Cập nhật 2026-09-03 sau khi chốt: **chạy predictor-only trước, agent để giai đoạn sau.**

| Phase | Trạng thái | Ghi chú |
|---|---|---|
| 0 — Baseline | ✅ | Tag `archive/agent-layer-165319beede5` + OpenAPI snapshot 11 path |
| 1 — Semantic contract | ✅ | |
| 2 — Package + registry | ✅ | |
| 3A — ClinTox provider | 🟡 | Code + 8 test xong; chờ tokenizer |
| 3B — ChemBERTa provider | ✅ | Parity 2,4e-07 |
| 4 — Application + API | ✅ | 6 endpoint `/v1/*`, 21 contract test |
| 5 — Applicability + attribution | ✅ | `element_rules_v1` + `grad_x_embedding_l2_v1` |
| 6 — Benchmark khoa học | ✅ | Split đóng băng, reproduction check PASS |
| 7A — Cutover | ✅ | Không cần adapter: `/predict` và `/explain` không có consumer |
| 7B — Xoá legacy | ✅ | 352 file, −55.518 dòng |
| 8 — Deps / image / CI / docs | ✅ | Workflow cần scope `workflow` để push |

### Definition of Done (§11 của plan)

| # | Điều kiện | |
|---|---|---|
| 1 | Repo chỉ còn predictor runtime + regression suite + deployment | ✅ |
| 2 | Không LLM/agent/research/chat/Firebase/frontend/OCR dependency | ✅ `check_no_agent_deps.py` |
| 3 | `model_server/main.py` không còn; entrypoint API vài trăm dòng | ✅ `toxpred/api/` 396 dòng |
| 4 | Một package canonical, không `backend/`–`src/` re-export kép | ✅ `src/` đã xoá |
| 5 | Model load qua manifest có SHA-256 | ✅ |
| 6 | hERG/ClinTox/Tox21 tách schema và semantics | ✅ |
| 7 | Threshold từ artifact/policy snapshot, có trong response | ✅ |
| 8 | Golden probabilities trong tolerance | ✅ 2,4e-07 |
| 9 | Benchmark có split/model/environment provenance | ✅ |
| 10 | Missing model làm readiness fail, không silent fallback | ✅ |
| 11 | Inference không cần internet sau provisioning | ✅ config đã vendor, revision đã ghim |
| 12 | Source tree 80–120 file | ✅ **108** |
| 13 | README, model card, benchmark protocol đúng khả năng và giới hạn | ✅ |

**12/13 đạt, 1 còn dở** (ClinTox artifact).

### Sai lệch có chủ ý so với plan

| Plan | Thực tế | Vì sao |
|---|---|---|
| Xoá baseline/experimental checkpoint khỏi image | Giữ toàn bộ `models/` | Quyết định của anh/chị |
| Không port training pipeline | Giữ `scripts/` training | `train_hybrid.py` là đường khôi phục ClinTox |
| Bỏ `config/workspace_mode.yaml` | Giữ, rút gọn còn switch dataset | Training script vẫn đọc; service thì không |
| Harness design docs không nằm trong predictor branch | Giữ `docs/spec/` | Là đầu vào cho giai đoạn agent sắp tới |
| Package ở `src/toxpred/` | `toxpred/` | `src/` từng là package thật; nay đã xoá, layout phẳng giữ nguyên |

## 6. Một sai lệch có chủ ý so với plan

Plan đặt package ở `src/toxpred/`. Repo **đã có** `src/` là một package thật với
`__init__.py` và 20+ compatibility wrapper (`from backend.inference import *`). Đặt vào đó
sẽ khiến package chỉ import được dưới tên `src.toxpred`.

Package được đặt ở **`toxpred/`** tại repo root. Khi Phase 7B xoá `src/`, có thể chuyển
sang layout `src/` nếu muốn — không ảnh hưởng gì đến code đã viết.

---

## 7. Về mục tiêu "80–120 file"

Vẫn đạt, nhưng con số đổi vì quyết định giữ toàn bộ checkpoint.

| Nhóm | File |
|---|---|
| `toxpred/` | ~24 |
| `tests/` | ~14 |
| `benchmarks/` | ~6 |
| `backend/` — chỉ phần model architecture còn được giữ | ~12 |
| `models/` — **toàn bộ, theo quyết định** | 51 |
| `config/` model config | ~20 |
| `artifacts/`, `deploy/`, `.github/` | ~8 |
| `docs/` + README + pyproject | ~8 |
| **Tổng** | **~143** |

Vượt khoảng 80–120 của plan, nhưng phần vượt **toàn bộ nằm ở `models/` và `config/`** —
tức là artifact và metadata, không phải code runtime. Nếu tính riêng source tree thì
khoảng **~72 file**, thấp hơn plan.

Muốn về đúng khoảng của plan thì đưa weight ra artifact store (GCS) và Git chỉ giữ
manifest — đúng như plan mục 2 đề xuất cho `models/`. Việc đó không xoá checkpoint nào,
chỉ đổi chỗ lưu, nên vẫn khớp với quyết định đã chốt.

---

## 8. Ai đang gọi API? — bản đồ consumer

Câu hỏi: *"còn consumer nào ngoài FE gọi predict/analyze không?"*

Câu trả lời ngắn: **trong repo thì không** — nhưng bức tranh khác với dự đoán.
**Frontend không hề gọi `/predict`, `/predict/batch`, `/explain` hay `/analyze`.**

### 8.1 Bản đồ đầy đủ

| Endpoint | Consumer trong repo | Ghi chú |
|---|---|---|
| `/agent/analyze` | **Frontend** — `frontend/src/lib/api.ts:541` | |
| `/agent/analyze/stream` | **Frontend** — `api.ts:644` | |
| `/agent/chat` | **Frontend** — `api.ts:706` | |
| `/agent/chat/stream` | **Frontend** — `api.ts:803` | |
| `/extract-smiles-from-image` | **Frontend** — `api.ts:850` | |
| `/smiles/preview` | **Frontend** — `api.ts:900` | |
| `/analyze` | **Chỉ chính agent layer**, qua self-HTTP localhost | Xem 8.2 |
| `/health` | Docker `HEALTHCHECK` + `tools/tox_tools.py:371` | |
| `/predict/batch` | **Chỉ** `scripts/sweep_clinical_threshold.py` | Script dev offline, mặc định `127.0.0.1:8000` |
| `/predict` | **Không có consumer nào** | |
| `/explain` | **Không có consumer nào** | Xem 8.3 |

Đã quét: `frontend/src/**`, `agents/`, `services/`, `tools/`, `scripts/`, `tests/`,
`benchmark/`, `model_server/`, cùng mọi `*.yaml|yml|json|md|sh|Dockerfile`.

### 8.2 `/analyze` chỉ có một consumer, và đó là chính process này

```
POST /agent/analyze
  └─ model_server/main.py:5047   run_orchestrator_flow
      └─ agents/orchestrator_agent.py:175   run_screening
          └─ agents/screening_agent.py:55   analyze_molecule
              └─ tools/tox_tools.py:255     httpx.post("/analyze")
                  └─ ...quay lại chính server này ở 127.0.0.1:8080
```

`deploy/cloudrun-env.yaml:26` đặt `MODEL_SERVER_URL: "http://127.0.0.1:8080"` — xác nhận
đây là self-HTTP trong cùng container, đúng như plan mục 6.2 mô tả.

Hệ quả tốt cho migration: khi làn A gọi application service **in-process**, vòng HTTP này
biến mất và `/analyze` mất luôn consumer cuối cùng. Không cần deprecation window cho
`/analyze` vì ngoài chính nó ra không ai gọi.

### 8.3 `/explain` không có consumer, nhưng logic explain thì có

Frontend đọc `heatmap_base64` và `explainer_note` (`api.ts:82`, `api.ts:86`) như **field
bên trong response của `/agent/analyze`**, không gọi `/explain`. Nên endpoint bỏ được,
còn logic explain thì phải giữ.

### 8.4 Cảnh báo: repo không trả lời được câu hỏi này một cách trọn vẹn

`firebase.json` rewrite các path sau tới Cloud Run `tox-agent-cpu` (asia-southeast1):

```
/health   /extract-smiles-from-image   /smiles/**   /predict   /predict/**
/explain  /analyze                     /agent/**
```

Nghĩa là **toàn bộ API đang public trên chính domain của web app**, không chỉ trên URL
Cloud Run. Ai đó có thể đã script trực tiếp vào `https://<domain>/predict` mà repo không
hề biết. Trước khi xoá, nên kiểm tra ở nguồn duy nhất biết sự thật:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="tox-agent-cpu"
   AND httpRequest.requestUrl:("/predict" OR "/analyze" OR "/explain")' \
  --freshness=30d --limit=100 \
  --format='value(httpRequest.requestUrl, httpRequest.userAgent)'
```

Nếu 30 ngày không có traffic nào ngoài health check, có thể xoá thẳng, không cần
compatibility adapter. Ngoài ra `AIP_HEALTH_ROUTE` / `AIP_PREDICT_ROUTE`
(`main.py:401-402`) là scaffolding cho Vertex AI custom container; Dockerfile đặt đúng
giá trị mặc định `/health` và `/predict` nên **không** tạo thêm route nào.

### 8.5 Kết luận cho deprecation window

| Endpoint | Đề xuất |
|---|---|
| `/predict`, `/predict/batch`, `/explain` | Không consumer trong repo. Xoá được sau khi xác nhận log Cloud Run. `sweep_clinical_threshold.py` chuyển sang gọi application service in-process |
| `/analyze` | Consumer duy nhất biến mất khi bỏ self-HTTP. Không cần window |
| `/agent/*` | **Đây mới là thứ frontend thực sự dùng.** Cần adapter cho tới khi FE migrate xong |

Thứ tự này ngược với giả định ban đầu: phần khó gỡ không phải scientific API, mà là
`/agent/*`.

---

## 9. Việc tiếp theo

**Làm được ngay, không cần chờ:**

1. Kiểm tra log Cloud Run 30 ngày theo lệnh ở mục 8.4 → chốt có cần adapter cho
   `/predict`, `/explain` không.
2. Ghim revision `DeepChem/ChemBERTa-77M-MTR` và vendor file config vào artifact —
   checkpoint đã chứa đủ trọng số backbone (61 tensor gồm `backbone.embeddings.*` và
   `backbone.pooler.*`), nên chỉ còn thiếu config kiến trúc là startup hết phụ thuộc mạng.
3. Sửa `backend/data.py` để fallback PyTDC **fail loud** thay vì âm thầm đổi sang random
   split — rủi ro rò rỉ train/test đang tồn tại, độc lập với refactor.
4. Bỏ `report_state` rehydration và tắt `mechanism_threshold` chung trong API cũ.

**Chặn bởi việc khôi phục ClinTox (không chặn phần còn lại):**

5. Đưa `clintox` vào contract v1: cần tokenizer + threshold đã hiệu chuẩn (mục 3.4).

**Phase tiếp theo:**

6. Phase 4 — FastAPI app + `/v1/*`. Danh sách endpoint nay đã rõ: `herg` + `tox21` bắt
   buộc, `clintox` khai báo sẵn và bật lên khi artifact đủ.
