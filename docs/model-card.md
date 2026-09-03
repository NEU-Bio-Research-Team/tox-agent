# Model card — ToxPred

**Service version:** 0.1.0.dev0 · **Report generated:** 2026-09-03T19:10:18+0800
**Benchmark commit:** `918c7852259b` · **Split hash:** `0f0a45cff673d92a…`

---

## 1. What this service does, and what it does not

ToxPred returns three kinds of probability. They come from different training
data and mean different things, so the service never combines them into a single
score and neither should a caller.

| Endpoint | Question it answers | Model |
|---|---|---|
| `herg` | Is this molecule likely to block the hERG potassium channel? | `herg-tox21-chemberta-v1` |
| `tox21` | For each of 12 Tox21 assays, is it likely active? | `herg-tox21-chemberta-v1` |
| `clintox` | Was this compound associated with clinical-trial toxicity? | `clintox-smilesgnn-v1` — **not currently served**, see section 6 |

**This is a screening aid, not a safety assessment.** A `non_blocker` label is a
model output on a scaffold-split benchmark, not evidence that a compound is
safe. Nothing here substitutes for an assay.

## 2. Served model

`herg-tox21-chemberta-v1`

| | |
|---|---|
| Architecture | ChemBERTa backbone with two heads: one hERG logit, twelve Tox21 logits |
| Base model | `DeepChem/ChemBERTa-77M-MTR` @ `66b895cab8adebea0cb59a8effa66b2020f204ca`, config vendored |
| Weights | Bundled in the checkpoint, including the backbone; the Hugging Face repo supplies only the architecture config |
| Tokenizer | Shipped with the artifact, SHA-256 checked at load |
| Max input | 128 tokens; longer SMILES are truncated and the response says so |
| Weights SHA-256 | `c851e81541f8975f66589879ba9bd35c3068c3fbd57417bb7939214183f62690` |

## 3. Training data and split

Scaffold split, seed 42, for both datasets — reproduced exactly in
`benchmarks/manifests/eval-split-v1.json` and content-hashed, so the benchmark
cannot score a molecule the model trained on.

| Dataset | Source | Test molecules |
|---|---|---|
| hERG | hERG_Karim (PyTDC) | 2 690 |
| Tox21 | Tox21 (DeepChem) | 783 |

Tox21 is sparsely labelled. Missing labels stay missing: they are masked per
task, never coerced to a negative, and the surviving count is reported below.

## 4. Measured performance

Frozen split, artifact thresholds applied — nothing fitted on the test set.
95% CIs are seeded percentile bootstrap, 1 000 resamples.

### hERG (n = 2690, prevalence 0.493, threshold 0.4133 from the artifact)

| Metric | Value | 95% CI |
|---|---|---|
| AUROC | 0.8372 | [0.821, 0.851] |
| PR-AUC | 0.8310 | [0.810, 0.849] |
| F1 | 0.7644 | [0.746, 0.783] |
| MCC | 0.5230 | [0.489, 0.557] |
| Balanced accuracy | 0.7613 | |
| Sensitivity | 0.7860 | |
| Specificity | 0.7366 | |
| Brier | 0.1816 | |
| **ECE (10 bins)** | **0.1200** | |

Confusion at the artifact threshold: TP 1043, FP 359,
TN 1004, FN 284.

> **Calibration.** ECE of 0.12 means the reported probabilities are
> **not** calibrated risks. Rank molecules by them; do not read `0.7` as a 70%
> chance of blockade.

### Tox21 (macro AUROC 0.7594 over 12/12 tasks)

| Task | n scored | n masked | Prevalence | Threshold | AUROC | PR-AUC |
|---|---|---|---|---|---|---|
| `NR-AR` | 715 | 68 | 0.038 | 0.94 | 0.782 | 0.392 |
| `NR-AR-LBD` | 624 | 159 | 0.030 | 0.49 | 0.751 | 0.390 |
| `NR-AhR` | 629 | 154 | 0.146 | 0.68 | 0.840 | 0.488 |
| `NR-Aromatase` | 523 | 260 | 0.090 | 0.71 | 0.766 | 0.308 |
| `NR-ER` | 554 | 229 | 0.126 | 0.51 | 0.734 | 0.380 |
| `NR-ER-LBD` | 653 | 130 | 0.032 | 0.68 | 0.725 | 0.231 |
| `NR-PPAR-gamma` | 575 | 208 | 0.038 | 0.42 | 0.742 | 0.092 |
| `SR-ARE` | 481 | 302 | 0.245 | 0.50 | 0.696 | 0.475 |
| `SR-ATAD5` | 672 | 111 | 0.049 | 0.48 | 0.699 | 0.168 |
| `SR-HSE` | 572 | 211 | 0.082 | 0.51 | 0.779 | 0.254 |
| `SR-MMP` | 520 | 263 | 0.185 | 0.54 | 0.831 | 0.469 |
| `SR-p53` | 630 | 153 | 0.114 | 0.35 | 0.770 | 0.376 |

Macro-averaged over tasks, never pooled over molecules. PR-AUC is the honest
figure for the rare assays: `NR-PPAR-gamma` has 3.8% positives, so its AUROC of
0.742 sits against a PR-AUC of 0.092.

## 5. Reproduction check

Recomputing from the frozen split through the serving code path reproduces the
metrics recorded when the artifact was trained:

| Metric | Recorded | Recomputed | Δ |
|---|---|---|---|
| `herg.auc_roc` | 0.8372 | 0.8372 | 0.00002 |
| `herg.pr_auc` | 0.8310 | 0.8310 | 0.00003 |
| `tox21.macro_auc_roc` | 0.7595 | 0.7594 | 0.00007 |

This runs on every full benchmark, so a drift in the split, the provider or the
scoring surfaces as a failure rather than a slowly moving number.

## 6. Known limitations

1. **ClinTox is not served.** Its tokenizer (`models/smilesgnn_model/tokenizer.pkl`)
   is absent and was never committed. The checkpoint's embedding is (69, 96) — a
   vocabulary derived from the ClinTox corpus — and the other SMILES tokenizers
   on disk have 80 tokens, so they do not fit. The checkpoint and provider are
   kept; the endpoint returns as soon as a matching tokenizer exists. Until then
   the service answers 503 for `clintox` rather than substituting another model.
2. **The ClinTox threshold, if the endpoint returns, is not calibrated.** The
   checkpoint ships none; the manifest declares 0.35 and every response labels
   it `threshold_source: manifest_declared`. Recalibrate before trusting a label.
3. ~~The base-model revision is not pinned.~~ **Closed.** The architecture
   config is vendored at `models/pretrained_2head_herg_chemberta_model/base_model/`
   from pinned revision `66b895cab8adebea0cb59a8effa66b2020f204ca`, checksummed in
   the manifest. With every backbone weight already in the checkpoint, the
   service starts and serves with no network; a test asserts it.
4. **Probabilities are not calibrated** (section 4).
5. **Applicability is an element whitelist**, not a learned OOD detector. It can
   flag an unusual element; a status of `ok` is not evidence that a molecule
   resembles the training distribution, and the response says so.
6. **Domain of validity.** Both datasets are small, drug-like and scaffold-split.
   Organometallics, inorganics and very large molecules are outside it — those
   get `out_of_domain`, but a prediction is still returned and should be ignored.

## 7. Intended use

Prioritising compounds for assay, and flagging liabilities early in a screening
cascade. Not for regulatory submission, clinical decisions, or any use where a
false negative carries physical risk.
