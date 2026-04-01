# Core Predict Model Research Brief

Date: 2026-04-01
Scope: why several toxic samples can be predicted as non-toxic even when reported metrics are very high.

---

## 1) Problem Statement

You observed this pattern:
- Reported model metrics are very high.
- Some toxic molecules in test data are still predicted as non-toxic.

This is not automatically a contradiction. In this workspace, the issue can be caused by a combination of:
- metric-definition effects,
- dataset imbalance,
- threshold policy,
- chemistry out-of-distribution (OOD),
- and pipeline/mode mismatch between legacy ClinTox artifacts and current Tox21-first runtime.

---

## 2) Evidence Collected From Workspace

### 2.1 Test set imbalance is strong

From `test_data/full_test_set.csv`:
- total = 148
- toxic = 10
- non-toxic = 138
- toxic rate = 6.76%

Implication:
- A model can still show very high AUC while missing a few toxic cases.

### 2.2 Existing high metrics are real artifacts in repo

From `models/smilesgnn_model/smilesgnn_model_metrics.txt`:
- test_auc_roc: 0.9993
- test_accuracy: 0.9865
- test_f1: 0.9091
- test_pr_auc: 0.9909

From `results/overall_results.csv`:
- SMILESGNN is recorded as top performer on ClinTox benchmark metrics.

### 2.3 Repo already records this exact paradox

From `XSMILES_PERFORMANCE_BRAINSTORM.md`:
- AUC remains near 1.0 while at least one toxic false negative still exists.
- One highlighted false negative is a Pt-containing compound (Carboplatin-like).

From `test_data/README.md`:
- `full_test_set.csv` is a preserved ClinTox-era artifact.
- Expected behavior explicitly mentions one Pt-complex likely missed.

### 2.4 Current workspace mode is tox21_only

From `config/workspace_mode.yaml`:
- mode: tox21_only
- clintox_enabled: false
- tox21_enabled: true

From `README.md` and `app.py`:
- ClinTox Streamlit workflow is intentionally disabled in current mode.

Important nuance:
- `model_server/main.py` still loads XSmiles checkpoint if model folder exists (`XSMILES_LOAD_REQUESTED = CLINTOX_ENABLED or MODEL_DIR.exists()`).
- So runtime may still serve clinical prediction from legacy checkpoint even while training workflows are tox21_only.

This can create confusion about which "core model" is actually being evaluated.

---

## 3) Why High Metrics And Toxic Misses Can Coexist

## 3.1 AUC is ranking, not decision-threshold quality

AUC evaluates ordering of toxic vs non-toxic scores globally.
It does not guarantee zero false negatives at threshold 0.5.

With only 10 positive samples:
- a small number of ranking mistakes can still produce near-perfect AUC.

## 3.2 Decision policy is threshold-dependent

In clinical branch (`backend/inference.py`):
- label is assigned by `p_toxic >= threshold`.
- default threshold is 0.5.

So a sample with true toxicity but score below 0.5 is a false negative even if global ranking metric is strong.

## 3.3 Data imbalance amplifies this mismatch

When positive rate is low:
- accuracy and even AUC can look excellent,
- while toxic recall remains fragile.

For safety screening, recall on toxic class is often more important than headline AUC.

---

## 4) Root-Cause Map (By Layer)

## A) Data layer

1. Severe class imbalance in ClinTox test split.
2. Potential label ambiguity/noise in benchmark-level curation (already noted in repo brainstorm).
3. OOD chemistry in organometallics (Pt/Bi) that differs from typical organic drug-like training distribution.

## B) Representation layer

From `backend/graph_data.py` atom features:
- only a small common-element set is one-hot encoded (C, N, O, F, P, S, Cl, Br, I),
- everything else is collapsed into a single "other" flag.

Implication:
- rare elements (for example Pt) can lose discriminative signal.

## C) Objective/training layer

ClinTox legacy branch (`scripts/train_hybrid.py` + `backend/graph_train.py`):
- uses focal loss and weighted sampler options,
- but still relies on threshold-based F1/accuracy reporting at 0.5 in core evaluator.

Tox21 branch (`scripts/train_tox21_gatv2.py`):
- includes per-task threshold calibration,
- but this is for mechanistic multi-task outputs, not automatically equivalent to ClinTox binary branch behavior.

## D) Serving/inference layer

Unified endpoint (`/analyze`) combines:
- clinical branch output,
- mechanistic branch output,
- final aggregated verdict categories.

If you inspect only one top-level verdict without checking clinical `p_toxic` and threshold_used, you can misread model behavior.

## E) Workflow/mode layer

Current mode is tox21_only.
Legacy ClinTox evaluation files still exist and are useful for analysis, but retraining/iterating ClinTox branch is guarded by workspace mode.

---

## 5) Critical Diagnostic Questions Before Any New Training

For each toxic sample you believe is wrongly predicted, verify all five points:

1. Same checkpoint?
   - Is prediction actually from `models/smilesgnn_model/best_model.pt`?

2. Same endpoint semantics?
   - Did you test `/predict` (clinical only) or `/analyze` (aggregated logic) or `/agent/analyze` (agent-layer report)?

3. Same threshold?
   - What exact `threshold_used` was applied?

4. Same input canonicalization?
   - Did raw SMILES canonicalize to an equivalent structure?

5. Is sample OOD?
   - Rare elements (Pt/Bi/etc), unusual salts, organometallic fragments, or chemistry not represented in training distribution.

Without these checks, false negatives can be over-attributed to "model weakness" when part of the issue is pipeline mismatch.

---

## 6) Research Plan (Serious, Structured, Actionable)

## Phase 0: Define the target metric policy

Primary objective for safety use:
- maximize toxic recall under acceptable false-positive budget.

Track at least:
- toxic recall (sensitivity),
- toxic precision,
- PR-AUC,
- false-negative count on known-toxic panel,
- calibration metrics (Brier/ECE/reliability bins).

## Phase 1: Build a Failure Registry

Create a table for every toxic miss with:
- sample id,
- raw and canonical SMILES,
- predicted p_toxic,
- threshold_used,
- endpoint used,
- model checkpoint hash/version,
- element set present,
- assay hit profile,
- final verdict,
- notes (OOD/parse/label ambiguity).

Goal:
- turn anecdotal misses into reproducible failure cohorts.

## Phase 2: Threshold and decision analysis (no retraining needed)

Using existing prediction scores:
- sweep threshold from 0.05 to 0.95,
- plot recall-toxic vs precision-toxic,
- choose policy threshold by business cost of FN vs FP.

Practical option:
- introduce triage zone (for example low / review / high risk) instead of hard binary at one threshold.

## Phase 3: OOD and chemistry subgroup audit

Split evaluation by subgroup:
- molecules containing rare elements (Pt, Bi, etc),
- high molecular weight/salts,
- charged species complexity,
- ring/aromatic-rich vs simple scaffolds.

Report per-group FN rates.

If one subgroup dominates FN, treat as domain-gap, not generic underfitting.

## Phase 4: Label quality and split robustness audit

1. Canonical-duplicate conflict check (same canonical SMILES with conflicting labels).
2. Multi-seed and multi-split evaluation (not one seed only).
3. External holdout panel for reality-check beyond benchmark split.

## Phase 5: Model improvements (after audit, not before)

Only after root cause is localized:
- add rare-element-aware atom encoding,
- add uncertainty/OOD gate,
- hard-case mining focused on toxic false negatives,
- reweight objective explicitly for toxic recall target.

---

## 7) Immediate Quick Wins (Low Risk)

1. Log complete prediction context in API output capture (`p_toxic`, `threshold_used`, checkpoint id).
2. Add warning flag for rare-element molecules (OOD risk banner).
3. Evaluate a higher-recall threshold policy for toxic screening workflows.
4. Report toxic recall/FN count side-by-side with AUC in dashboards and docs.

These changes improve decision quality before any expensive retraining cycle.

---

## 8) Key Workspace References

- `test_data/full_test_set.csv`
- `test_data/README.md`
- `models/smilesgnn_model/smilesgnn_model_metrics.txt`
- `results/overall_results.csv`
- `XSMILES_PERFORMANCE_BRAINSTORM.md`
- `backend/graph_data.py`
- `backend/inference.py`
- `backend/graph_train.py`
- `scripts/train_hybrid.py`
- `scripts/train_tox21_gatv2.py`
- `config/workspace_mode.yaml`
- `model_server/main.py`

---

## 9) Bottom Line

Current evidence supports this interpretation:
- high benchmark metrics and toxic false negatives can both be true,
- especially under strong class imbalance and fixed thresholding,
- and this repo adds extra complexity because legacy ClinTox artifacts coexist with tox21_only workflow.

So the right next step is not blind framework expansion.
The right next step is a disciplined failure-analysis program focused on toxic recall, threshold policy, and subgroup/OOD behavior.
