# Bioactivity branch — V1 fixed-target runbook

Implements phases P0–P2 of `TOXAGENT_BIOACTIVITY_BENCHMARK_PLAN_VI.md`: the
`toxact-chembl37-hq-v1` data contract, the four frozen split views, and the
B0–B2 baselines with the full metric suite.

Nothing here is admitted for serving. These are research artifacts; a checkpoint
becomes servable only through the registry admission path in
`backend/predictor/registry/`, after the gates in plan section 9.

## Environment

Use the existing `drug-tox-env` conda environment — do not create a new one:

```bash
PY=/home/mluser/.conda/envs/drug-tox-env/bin/python
```

It already carries RDKit, PyTorch (cu121), scikit-learn, LightGBM, XGBoost and
PyTorch Geometric. `pyarrow` is absent, so tables are `.csv.gz`, matching the
existing `data/tox21.csv.gz` convention.

## Why the REST API rather than the ChEMBL dump

The data contract pins ChEMBL 37. The full SQLite dump needs ~25 GB unpacked
and this machine has under 10 GB free, so the pipeline reads the REST API and
**asserts** the release it is served (`ChemblClient.assert_release`) rather than
trusting a filename. A release bump fails the run instead of silently changing
the data under a frozen split.

The whole HQ-Exact predicate set pushes server-side — including
`assay_confidence_score=9`, which the activity endpoint accepts as a passthrough
— so each page carries about half the bytes and the filter lives in one place
(`chembl_client.HQ_EXACT_FILTERS`).

## Pipeline

All commands run from `backend/predictor`.

### P0 — profile the target universe, then select the panel by rule

```bash
cd backend/predictor
PYTHONPATH=research $PY -m bioactivity.ingest.profile_targets \
    --out ../../data/bioactivity/profile --workers 16
```

Counts HQ-Exact records for all ~5,900 human `SINGLE PROTEIN` targets, then
breaks the survivors down per `standard_type`. Roughly 15–25 minutes at 16
workers (about 11 requests/second, no throttling observed).

```bash
PYTHONPATH=research $PY -m bioactivity.ingest.select_panel \
    --profile ../../data/bioactivity/profile/target_profile.csv \
    --out evals/bioactivity/manifests
```

Selection consults **only** dataset properties — never a model score — and is a
deterministic family-balanced round robin, so it reproduces exactly. Every
target that qualified is written to `qualifying_targets.csv`, including the ones
the rule passed over.

### P1 — extract, standardize, aggregate, split

```bash
PYTHONPATH=research $PY -m bioactivity.ingest.build_dataset \
    --panel evals/bioactivity/manifests/panel-v1.json \
    --out ../../data/bioactivity/hq-v1 --workers 10

PYTHONPATH=research $PY -m bioactivity.ingest.split \
    --dataset ../../data/bioactivity/hq-v1/hq_exact.csv.gz \
    --out evals/bioactivity/manifests/splits
```

The split assignment is written straight to the **tracked** manifests directory,
not to `data/` (which `.gitignore` excludes wholesale, the same as the rest of
`data/`). This matches the toxicity branch's convention
(`evals/benchmark/manifests/eval-split-v1.json`): a split is a frozen
reproducibility artifact, not a regenerable cache file, so it belongs in git
even though the dataset it was built from does not. `dataset_manifest.json`
(written by `build_dataset`, see above) should be copied there too after a run
you intend to keep -- `cp ../../data/bioactivity/hq-v1/dataset_manifest.json
evals/bioactivity/manifests/`.

`build_dataset` caches the raw extract to `raw_activities.json.gz`; delete it to
re-query. It writes a filter ledger recording how many records each filter
dropped and why — `dataset_manifest.json`.

Aggregation is median pChEMBL per `(parent, target, standard_type, assay
context)`. Replicates spanning more than one log unit are flagged
`high_disagreement` and held out by default.

### P2 — benchmark

```bash
$PY evals/bioactivity/run_benchmark.py \
    --dataset ../../data/bioactivity/hq-v1/hq_exact.csv.gz \
    --splits evals/bioactivity/manifests/splits \
    --out evals/bioactivity/results \
    --models b0,b1,b2a,b2b --views temporal,cluster,scaffold,random
```

The runner **reads** the frozen split and verifies the dataset hash against the
manifest. It never re-splits; a mismatch is a hard failure.

## Reading the output

`temporal` is the primary view and the only one release decisions may use.
`random` is labelled a diagnostic in the report itself. Per task the report
carries the label distribution next to every metric, because a low MAE on a
task whose IQR is 0.3 is not a good model.

Two numbers deserve attention before any aggregate:

- `n_tasks_unscorable` — tasks in test but absent from train. A fixed-target
  model has no claim on these and they are not scored.
- `cliffs.direction_accuracy` — a model can post a respectable RMSE while
  ordering every activity-cliff pair backwards.

`paired_vs_best` compares each model against the strongest one by macro MAE,
bootstrapping over **compounds** rather than rows, since a compound's
measurements across targets are correlated.

## P2 results (B0–B2b, 2026-09-14)

Full run, frozen: `evals/bioactivity/manifests/benchmark_report.json` and
`evals/bioactivity/manifests/ood_similarity.json` (the `results/` directory
itself is gitignored like the toxicity branch's, so a run worth keeping gets
copied into `manifests/` -- see `run_benchmark.py` / `stress/ood.py` for how to
regenerate either from the frozen splits above). 15/27 tasks clear the
eligibility floor and feed the macro; the other 12 stay in the dataset for
training signal but are reported per-task only.

| view | role | b0 median | b1 kNN | b2a RF | b2b LightGBM |
|---|---|---:|---:|---:|---:|
| temporal | **PRIMARY** | 1.060 | 0.991 | 0.945 | 0.943 |
| cluster | secondary | 1.025 | 0.770 | 0.725 | 0.706 |
| scaffold | secondary | 0.982 | 0.595 | 0.582 | 0.551 |
| random | diagnostic | 0.940 | 0.511 | 0.509 | 0.470 |

(macro MAE, pChEMBL units, lower is better)

Two things worth registering before any of this is cited:

**The temporal (release-deciding) improvement over the dummy median is small.**
Best model beats B0 by ~11% macro MAE on temporal, versus ~50% on random. A
fixed-target classical model here is doing real but modest work on genuinely
prospective compounds; anything claiming much more should be checked against
this floor.

**Cliff direction accuracy sits at 0.50–0.58 on temporal** — barely above
chance for a binary "which of the two is more potent" call, reproducing
MoleculeACE's finding on this panel. A candidate architecture's cliff-loss
ablation (plan A6, section 8.3) has a low bar to clear here, but clearing it
should still be treated as a real result given how weak these baselines are.

**Difficulty ordering validates the split design, with one caveat.** Measured
directly via each test compound's max ECFP4 Tanimoto to train
(`stress/ood.py`, `results/ood_similarity.json`):

| view | median max-Tanimoto to train | % with a close (>0.7) train neighbor |
|---|---:|---:|
| temporal | 0.416 | 5.6% |
| cluster | 0.526 | 17.2% |
| scaffold | 0.714 | **53.7%** |
| random | 0.790 | 79.4% |

This orders exactly as model difficulty does (temporal hardest, random
easiest), which is the sanity check a split suite should pass. The caveat is
**scaffold**: despite being the split most QSAR papers treat as a
generalization test, over half its test compounds have a near-duplicate in
train here — Bemis-Murcko scaffolds change on the ring core, not on
substituents, so two very similar molecules routinely land on opposite sides.
Treat scaffold as closer to an interpolation view than an OOD one for this
panel; **cluster is the only non-diagnostic view doing substantial OOD work**
besides the primary temporal view itself.

## B3.5: CheMeleon-initialized Chemprop (2026-09-15)

Frozen: `evals/bioactivity/manifests/benchmark_report_chemeleon.json`. Added
after a literature audit found CheMeleon (Burns, 2026 — plan footnote
[^19]) reports a 97% win rate against Chemprop/RF/fastprop specifically on
MoleculeACE activity-cliff pairs, which is exactly where B0–B2b are weakest on
this panel (cliff direction accuracy 0.50–0.58 on temporal, see above).

This baseline differs from B0–B2b in two ways worth flagging before reading
the table: (1) it is **one multi-task D-MPNN shared across all 27 tasks**, not
27 independent per-task fits — the whole point of fine-tuning a foundation
checkpoint — which means assay-context conditioning is lost (median-aggregated
per task_unit_key; see `models/chemeleon_chemprop.py` docstring); (2) it runs
under a **different Python interpreter** (`comosa_phase1`, Python 3.11 — see
"CheMeleon environment" below), never in-process with `run_benchmark.py`, via
the standalone driver `bench_chemeleon.py` (which reuses `run_benchmark.py`'s
scoring functions directly rather than duplicating them). 40-epoch budget,
patience-5 early stopping (all four views stopped between ~26 and ~38 epochs,
none hit the cap); ~45–70 min fit time per view on one RTX 3090.

| view | b0 | b1 kNN | b2a RF | b2b LightGBM | **CheMeleon** | cliff dir-acc (b2b → CheMeleon) |
|---|---:|---:|---:|---:|---:|---:|
| temporal (**PRIMARY**) | 1.060 | 0.991 | 0.945 | 0.943 | 0.971 | 0.527 → **0.560** |
| cluster | 1.025 | 0.770 | 0.725 | 0.706 | 0.706 | 0.655 → **0.703** |
| scaffold | 0.982 | 0.595 | 0.582 | 0.551 | 0.548 | 0.690 → **0.718** |
| random | 0.940 | 0.511 | 0.509 | 0.470 | 0.478 | 0.766 → **0.824** |

(macro MAE, pChEMBL units; cliff direction accuracy 0.5 = chance)

**Read this carefully before citing it.** On macro MAE, CheMeleon is
essentially tied with LightGBM — marginally worse on temporal and random,
marginally better on cluster and scaffold, all within a few percent. It does
**not** clearly win the panel's primary release metric. What it does do,
consistently across all four views with no exceptions, is improve cliff
direction accuracy by roughly +3 to +6 percentage points over the best
classical baseline. That is a real, reproducible effect in the direction the
literature predicted, but it is a modest one here, not the dramatic
MoleculeACE result — the difference is architecture of the test itself:
MoleculeACE evaluates single, curated, matched-pair assays, while this run is
multi-task across 27 heterogeneous ChEMBL tasks with real curation noise and
assay-context collapsed by aggregation. **Conclusion for this panel:**
CheMeleon is worth keeping in the model tournament specifically for
cliff-sensitive use, not as a replacement for LightGBM on aggregate MAE.

### CheMeleon environment

Chemprop 2.x (required for `--from-foundation CHEMELEON`) needs Python >=3.11;
`drug-tox-env` (home to `run_benchmark.py` and B0–B2b) is 3.10 and cannot
import it. This machine's only Python 3.11 env is `comosa_phase1`, otherwise
unrelated to this project — rdkit + chemprop>=2.2 were installed into it after
a `pip install --dry-run` confirmed zero changes to its existing torch/numpy/
pandas/sklearn pins. Reproduce with:

```bash
/home/mluser/.conda/envs/comosa_phase1/bin/python evals/bioactivity/bench_chemeleon.py \
    --dataset ../../data/bioactivity/hq-v1/hq_exact.csv.gz \
    --splits evals/bioactivity/manifests/splits \
    --out evals/bioactivity/results \
    --views temporal,cluster,scaffold,random --epochs 40
```

## Known limitations

- The cluster-OOD view uses MiniBatchKMeans over ECFP4, not Butina. Butina needs
  the full pairwise similarity matrix, which is quadratic and infeasible at
  panel scale. The manifest records the substitution; do not cite this view as
  literature-standard Butina.
- `target_type = SINGLE PROTEIN` cannot be expressed on the activity endpoint,
  so it is enforced by restricting the panel at selection time.
- The `Censored-Extended` track (`<`, `>` relations) is not implemented. It
  needs a censored likelihood; do not add those records to HQ-Exact.
- B3–B6 and the ToxAct-TAC-MoE candidate (plan sections 5–6) are not built.
  Chemprop is not installed in `drug-tox-env`.

## Layout

```
research/bioactivity/
├── data/
│   ├── chembl_client.py      release pin, HQ-Exact filters, paging
│   ├── target_annotation.py  protein family + gene symbol resolution
│   ├── profile_targets.py    P0 pass 1 and 2
│   ├── select_panel.py       pre-registered panel rule
│   ├── standardize.py        parent structure, connectivity key
│   ├── build_dataset.py      extract + aggregate + manifest
│   └── split.py              the four frozen views
└── models/
    ├── featurize.py          ECFP4 + descriptors, cached
    └── ecfp_baselines.py     B0 median, B1 kNN, B2a RF, B2b LightGBM

evals/bioactivity/
├── metrics.py                regression, ranking, cliffs, calibration
├── stress/activity_cliffs.py held-out cliff pair construction
├── manifests/                panel + split manifests
└── run_benchmark.py          reads frozen splits, never re-splits
```

Both `bioactivity` directories are PEP 420 namespace packages (no
`__init__.py` at their top level) so `bioactivity.models` and
`bioactivity.metrics` resolve from their separate trees.

Tests: `$PY -m pytest tests/unit/test_bioactivity_data_contract.py`
