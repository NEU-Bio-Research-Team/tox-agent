# Benchmark protocol

How the numbers in the [model card](model-card.md) are produced, and the rules
that keep them meaningful.

## Two commands

```bash
# Once. Freezes the evaluation split; needs DeepChem and PyTDC.
python benchmarks/build_split_manifest.py

# Every time. Needs only the frozen manifest and the artifact.
python benchmarks/run_benchmark.py
python benchmarks/run_benchmark.py --limit 150      # smoke size
```

Output lands in `benchmarks/results/`: `benchmark_report.json` (tracked) plus
per-sample dumps (gitignored, regenerated on demand).

## Rules

**The split is frozen, and nothing re-splits at run time.**
`benchmarks/manifests/eval-split-v1.json` holds the test molecules with a content
hash, reproducing scaffold/seed 42 — the values in the artifact's `config.yaml`.
A unit test recomputes the hash, so an edited split is a failure rather than a
quietly moved goalpost.

This rule exists because of a real defect: `backend/data.py` falls back from
DeepChem to PyTDC when the former is missing, and the fallback path replaces the
scaffold split with a random 80/10/10. Any metric computed through it would put
training molecules in the test set. `build_split_manifest.py` fails rather than
substituting a loader.

**Thresholds are applied, never fitted.** Fitting on the test split is how a
calibration silently becomes a leak. Thresholds come from the artifact and the
report records the source.

**Missing Tox21 labels are masked, not imputed.** Tox21 is sparsely labelled;
coercing a missing assay to 0 invents negatives and inflates specificity. Each
task reports its own `n` and `n_masked_out` — `SR-ARE` is scored on 481 of 783
molecules, and the report says so.

**Endpoints are scored separately.** No pooled accuracy across hERG and Tox21;
averaging over chemically unrelated assays produces a number that means nothing.
Tox21 is macro-averaged over tasks, never pooled over molecules.

**Benchmarking goes through the application service** — the same code path the
API serves, so a benchmark number and an API answer cannot disagree. It chunks to
respect the 256-molecule batch cap rather than lifting it.

## Reported metrics

Per endpoint (and per Tox21 task): n, positives, prevalence, threshold and its
source, AUROC, PR-AUC, F1, MCC, balanced accuracy, sensitivity, specificity,
Brier, 10-bin ECE, confusion counts, and seeded percentile bootstrap 95% CIs
(1 000 resamples, seed 20260903) for AUROC, PR-AUC, F1 and MCC. Plus per-molecule
latency.

Read PR-AUC, not AUROC, on the rare assays. `NR-PPAR-gamma` has 3.8% positives:
AUROC 0.742 against PR-AUC 0.092.

## Reproduction check

Every full run recomputes the metrics the artifact recorded at training time and
compares them at 5e-3 tolerance. It passed at Δ ≤ 7e-5, which checks the split,
the ported provider and the scoring in one shot. A failure means one of the three
drifted.

## Provenance

Each report carries the git commit, split hash, Python and torch versions,
bootstrap settings and the threshold policy. A number without its manifest is not
a result.

## What is not benchmarked

**ClinTox** — not served (model card §6). The report records the reason rather
than omitting the endpoint.

**Cross-model comparison** — this protocol measures one artifact. Comparing
models requires the same split and the same protocol; a comparison across
different splits is not a comparison.
