# Committed eval evidence

Runs that a document, a paired comparison or the scorecard cites. Ad-hoc runs
do not land here: `python -m evals.runner` writes to `../results/` by default,
which is git-ignored as a whole (see `../README.md`, "Code, evidence and
scratch").

A run becomes evidence by being written here explicitly (`--out
manifests/<run-name>`), and is committed as the whole directory:

- `manifest-<ts>.json` — the run manifest: eval-suite hash, toxagent/toxpred
  commits, runtime kind, trial count, and the summary.
- `results-<ts>.json` — per-task pass/fail with grader reasons.
- `traces-<ts>.jsonl` — the `eval-trace-v1` projection the graders read.
- `config.env` — the flags and provider settings the run was launched with.
- `stdout.json`, `stderr.log` — the runner's own output.

`paired-<a>-<b>.json` files are `python -m evals.paired` comparisons between two
runs in this directory.

Loose files directly under this directory (`manifest-*.json`, `results-*.json`,
`_work/`) are still ignored, so a run written here without a run directory is
not committed by accident.
