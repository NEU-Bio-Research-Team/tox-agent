# Eval run output

`python -m evals.runner` writes two files here per run:

- `manifest-<ts>.json` — the run manifest (plan §16.9): eval-suite hash,
  toxagent/toxpred commits, runtime kind, trial count, and the summary.
- `results-<ts>.json` — per-task pass/fail with grader reasons.

Both are git-ignored (`.gitignore`), along with the `_work/` scratch databases.
The tracked artifacts are the task set (`../tasks/`), the frozen fixtures
(`../fixtures/`), the graders (`../graders/`) and the schema (`../schema/`).

Promote a manifest into a decision record by copying it out of this directory
under a stable name.
