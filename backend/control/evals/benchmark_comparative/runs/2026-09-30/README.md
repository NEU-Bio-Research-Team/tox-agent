# ToxBench run, 2026-09-30

First three-way run: ToxAgent against the ChatGPT and Gemini web answers
collected by Duc Minh (commit `b5a4452`). The table is in
[comparative_report.md](comparative_report.md).

## What each arm is

| Arm | Input it saw | Source |
|---|---|---|
| `toxagent` | SMILES only, via `POST /v1/predict` on the local stack (`./bin/toxagent up`, repo at `9cb4ae1`) | `herg-tox21-chemberta-v1`, predictor `0.1.0.dev0`, policy `tox-policy-v1`; artifact hashes in each raw response |
| `chatgpt` | `prompts_web/` (unblinded) pasted into the web UI | `web_results/chatgpt_batch_*.json`; model id and date were not recorded |
| `gemini` | same | `web_results/gemini_batch_*.json`; model id and date were not recorded |

## Why the arms are not yet comparable

1. **The web prompts leak the answer.** Each case is shown with its dataset id
   (`herg_pos_01`, `herg_neg_05`), the compound name, and for borderline cases
   a vignette saying the activity is borderline. Both web models were 100%
   accurate on the hERG calls they committed to, and answered `uncertain` on
   all ten borderline cases. That measures recall of the label, not
   prediction. `prompts_web_blind/` removes all three; the web arms should be
   re-collected from it.
2. **The ToxAgent arm is the predictor, not the agent.** `/v1/predict` returns
   structured predictions. The driver builds a one-line text from them, so the
   regex hallucination traps and safety gates pass it by construction, and it
   never emits `screening_not_safety_assessment`, which only the agent's report
   path adds.
3. **Every metric here is an automatic proxy.** Limitations coverage is
   lexical; hallucination and safety are regex checks. The lab's blind grading
   is the quality read.

## What the numbers do show

- ToxAgent commits a hERG call on all 50 labelled cases: 16/20 known
  blockers, 20/20 known non-blockers, 6/10 borderline.
- ToxAgent Tox21 precision is low (0.186, recall 0.615). It calls many assays
  active on the 20 labelled Tox21 cases. The web models are more selective,
  but they had the compound names.
- Two ToxAgent inputs were classed `out_of_domain` and three `limited` by the
  applicability check. The invalid SMILES (`edge_05`) was refused with HTTP 400.

## Files

- `toxagent-*.json`: every request's raw response, with its per-case evaluation.
- `cases_<system>_*.json`: each web answer with its per-case evaluation. Refusal
  matches that were not counted as hallucinations are under `suppressed_spans`.
- `scorecard_<system>_*.json`: the aggregates that `compare_scorecards` reads.
