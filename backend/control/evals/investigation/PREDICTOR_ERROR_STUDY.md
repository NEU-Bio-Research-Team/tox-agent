# Predictor-error study: what happens when ToxPred is wrong

The comparison study in this directory asks whether a system reasons well about
a compound. This set asks one narrower question:

> When the served predictor's output is wrong or misleading, does the system
> notice it, or does it pass the prediction through as a finding?

That question is about the **agent**, not the predictor. It cannot be answered
by `POST /v1/predict`, which is what `evals/benchmark_comparative` measures:
that path returns structured numbers and runs none of the agent's reasoning,
retrieval or drafting. A predictor benchmark and this study measure different
products and neither substitutes for the other.

## How the cases were chosen

Candidates came from a **recorded ToxPred run over the ToxBench compounds**
(`evals/benchmark_comparative/runs/2026-09-30/`), not from memory: compounds
where the served predictor's own output diverges from published pharmacology.
Each case was then checked against the live stack before the study ran, so the
`predictor_error` block in each case sheet describes what the predictor
actually does, not what it was assumed to do.

Structures are resolved from PubChem by `cases.py`; no SMILES is typed by hand.

| Case | Compound | What ToxPred says | Why that misleads |
|---|---|---|---|
| `inv-11` | sotalol, named | hERG p≈0.31, non-blocker | Sotalol's therapeutic mechanism **is** IKr/hERG blockade, and it causes torsades (~1% below 320 mg/day, up to ~5% above). It is genuinely a *low-affinity* blocker, so the score is defensible on affinity — but reading it as "hERG is not a concern for sotalol" is false reassurance about an established torsadogen. |
| `inv-12` | sotalol, anonymous | same | Same compound with the name withheld: can the system say what it cannot know? |
| `inv-13` | moxifloxacin | hERG p≈0.14, non-blocker | The question proposes it as a *negative* control. Moxifloxacin 400 mg is the standard **positive** control in ICH E14 thorough-QT studies (mean QTc +7.5–12.5 ms). The premise is wrong and the low score appears to endorse it. |
| `inv-14` | catechin | 8 of 12 Tox21 assays active | Eight of twelve, spanning unrelated nuclear-receptor and stress-response readouts, is the signature of a non-specific or cytotoxicity-driven call, not twelve independent findings. |
| `inv-15` | uric acid | 4 of 12 Tox21 assays active | An endogenous purine end product, present in plasma at hundreds of µM, reported as having toxicity findings. |
| `inv-18` | 4,4′-biphenol | 10 of 12 Tox21 assays active | Same breadth problem, but harder: a bisphenol-like structure has a real basis for oestrogen-receptor activity *specifically*. The discriminating behaviour is to keep that one and question the rest. |
| `inv-16` | succinic acid | nothing active, hERG far below threshold | **Control.** Predictor and chemistry agree. |
| `inv-17` | fexofenadine | hERG p≈0.25, non-blocker | **Control.** The drug that replaced terfenadine precisely because it lacks the cardiac liability. |

The two controls are not padding. Without them the study rewards a system that
hedges on everything, which is not the capability being measured. A system
should be *more* definite on `inv-16` and `inv-17` than on `inv-14` and
`inv-15`.

### What the questions do not say

No question mentions the predictor, calls anything wrong, or uses the words
that would hand over the finding (`torsade`, `positive control`, `endogenous`,
`overcall`, …). A test enforces this (`test_the_question_never_gives_the_finding_away`).
This is deliberately unlike pilot case `inv-08`, which told the system that
moxifloxacin blocks hERG at tens of micromolar; `inv-13` gives nothing away.

## What this study does not establish

- **Tox21 ground truth is not asserted.** For catechin, uric acid and
  4,4′-biphenol the assay-level truth could not be confirmed from a primary
  source, so those references are `reasoning_expectation`: they describe the
  reasoning a competent answer shows, not which assays are truly active. The
  ToxBench label set was **not** used as truth here — it is hand-entered
  without source ids.
- **Nothing here grades quality.** As with the pilot, the harness runs, logs
  and blinds; the chemistry lab grades. Any number this produces about process
  (turns, tool calls, refusals) is not a quality score.
- **Eight cases is small.** Intervals will be wide. That is the honest result.

## The rubric flag

`rubric.json` gains one critical-error flag:

> `predictor_error_uncaught` — The served predictor's output for this case is
> wrong or misleading (the case sheet says so under `predictor_error`), and the
> response passes it through as a finding without questioning it.

It is bounded on purpose: a case whose `kind` is `none` can never raise it, and
doubting a correct prediction is not this flag. Otherwise the flag would
measure hedging.

## Running it

```bash
# 1. Build the cases (once; network, PubChem)
python -m evals.investigation.cases --build \
    --specs evals/investigation/case_specs_predictor_error.json \
    --cases-dir evals/investigation/cases_predictor_error

# 2. One control plane per ToxAgent arm
backend/control/evals/scripts/study_arm.sh C 8011 \
    TOXAGENT_FLAG_SCIENTIFIC_CASE_V1=0 TOXAGENT_FLAG_SCIENTIFIC_SKILLS_V1=0 \
    TOXAGENT_FLAG_CLAIM_REVIEWER_V1=0
backend/control/evals/scripts/study_arm.sh D 8012 \
    TOXAGENT_FLAG_SCIENTIFIC_CASE_V1=1 TOXAGENT_FLAG_SCIENTIFIC_SKILLS_V1=1 \
    TOXAGENT_FLAG_ANSWER_DRAFT_V2=1 TOXAGENT_FLAG_CLAIM_REVIEWER_V1=0

# 3. Run each ToxAgent arm ONE AT A TIME (the host has 7.8 GiB; three runners
#    at once exhausted it once before). Platform arms can run alongside.
export TOXAGENT_STUDY_TOKEN=...
python -m evals.investigation.run --study predictor-error-2026-09-30 \
    --cases-dir evals/investigation/cases_predictor_error \
    --systems A_predictor_template,D_toxagent_investigator \
    --toxagent D_toxagent_investigator=http://127.0.0.1:8012 \
    --snapshot-from http://127.0.0.1:8012 --trials 1

# 4. Packet for the lab, then the scorecard once grades come back
python -m evals.investigation.packet --study predictor-error-2026-09-30 \
    --cases-dir evals/investigation/cases_predictor_error --packet-id lab-pe-1
```

`--cases-dir` is what keeps this study's records, manifest and packet separate
from the pilot's. Passing the pilot directory here would join two case sets
into one study.
