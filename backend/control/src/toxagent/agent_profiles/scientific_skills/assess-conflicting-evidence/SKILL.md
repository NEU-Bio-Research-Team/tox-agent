---
name: assess-conflicting-evidence
description: Use when sources in the case disagree about the same hypothesis, or when a retrieved study seems to contradict the predictor's signal. Helps decide whether the disagreement is real or a difference of compound, endpoint, assay, species or exposure, what to record, and when to ask the researcher instead of searching again.
---

# Assess conflicting evidence

Two sources that look contradictory usually disagree because they measured
different things. Your job is to find out which, not to pick a winner so the
answer reads cleanly.

## When this applies

- Two ledger entries bear on the same hypothesis with opposite stances.
- A retrieved record points one way and the predictor's score the other.
- The researcher's own result disagrees with the literature.

It does not apply to a question that only asks for a predicted value, or when
every source agrees. Do not load it for those.

## Questions that usually settle it

Work through what the sources you already hold can answer; the order below is
the one that most often resolves a conflict early, not a script.

1. **Same compound?** The compound itself, a salt or prodrug, a structural
   analogue, or a class statement. Only the first is `direct`.
2. **Same endpoint?** For hERG, a binding displacement assay, a thallium-flux
   assay and patch clamp measure different things (see the reference). A
   Tox21 assay call is an in-vitro reporter readout, not an in-vivo effect.
3. **Same system?** Species, cell line, in vitro vs in vivo, temperature.
4. **Same exposure?** A potency number means little without the concentration
   the compound reaches. For hERG, compare the IC50 with the *free* plasma
   Cmax where it is known.
5. **Same measure?** An IC50 and a percent inhibition at one concentration are
   not comparable without the concentration.

Read the records' metadata (`get_evidence_record`) before searching for more
abstracts that say the same thing. A new source is worth its budget only if it
can separate the explanations you already have.

## Signs to change direction

- **The conflict dissolves on scope.** Record each source with its `scope`
  (endpoint, species, assay, dose) and the stance it has *within* that scope.
  An analogue's result against a direct result is `contextual` for the direct
  hypothesis, not `contradicts`.
- **Exposure decides it and nobody has it.** Record an uncertainty of kind
  `missing_exposure` (severity `blocking` if the conclusion turns on it) and
  ask the researcher. Searching the literature for the programme's own
  exposure is wasted budget.
- **Two direct sources really disagree.** Keep both in the ledger, leave the
  hypothesis `open` or mark it `unresolvable` with the reason, record a
  `conflicting_sources` uncertainty, and propose the test that would separate
  them (`discriminates` naming both hypotheses).
- **Only the model disagrees with direct experimental data.** The experiment
  wins on its own scope; the score is a `predictor_fact`, a signal about the
  model. Say so plainly.

## When to stop

Stop searching when the remaining disagreement is explained by scope, or when
the only thing that would settle it is data the researcher has to supply.
Record the reason with `record_action` (decision `answer` or `ask_user`).

## What to leave in the case

- `record_evidence` for each source that bears on the conflict, with `scope`
  and honest `directness`.
- An uncertainty for what is still unresolved, linked to the hypotheses.
- A hypothesis status change only with a ledger entry of that stance behind it.
- If a test would settle it: `propose_next_test` with expected readouts.
