---
name: explain-predictor-results
description: Request an explanation for one endpoint or assay and connect its figure to the extracted contributors, without turning attribution into mechanism.
---

# Explain predictor results

One explanation, one endpoint — and for Tox21, one assay. Call
`get_or_create_explanation` per target. It returns an `explanation_id`, the
`observation_id` every explanation claim must cite, a `figure_id` for the stored
diagram, the ranked positive and negative contributors, and
`unmapped_importance`.

The figure and the contributor lists come from the same computation. Reference
the figure only in the section that discusses that endpoint, and never describe
a figure as showing something the contributor list does not say.

## Attribution is not mechanism

Write "the model's score for this endpoint responded most to the region around
atoms 7–9" — not "the piperazine ring causes hERG blockade". Attribution
describes the model's behaviour on this input. It is not evidence about
chemistry, it does not identify a pharmacophore, and a reader who takes it that
way has been misled by the wording, not by the number.

Every explanation claim carries `attribution_not_causality`.

## Both directions

Positive and negative contributors are separate lists and are discussed
separately. An atom pushing the score *down* is a real finding. Collapsing both
into "the most important atoms" deletes half the explanation.

## Say what the explanation does not cover

`unmapped_importance` is the share of attribution mass that landed on no atom —
tokenizer artifacts, structure the alignment could not place. When it is
non-null and greater than zero, the report says so. Silence there overstates how
much of the score the picture accounts for, and the validator will reject a
draft that shows a partial explanation without mentioning it.

Status `partial` is disclosed the same way.

## A failed explanation

`status: failed` means no figure and no contributors. Record a gap with reason
`explanation_failed` on `explanation_and_visuals`, naming the endpoint and task,
and keep the section. Do not describe an explanation you did not get, and do not
substitute the explanation of a different endpoint.

See `references/explanation-policy.md`.
