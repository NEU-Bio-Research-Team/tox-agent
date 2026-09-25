---
name: interpret-model-attribution
description: Use when the question asks why the model scored a compound the way it did, or before any atom/token highlight is used in a conclusion. Helps state what an attribution can and cannot show, read its explainer_validation verdict, and record it in the case without turning a highlight into a mechanism.
---

# Interpret a model attribution

An attribution answers one narrow question: which parts of the input moved
*this model's* score for *this endpoint*. It is not a mechanism, not evidence
that a hazard exists, and not a check on the score itself.

## When this applies

- The researcher asks why the prediction came out as it did, or which part of
  the structure "drives" the risk.
- You are about to mention highlighted atoms or tokens in an answer or in the
  case conclusion.

Not needed when the question is only about the predicted value, or when no
attribution exists and none is asked for.

## What to check before you say anything

1. **Which model and head.** The attribution belongs to the model id and the
   endpoint (and Tox21 assay) it was computed for. It says nothing about
   another endpoint.
2. **The explainer's verdict.** Every attribution carries
   `explainer_validation`. Read `faithfulness_vs_random_control`:
   - `not_better_than_random_control`: on the measured panel the highlighted
     atoms did not move the score more than random atoms. Say that the
     highlight's faithfulness is not established.
   - `not_measured`: nobody has checked it for this target or method.
   - `better_than_random_control`: still a model signal only.
3. **Status.** A `partial` attribution is incomplete; do not treat its top
   tokens as a complete list.
4. **Applicability.** If the analysis is `limited` or `out_of_domain`, the
   score being explained is itself less reliable.

## How to word it

Say "the model's gradient attribution concentrates on …" — never "this group
causes …" or "the toxicophore is …". If the highlighted region overlaps a
known structural alert, that overlap is at most a *hypothesis*
(`add_hypothesis` with kind `mechanism`) that needs independent evidence; the
attribution cannot support it on its own.

## Signs to change direction

- The researcher wants a mechanism: say that attribution cannot provide one
  and look for experimental or literature evidence instead, or propose a test.
- The conclusion leans on the highlight: move that line to `cannot_say`, or
  back it with an independent source.

## What to leave in the case

- `record_evidence` with `source_class` `explanation_fact`, the attribution's
  observation ref, stance `contextual` (a highlight does not support or refute
  a hypothesis about the compound), and the endpoint in `scope`.
- When the attribution matters to the question and its verdict is
  `not_better_than_random_control` or `not_measured`: `record_uncertainty`
  with kind `explainer_faithfulness`.
- Declare the `attribution_not_causality` limitation in the answer.
