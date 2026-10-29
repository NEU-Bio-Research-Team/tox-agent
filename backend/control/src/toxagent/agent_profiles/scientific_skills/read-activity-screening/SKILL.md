---
name: read-activity-screening
description: Use when the run's subject is an activity-guided screening (a ligand on one target in one assay context) and the question is about its estimate, interval, policy decision or stage status, or about resolving a REVIEW decision. Helps read what each artifact is, what a REVIEW means, what evidence could settle it, and what never to infer from a stage that did not run.
---

# Read an activity screening

A screening is three different kinds of fact about one ligand on one target:
a model's estimate, a policy's decision, and the workflow's state. Keep them
apart in what you write and in what you record.

## When this applies

- The run's subject is a screening (`get_artifact_inventory` says
  `activity_screening`) and the question asks what it says, why it was
  prioritised, deprioritised or sent to review, or what would resolve it.

It does not apply to a toxicity-only question about an analysis, and a plain
lookup of one value needs no search at all: read the artifact and answer.

## What each artifact is

- **Activity estimate** (`get_activity_snapshot`): a predicted pIC50 with a
  prediction interval. The interval is marginal coverage on validation
  residuals across many compounds, not a guarantee for this one. A model that
  has not passed admission gives an unvalidated estimate; say so.
- **Policy decision** (`get_prioritisation_decision`): the output of a fixed
  rule applied to the estimate -- a threshold and its source. REVIEW means the
  interval crosses the threshold, so the estimate alone cannot place the
  ligand on either side. It is not a measurement and not your conclusion.
- **Stage status** (`get_screening_state`): which stages ran. A stage that did
  not run, or failed, is a state. It is never a result in either direction,
  and never a reason to call a ligand clear of an endpoint nobody measured.

A pIC50 is an in-vitro potency on this target in this assay context. It is
not efficacy, not selectivity, and not a value for another target.

## What settles a REVIEW

A measured potency for **this ligand on this target**, in a comparable
readout, can. Check, in the source's own table row where the text is held:

1. Same ligand (not an analogue or the series).
2. Same target, and the same variant where the screening names one.
3. Same kind of readout: an IC50, a Ki and a percent inhibition at one
   concentration are not interchangeable.
4. The unit. An IC50 in nM is a pIC50 of 9 - log10(IC50 in nM); do not
   convert in your head when a value decides the answer -- quote the row and
   state the conversion.

Record the measurement with `record_evidence`, `scope` naming the target and
the assay context, and honest `directness`. Compare it with the policy's
threshold in words; do not rewrite the estimate, the interval or the
decision -- they stay what the artifacts say.

## Signs to change direction

- **The source is about the ligand on another target.** It is context, not
  support or contradiction for this target. Another screening of the same
  ligand is another subject.
- **The source is about the target with another ligand.** It does not bear on
  this ligand's potency.
- **Only the abstract was read.** A number that sits in a table you have not
  read cannot be quoted as read; read the text or owe the limitation.
- **Nothing comparable exists in the corpus.** Say what measurement would
  settle it (target, readout, assay) and stop; a generic "not enough data" on
  a question the corpus settles is a failure, and so is searching past the
  point where the remaining gap is data only the lab can produce.

## What to leave in the case

- `record_evidence` for each measurement read, scoped to target and assay.
- An uncertainty when the decision still rests on the interval alone.
- A conclusion that says what the estimate and the evidence support for this
  target, and what would change it.
