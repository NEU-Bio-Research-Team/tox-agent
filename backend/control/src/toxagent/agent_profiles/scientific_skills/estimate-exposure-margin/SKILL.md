---
name: estimate-exposure-margin
description: Use when the question turns on how a potency compares with the concentration a compound reaches -- an IC50 against a free plasma Cmax, a margin at an intended dose. Helps decide whether the ratio can be computed from what the case holds, compute it with the server's tool rather than by hand, and ask the researcher for the exposure instead of assuming one.
---

# Estimate an exposure margin

A potency on its own does not answer a question about an intended dose. A
margin is a ratio of two sourced numbers in compatible units, and it is only
as informative as the weaker of the two.

## When this applies

- The question asks whether a potency matters at the intended dose, or asks
  for a margin or a ratio.
- The case holds, or the researcher has supplied, a free Cmax (or a total
  Cmax with a fraction unbound).

It does not apply when the question only asks what a value is, and it does
not apply when no exposure exists anywhere in the case -- that is the next
section, not a reason to pick a number.

## No exposure in the case

The exposure of a programme's own compound is not in the literature. Do not
search for it, and do not carry one over from another compound or another
species.

Instead: record an uncertainty of kind `missing_exposure` (severity
`blocking` when the question turns on it), say in the answer which exposure
you need (free plasma Cmax at the intended dose, or total Cmax with the
fraction unbound), and stop there. On a question the case cannot settle,
naming the missing input exactly is the correct answer.

## Computing it

Use `compute_exposure_margin`: it takes the IC50 and the Cmax each as a value
with a unit, and a fraction unbound when the Cmax is total. The server
converts the units and returns the ratio with what it used.

- Never compute the ratio in your head or in prose, and never state a margin
  the tool did not return.
- Both numbers need a source: the IC50 from a record you read (quote the
  table row), or the screening's own estimate as what it is -- a prediction,
  not a measurement; the Cmax from the researcher's context, cited as
  user-supplied.
- A pIC50 is not a concentration. Convert it to an IC50 first (IC50 in nM is
  10 ** (9 - pIC50)) and say that you did.
- The two must be the same kind of quantity. A binding affinity against a
  functional block, or a free concentration against a total one, is not a
  margin -- it is a mismatch to name.

## What the ratio does and does not say

It says how far apart two numbers are, in one assay, for one endpoint, at
one exposure. It is not a safety assessment, not a probability of an effect
in a person, and not a clinical or regulatory conclusion. A margin computed
from a predicted potency inherits the prediction's interval: give the range,
not only the point ratio, and owe the limitation.

## What to leave in the case

- `record_evidence` for the potency and for the exposure, each with its
  scope, directness and source class.
- The `missing_exposure` uncertainty resolved when the researcher supplied it,
  or recorded when they have not.
- A conclusion that states the ratio, the inputs it came from, and what would
  change it.
