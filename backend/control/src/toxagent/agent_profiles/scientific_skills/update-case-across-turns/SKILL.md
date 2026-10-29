---
name: update-case-across-turns
description: Use when the researcher adds something to the case between turns -- a measured value, an exposure, an assay condition -- or turns to another screening of the same ligand. Helps decide whether the new information is about this subject, what it changes in the assessment and what it must leave untouched, and when a different target is a different case.
---

# Update the case across turns

A later turn is the same investigation only while it is about the same
subject. New information changes the assessment; it does not change what the
model predicted or what the policy decided.

## When this applies

- The case shows researcher context added since the last turn (an item with
  a key, a value, a unit, a source and a scope).
- The researcher says they measured something, or gives an exposure, a dose,
  a species or an assay condition.
- The run's subject is a screening of a ligand you have already discussed on
  another target.

## Is it about this subject?

Read the item's `scope` and unit before anything else:

- A value scoped to another target, variant or assay context is not evidence
  about this subject. Say so; do not fold it in.
- A value with no unit, or in a unit that does not fit the quantity (a
  pIC50 given in nM), is a question for the researcher, not a number to fix.
- Researcher-supplied context is what the researcher states. It is citable as
  such (`source_class` user_supplied, `source_ref` the context id); it is not
  an SME-verified assay unless the case says it is.

## What changes, what stays

Changes, when the new information bears on it: the hypothesis statuses, the
uncertainties (a `missing_exposure` that is now supplied is resolved), the
conclusion and the next steps -- each with the ledger entry behind it.

Stays, always: the predicted value, the prediction interval, the policy's
decision and its threshold, any stage's status, and what was authorised. A
measured value below the threshold weakens the case for prioritising; it does
not turn the policy's decision into another one, and only the policy's owners
can change a threshold.

State in the answer what changed since the last turn and what did not.

## Another target is another case

When the subject becomes the same ligand on another target, the server opens
a different case. Nothing concluded about the first target carries over: not
the estimate, not the measured value, not the conclusion. What is about the
molecule itself (its toxicity analysis, when the screening links one) may be
read again for the new subject; anything about the first target's activity
may not.

## What to leave in the case

- `record_evidence` for the supplied value, scoped as the researcher scoped it.
- Resolve or add the uncertainty it bears on.
- `record_action` naming what changed and why, and a revised conclusion only
  with ledger support.
