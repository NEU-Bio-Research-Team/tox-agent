# ADR 0013: Activity-guided screening is a sequential, typed workflow

**Status:** Proposed
**Date:** 2026-10-20

## Context

ToxAgent currently serves hERG and Tox21 classification from a toxicity
predictor whose endpoint contract is probability-based.  The proposed SAIR
activity capability is a target-conditioned pIC50 regression.  pIC50 and
toxicity probabilities do not share a scale or scientific meaning, so a late
fusion score would be uninterpretable.

The SAIR dataset is publicly distributed but is gated on Hugging Face and is
not present in this repository.  We can therefore define and test the boundary
without claiming that a trained SAIR artifact exists.

## Decision

1. Activity prediction has a separate typed contract and model registry from
   toxicity prediction.
2. The existing hERG/Tox21 predictor contract is not widened to accept pIC50.
3. The control plane owns a sequential workflow:
   activity -> target-specific prioritisation -> optional toxicity.
4. Prioritisation is a versioned policy over pIC50 uncertainty and
   applicability, not a model-authored active/inactive label.
5. Activity and toxicity persist as separate immutable snapshots linked by an
   `ActivityGuidedScreening` aggregate.
6. A toxicity stage that was not run is represented explicitly and is never
   projected as a negative or safe result.
7. Until a real SAIR file is inspected, only contract/domain code and synthetic
   tests may be called complete.  No production route is advertised.

## Consequences

- The activity service can later move to a separate process without changing
  workflow semantics.
- A new persistence migration, repository implementation, client, route and UI
  are still required before end-to-end use.
- SAIR schema, license acceptance, dataset identity, splits, model performance,
  uncertainty and artifact admission remain scientific release gates.
- Existing quick prediction and toxicity analysis behavior remains unchanged.

## Rejected alternatives

### Add `activity` to the toxicity endpoint enum

Rejected because the surrounding contracts assume probabilities, thresholds
and endpoint-specific classification semantics.

### Feed pIC50 into the toxicity model

Rejected because activity does not provide toxicity ground truth.  It selects
which candidates receive additional screening.

### Compute one activity-toxicity fusion score

Rejected because the inputs are not commensurate and the resulting number has
no validated scientific interpretation.
