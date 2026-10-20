# ADR 0014: Activity artifacts are bound to a subject, and a policy is scoped to one context

**Status:** Accepted
**Date:** 2026-10-20
**Amends:** ADR 0013

## Context

ADR 0013 established the sequential activity -> prioritisation -> optional
toxicity workflow.  A review of the first implementation found three gaps that
let correct-looking artifacts describe the wrong thing:

1. `ActivityGuidedScreening.record_activity` verified the session, the run and
   `target_id`.  It never compared the ligand.  A screening opened for `CCO`
   accepted a snapshot for `CCN` and advanced to `activity_completed`.
2. `TargetPriorityPolicy` was keyed on `target_id` alone.  Any `assay_context`
   was accepted, so a threshold calibrated against one kind of measurement
   could decide a snapshot from another.
3. Nothing recorded why a target was being screened.  The rule "the whole
   interval clears the threshold means prioritise" assumes the target is the
   intended one; for an anti-target the same number argues the opposite.

These are not agent errors.  They are reachable from the domain layer, so no
downstream benchmark could attribute them correctly.

## Decision

1. **Identity lives in a subject.**  `ActivitySubject` carries the canonical
   SMILES, the `TargetIdentity` and the assay context.  `ActivityGuidedScreening`
   holds one, and `ActivitySnapshot` exposes the subject it claims.  Every
   artifact a screening accepts must match on that triple.
2. **The domain compares canonical forms; the boundary produces them.**  The
   screening keeps `input_smiles` as provenance of what the caller sent, and
   the application boundary normalises it into the subject.  The domain never
   infers that two spellings are the same molecule.
3. **Target fields are compared when both sides declare them.**  A caller may
   open a screening knowing only `target_id` and let the predictor resolve the
   UniProt accession and sequence digest.  Once both sides carry a value, a
   difference is a mismatch.
4. **The policy key is `(target_id, assay_context)`.**  A policy refuses a
   snapshot measured in another context rather than comparing two different
   quantities.
5. **A policy declares the target role it is valid for.**  Only
   `TargetRole.INTENDED_TARGET` has a defined rule today.  Other roles raise
   `UnsupportedTargetRole` instead of being silently scored with the intended
   target's rule.
6. **The decision carries its own scope.**  `PrioritisationDecision` records
   the assay context and the role it assumed, and a screening rejects a
   decision whose scope differs from its own.

## Consequences

- `ActivityGuidedScreening.create` now takes `subject` and
  `intended_target_role` instead of `target`.  A `target` property is kept for
  readers.  The aggregate has no persistence yet, so no migration is owed.
- `ActivityPredictionRequest` gains a required `assay_context`.  The response
  already required one, so the caller previously could not see or check the
  context the server chose.
- Mismatches raise `ActivityIdentityMismatch`, which names the fields that
  differed, so a boundary can report which identity was wrong.
- Supporting off-target screening means adding a rule for that role, not
  relaxing the guard.
- Context routing is now explicit rather than implicit.  A deployment that
  wants to serve one context only can do so by publishing one policy per
  target, and that choice is visible in the policy id.

## Rejected alternatives

### Compare the raw `input_smiles` strings

Rejected because two spellings of one molecule would read as a mismatch while
a canonicalisation difference between caller and predictor would read as
agreement.  The comparison has to happen on canonical forms.

### Let the domain canonicalise

Rejected because it would put a cheminformatics dependency behind the domain
boundary that ADR 0001 keeps clear, and because the application layer already
owns normalisation for toxicity analyses.

### Default the target role to `intended_target`

Rejected because the silent assumption is the defect.  A caller that has not
decided what the target is for has not yet established that the threshold
points the right way.

### Treat a resolved sequence digest as a mismatch

Rejected because it would force every caller to know the sequence hash before
opening a screening, which the predictor is better placed to resolve.
