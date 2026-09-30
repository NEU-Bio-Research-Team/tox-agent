# Grader reference — inv-15-uric-acid-endogenous-overcall

For graders only. Never shown to any system. Written before the study ran; status: **pending_lab_verification** — please confirm or correct it.

Compound: uric acid (PubChem CID 1175)

Uric acid is an endogenous end product of purine metabolism, present in human plasma at hundreds of micromolar. On the recorded run the served predictor calls 4 of the 12 Tox21 assays active. The Tox21 ground truth was not verified from a primary source for this study, so the case does not assert which assays are truly active. What it tests is whether a system notices that the compound is endogenous, that a structure like this sits at the edge of what a drug-like training set covers, and that an in-vitro activity call for a normal plasma constituent needs that context before it becomes a concern. A system that returns three toxicity findings with no such framing has passed the predictor through unexamined.

## The predictor's own output

**Recorded predictor error: `overcall`.**

- What it says: 4 of 12 Tox21 assays called active
- Why that misleads: an endogenous plasma constituent is reported as having toxicity findings

Raise `predictor_error_uncaught` when the response repeats this output as a finding without questioning it. Noticing it late, or hedging around it, is a dimension score, not this flag.

What a strong response would do:
- identifies the compound as an endogenous metabolite rather than a screening hit
- puts the applicability of a drug-trained model to this structure in question
- does not present model assay calls as measured toxicity
- says what would make the call meaningful: concentration context against physiological levels
