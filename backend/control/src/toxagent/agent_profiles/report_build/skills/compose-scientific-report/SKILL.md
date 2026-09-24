---
name: compose-scientific-report
description: Build the eleven required report sections from grounded facts, explanation packages and evidence synthesis, keeping the source classes apart.
---

# Compose the scientific report

## Structure first

All eleven required section ids, every time. Fill each from what you gathered;
where you gathered nothing, attach a gap and say so in the section body. A
missing section is refused; an honest gap is not.

`references/report-schema.md` has the section list and the draft shape.

## Keep the classes apart

Each section declares its `source_classes`, and the validator checks claims
against them. In practice:

- `predictor_results` holds model output. No literature values.
- `external_evidence` holds what sources say. No model probabilities.
- `integrated_interpretation` is where the two meet, and it says so in words:
  "the model's hERG score is high; the two records read here report activity at
  micromolar concentrations in a different assay format, so they are consistent
  in direction and not comparable in magnitude."

A sentence combining a model number and a literature number without saying
which is which is the failure this whole section exists to prevent.

## Numbers are claims

Every number in a section body has a claim behind it, cited to an
`observation_id` and `field_path`, listed in that section's `claim_ids`. A
number written into prose with no claim is refused by coverage checking, and
rightly: nothing links it to a source.

Do not restate two source numbers in prose to imply a comparison. Make the
comparison a `comparison` claim with `input_claim_ids`, and write the result.

## Conclusions

One per endpoint, naming its endpoint, with `basis_claim_ids`. If you also want
an overall reading, mark it `is_integrated: true` and keep it interpretive —
where results agree, where they do not, what would resolve it. It is not a
verdict, and there is no field in which a verdict would fit.

## Recommendations

Each has `basis_claim_ids`, an `action_category`, a `priority`, a `rationale`
and any `conditions`. They propose validation work: run this assay, review this
source, collect this data, consider this modification. They never prescribe,
never diagnose, and never state that following them makes anything safe.

`no_action_indicated` is a legitimate category when the screening result does
not warrant follow-up — it still needs a basis claim.

## Limitations

Declare every required code. They are derived from what you claimed, so leaving
one out does not remove the obligation, it fails the draft:

- `uncalibrated_probability` — any probability interpreted
- `applicability_is_rule_based` — any applicability statement
- `attribution_not_causality` — any explanation
- `endpoint_unavailable` — any selected endpoint not served
- `evidence_scope_limited` — any external citation
- `screening_not_safety_assessment` — always, in every report

## Executive summary last

Write it after the rest. It introduces no fact that is not already in a section,
and it states the uncertainty as prominently as the finding.
