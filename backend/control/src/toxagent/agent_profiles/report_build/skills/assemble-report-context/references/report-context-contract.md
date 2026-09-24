# Report context contract

## Field paths a claim may cite

Values come from `get_analysis_slice`, which returns each value with its own
`field_path` and `observation_id`. These are the declared paths:

| Section | Fields |
|---|---|
| `herg` | `probability_blocker`, `label`, `threshold`, `threshold_source`, `model_id` |
| `clintox` | `probability_clinical_toxicity`, `label`, `threshold`, `threshold_source`, `model_id` |
| `tox21` | `task_order_version`, `model_id`; per assay: `probability_activity`, `active`, `threshold`, `threshold_source` |
| `applicability` | `status`, `method`, `reasons` |
| `provenance` | `git_commit`, `service_version`, `artifact_hashes`, `model_ids` |

A Tox21 slice requires a `task`. The twelve assays are independent
measurements; there is no combined Tox21 probability and no path that resolves
to one.

## Claim shapes

A **numeric** claim: `observation_id` + `field_path` resolving to exactly one
numeric field, `source_value` equal to that field, `rendered_value` a single
number, `transform` one of `identity`, `round:0-6`, `percent:0-6`.

A **classification** claim: `observation_id` + `field_path` resolving to the
label or the active flag. No difference/ratio transform.

A **comparison** claim: `kind=comparison`, `transform=difference` or `ratio`,
`input_claim_ids` naming exactly the two numeric claims it is computed from, in
order. Do not write a difference as a `numeric` claim — a numeric claim's
`field_path` must resolve to one field, and a difference between two is not one.

A **scientific** claim: needs either an observation basis (a resolvable
`field_path`, or an explanation observation) or at least one evidence
`citation_id`. Neither is optional; a scientific claim with no basis is refused.

## Compound record

`resolve_compound_record` returns `resolved`, `preferred_name`, `synonyms`,
`identifiers` (`pubchem_cid`, `inchikey`, `cas`), `properties` (each with
`name`, `value`, `unit`, `source_field`), `canonical_url` and `source_ref`.

`cas` is null from this provider. That is a fact about the provider, not an
invitation to find one elsewhere.


## Field names come from the manifest, never from a guess

`get_report_context` returns `analysis_slice_fields` — the exact field names
each section of `get_analysis_slice` exposes — plus `tox21_assay_fields` and
`tox21_assays_available`. Read arguments out of those.

Guessing a field name is not free. The refusal names the allowed list, so a
guess does teach you the answer, but it spends one of the run's tool calls to do
it. A live build spent twelve of its forty that way and had none left to submit
the correction it was offered.

Only served endpoints appear in `analysis_slice_fields`: a section listed there
has data, and one absent from it has none and belongs in the report as a gap.
