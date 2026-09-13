# Violation codes and what to do about them

A rejected draft returns typed violations with a `path`. Fix exactly what they
name — a rewrite of something they did not name spends the attempt without
addressing the rejection.

## Structure

| Code | Fix |
|---|---|
| `missing_required_section` | Add the named section ids. |
| `duplicate_section` | One entry per section id. |
| `required_section_empty` | This section cannot be empty; write it. |
| `section_without_content_or_gap` | Add content, or attach a gap explaining the absence. |
| `gap_not_shown` | Reference the gap id from the section's `gap_ids`. |
| `gap_section_unknown` / `section_references_unknown_gap` | The gap's `section_id` and the section's `gap_ids` must agree. |

## Coverage

| Code | Fix |
|---|---|
| `endpoint_not_reported` | Add claims reporting the named served endpoints. |
| `unavailable_endpoint_hidden` | Add a gap for each requested, unserved endpoint. |

## Claims

| Code | Fix |
|---|---|
| `numeric_value_mismatch` / `classification_mismatch` | `source_value` must equal the canonical field exactly. Re-read the slice; do not adjust. |
| `rendered_value_mismatch` | `rendered_value` must be a single number consistent with `transform`. Move prose into `text`. |
| `claim_has_no_basis` | Add a resolvable `field_path`, or a read evidence `citation_id`. |
| `claim_transform_invalid_for_kind` | A classification claim cannot use difference/ratio. |
| `duplicate_claim_id` / `claim_id_not_unique` | Generate fresh 32-hex ids. |
| `claim_not_placed` | Reference the claim from a section's `claim_ids`. |
| `section_cites_unknown_claim` | The claim id is not defined in `claims`. |
| `source_class_mismatch` | The claim's kind or its citations do not belong in that section's declared classes. Move the claim, or correct the classes. |
| `unclaimed_number_in_markdown` | Every number in prose needs a claim in that section. |

## Explanations and figures

| Code | Fix |
|---|---|
| `explanation_not_found` | Use an `explanation_id` returned by a tool. |
| `explanation_target_mismatch` | The ref's endpoint/task must match the package's. |
| `figure_target_mismatch` / `figure_observation_mismatch` | The figure belongs to a different endpoint or computation. Do not reference it here. |
| `figure_not_found` | Reference only figures this build produced. |
| `unmapped_importance_suppressed` | State the partial or unmapped attribution in `explanation_and_visuals`. |
| `explanations_requested_but_absent` | Produce explanations, or record why not. |

## Evidence

| Code | Fix |
|---|---|
| `citation_not_found` | The evidence id does not exist in this session. Applies to a claim's `citation_ids` and to an `[@evd_...]` token in prose. |
| `citation_not_read` | Open it with `get_evidence_record` before citing it. |
| `citation_not_citable` | The record is rejected or superseded, not accepted. |
| `section_cites_without_declaring_external_evidence` | A section carrying an `[@evd_...]` token must declare the `external_evidence` source class. |
| `synthesis_without_evidence` | Only `insufficient` may name no record. |
| `evidence_conflict_suppressed` | Discuss the contradiction in prose. |
| `research_outcome_missing` | Record a synthesis, or a gap saying none was found. |
| `references_section_empty` | Write something in the References section; the numbered list itself is built by the server. |

## Conclusions and recommendations

| Code | Fix |
|---|---|
| `conclusion_scope_missing` | Name an endpoint, or set `is_integrated: true`. |
| `conclusion_endpoint_not_served` | This analysis has no such endpoint. |
| `conclusion_basis_unknown` / `recommendation_basis_unknown` | Basis claim ids must exist in `claims`. |
| `recommendation_without_basis` | Add `basis_claim_ids`. |
| `recommendation_guarantees_safety` | Remove the promise. Propose work, not an outcome. |

## Wording and safety

| Code | Fix |
|---|---|
| `prohibited_safety_verdict` / `prohibited_aggregate_verdict` | Remove the overall verdict. Endpoint-level statements only. |
| `prohibited_clinical_claim` | No diagnosis, dose or clinical instruction. |
| `prohibited_mechanism_claim` | Attribution is model behaviour, not mechanism. |
| `hitcount_severity` | Active-assay counts are not a severity. |
| `raw_html_in_report` / `html_event_handler_in_report` / `remote_image_in_report` | Report prose is plain text and locally stored figures only. |

## Limitations

| Code | Fix |
|---|---|
| `missing_required_limitation` | `expected` lists the codes. Add them. |
| `unknown_limitation_code` | Use a declared code. |
