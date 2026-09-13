---
name: assemble-report-context
description: Select and read the minimum complete set of substance, predictor, applicability and provenance facts for a report build.
---

# Assemble report context

Read what the report is about before writing any of it. This step is cheap and
its omissions are expensive: a number you did not read is a number you will
either leave out or invent.

## Order

1. `get_report_context` — the manifest. It tells you the analysis, the selected
   endpoints, **which of them the analysis actually served**, the Tox21 assays
   chosen, whether explanations and evidence were asked for, the required
   section ids, and how many correction attempts remain.
2. `get_analysis_bundle` — the prediction summary for every served endpoint,
   the explanations that already exist, and full provenance, in one call.
3. `get_analysis_slice` — per section, for the exact `field_path` values a
   numeric or classification claim must cite. The bundle tells you what exists;
   the slice is what makes it citable.
4. `resolve_compound_record` — identity and bulk properties, by structure.

## Rules

**Include every requested and served endpoint.** Not the interesting ones, not
the ones with high scores. A report that quietly covers two of three selected
endpoints is wrong in a way its reader cannot detect.

**Keep unavailable endpoints visible.** `unavailable_endpoints` in the manifest
is the list of things the user asked for and this analysis does not have. Each
becomes a gap with reason `endpoint_not_served`, attached to
`predictor_results`. Never substitute a related endpoint, and never let the
report imply the endpoint was fine.

**Preserve exact numbers and field paths.** Copy `source_value` from the slice
into the claim unchanged. Rendering is a transform (`round:3`, `percent:1`) that
the server re-applies; the raw value is not yours to adjust.

**Never use a literature value as a predictor output.** If a paper reports an
IC50 and the model reports a probability, those are two facts from two sources
in two sections. Merging them is the single most damaging thing this step can do.

**An unresolved identity is a gap, not a guess.** When
`resolve_compound_record` answers `resolved: false`, or leaves a field null,
that field stays null and `substance_profile` carries a
`compound_identity_unresolved` gap. Never fill it from a compound with a
similar structure or a similar name.

## What to carry forward

For each served endpoint: the `observation_id`, every `field_path` you will
cite, the probability, the label, the threshold and its source, the model id,
and the applicability status. For the compound: whichever identity fields
resolved, and the source ref for each.

See `references/report-context-contract.md` for the exact field paths.
