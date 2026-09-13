---
name: preflight-report-draft
description: Check a draft against the known server validation rules before submitting it, to spend the single correction attempt on something real.
---

# Preflight the draft

## Save once, then patch by version

Call `save_report_draft` once with the complete candidate. It stores a durable
working copy and runs the same validator as final submission. It returns a
`draft_version` and typed violations while consuming **none** of the correction
attempts.

Fix only those paths with `patch_saved_report_draft`, passing the latest
`draft_version`. The patch is saved and revalidated in the same call. Never
resend or regenerate the complete report after the initial save. Its
`submit_if_valid` default is true, so the patch call performs final submission
itself as soon as every violation is cleared.

Do this even when the draft looks finished. There is exactly one correction
attempt and no fallback report, so an unchecked submission risks the whole
report on a bookkeeping slip: a claim no section references, a duplicate claim
id, a derived limitation code left undeclared. A live build lost an otherwise
complete eleven-section report to one missing `evidence_scope_limited`.

When a saved version returns `ok`, immediately call
`submit_saved_report_draft` with that exact version. Passing the check is not
acceptance — submission re-validates and remains the boundary. The old
`check_report_draft`/`submit_report_draft` pair is compatibility-only.

## The checklist below is for writing, not for verifying

Use it while assembling, so the first check comes back short. The list is not an
authority and it is not exhaustive; `check_report_draft` is what actually knows.

## Walk the list

**Structure**
- [ ] All eleven required section ids present, none duplicated.
- [ ] `executive_summary`, `substance_profile`, `predictor_results`,
      `conclusions`, `limitations`, `provenance_appendix` have real content.
- [ ] Every other section has content or a referenced gap.
- [ ] Every declared gap is referenced by the section it names.
- [ ] Every referenced gap id is declared.

**Coverage**
- [ ] Every selected *served* endpoint has at least one claim whose
      `field_path` names it.
- [ ] Every selected *unserved* endpoint has a gap naming it.
- [ ] Every Tox21 assay you reported has its own claims.

**Claims**
- [ ] Every `claim_id` is `clm_` + 32 fresh hex characters, unique.
- [ ] Every numeric/classification claim has `observation_id` + `field_path`.
- [ ] Every `source_value` equals the value the slice returned, unmodified.
- [ ] Every `rendered_value` is a single number, matching its transform.
- [ ] Every comparison uses `input_claim_ids`, not a hand-written difference.
- [ ] Every claim appears in exactly one section's `claim_ids`.
- [ ] Every number in a section body has a claim in that section.

**Explanations**
- [ ] Every `explanation_id` came from a tool result.
- [ ] Each ref's `endpoint`/`task` match the package's.
- [ ] Figures referenced only by the section discussing that endpoint.
- [ ] Partial status or non-zero `unmapped_importance` stated in
      `explanation_and_visuals`.
- [ ] Failed explanations recorded as gaps.

**Evidence**
- [ ] Every cited `evidence_id` was opened with `get_evidence_record`.
- [ ] Non-`insufficient` syntheses name at least one record.
- [ ] Any `contradicts` relation is discussed in words.
- [ ] Organism/assay/dose recorded where the record states them.
- [ ] Absence recorded as `no_relevant_evidence` (or the right gap reason),
      never omitted.
- [ ] `references` section is non-empty if anything is cited.

**Wording**
- [ ] No aggregate safe/unsafe/toxic verdict anywhere.
- [ ] No severity derived from counting active assays.
- [ ] No attribution described as mechanism or causality.
- [ ] No recommendation promising safety, prescribing a dose, or diagnosing.
- [ ] No raw HTML, event-handler attributes, or remote images.

**Limitations**
- [ ] `screening_not_safety_assessment` present.
- [ ] Every other code your claims trigger is present.

`references/validation-codes.md` maps each violation code to its fix.
