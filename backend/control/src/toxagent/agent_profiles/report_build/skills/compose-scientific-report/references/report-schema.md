# Report draft schema

```json
{
  "schema_version": "report-draft-v1",
  "report_build_id": "rpb_...",
  "title": "Toxicity screening report: <compound>",
  "sections": [
    {
      "section_id": "predictor_results",
      "heading": "Predictor results",
      "body_markdown": "...",
      "claim_ids": ["clm_..."],
      "table_ids": ["t_predictions"],
      "figure_ids": [],
      "gap_ids": ["g_clintox_unserved"],
      "source_classes": ["predictor_fact"]
    }
  ],
  "claims": [...],
  "tables": [...],
  "explanations": [{"explanation_id": "xpl_...", "endpoint": "herg", "task": null,
                    "narrative_claim_ids": ["clm_..."]}],
  "evidence_synthesis": [...],
  "conclusions": [{"conclusion_id": "c_herg", "text": "...",
                   "basis_claim_ids": ["clm_..."], "endpoint": "herg",
                   "is_integrated": false}],
  "recommendations": [{"recommendation_id": "r_1", "text": "...",
                       "basis_claim_ids": ["clm_..."],
                       "action_category": "in_vitro_assay", "priority": "high",
                       "rationale": "...", "conditions": ""}],
  "limitations": [{"code": "uncalibrated_probability", "text": ""}],
  "gaps": [{"gap_id": "g_clintox_unserved", "reason": "endpoint_not_served",
            "detail": "ClinTox was requested and this analysis does not serve it.",
            "section_id": "predictor_results", "endpoint": "clintox"}]
}
```

## Section ids

`executive_summary`, `substance_profile`, `predictor_results`,
`explanation_and_visuals`, `external_evidence`, `integrated_interpretation`,
`conclusions`, `recommendations`, `limitations`, `references`,
`provenance_appendix`.

## Source classes

`structure_fact`, `predictor_fact`, `explanation_fact`, `external_evidence`,
`agent_synthesis`, `recommendation`.

A section may declare more than one, and the validator checks each claim's kind
against what its section declares. A classification claim belongs only in a
section declaring `predictor_fact`. A claim carrying `citation_ids` belongs only
in a section declaring `external_evidence` or `agent_synthesis`.

## Citing a source inside a sentence

Put `[@evd_...]` — the evidence id in square brackets after an `@` — directly in
`body_markdown` where the claim it supports is made. The server turns it into a
numbered marker the reader can click, using numbering it assigns itself from the
order the tokens appear in the report.

Never write the number yourself, and never write the URL. The number belongs to
the artifact, not to the draft: numbering written into prose disagrees with the
numbering the References section, the Markdown export, the HTML and the PDF all
share. A URL written into prose is unvalidated text in something the renderers
turn into a link, and it is refused by the content-safety gate.

The token is held to the same three rules as a claim's `citation_ids`:

- the record must exist in this session;
- it must have been opened with `get_evidence_record` — a search result is a
  byline, not a source;
- the section carrying it must declare the `external_evidence` source class.

A token naming a record you did not read comes back as `citation_not_read`; a
section citing without declaring gets
`section_cites_without_declaring_external_evidence`.

The References section needs no list from you. The server builds it from the
resolved snapshot of every source cited anywhere in the report — title, authors,
provider, identifier, canonical URL and retrieval date — so write that section's
`body_markdown` as a short note about scope, not as a bibliography.

## Ids you choose, and ids you do not

You choose: `claim_id` (must be `clm_` plus 32 lowercase hex characters, freshly
random — a short label or a repeated digit is refused and can collide with an
unrelated stored answer), `table_id`, `conclusion_id`, `recommendation_id`,
`gap_id`.

You do not choose, and must copy verbatim from tool results: `observation_id`,
`evidence_id`, `explanation_id`, `figure_id`, `report_build_id`.

## Leave `text` empty on a limitation

The server fills the canonical wording. Supplying your own risks weakening it by
paraphrase.
