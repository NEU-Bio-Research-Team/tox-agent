# Required sections

Eleven stable section ids. Every accepted report contains all of them, in this
order. The heading a reader sees is separate from the id and may be worded
naturally.

| id | Contains |
|---|---|
| `executive_summary` | What was analysed, what was found, what is uncertain. No new facts. |
| `substance_profile` | Canonical SMILES, structure figure, resolved identity and properties with sources, or an identity gap. |
| `predictor_results` | Every selected served endpoint, with exact values, labels, thresholds, applicability, model ids. Unserved endpoints appear as gaps. |
| `explanation_and_visuals` | Explanation figures, positive and negative contributors, unmapped attribution mass, failed explanations. |
| `external_evidence` | Read records, what each supports or contradicts, organism/assay/dose context, conflicts, gaps. |
| `integrated_interpretation` | Where model output and literature agree and disagree. Explicitly labelled as interpretation. |
| `conclusions` | Endpoint-level conclusions, each naming its endpoint; any integrated conclusion marked `is_integrated`. |
| `recommendations` | Follow-up work, each with basis claims, category, priority, rationale, conditions. |
| `limitations` | Every required limitation code. |
| `references` | Every cited evidence record. |
| `provenance_appendix` | Analysis hash, predictor version, artifact hashes, model ids, policy snapshot. |

## A section is never omitted

If the content for one is unavailable, the section stays and carries a gap:

```
gaps: [{ gap_id, reason, detail, section_id, endpoint?, task? }]
```

and the section references that `gap_id`. A gap that is declared but not
referenced by its section is refused — a reader would never see it.

Gap reasons: `endpoint_not_served`, `explanation_failed`, `explanation_partial`,
`compound_identity_unresolved`, `no_relevant_evidence`, `provider_unavailable`,
`budget_exhausted`.
