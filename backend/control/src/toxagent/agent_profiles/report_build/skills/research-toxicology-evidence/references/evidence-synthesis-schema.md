# Evidence synthesis schema

```json
{
  "proposition": "hERG inhibition has been reported for this compound class at micromolar concentrations",
  "relation": "contextualizes",
  "evidence_ids": ["evd_..."],
  "endpoint": "herg",
  "assay": "patch clamp, HEK293",
  "organism": "human",
  "dose_context": "IC50 12 uM",
  "quality_notes": ["single study", "class-level rather than compound-specific"],
  "conflict_id": null
}
```

| Field | Rule |
|---|---|
| `proposition` | One statement. Not a summary of a paper; a claim the evidence bears on. |
| `relation` | `supports`, `contradicts`, `contextualizes`, `insufficient`. |
| `evidence_ids` | Records opened with `get_evidence_record`. Required unless `relation` is `insufficient`. |
| `endpoint` / `assay` | Which measurement this bears on. |
| `organism` / `dose_context` | As the record states them. Omit rather than infer. |
| `quality_notes` | Anything that limits the weight of this record. |
| `conflict_id` | Links records that disagree with each other. |

## Conflicts must be visible

When any synthesis has `relation: contradicts`, the `external_evidence` or
`integrated_interpretation` section must discuss the conflict in words. A draft
that records a contradiction in structured data and writes only agreement in
prose is refused.

## Citations at claim level

A scientific claim in the report carries `citation_ids` naming the records it
rests on. Section-level "see references" is not a citation; a reader must be
able to go from the sentence to the record.
