# Audit fixtures — 2026-09-13

Sanitized captures from the live audit run
(`docs/audit/AGENTIC_FLOW_AUDIT_2026-09-13_VI.md`, session
`ses_cc304429ec9d49cb8f51a8e05bba89ac`). Each file reproduces one finding so a
regression can fail a test instead of a report.

| File | Finding | Used by |
|---|---|---|
| `opencode_duplicate_usage_sse.json` | P1-1: the same cumulative usage snapshot arrives three times | `tests/unit/test_runtime_usage_normalization.py` |
| `evidence_ethanol_herg_false_matches.json` | P1-2: five unrelated papers persisted as `accepted` | `tests/unit/test_evidence_relevance.py` |
| `report_contradiction_artifact.json` | P0-2: an executive summary that denies its own explanation data | `tests/unit/test_report_semantic_consistency.py` |
| `explanation_cco_herg_attribution.json` | P1-6: 35.84% of the attribution mass lands on special tokens | `tests/unit/test_xai_coverage.py` |
| `baseline_manifest.json` | The measured numbers every KPI gate is a delta against | `tests/unit/test_audit_baseline_manifest.py` |

## Sanitization rules

No credentials, no bearer tokens, no raw system prompts, no full external
provider payloads. Identifiers are the audit's own, which are development
session ids and are not secret; token counts, timings and importance values are
verbatim, because a rounded fixture would stop reproducing the arithmetic the
findings are about.
