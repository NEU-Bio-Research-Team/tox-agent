# Agent benchmark master plan — execution log

Source plan: `ToxAgent_Agent_Implementation_and_Benchmark_Master_Plan_2026-10-21.md`
(repo root, untracked), which follows the 2026-10-20 proposal
(`docs/internal/spec/TOXAGENT_AGENT_LAYER_AND_BENCHMARK_PROPOSAL_2026-10-20_VI.md`).

Branch: `feat/agent-benchmark-master-plan`, cut from
`fix/activity-identity-guards` at `025db84`. One commit per planned PR slice
(P01–P12), so the stack can be split into the stacked PRs §16 describes.

Baseline before any change: control suite 2686 passed, 19 skipped.

## Status by slice

| Slice | Issue | State | Notes |
|---|---|---|---|
| P01 subject-binding | #44 | done | see below |
| P02 frozen-full-text | #43 | done | see below |
| P03 subject-evidence | AG-EVIDENCE | in progress | |
| P04 strict-release | #47 | todo | |
| P05 budget-phases | #46/#53 | todo | |
| P06 case-inputs | #51 | todo | |
| P07a shared-replay-A | #50 | todo | |
| P07b strong-B-fairness | #50 | todo | |
| P08 stories-skills | #46 | todo | |
| P09 research-UI | AG-UI | todo | |
| P10 scoring-packets | #53 | todo | |
| P11 pilot-24 | #52 | todo | needs SME-verified gold; engineering side only |
| P12 freeze-agent | #56 | todo | needs the pilot; engineering side only |

## P01 — subject binding (#44)

Flag `subject_binding_v1` (default off).

- `domain/subject.py`: `RunSubjectBinding` (run-subject-binding-v1). Case key
  `analysis:<id>` (unchanged) or `activity:<40 hex>` digest of ligand, target,
  variant, assay context. Optional enrichment (UniProt, sequence) checked by
  `mismatches`, never part of the key. `TargetIdentity.variant` added, not
  serialised when unset (existing hashes unchanged).
- Resolved at admission (`application/conversation/subject_resolution.py`):
  explicit `subject_ref` on `POST /messages`, the legacy `analysis_id`, a new
  molecule (pending until its snapshot), or the session's only subject. Two
  subjects and no selection is the `subject_ambiguous` clarification.
- Frozen into the run: `RunContext.subject`, run envelope v2 (a v1 worker
  refuses to claim it), `run_configuration_snapshots.subject_binding`
  (migration 0026), `run.subject_bound` event. Recovery runs keep it.
- Tools receive it through `ToolContext.subject` (MCP and local);
  inventory/report summary default to it, never to the session's selection.
- Case: keyed by the binding, stores `subject`; each run records its binding
  hash and artifact refs. `SubjectMismatch` on an incompatible open case.
  With a binding, a case that cannot be opened fails the run
  (`case_unavailable`) instead of silently running without it.
- Ledger: `policy_fact` and `workflow_fact` source classes (not independent).
  Activity estimate → predictor_fact scoped by target and unit `pIC50`, never
  an endpoint. Full EvidenceScope axes kept; quoted spans and server
  provenance kept per entry. Another subject's artifact is left out.
- Replay: a frozen screening may carry its toxicity snapshot
  (`frozen-toxicity-snapshot-v1`); binding makes one session-local copy with
  its prediction observation, hash-checked, never calling the predictor.
- `GET /v1/sessions/{sid}/screenings/{id}`: screening read model with every
  stage stated.
- Effective product: `contracts.run_envelope_version`,
  `contracts.run_subject_binding`.

Tests: `tests/unit/test_subject_binding.py`,
`tests/e2e/test_subject_binding_through_the_product.py`,
`tests/integration/test_subject_binding_queue.py`.

## P02 — frozen full text (#43)

- `research/fulltext_store.py`: `frozen-full-text-v1` store -- each
  document's passages (rows with cells), the outcome (retrieved /
  access_failed), the capture (provider, time, sha256 of the bytes received)
  and the parser version (`jats-passages-v1`). Every document re-hashed on
  load; a sidecar file is pinned by sha256.
- `FrozenFullTextProvider` wraps the snapshot or corpus provider when a store
  is configured (`TOXAGENT_RESEARCH_FULL_TEXT_PATH`/`_SHA256`, or inline in a
  snapshot fixture under `evidence.full_text`). Only then is the provider
  text-capable, so `read_evidence_text` exists on a frozen deployment. A
  record the store does not hold is answered "not held", never fetched; an
  injected snapshot outage reaches the text as transient.
- Live Europe PMC: bounded per-provider cache of definitive answers and
  single-flight per PMCID; an outage is never cached.
- Provenance of every read text is stored (`evidence_full_texts.provenance`,
  migration 0027), shown by `read_evidence_text` (`text_provenance`) and by
  the evidence API (`full_text.provenance`).
- `python -m evals.investigation.capture_full_text`: capture once from
  Europe PMC into a store; transient fetches are reported, not frozen.

Tests: `tests/unit/test_frozen_full_text.py`,
`tests/e2e/test_frozen_full_text_through_the_product.py` (offline table-row
quote validated; captured access failure owes `evidence_full_text_not_read`;
unknown locator refused). Suite: 2732 passed, 19 skipped.
