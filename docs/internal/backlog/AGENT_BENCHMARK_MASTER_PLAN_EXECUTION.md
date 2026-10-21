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
| P03 subject-evidence | AG-EVIDENCE | done | see below |
| P04 strict-release | #47 | done | see below |
| P05 budget-phases | #46/#53 | done | see below |
| P06 case-inputs | #51 | done (engineering part) | SME gold still needed |
| P07a shared-replay-A | #50 | in progress | template v2 for arm A drafted, not wired |
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

## P03 — subject-aware evidence retrieval (AG-EVIDENCE)

- `research/target_catalog.py` (`benchmark-targets-v1`): KCNH2 (hERG,
  CHEMBL240, live mapping verified), EGFR (CHEMBL203), ABL1 (CHEMBL1862),
  each with the vocabulary a relevance check looks for. An entry whose UniProt
  disagrees is not the target. EGFR/ABL1 live ChEMBL mappings are recorded as
  unverified.
- Relevance over a target (`evidence-relevance-target-1`): ligand on the
  target → direct (with `assessed_scope` ligand + target); ligand on another
  target → contextual; target with another ligand → irrelevant; target not
  in the catalog → uncertain at best; thin metadata → uncertain.
- `search_toxicology_evidence` under `subject_binding_v1` takes
  `screening_id`, defaults to the run's subject, refuses another subject
  (`subject_mismatch`), and passes the same data-scope guard keyed by the
  bound subject. The ligand's identity is the frozen record's
  (`subject_names`, server-held), never the query's words.
- `get_chembl_activities` follows the binding; a screening whose target is
  not hERG is `capability_unavailable`, never a string passed to the hERG
  lookup.
- Answer validation: a relation may not restate the target an activity
  artifact or an assessed record states
  (`evidence_relation_scope_contradicts_source`), and a record about the
  ligand on another target cannot be direct support/contradiction
  (`evidence_relation_direct_out_of_subject`).
- Arm B (`fixed-workflow-b-v2`): activity branch with queries from the
  binding (ligand name, target, readout); REVIEW triggers a search; toxicity
  branch whose endpoint comes from the question and reads the linked
  toxicity analysis; provider-failure branch said as such.
- Evidence API exposes `relevance_assessment`.

Tests: `tests/unit/test_subject_relevance.py`,
`tests/e2e/test_subject_evidence_through_the_product.py`. Suite: 2745 passed.

## P04 — strict publication policy (#47)

`TOXAGENT_RELEASE_POLICY` = `legacy` (default, unchanged) |
`strict-reviewed-v1` | `hard-only-v1`, stated in the effective product with the
hash of its decision table.

- `domain/run_result.py`: `RunResultV1` keeps publication disposition
  (publish / partial_facts / withhold), review status, execution outcome and
  knowledge outcome apart; task success is none of them. The policy is a pure
  table (`decide_publication`).
- One writer: the gateway's publication coordinator records the run result
  (`run_results`, migration 0028) and, only for a published answer, the
  compatible `answer_releases` row and `answer.accepted`. Idempotent on
  run_id. A run that ends without committing (failure, cancel) records a
  withheld result with its typed execution outcome.
- The assistant message of a non-published run carries no candidate prose:
  server-resolved facts, the reason, needed inputs, and a `result_ref` part.
  The answer route serves only the run's public answer.
  `GET /runs/{rid}/result`; drafts carry their phase (initial /
  hard_correction / semantic_repair).
- Case isolation: under a declared policy a run's conclusion is `proposed`
  and is promoted only with its published answer; the last released
  conclusion is kept beside it; checkpoints and report views never present a
  proposal as the case's conclusion; hypothesis statuses set by an
  unpublished run are marked as proposals; runs record their disposition.
- `hard-only-v1`: the declared no-review ablation; publishes the
  hard-validated candidate with review status `not_reviewed_by_design`.

Tests: `tests/unit/test_run_result_policy.py`, `tests/e2e/test_strict_release.py`.
Suite: 2773 passed; `test_worker_quotas_and_drain::test_a_draining_worker...`
failed once under full-suite load and passes in isolation (timing).

## P05 — budget v2 and phase attribution (#46/#53)

- Preset `investigator-pilot-v2` (`total-logical-invocations-v2`): 360 s root
  deadline; 20 tool calls in total -- case, skills, reads, searches,
  candidates and reviewer submissions all count; 6 reserved for the
  submissions (gathering may use 14); searches 3, records 8, full-text pages 4
  within the total; one semantic repair; two hard submissions per phase;
  6 invalid/denied calls then everything but the submissions is refused;
  120 s of the deadline kept for review and repair; token/cost limits
  declared `observed_only`. `investigator-pilot-v1` is unchanged and now
  named `action-read-ceiling-v1`.
- The effective budget states all of it and has a `sha256`, so two arms with
  one preset label but different limits are told apart.
- Tool calls and runtime bindings carry their phase (`investigator`,
  `workflow_acquisition`, `workflow_synthesis`, `hard_correction`,
  `review_initial`, `semantic_repair`, `review_final`; migration 0029); MCP
  calls take it from their binding. Invalid calls are recorded.
- `GET /runs/{rid}` reports `phase_usage` (phase-usage-v1): per phase,
  dispatches, calls by status, tokens (cumulative latest once, deltas added,
  unknown kept unknown), cost by currency (missing cost is unknown).
- The investigator turn ends before the review's reserve; running into it is
  `phase_budget_exhausted`, and the run result says `budget_exhausted`.
- Under v2 a recovery continues the original's deadline and call budget.

Tests: `tests/unit/test_budget_v2_and_phase_usage.py`,
`tests/e2e/test_budget_phases_through_the_product.py`. Suite: 2785 passed.

## P06 — protocol v2 cases, inputs and gold (#51, engineering part)

- Three files per case, three schemas: `investigation-case-v3` (what the case
  measures, `workflow_context` separate from `cohort`, pins of the other two),
  `frozen-case-inputs-v1` (everything a system is given, the same for every
  arm: subject by reference, frozen artifacts with hashes, corpus and
  full-text store, contract versions, turns with `add_context` /
  `attach_evidence_ref` / `switch_subject`, provenance, dependency groups) and
  `case-gold-v1` (never model-facing).
- `case_inputs.py`: loader, audit (shapes, hashes, artifacts against the
  replay file, spans against the frozen text or a declared access failure,
  pairing, answerability, T3 polarity, T6 faults, gold isolation, SAIR
  availability) and `dependency_split` by whole dependency components.
- The replay file may hold frozen toxicity analyses that belong to no
  screening (`analyses`), for toxicity-only cases; each is hash-checked on
  load, and `materialise_frozen_analysis` makes the session-local copy.
- `cases_protocol_v2_technical/` (built by `technical_seed.py`): 12 cases,
  T1-T6 × {toxicity_only, activity_guided}, both T3 polarities, a multi-turn
  update with a target switch, two T6 faults -- fictional compounds and
  invented numbers, labelled `synthetic_engineering_fixture` /
  `synthetic_engineering_gold` on every file. Audits clean.

Not done here (needs the SME): real cases, acceptable conclusions, critical
errors, verified T3 labels and source checks for the 24-case pilot.

Tests: `tests/unit/test_case_inputs.py`. Suite: 2794 passed; the one failure
(`describe()` gained `analyses: 0` for every replay file) is fixed and its
file rerun.
