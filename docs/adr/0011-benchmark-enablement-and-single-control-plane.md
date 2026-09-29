# ADR 0011 — One control plane, server-owned decision state, and an evaluation spine that says what it graded

**Status:** accepted · **Date:** 2026-09-16 · **Source:**
`docs/TOXAGENT_AGENT_ARCHITECTURE_AND_BENCHMARK_ENABLEMENT_REVIEW_VI.md`
(companion to `docs/TOXAGENT_AGENT_BENCHMARK_REVIEW_AND_PROPOSAL_VI.md`,
TAB-Suite v3)

## Context

The 2026-09-16 architecture review found the gap between ToxAgent and a
benchmark was not agent capability but three truths that did not meet:

* **Implementation truth** — two agent architectures looked canonical: the
  live `AgentRuntimeGateway` and the dormant `ScientificAgentKernel`
  (ADR 0010 already chose the gateway for decision support, without retiring
  the kernel).
* **Evaluation truth** — the runner globbed one directory, so the ADS
  regression task changed no hash and ran nowhere; results could not tell a
  product failure from an infrastructure one; a fallback answer counted as an
  agent pass; the semantic rubric was a declaration with no judge contract.
* **Release truth** — a manifest named a commit and a runtime, not the flags,
  profiles, budgets and topology that decide which product ran.

## Decision

1. **`backend/*` is the canonical product tree.** `docs/architecture-inventory.json`
   is generated from code and a unit test fails on drift; tracked paths may
   not differ only by case; `devops/scripts/check_workspace.py --strict` runs
   in CI.
2. **TAB-Suite v3 is the evaluation contract; this repository implements it.**
   Declared task packs (`evals/packs`), `eval-task-v3` with a v1 adapter,
   statuses `pass | fail | invalid | skipped | infra_error` with task
   conservation, `eval-trace-v1`, a versioned grader registry, and
   `eval-manifest-v2` embedding the deployment's `effective-product-v1`
   (`GET /v1/system/effective-product`).
3. **`AgentRuntimeGateway` with server-owned state is the only live agent
   control plane.** Decision support keeps a `DecisionSupportStateV1`
   (goal, propositions, coverage, usage, stop reason) whose transitions are
   pure and whose writes are revision-checked. The model may propose a plan
   (`record_decision_plan`, behind `decision_state_plan_tool`); the server
   issues ids and decides coverage and stop reason.
   `ScientificAgentKernel`, `agent/budget.py` and the `case_states` /
   `investigation_steps` / `kernel_transitions` tables are **superseded**:
   what they contributed (budget snapshot, stop reasons, coverage) now lives
   in `application/run_budget.py` and `domain/decision_state.py`.
   `tests/unit/test_eval_paired.py` forbids any module outside their own
   persistence adapters from importing them. Deleting the code and retiring
   the tables is a separate, destructive change, taken after the Wave 4
   paired benchmark and never in the same release as a schema drop.
4. **Report orchestrator v2 is the candidate report path.** Report benchmarks
   run with `report_orchestrator_v2=1`; the legacy agent-driven path is not
   tuned further. Cutover is decided by `evals/paired.py` on paired task ids
   (no critical pass→fail, no confound outside the flag), then the flag and
   the old path are removed before `remove_by`.
5. **Evidence-derived claims carry lineage and trust.** One canonical
   evidence ontology (`domain/evidence_ontology.py`) with relation and
   relevance as separate axes and a recorded assessor; `agent_synthesis` must
   name resolvable inputs and cannot alone carry a committing development
   posture; provider free text reaches the model in trust envelopes
   (`trust_envelope_v1`).
6. **Single agent with typed tools remains the default.** No multi-agent split,
   no framework migration, no shell/filesystem/raw web for the scientific
   agent unless a paired eval shows the complexity pays for itself.
7. **No single ToxAgent score.** Control conformance, capability, scientific
   communication and safety/reliability are reported separately; a fallback is
   a containment success and a capability failure at the same time; a semantic
   judge gates only a rubric calibrated against the adjudicated SME set.

## Consequences

* New rollout flags `trust_envelope_v1` and `decision_state_plan_tool` follow
  the flag rules (owner, `remove_by` ≤ 90 days). Both default off: the default
  tool surface and model view are unchanged until the paired security and
  ads-plan packs are run live.
* Migration 0018 is additive only (a new table, nullable columns).
* A live benchmark needs a stack that serves `/v1/system/effective-product`;
  frozen-evidence runs use `TOXAGENT_RESEARCH_PROVIDER=snapshot`, which the
  manifest records.
* Still open and deliberately not decided here: which provider packs
  (regulatory, bioactivity) the product claims, external-worker cutover (needs
  the multi-replica drills), and the SME gold set itself.
