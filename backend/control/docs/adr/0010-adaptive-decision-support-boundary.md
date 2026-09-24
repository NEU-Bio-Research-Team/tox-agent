# ADR 0010 — Adaptive decision support builds on the gateway, not the dormant kernel

**Status:** accepted · **Date:** 2026-09-15 · **Source:**
`docs/spec/TOXAGENT_ADAPTIVE_DECISION_SUPPORT_PLAN_VI.md`

## Context

The motivating run (`run_21a5bb38922a4fffb83859841ee11b65`,
`ses_74a7e689b5ac4b88a6e6e8d0a8e640d4`) asked, in Vietnamese, whether a
compound should be used in drug development after a report had already been
built for it. The router (`application/router.py`, router-2) put the
follow-up in `report_qa` because it contained no research/attribution
keyword. That profile carries no search or evidence-read tool
(`tools/registry.py::PROFILES["report_qa"]`), no pinned report
(`harness/gateway.py::_prepare_context`, one analysis + up to five evidence
records pinned, no report), and the system prompt describes a fixed predictor
Q&A assistant with no license to broaden search or reconcile conflicting
sources. Two `submit_grounded_answer` candidates were rejected on
out-of-scope safety-verdict wording; the run ended on the deterministic
fallback. This is RC-01 through RC-09 in the plan's section 2.3, and is
reproduced as the frozen baseline task
`evals/regression/tasks/ads-00-benzene-drug-decision-baseline-vi.json`
against fixture `benzene-motivating-run` (W0-01/W0-02).

A pre-implementation codebase survey (2026-09-15) found the target shape —
an agent that plans, checks sufficiency/coverage, and reasons about conflict
within a bounded budget — already has a scaffold: `domain/investigation.py`
(`CaseState`, `InvestigationPlan`, `Coverage`, `EvidenceConflict`,
`EvidenceGap`, `GoalType`) plus `agent/kernel.py::ScientificAgentKernel`
(explicit `KernelState` machine, resumable plan execution) plus
`capabilities/registry.py` (`CapabilityDefinition`,
`CapabilityRegistry.default_capabilities()`, `validate_plan()`), backed by
real tables (`cases`, `case_revisions`, `investigation_plans`,
`investigation_steps`, `kernel_transitions`) in `persistence/schema.py`. None
of it is reachable from `submit_message` → `router.route()` →
`AgentRuntimeGateway.execute()`, which is the only live path today and drives
OpenCode directly through per-intent MCP tool profiles.

## Decision

**`decision_support` is built as a new capability profile and runtime profile
on the existing `AgentRuntimeGateway` / OpenCode path, not on
`ScientificAgentKernel`.** Concretely:

1. `Intent.DECISION_SUPPORT` is added to `domain/run.py`; `router.py` routes
   open conversational questions to it (W1).
2. `tools/registry.py` gains a `decision_support` profile per plan section
   7.2 — a superset read/research surface, still closed and still enforced by
   `ToolRegistry.register()` (W2).
3. `harness/gateway.py::_prepare_context` is extended to build and pin
   `ArtifactInventoryV1` (report, explanations, accepted evidence — plan
   section 8.1), not replaced (W3).
4. Bounded planning (proposition list, sufficiency check, query-expansion
   ladder, stop reasons) is enforced as budget/tool-call bookkeeping in the
   existing single-OpenCode-turn model, the same way `agent/budget.py`
   already enforces `report_qa`/`evidence_research`/`attribution` budgets
   today — extended with a `decision_support` entry, not a new execution
   engine (W4).
5. `ScientificAgentKernel`, `domain/investigation.py`, and
   `capabilities/registry.py` are left untouched. They are not deleted (their
   tables are live schema and may back something else later), but no ADS
   work package reads or writes them.

## Why not the kernel

The kernel scaffold is a closer conceptual match, but reviving it would mean
answering questions this plan does not need answered to hit its goals: why it
was never wired in, whether its `KernelState` machine and OpenCode's own turn
loop would double-govern the same execution, and whether its persistence
tables are safe to resume writing to without a schema audit. `AgentRuntimeGateway`
is the substrate every other capability (`report_qa`, `evidence_research`,
`attribution`, `build_report`) already runs on, with its budget, pinning, and
commit path exercised in production and covered by the existing eval harness.
Extending it keeps `decision_support` consistent with its siblings and keeps
this workstream's surface area to what sections 5-14 of the plan actually
specify. If a later audit concludes the kernel should be revived, that is a
separate ADR with its own migration story — not a precondition for W1-W6.

## Evidence-relation vocabulary

Two relation enums already exist and are narrower than the plan's target
(`supports | contradicts | contextual | insufficient | not_applicable`):
`domain/report.py::EvidenceRelation` (`SUPPORTS, CONTRADICTS, CONTEXTUALIZES,
INSUFFICIENT`) and `research/relevance.py::Relevance` (`DIRECT, CONTEXTUAL,
UNCERTAIN, IRRELEVANT`), which answers a different question (is this hit
citable) than the one `EvidenceRelation` (W5) needs to answer (what does this
source say about this proposition). W5 reconciles these into one type; no PR
before W5 may introduce a third parallel enum. See
`docs/glossary/ads-glossary.md` for the full definition.

## Baseline (W0-03)

No live rerun was performed for this ADR: reproducing the motivating run
would require a running model + OpenCode + predictor stack, which is a
heavier job than a documentation/fixture freeze warrants on this machine. The
baseline is instead the trace the plan document itself records for the
follow-up turn, taken as ground truth for what W1-W6 must change:

| Metric | Baseline value | Source |
|---|---|---|
| Routed intent | `report_qa` | plan section 2.1 |
| Tool calls | `get_analysis_slice` x4 | plan section 2.1 |
| Evidence search calls | 0 (tool not in profile) | plan section 2.1, RC-03 |
| Report pinned | no | RC-04 |
| `submit_grounded_answer` candidates | 2, both rejected | plan section 2.1 |
| Final answer | deterministic fallback (`is_fallback=true`) | plan section 2.1, RC-08 |

`evals/regression/tasks/ads-00-benzene-drug-decision-baseline-vi.json`
reproduces the routing/tool-availability portion of this (intent, zero
evidence-tool calls, one committed answer) against the frozen
`benzene-motivating-run` fixture; it does not yet assert `is_fallback`
because no grader field exists for it (tracked as a W7 gap, since W7-02/W7-03
already extend validator/trace diagnostics). Running this task through the
live/scripted harness to confirm it reproduces exactly is deferred to when a
model + OpenCode run is actually needed for W4 development, rather than run
here as a standalone heavy job.

## Non-goals (carried from the plan, section 4.2)

No aggregate toxicity/safety score, no attribution-as-mechanism, no shell/
filesystem/arbitrary-URL access for the runtime, no chain-of-thought
persistence, no diagnosis/dose/regulatory-approval output, no ToxPred
retraining.

## Consequences

- `agent/budget.py::PROFILES` and `tools/registry.py::PROFILES` grow one more
  named entry each; `report_qa`/`evidence_research`/`attribution` keep
  working unchanged during the compatibility window (plan section 7.1,
  17.1).
- The dormant kernel tables stay dormant. If they turn out to be dead weight,
  that is a separate, later cleanup — out of scope here.
- The frozen 50-task eval set (`evals/tasks/`, locked by
  `tests/unit/test_eval_tasks.py`) is not touched by this or any W0-W8 work;
  new ADS tasks live under `evals/regression/tasks/` until W8 decides whether
  they graduate into the main suite.
