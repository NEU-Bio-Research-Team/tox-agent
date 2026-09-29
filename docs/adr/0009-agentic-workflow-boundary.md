# ADR 0009 — The agentic workflow boundary

**Status:** accepted · **Date:** 2026-09-13 · **Source:** `docs/audit/AGENTIC_FLOW_AUDIT_2026-09-13_VI.md`

## Context

The 2026-09-13 live audit measured a working substrate — deterministic
analysis, immutable observations, outbox/SSE, lease/fencing, run-scoped
capability tokens — carrying a workflow the model owns end to end. A minimal
hERG report loaded 18,149 input tokens on the first turn, finished at 28,408,
took 155.5 s, called tools out of order, minted its own database identifiers,
and used the validator as a reactive planner. The same report shipped an
executive summary that denied the explanation data sitting in its own
artifact.

None of that is an OpenCode defect. It is a boundary defect: the runtime shell
was handed the scientific workflow.

## Decision

**OpenCode remains the execution shell. It does not own the scientific
workflow.** Four rules follow, and every workstream in the remediation plan is
an instance of one of them.

### 1. The control plane owns order of work

A report is a server-driven state machine (`ReportOrchestrator`), not a
sequence the model chooses. Stages have their own input, output, checkpoint,
idempotency key and timeout. The model is invoked at exactly one stage
boundary — synthesis — behind a runtime profile that exposes only the submit
tool for that stage.

### 2. The server owns identity and canonical rendering

The model never mints a deployment-global identifier and never sends a
scientific value the server can read for itself. It sends a *local* reference
plus a field path into an immutable observation; the server resolves it,
renders the number under a pinned locale/transform policy, and issues the
`claim_id`. ADR 0005 (canonical rendered value) extends from answers to every
compiled artifact.

### 3. Facts have one source per artifact

Every repeated statement — endpoint result, contributor counts, unmapped
importance mass, evidence search scope, applicability limitation — is compiled
from a single `ReportFactBundle` entry. Sections reference `fact_id`; they do
not restate. Semantic validation runs on the normalized representation, never
as a regex over prose.

### 4. Telemetry is normalized before it is aggregated

A provider usage report is raw evidence. It becomes a normalized usage fact
only with source identity (`source_event_id`, `source_event_type`,
`provider_message_id`, `provider_step_id`, `revision`) and an explicit
`cumulative | delta | unknown` semantics tag. Aggregation happens over
normalized facts, and never sums two cumulative snapshots.

## Non-goals

- Replacing OpenCode with another framework in this epic. A paired evaluation
  against a direct provider adapter comes after Wave 4; migration opens only
  if that data justifies it.
- Trading grounding or safety gates for token streaming.
- An aggregate `safe/unsafe` verdict or whole-compound risk score (ADR 0002
  stands).
- Calling attribution a chemical mechanism. It is model sensitivity.

## Topology consequence

The FastAPI process stops being the agent scheduler. Runs are claimed by
independent worker processes over the existing lease/fencing tables, with
queue classes `interactive | report | deterministic` and DB-held concurrency
slots. A web replica restart must not cancel an agent run, which it does
today.

## Compatibility consequence

Expand/contract on every schema. New readers accept old rows; old readers
ignore new fields. Historical answers, reports and evidence records are never
mutated or re-hashed — an evidence record accepted under the old lifecycle
stays readable, but it is not citable in a new run without a fresh relevance
assessment.

## Consequences accepted

- More server code, and a narrower model surface. The model's remaining job —
  selection, interpretation, synthesis — is the part it is actually good at.
- Deterministic stages must be individually recoverable, which is more
  machinery than one long agent turn but the only way a 155 s build can be
  resumed instead of replayed.
- Feature flags carry an owner and a removal deadline (two releases maximum),
  or compatibility becomes the architecture.
