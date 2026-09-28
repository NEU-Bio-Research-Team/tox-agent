# Adaptive Decision Support — glossary (W0-04)

Source: `docs/spec/TOXAGENT_ADAPTIVE_DECISION_SUPPORT_PLAN_VI.md`. Terms are
pinned here so later PRs (W1-W8) use one vocabulary instead of each
reinventing wording for the same concept.

## Decision support

The conversational capability (target `Intent.DECISION_SUPPORT`) that answers
an open scientific or R&D question about an already-analyzed compound by
reading whichever artifacts (analysis, explanation, report, evidence) are
relevant and, when those are insufficient, searching for more — as opposed to
the current closed per-intent tool profiles (`report_qa`, `evidence_research`,
`attribution`) that fix the tool set before the model sees the question.

## Development posture

A structured, scoped R&D recommendation (`proceed | hold | deprioritize |
insufficient | not_applicable`) attached to a decision-support answer. It
always carries `scope`, `basis_claim_ids`, `confidence_band`, `conditions`,
and `rationale`. It is deliberately **not** a safety verdict: `proceed` never
means "safe", and `deprioritize` never means "unsafe" — see ADR 0002 (no
aggregate verdict), which this field must not reopen.

## Safety verdict

A prohibited output class: any claim that a compound is safe/unsafe, that
model output indicates clinical risk directly, or any diagnosis/dosing/
regulatory-approval statement. `validation/prohibited_claims.py` gates this
today; the plan's decision to fix (section 12.2) is that the gate must
distinguish an absolute safety claim, a negated limitation, a cited hazard
fact, and a scoped development-posture recommendation — not reject on a bare
token regardless of context.

## Evidence relation

The classification of one source's bearing on one proposition:
`supports | contradicts | contextual | insufficient | not_applicable`, each
with a `strength` band (`weak | moderate | strong | not_assessed`) and reason
codes — never a single pseudo-precise score. Two existing, narrower types
already exist in the codebase and must be reconciled with this vocabulary
rather than left as a third parallel enum:

- `domain/report.py::EvidenceRelation` — `SUPPORTS, CONTRADICTS,
  CONTEXTUALIZES, INSUFFICIENT` (missing `NOT_APPLICABLE`; "contextual" vs
  "contextualizes" wording differs from the plan).
- `research/relevance.py::Relevance` — `DIRECT, CONTEXTUAL, UNCERTAIN,
  IRRELEVANT`, which answers a narrower question ("is this hit citable at
  all") than "what does this source say about this proposition".

W5 is where this reconciliation must actually happen (evidence relation
schema/persistence); until then, treat the plan's five-value vocabulary as
the target and the two existing enums as inputs to a future merge, not as
already-satisfying it.

## Artifact inventory

A server-authored, compact projection (`ArtifactInventoryV1`, plan section
8.1) of what already exists for the active analysis — pointers only (ids,
status, coverage), never numeric values or evidence content, which still must
be read through their owning tool. Distinct from a `PinnedReference`
(`harness/context.py`), which is the existing single-analysis / up-to-5-
evidence pinning mechanism; the inventory is meant to widen what a follow-up
turn can see without spending tool budget probing for it blindly.

## Bounded autonomy

The agent chooses which tools to call and in what order, but only inside a
closed MCP surface, per-run budgets (`superseded/budget.py::BudgetLimits`), an
approved provider/source allowlist, and a deterministic final validator — as
opposed to unrestricted tool access. `agent_profiles/opencode/toxagent.json`
already denies every non-`toxagent_*` tool; that boundary does not change.

## Source class

The epistemic role of a source, independent of whether it currently agrees
with anything: `predictor_fact`, `explanation_fact`, `external_experimental`,
`external_regulatory`, `report_fact`, `agent_synthesis` (plan section 5.2).
`domain/report.py::SourceClass` already has six values enforcing claim/section
separation for reports; the decision-support answer contract reuses the same
role distinctions rather than inventing new ones.
