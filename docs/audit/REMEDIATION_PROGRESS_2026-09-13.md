# Remediation progress — 2026-09-13 audit

Tracks `AGENTIC_FLOW_REMEDIATION_IMPLEMENTATION_PLAN_2026-09-13_VI.md` against
what is actually merged. One row per PR in that plan's §7 sequence.

Measured state of the suites at the time of writing: **1,128 backend unit +
contract, 251 integration + e2e, 155 frontend** — all passing. The audit's
baseline was 1,019 backend and 155 frontend.

## Merged

| PR | What closed | Findings |
|---|---|---|
| PR-00 | Audit fixtures, ADR 0009, rollout-flag catalogue with owners and expiry | all — every finding now has a reproducing fixture |
| PR-01 | Runtime profile registry; the adapter dispatches the intent's named agent and records requested *and* effective step caps | P0-1 |
| PR-02 | Usage source identity and normalization; cumulative snapshots collapse, zero-only lifecycle events are dropped, partial unique index survives a restart | P1-1 |
| PR-03/04 | `GroundedAnswerDraftV2`: local refs, server-issued claim ids, server-resolved values and rendering; narrowed safety-negation exemption | P1-4, P1-5 |
| PR-05 | Content-Encoding stripped from a decoded body; bounded retry for transport failures only | P2-5 |
| PR-06/07 | Deterministic relevance assessment; only direct/contextual become citable; the user limit is a promotion ceiling | P1-2 |
| PR-08 | XAI coverage: mapped / special-token / other-unmapped fractions, coverage bands, no renormalization | P1-6 |
| PR-09 | `ReportOrchestrator` skeleton: true per-stage events, typed checkpoints, recovery that does not repeat a settled stage, skipping as a recorded outcome | P1-3 (part), P0-2 (evidence-state half) |
| PR-10 | `ReportFactBundle` — every assertable fact once, with a stable id, source class and server-owned rendering — and the deterministic stage handlers that fill PR-09's slots | P1-3, P0-2 (structural half) |
| PR-11 | Cross-section semantic gates; the audit's contradiction artifact is refused | P0-2 |
| PR-13 | Word-boundary intent matching, bounded negation, `IntentDecision` persisted on the run | P1-9 |
| PR-16 | Prompt measured by component, with a cacheable-prefix hash, recorded on the binding | P1-10 (measurement half) |
| PR-17 | `make test` from a fresh clone, Docker `test` stage, production image stops shipping tests; canvas stub and a console.error gate in the frontend suite | P2-1, P2-2, P2-3, P2-4 |

## Not started

| PR | What it needs | Blocking gate |
|---|---|---|
| PR-12 | Report renderers/UI compatibility and the progress timeline (frontend) | G3 |
| PR-14 | Worker composition root, entrypoint and queue classes | G4 |
| PR-15 | Global/per-tenant quotas, graceful drain, recovery drills | G4 |
| PR-18 | Metrics, dashboards, live eval, alpha rollout | G5 |

`PR-16`'s second half — actually shrinking the report profile once the
orchestrator owns the workflow — depends on PR-10, because the tokens to cut are
the skills that describe work the orchestrator will be doing.

## Flags, and when they must be gone

Every rollout flag is declared in `toxagent.flags` with an owner and a removal
date at most two releases out; `tests/unit/test_rollout_flags.py` fails the
build when one is not. Current defaults:

| Flag | Default | Removable when |
|---|---|---|
| `runtime_profile_selector_v2` | on | every deployment ships both agent profiles |
| `normalized_usage_v2` | on | dashboards are off raw sums |
| `answer_draft_v2` | off | first-pass acceptance holds on the QA eval for two releases |
| `evidence_pipeline_v2` | off | promoted-evidence precision ≥ 0.9 on the curated set |
| `report_orchestrator_v2` | off | 100% and zero fallback for 14 days |
| `router_v2` | off | bilingual corpus at 100% golden |
| `external_worker_mode` | off | multi-replica drills pass |

`answer_draft_v2` and `evidence_pipeline_v2` are written and tested but default
off: both change what a model is asked to produce, and the plan's own rule is
that a new hard-reject class runs in shadow before it refuses anything.

## Known limitations, recorded rather than discovered later

**The semantic gates read denials, not meaning.** `report_semantics` checks a
closed set of sentences that assert the absence of a fact the server holds. That
is the exact shape both audit contradictions took, and it is much narrower than
"does this paragraph agree with the data". The structural fix — every repeated
fact compiled from one entry and referenced by id — needs PR-10.

**Router negation reaches four tokens.** Longer-range negation is a classifier
question (WS07 step 7), and widening the window trades one class of false
positive for another. `test_router_corpus.py` pins both directions.

**Prompt token counts are estimates.** A real tokenizer belongs to the provider
and differs per model. What the estimate has to be is *stable*, so a diff
between two manifests means the prompt changed; it is never presented as a
provider count.

**The orchestrator is not wired.** `report_orchestrator_v2` is off, so the
report path is unchanged: the loop, the checkpoints, the handlers and the fact
bundle exist and are tested, but nothing dispatches through them yet. Wiring
them is PR-11's narrow synthesis schema plus PR-12's renderers — until both
land, switching the flag on would produce a build with no way to write or
render its output.
