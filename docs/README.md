# Documentation index

Every Markdown file in this repository, with its status. The audit (I32) found
operational runbooks and historical records interleaved with no way to tell
which was which, so a deployer could follow a document describing a system that
no longer exists.

Three statuses, and they mean different things:

- **Current** — describes the tree as it is now. Commands must run from a clean
  clone. `devops/scripts/check_docs.py` enforces that these name no retired
  path and that every relative link resolves.
- **Historical** — evidence of what was true when it was written. Not edited to
  match today's tree; its value is that it does not change. Exempt from the
  retired-path check, never from the link check.
- **Superseded** — was operational, is now wrong, and carries a banner saying
  so plus a pointer to what replaces it.

## Start here

| Document | Status | What it is for |
|---|---|---|
| [`../README.md`](../README.md) | Current | Repository overview |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Current | First run via Docker Compose |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Current | Service boundaries across the whole product |
| [`WORKSPACE_LAYOUT.md`](WORKSPACE_LAYOUT.md) | Current | Where each service lives |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Current | Environment variables |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Current | Per-service test commands |
| [`OPERATIONS.md`](OPERATIONS.md) | Current | Running the stack |

## Per-service

| Document | Status | Scope |
|---|---|---|
| [`architecture.md`](architecture.md) | Current | **ToxPred only** — the predictor's internal layering. Distinct from `ARCHITECTURE.md` above, which is the whole product. |
| [`../backend/control/README.md`](../backend/control/README.md) | Current | Control plane |
| [`../backend/control/docs/adr/`](../backend/control/docs/adr/) | Current | Architecture decision records (ADR 0001–0008) |

`frontend/` has no README; its commands are in `package.json` and in
[`DEVELOPMENT.md`](DEVELOPMENT.md).

## Scientific

| Document | Status | Scope |
|---|---|---|
| [`model-card.md`](model-card.md) | Current | **Generated** full ToxPred model card, with split hash and benchmark commit. Regenerate rather than hand-edit. |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Current | Short customer-facing summary of the same measurements |
| [`benchmark-protocol.md`](benchmark-protocol.md) | Current | Frozen split and metric definitions |
| [`artifacts/herg-tox21-chemberta-v1.md`](artifacts/herg-tox21-chemberta-v1.md) | Current | Admitted model card |
| [`artifacts/clintox-smilesgnn-v1.md`](artifacts/clintox-smilesgnn-v1.md) | Current | **Blocked** artifact — records why v1 cannot be served (I07) |
| [`BM1_explainer_benchmark_analysis.md`](BM1_explainer_benchmark_analysis.md) | Historical | Explainer analysis of the pre-refactor monolith |

## Runbooks

| Document | Status | Scope |
|---|---|---|
| [`runbooks/DOCKER_TEST_RUNBOOK.md`](runbooks/DOCKER_TEST_RUNBOOK.md) | Current | Build and smoke-test the predictor image |
| [`runbooks/TOXAGENT_OPERATIONS_RUNBOOK.md`](runbooks/TOXAGENT_OPERATIONS_RUNBOOK.md) | Current | Secrets, outages, stuck runs, backup |
| [`runbooks/TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md`](runbooks/TOXAGENT_DATABASE_MIGRATION_RUNBOOK.md) | Current | Forward-only migration policy |
| [`runbooks/EXPERIMENT_RUNBOOK.md`](runbooks/EXPERIMENT_RUNBOOK.md) | Current | Research-side reproducibility |
| [`runbooks/CHAT_PERSISTENCE_E2E_CHECKLIST.md`](runbooks/CHAT_PERSISTENCE_E2E_CHECKLIST.md) | Historical | Written against Firestore; acceptance moves to SQL/REST/SSE under K07/K08 |
| [`runbooks/CICD_AUTO_DEPLOY_MASTER_PLAN.md`](runbooks/CICD_AUTO_DEPLOY_MASTER_PLAN.md) | Historical | Single-backend/Firestore/ADK topology, replaced by K12 |
| [`runbooks/DEPLOY_FIREBASE_APP_RUNBOOK.md`](runbooks/DEPLOY_FIREBASE_APP_RUNBOOK.md) | **Superseded** | Pre-refactor monolith deployment. Replacement is K12 (#22). |

## Plans and specifications

Source documents. Held immutable so a later report cannot be read back into an
earlier one. The current consolidated view is the audit pair.

| Document | Status |
|---|---|
| [`audit/SYSTEM_ISSUES_VI.md`](audit/SYSTEM_ISSUES_VI.md) | Current — the 32 issues, I01–I32 |
| [`audit/REMAINING_IMPLEMENTATION_PLAN_VI.md`](audit/REMAINING_IMPLEMENTATION_PLAN_VI.md) | Current — packages K01–K13 |
| [`spec/`](spec/) | Historical — rebuild, capabilities, quick-predict, UI, landing, handoff, GPU plans |
| [`spec/TOXAGENT_AGENTIC_LAYER_PROGRESS_VI.md`](spec/TOXAGENT_AGENTIC_LAYER_PROGRESS_VI.md) | Historical — a running log. Later entries correct earlier ones; do not sum runs or carry a score forward to a newer HEAD. |
| [`unified-v2/ARCHITECTURE.md`](unified-v2/ARCHITECTURE.md), [`unified-v2/DECISIONS.md`](unified-v2/DECISIONS.md), [`unified-v2/MIGRATION.md`](unified-v2/MIGRATION.md) | Current — typed boundary invariants |
| [`unified-v2/BASELINE.md`](unified-v2/BASELINE.md), [`unified-v2/IMPLEMENTATION_STATUS.md`](unified-v2/IMPLEMENTATION_STATUS.md) | Historical — G0–G12 evidence at the time of writing |
| [`refactor/PREDICTOR_ONLY_STATUS_VI.md`](refactor/PREDICTOR_ONLY_STATUS_VI.md) | Historical — audit of the monolith it replaced |
| [`archive/`](archive/) | Historical — ADK checklist, v1.0 overview, brainstorms |
| [`slides/README.md`](slides/README.md) | Historical — deck source document was removed at `b79036d` |

## Test numbers cited anywhere in these documents

A count belongs to the commit, environment and configuration it was measured
on. Do not add counts from different runs together, and do not treat a figure
recorded in a progress log as a statement about a later HEAD. Baselines for a
release carry commit, image digest, artifact hashes and skip reasons.
