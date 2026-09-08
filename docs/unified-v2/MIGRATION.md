# Unified v2 migration

Migration is gate-based. Predictor typed ownership and provenance v2 land
before agent cutover. The new `toxagent.runtime` namespace is introduced while
`toxagent.harness` remains a compatibility layer; consumers migrate before the
old namespace is archived. Migration `0004_investigation_kernel` adds durable
case/plan/step/transition and model-connection storage without rewriting
existing sessions.

Physical repository relocation into `packages/apps/services/research` is the
last compatibility step. It is intentionally not performed until import-graph
and deployment consumer checks prove every legacy path has zero consumers.
