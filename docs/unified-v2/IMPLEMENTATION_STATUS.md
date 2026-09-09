# Unified v2 implementation status

Updated: 2026-09-08. A gate is marked complete only when its required evidence
exists; implemented code without a production-path benchmark remains partial.

| Gate | Status | Evidence / remaining work |
|---|---|---|
| G0 baseline | Partial | Unified/golden/API/runtime manifests and live-eval driver exist. Offline target-image cold start, memory, image size and batch `1/8/32/128/256` are recorded; the credentialed OpenCode live baseline remains. |
| G1 predictor correctness | Complete | Typed owner identity, finite/range/shape validation and per-sample truncation are implemented. The offline Python 3.10 container matched all 546 golden values across 42 molecules within `1e-6`. |
| G2 typed boundary | Implemented | Immutable provider rows, provider-owned batches and `toxpred-provenance-v2`; static/unit checks exist. |
| G3 standalone predictor | Complete | Serving imports no backend/control/agent modules; the inference-only model is wheel-packaged, the clean wheel builds, immutable artifacts verify, offline readiness passes, and 42-case numerical parity passes. ClinTox v1 is explicitly blocked until its exact tokenizer is recovered or v2 is retrained. |
| G4 agent foundation | Implemented, integration pending | Durable CaseState revisions, plans/steps, capability validation, coverage/budget/stopping, migration and restart reconstruction test exist. |
| G5 agent kernel | Partial | Runtime-neutral kernel, evidence types, conflict/gap state and server answer compiler exist. Current OpenCode production flow still uses the compatibility gateway; cutover needs the credentialed baseline variance run. |
| G6 calibration | Infrastructure only | Data-role guard and artifact/calibrator contracts exist. No disjoint calibration split or admitted fitted artifact exists, so policy-v1 remains unchanged. |
| G7 AD/uncertainty | Infrastructure only | Element guard semantics, ECFP, embedding AD and split-conformal contracts exist. Cutoffs/reference artifacts and coverage/risk benchmarks are not fitted. |
| G8 XAI | Partial | Signed logit-targeted gradient × input and integrated gradients, versioned mapping and conservation fields exist. Full faithfulness/stability benchmark has not been rerun. |
| G9 provider independence | Partial | Runtime/model/auth concepts, audit columns, secret-reference store, connection CRUD/probe API and runtime namespace exist. OpenCode/BYO live matrix remains credentialed work. |
| G10 runtime candidates | Complete as an admission decision | Codex and DSH are `EXPERIMENTAL`, not supported. DSH surface snapshot records the missing authenticated roundtrip; no unsafe adapter is shipped. |
| G11 evaluation | Partial | Existing task suite plus required runtime-provider matrix/categories exist. Credentialed OpenCode/Codex/DSH trials and admission results remain absent. |
| G12 hardening | Partial | Five CI job classes, artifact/model cards, migration docs, release-image build, offline smoke and operational measurements exist. Runtime cleanup suites and zero-consumer repository relocation remain release work. |

## External inputs required to close remaining gates

1. A disjoint, hash-pinned calibration split and training reference embeddings/
   fingerprints; the untouched test manifest cannot legally substitute.
2. Approved provider credentials/quota for three-trial OpenCode and optional
   Codex/DSH live matrices.
3. A release runner for the runtime start/cancel/cleanup suite and final
   zero-consumer migration proof.

Until these are supplied, no status document or aggregate score may claim the
corresponding scientific/runtime gate passed.
