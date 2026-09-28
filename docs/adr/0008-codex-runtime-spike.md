# ADR 0008 — Codex runtime spike

**Status:** experimental · **Date:** 2026-09-08

## Decision

Codex is not admitted as a supported ToxAgent runtime. The repository has no
version-pinned adapter, no product-managed ChatGPT authentication lifecycle,
no captured deny-all composed surface, and no authenticated capability-token
roundtrip/recovery result for Codex. Implementing an adapter before those facts
exist would turn assumptions into authority.

The common `AgentRuntime` contract and product-owned kernel are sufficient to
run a future isolated spike. Admission requires: an allowlisted environment,
only ToxAgent MCP capabilities, normalized events, known cancel/close
semantics, reliable process/workspace reaping, and the full runtime-provider
evaluation matrix. Until then the verdict is `EXPERIMENTAL`, not `ACCEPT`.
