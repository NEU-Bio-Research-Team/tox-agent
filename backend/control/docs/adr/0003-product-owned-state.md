# ADR 0003 — Product-owned state, relational store, transactional outbox

**Status:** accepted · **Date:** 2026-09-04 · **Plan:** §13, PROD-04, PROD-05

## Decision (DEC-02)

PostgreSQL is the production store; large or raw payloads go to an object
store. Dev and CI run the identical SQLAlchemy Core schema on SQLite, so the
mapping under test is the mapping that ships. Redis is not introduced without a
measured need, and never as a source of truth.

Relational was chosen for what this product actually needs: one transaction
spanning a state change and its event, foreign keys from a claim to the
observation that backs it, unique idempotency constraints, and a monotonic
per-session sequence.

## Event delivery

Every mutation writes its events into `event_outbox` **in the same
transaction**. A dispatcher reads the outbox and feeds SSE. Delivery is
at-least-once and clients dedupe on `(session_id, sequence)`. A stream that
dies loses nothing: `GET /v1/sessions/{id}` plus `GET .../messages` reconstruct
the session, and `?after_sequence=` replays the tail.

## Runtime state is not product state

The runtime's transcript is disposable. Sessions, messages, runs, snapshots,
observations, evidence, and answers live here. When a runtime binding is lost,
the product rebuilds a prompt from its own checkpoint and opens a **new** run
with `recovery_of_run_id` set — it never silently appends to a failed run
(PROD-10).
