# ADR 0004 — Runtime pinning and the provider contract

**Status:** accepted · **Date:** 2026-09-04 · **Plan:** §10, §11, §12 (DEC-05, DEC-06)

## Decision

Runtimes are reached only through `AgentRuntimeProvider`. Two adapters ship
today:

| Adapter | Pin | Role |
|---|---|---|
| `scripted` | in-process | deterministic runtime for frozen evals and CI; no provider, no network |
| `opencode_v1` | `1.17.11` (the version resolved on this machine) | primary app-facing candidate |

**Corrected 2026-09-06 (see ADR 0007):** this row previously listed a `dsh`
adapter pinned at `0.1.1-rc.2` as shipped. That was never true — no
`harness/adapters/dsh.py` exists. `domain/runtime.py::RuntimeKind.DSH` and
`config.py`'s `dsh_command`/`dsh_version` are scaffolding for a runtime kind
nothing implements yet; `TOXAGENT_RUNTIME_KIND=dsh` fails to start a
deployment today, by construction. ADR 0007 records the actual spike
(package/binary identity, a real smoke run, findings) and DEC-06's current
status. This correction is a documentation fix, not a design change — the
rules below always applied to `dsh` as an intended future adapter, and still
do once one is written.

OpenCode V2 is an evaluation track only while it is beta; it is not a
production candidate and no code here depends on a floating contract.

## Rules

- Capability is **probed**, never inferred from the adapter's name. An adapter
  that cannot cancel a turn reports `cancel_turn: false`, and the product API
  then tells the client exactly what it did instead (terminate the owned worker)
  rather than reporting a cancellation that did not happen.
- One session binds one runtime at a time. A run never changes runtime
  mid-flight. Bindings pin runtime kind/version, provider, model, profile hash,
  tool-schema hash, and system-prompt hash, and those hashes appear in the run
  audit.
- Runtimes hold no product authority. They authenticate to the tool plane with a
  short-lived capability token scoped to one session, one run, and one exact
  tool allowlist; a `session_id` argument that disagrees with the token loses.
- Adapters are deny-all: no shell, edit, read, glob, grep, list, subagent,
  skill, webfetch, websearch, or execute. Only the ToxAgent MCP namespace.

## Verification status

The `scripted` adapter is exercised by the full e2e suite. `opencode_v1` is
written against the pinned version above and is covered by contract suites
marked `live_runtime`; those suites, not this ADR, are what certifies it. DSH
has no adapter yet to certify (see the correction above and ADR 0007) — until
one exists and its own contract suite runs green on a given host, DSH is
"unsupported here", not silently treated as equivalent to `opencode_v1`.
