# DSH custom profile — spike artifact, not a supported runtime

**This is not wired to any adapter.** `RuntimeKind.DSH` has no registered
`AgentRuntimeProvider` (see `harness/adapters/` — only `scripted.py` and
`opencode_v1.py` exist). Nothing in `toxagent/` reads this directory. It is
checked in only so the next DSH spike or adapter attempt does not have to
re-derive the Cordis patch syntax from scratch — see progress log §46 and
ADR 0007 for how it was produced and verified.

## What was actually confirmed with this profile (2026-09-06)

Booted for real (`deepseek-harness-sdk==0.1.2rc1`, isolated `DSH_HOME`, no
credential): the composed profile reports
`permission/preset: read-only` and `sandbox/mode: read-only` in its own
session events — not the shipped `sdk` profile's default `workspace-write`.
`@deepseek-ai/dsh-mcp-client` (an official, version-matched plugin,
`streamable-http` transport) loaded without crashing when pointed at a real
running ToxAgent control plane's `/internal/mcp`, using a deliberately
invalid placeholder bearer token and `failOnStartupError: false` — the
harness stayed up regardless of the connection outcome. **Not yet
confirmed:** a real, valid capability token was never minted for this spike
(that requires a live product run, which a standalone token has no
scope story for yet), so actual MCP tool discovery/registration against
ToxAgent's server is unverified.

## What is disabled and why

Mirrors `agent_profiles/opencode/toxagent.json`'s deny list: bash/pwsh, the
filesystem tools (read/edit/search), the string-replace editor, subagent
spawning, the skill catalog, web search/fetch, workflows, background jobs,
`ralph`, `goal`, and plan-mode. `sandbox-policy`/`approval` are pinned to the
`read-only`/`ask` preset pair explicitly — the loader itself refuses to boot
on a sandbox/approval combination that doesn't match one of its three named
presets (`read-only`, `workspace-write`, `danger-full-access`), which is a
real guardrail against a typo silently falling back to something permissive.

## Before this becomes a real adapter input

- Mint a real capability token scoped correctly and confirm tool
  registration and one real `mcp__toxagent__*` call round-trips.
- Decide whether `read-only`+`ask` is really desired end-state (`ask`
  without an answerer fails closed per ADR 0007 — fine for MCP-only tool
  calls that never hit the sandbox/approval path, but confirm that's true
  rather than assumed).
- Snapshot the composed `--dump-config` output the way
  `scripts/snapshot_opencode_contract.py` does for OpenCode, so a future SDK
  version bump has something to diff against.
