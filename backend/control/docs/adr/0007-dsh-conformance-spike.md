# ADR 0007 — DSH conformance spike (DEC-06)

**Status:** accepted (spike only — no adapter) · **Date:** 2026-09-06 · **Plan:** §12 (W7), DEC-06

## Context

ADR 0004 already asserts a `dsh` adapter "written against the pinned version
`0.1.1-rc.2`... covered by contract suites marked `live_runtime`". That claim
is false: `harness/adapters/` has only `scripted.py` and `opencode_v1.py`. The
DSH side of ADR 0004's table is enum/config scaffolding
(`domain/runtime.py::RuntimeKind.DSH`, `config.py`'s `dsh_command`/
`dsh_version`) that nothing implements or tests — this ADR's own conclusion
(remaining-plan §12, this repo's progress log §4.8) is that building an
adapter against a carrier no one has run here would be exactly the
"code no one can verify" this project's discipline refuses to check in. ADR
0004 is corrected alongside this one to stop asserting DSH verification that
never happened.

This ADR is the isolated spike the remaining-implementation-plan's W7-01/02
ask for: install the real, official carrier, run the documented handshake,
and record what is actually true about it. It does not open an adapter.

## What was actually run (2026-09-06, this session)

An isolated venv (not any environment this repo's tests or dev stack use),
network access to the real PyPI:

```
python -m pip install --pre deepseek-harness-sdk==0.1.2rc1
```

This is a real package: `pypi.org/pypi/deepseek-harness-sdk/json` returns
author `DeepSeek`, project URLs at `github.com/deepseek-ai/deepseek-harness`
— the correct upstream project (progress log §4.8 already ruled out the
same-named, unrelated `deepseek-harness` package by a different author; this
is the SDK meta-package the corrected §4.8/§2.3 entries point at). It pulls
in `deepseek-harness-runtime-bin==0.1.2rc1` (the matching platform wheel) and
`pydantic>=2.12`. No other network egress happened in this spike beyond the
`pip install` and one deliberate, credential-less model-turn attempt below.

**Package/binary identity, for pinning:**

| Artifact | Version | sha256 |
|---|---|---|
| `deepseek_harness_sdk-0.1.2rc1-py3-none-any.whl` | 0.1.2rc1 | `24689eec01e95233feb75ca08a0fe748b491e71682f80d2c04e6f9485f20488a` |
| `deepseek_harness_runtime_bin-0.1.2rc1-py3-none-manylinux_2_28_x86_64.whl` | 0.1.2rc1 | `670d8af06845cc2fc16738f1b030375321f3f8268837152aa3100e14a7b9887d` |
| `deepseek_harness_runtime_bin-0.1.2rc1-py3-none-manylinux_2_28_aarch64.whl` | 0.1.2rc1 | `5052ca9fea2304e28d82a71012a32e7d2399322e1884a065cb08cf5bc57f017f` |
| `deepseek_harness_runtime_bin-0.1.2rc1-py3-none-macosx_14_0_arm64.whl` | 0.1.2rc1 | `2cac2256cdebfda726c3378e24f4be31c0847aece7f2a7eb5a8523559f6b5894` |
| `deepseek_harness_runtime_bin-0.1.2rc1-py3-none-win_amd64.whl` | 0.1.2rc1 | `390bd8cd5f8700fc609c58e1ccb78091d5c8c6e11c21656e284e0f68da0e148f` |

Extracted binary on this machine (linux x86_64), `--version` reports
`0.1.2-rc.1`; own sha256
`8ae368a6c2bfbe46a1f11e4f80926c27f2bfe342bff1e370157d0a5240744302`
(267,455,680 bytes) — consistent with the wheel above, unpacked.

**Smoke: `initialize` → `session/prompt` → events → `close`, isolated
`dsh_home`, no credential.**

- `DeepSeekHarness.start()` (subprocess launch + JSON-RPC `initialize`)
  succeeded in **0.92s**, no `DEEPSEEK_API_KEY` set anywhere in the
  environment. `close()` returned cleanly; `pgrep` immediately after found
  no surviving process — **clean reaping**, matching the requirement W7-10
  states rather than assumes.
- `stderr` was empty for the whole session (`STDERR LINE COUNT: 0`) — clean,
  matching W7-03's purity concern; this spike does not confirm stdout carries
  *only* JSON-RPC (not scraped for it), only that nothing leaked to stderr.
- A real turn (`harness.run("Say hello in one word.")`) was attempted
  deliberately, specifically to observe the missing-credential path — this
  is a local, pre-flight check inside the runtime, not a request that ever
  reached DeepSeek's API (no cost, confirmed by the error itself):
  ```
  turn/end: {"reason": {"kind": "error", "error": {
    "code": "MISSING_CREDENTIAL",
    "message": "llm-deepseek: no API key for provider route
      \"deepseek-official\"; store DEEPSEEK_API_KEY through the credentials
      service (the web Models page writes it), or export DEEPSEEK_API_KEY in
      the launching environment"
  }}}
  ```
  Typed, clean failure — no hang, no stack trace, no ambiguity.

## Findings that bear directly on W7-05/06/09

- **The default `profile="sdk"` is not deny-all — confirmed, not assumed.**
  The very first system-prompt event the runtime injects for that profile
  states: *"Current DSH file policy: **workspace-write**. Any available
  operation enforced by the DSH file sandbox may modify files under the
  session workspace... Approval policy: ask... without an available
  answerer, the request fails closed."* This is a coding-agent profile with
  file-write capability and an approval escape hatch, exactly the
  "tool coding không phù hợp ToxAgent" progress log §2.3 already flagged —
  now with the literal prompt text confirming it, not just the profile's
  name. **W7-05's custom deny-all profile is not optional hardening — the
  shipped default is actively wrong for this product**, the same severity
  bar as OpenCode's own captured-surface requirement (ADR 0004's deny-all
  rule).
- **No cancel or session-close method exists in the SDK's public surface** —
  confirmed by reading `deepseek_harness/client.py` in full: `start`,
  `close` (kills the whole subprocess), `initialize`, `session_prompt`,
  `request`, `notify`, subscription helpers. No `cancel`, no per-session
  `close`. This matches progress log §2.3's correction exactly:
  `runtime_cancel_supported=false` is not a hedge, it is what the protocol
  actually offers at this pre-release. `close_session`-shaped granularity
  would need killing and restarting the whole runtime process per session,
  which is a real operational cost W7-14 (Track B) should weigh, not a
  detail an adapter can paper over.
- **`HarnessClient.start()` inherits the *entire* calling process
  environment by default** (`env = os.environ.copy()`, then
  `env.update(self.config.env)` only adds/overrides keys — it never starts
  from empty). Unlike `run_local_phase3.sh`'s `env -i` isolation for
  OpenCode, a naive DSH launch leaks every environment variable the control
  plane process itself has, including any predictor/database
  credentials that happen to be in that process's environment. `dsh_home`
  isolates the runtime's *own* config/state directory, not the process
  environment — a future adapter needs its own explicit allowlist
  construction, the same discipline `run_local_phase3.sh` already applies to
  OpenCode; the SDK does not do this for you.

## Decision

DEC-06, updated: the "no distribution channel" blocker is gone (progress log
§4.8 corrected this on 2026-09-05/06) — a real, official, installable
pre-release exists and now has one verified smoke run on this host, pinned
by hash above. Phase 4/W7 remains **not opened past the spike**:

- No `harness/adapters/dsh.py` is added by this ADR. Writing one now, before
  a custom deny-all profile exists and before the coding-profile file-write
  finding above is resolved, would ship a runtime whose captured model
  surface has not been asserted the way `assert_opencode_surface.py` asserts
  OpenCode's — exactly the gap this project's own discipline exists to
  catch, not create.
- `config.py`'s `dsh_version` default is corrected from the stale
  `0.1.1-rc.2` (a version this spike never found available, and ADR 0004
  never actually verified against) to `0.1.2rc1`, matching what is real and
  hash-pinned above. This is a documentation/default correction, not a
  claim that a `dsh` runtime is now usable — `RuntimeKind.DSH` still has no
  registered adapter, so `TOXAGENT_RUNTIME_KIND=dsh` still fails to start a
  deployment (by construction — nothing implements `AgentRuntimeProvider`
  for it).
- Before W7-05 onward: build the custom `cordis.yml`-equivalent minimal
  profile (deny-all except an eventual ToxAgent MCP namespace), and confirm
  the resulting system-prompt injection no longer claims `workspace-write`.
  That confirmation is what should gate writing the adapter, the same way
  `assert_opencode_surface.py`'s live check — not the profile file's
  contents alone — gates OpenCode's.

## Consequences

- ADR 0004's DSH row and "Verification status" section are corrected in the
  same change as this ADR: the table now says what actually exists (an enum
  value and unused config fields, not an adapter), and the false "written
  against... covered by contract suites" claim for DSH is removed.
- This spike used no real provider credential and made no billable request —
  the one model-turn attempt failed at a local, pre-flight credential check,
  confirmed by the error itself (`MISSING_CREDENTIAL`, no network stack
  trace). A future Track A/B comparison (W7-13/14) will need a real
  `DEEPSEEK_API_KEY` and its own cost/consent, same as every OpenCode live
  run in this progress log.

## Follow-up (2026-09-06, same day): W7-05 custom profile built and booted for real

Built and booted `toxagent-control/agent_profiles/dsh/` — see its own README
for scope and caveats; progress log §46 has the full session transcript this
condenses. Two things this resolves from "Decision" above:

- **The composed sandbox/approval posture is no longer `workspace-write`.**
  Confirmed by the harness's own `session.event` stream during a real boot +
  turn attempt (not by re-reading the system prompt's prose, which the
  earlier spike used): `permission/preset: read-only`,
  `sandbox/mode: read-only`. Deny-all now has a live-verified profile, not
  only a design intent.
- **`@deepseek-ai/dsh-mcp-client` is real, version-matched to the SDK
  (`0.1.2-rc.1`), and speaks `streamable-http` with per-request headers** —
  a bearer-token MCP client shaped exactly like OpenCode's remote MCP
  config, not a gap this ADR's first pass assumed silently. It is not
  bundled in any default profile (`sdk`/`web`/`headless`/`tui` all lack it);
  adding it needs an `insert` patch entry with no `id` (targeting the
  profile root, which is the one implicit "group"; a named existing row
  like `tools` is a leaf and refuses `insert` with a clear runtime error).
  Loaded successfully against ToxAgent's real, running control plane with a
  deliberately invalid placeholder token and `failOnStartupError: false` —
  the harness stayed up either way, so this doesn't yet prove a real,
  authenticated round-trip, only that the wiring and boot sequence survive.

Still not opened: no adapter, no real capability-token round-trip, no
contract snapshot of the composed config for version-bump diffing. DEC-06
stays at "spike, not adapter."
