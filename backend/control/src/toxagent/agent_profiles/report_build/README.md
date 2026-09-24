# Report-builder profile

`AGENTS.md` is the behavioural specification; `skills/` holds the five
instruction packages; `references/` holds the shared policy documents both
refer to. None of it is a permission. The OpenCode profile below denies the
runtime `skill` tool along with everything else, so a skill is not something
the model loads — `harness/report_profile.py` composes the selected packages
into the system prompt before dispatch and records each file's hash in the
runtime manifest.

That matters for audit: "which instructions produced this report" is answerable
from the manifest, not from whatever a runtime happened to have on disk.

`maxSteps` is 64 rather than the Q&A profile's 32. A report build legitimately
does more work in one turn — a context read, a compound lookup, one explanation
per selected target, several searches, several evidence reads, then the draft —
and the same failure the Q&A profile hit at 4 steps (progress log §4.6) would
reappear here at 32: the turn simply ends, and the product's own single
correction attempt becomes structurally unreachable. The server-side
`TOXAGENT_MAX_TOOL_CALLS` budget remains the real cost control; `maxSteps` only
has to be large enough not to truncate an honest workflow.

The deployment surface is unchanged from the Q&A profile: deny-all with the
`toxagent_*` MCP namespace re-enabled, launched under an isolated `HOME` and
XDG root, and gated on the live `GET /agent` surface. See
`../opencode/README.md`, which this profile does not repeat.
