# Pinned OpenCode V1 ToxAgent profile

`toxagent.json` is the only product agent profile.  It is deny-all and
re-enables only the ToxAgent MCP namespace.  OpenCode names a remote MCP tool
`<server>_<tool>`, so the allow rule is `toxagent_*` — an `mcp_`-prefixed rule
matches nothing, leaves `*: deny` in force, and the model is handed no tools at
all (progress log §3.2).  The profile does not inherit a coding, shell,
filesystem, web, skill, or subagent surface.

`serve --pure` with `OPENCODE_CONFIG` pointed here is **not** enough on its own:
the machine's `~/.opencode` and `~/.config/opencode` still resolved `read:
allow` and foreign MCP servers into the agent (progress log §4.2).  Launch the
worker with `HOME` and every XDG dir pointed at an isolated root (see
`scripts/run_local_phase3.sh`) and gate the run on the live `GET /agent`
surface with `scripts/assert_opencode_surface.py`.

`maxSteps` is `32`, not the plan's initial `4` (progress log §4.6). V1's
`POST /session/{id}/prompt_async` has no per-request step field — this static
value is the *only* enforced cap, for every intent, regardless of
`RuntimeSettings.max_steps_qa`/`max_steps_research`.  A live report-Q&A run hit
exactly `4` (three `get_analysis_slice` reads plus one `submit_grounded_answer`)
and the turn ended there — OpenCode does not resume a turn once its step count
is spent, so the product's own "one correction attempt" policy (plan §9.5) was
structurally unreachable whenever gathering context took more than one step.
The independent server-side `TOXAGENT_MAX_TOOL_CALLS` budget remains the cost
control: deploy report-Q&A with 12 by default and raise it to 24 only for a
measured evidence-research workload.

The remote-MCP template is not a shared server configuration.  At dispatch,
the V1 adapter creates the product binding first, then adds this exact MCP
configuration through OpenCode's private `/mcp` API with a short-lived,
binding-scoped capability token.  The capability must never be placed in a
prompt, a product event, or a checked-in profile.

Deploy the OpenCode worker with an isolated project/data directory per product
run.  OpenCode V1 persists MCP configuration at project scope, so sharing that
directory would retain a run capability beyond its intended authority.
