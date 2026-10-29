# Extending the agent layer

Written for engineers working on this repository.

Each recipe below ends with the tests that will tell you it is wired up.

## Add a tool

1. **Define it** in `tools/definitions/<area>.py`: a pydantic input model with
   `extra="forbid"`, a handler, and a `ToolDefinition` naming its profiles,
   timeouts and cost class. Return a `ToolOutput` with `canonical`,
   `model_view`, `ui_view` and `provenance`; attach `observation_ids` when the
   call produced something citable.
2. **Register it** in `tools/bootstrap.py`, behind a flag unless it is core.
3. **List it** in `src/toxagent/agent_profiles/tool_profiles.json`, under each
   profile that may call it, and under `flag_gated_tools` if it is gated. That
   file is the decision; the registry only exposes it.
4. **Regenerate** the inventory and the capability matrix.

Three things the input model decides, so put them there rather than in the
handler: an allowlist is an enum (an unknown value is then a schema refusal
before any code runs), a bound is a `Field` constraint, and whatever you do not
describe, the model learns only from the refusal. A field with no
`description` cost 17 refused submissions in one live run.

The description is the only documentation the model gets. Say what the tool
does, what it will not do, and which field paths a claim may cite.

Tests: `tests/unit/test_tool_registry.py` enforces the profile ceiling —
a conversational profile is a small closed set, and that is deliberate.

## Add a case operation

In `domain/scientific_case/operations.py`: write the transition, then add it to
`_OPS` with the actors allowed to perform it. Return `case` unchanged for an
idempotent repeat, so a caller can skip the write.

Decide the actor set first, and write the reason beside it:

- `Actor.MODEL` — only if the model may assert this. It then needs the
  operation in `update_scientific_case`'s schema and prompt guidance.
- `Actor.USER` — the researcher's, through an API route.
- `Actor.SERVER` — a consequence of something the server computed or observed.
  A derived numeric claim is the server's, because the server did the
  arithmetic.

Keep references, never copied values: a justification holding the number would
still look valid after the number changed.

Tests: a case must serialise and hash identically to the replay of its log, so
`tests/unit/test_scientific_case*.py` will catch a transition that writes state
the log cannot rebuild.

## Add a procedure

In `application/investigation/procedures.py`: a `ProcedureSpec`, a pure
function returning a `ProcedureResult`, and an entry in `CATALOG`. Then add the
name to the `Literal` in `tools/definitions/procedures.py` and load whatever it
reads.

A `ProcedureResult` has to carry the inconvenient half: the criteria it
applied, the records it excluded **with the reason**, the fields its inputs did
not state, and its sub-operation count. A record left out silently makes a
count mean something other than what the caller will read it as.

Procedures write nothing and reach no provider. Keep it that way: it is what
makes a repeat free and a frozen benchmark replayable.

## Add a stop reason or a gap

Both are enums read in several places, so the work is in the wiring, not the
value:

- A stop reason (`domain/decision_state.py`) needs its place on the knowledge
  axis, the execution axis, and — if the model may declare it — the draft
  schema's vocabulary in `validation/answer/draft_wire.py`.
- A gap (`domain/investigation_gap.py`) needs a line in the contract's
  `GAP_GUIDANCE` saying what it asks of the turn, or the model sees a label
  with no instruction.

The rule they both exist for: do not merge two outcomes that mean different
things. `provider_failure` and `insufficient_evidence` were one value once, and
a grader could not tell "the product could not look" from "the science is
unsettled".

## Add a rollout flag

In `platform/flags.py`, with an owner, `added_on`, and a `removal_condition`
written as the observation that would retire it — not "when it is stable".
Default off.

Then make the flag actually change behaviour, and write a test that asserts the
*prompt* differs, not only that the code path ran. A flag that changes what is
recorded but not what the model was told is a flag that does nothing.

Flags in this layer are also benchmark treatments. A flag that differs between
two arms and is not declared in `treatment_fields` makes the preflight refuse
the arms as confounded — see [benchmark.md](benchmark.md).

## Add a capability profile

Add it to `tool_profiles.json` as a closed set and wire whatever chooses it
(for the serving routes, `router.profile_for`). If it is meant to be cheaper
than an existing profile, assert the subset relation in a test: cheaper must
mean *can do less*, never *can do something the full profile cannot*.

Remember the repair turn. It reuses the investigator's prompt, so it must run
under the same profile — otherwise the prompt lists one tool set and the turn
is handed another.

## A rule for all of it

The model proposes; the server records. If a change would let a model write
state directly, decide a release, or produce a number that does not resolve to
a stored field, the answer stops being auditable — and an unauditable answer is
the one failure this layer exists to prevent.
