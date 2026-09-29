# Experiments

One-off probes that inform a design decision. They are not benchmarks and do
not feed a scorecard; each records its binary, model, configuration and raw
responses so the finding can be re-checked.

## OpenCode native `skill()` on an isolated profile (W9-14, RETHINK §4.9)

`python -m evals.experiments.opencode_native_skills --model openai/gpt-5.6-luna`

RETHINK §4.9 proposed trying OpenCode's own `skill()` on an isolated profile
that allows only the approved skills, before relying on the product's closed
MCP tools (`read_scientific_skill` / `read_skill_reference`, ADR 0012). The
product did not wait for it; this is the optional experiment.

**Setup.** OpenCode 1.17.11 (pinned), its own HOME/XDG directories, external
skill scans disabled, the production runtime's model catalog, the three active
decision-support skills of the shipped catalog copied into the workspace and
declared under `skills.paths`. Agent `skill-probe` denies every tool; `skill` is
allowed only for those three names (`read` stays denied, as in production).
Five prompts fixed before any run: one positive case per skill, two negative
controls. Three arms: `native` (the binary's own behaviour), `instructed` (plus
the one sentence the product's dynamic arm puts before its index), `denied`
(`skill` off, to measure what the native listing costs). One trial each.

**Result** (`runs/opencode-native-skills-20260926T033540Z/`, model
`openai/gpt-5.6-luna`, all 15 turns answered):

| Arm | skill calls | positive cases that loaded their skill | negatives clean | prompt tokens on a negative (input + cache read) |
|---|---|---|---|---|
| native | 0 | 0/3 | 2/2 | 2 512 / 2 516 |
| instructed | 0 | 0/3 | 2/2 | 2 567 / 2 571 |
| denied | 0 | — | 2/2 | 1 973 / 1 977 |

Findings, from the binary itself:

1. **Discovery works; the listing costs about 540 input tokens a turn** here
   (native − denied), paid on every turn whether or not a skill applies.
2. **The model never loaded a skill** in six positive opportunities, with or
   without the instruction, and answered directly. The same model through the
   product's closed MCP tools did read skills in the W8 pilot (the dynamic arm's
   trigger data is in the backlog). One trial per prompt is thin, but the
   difference is total, not marginal.
3. **The binary ships a built-in skill, `customize-opencode`**, listed next to
   the approved ones. An isolated profile has to deny it by name (the allowlist
   here does, with `"*": "deny"`).
4. **It scans `~/.claude/skills` and `~/.agents/skills` automatically** unless
   `OPENCODE_DISABLE_EXTERNAL_SKILLS=1` / `OPENCODE_DISABLE_CLAUDE_CODE_SKILLS=1`
   or an isolated HOME — the "never enable skills from HOME" rule of §4.9 needs
   one of them.
5. **References cannot be read natively with `read` denied**: the `skill` tool
   injects `SKILL.md` and points at files next to it, which only `read` can
   open. The closed MCP tool serves references without opening the filesystem.
6. `/api/skill` is the v2 API and takes a `location[directory]` object; the V1
   listing is `GET /skill?directory=…`. Probing `/experimental/tool` before the
   turns made every turn fail with an internal server error, so the harness
   probes after them.

**Decision:** keep the closed MCP tools (ADR 0012) as the production path. The
native route loaded nothing, costs context on every turn, needs extra switches
to stay isolated, and cannot serve references under the current permissions.
Re-run this probe if the runtime or model changes.

`runs/opencode-native-skills-20260926T033341Z/` is the native arm alone, run
just before, with the same result (0/3, 2/2). The two `-listing-unverified`
runs are earlier passes of the same prompts whose skill listing used the wrong
endpoint; their turns (0/3 loaded in each) agree with the result, but they do
not show which skills the server had. Passes that failed for harness reasons
(a crash, a model catalog without the deployed model) produced no model output
and were not kept.
