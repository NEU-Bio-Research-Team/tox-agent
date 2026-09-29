# Scientific skills

Investigation methods the decision-support agent can load when a situation calls
for them (ADR 0012, `docs/internal/rethink-agentic-research-evaluation.vi.md`
§4.5–§4.9, §5.5). Loaded and validated by `application/investigation/skill_catalog.py`.

## Package format

```text
<skill-id>/
  SKILL.md             # Agent Skills front matter (name, description) + instructions
  skill.manifest.json  # ToxAgent metadata (not part of the Agent Skills format)
  references/*.md      # optional; each one declared in the manifest
```

`skill.manifest.json` fields: `schema_version` (`scientific-skill-manifest-v1`),
`skill_id` (= `name` = directory), `version` (MAJOR.MINOR.PATCH), `status`
(`active` | `draft`), `owner`, `allowed_profiles`, `required_capabilities` (tool
names), `risk_tier` (`low` | `medium` | `high`), `output_contract`, `eval_set`
(`positive_tags` / `negative_tags` of the investigation cases that measure it),
`references`.

## Rules

- **Teach judgement, do not hide a workflow.** A skill says when it applies,
  what to check, the signs to change direction, when to stop and what to leave
  in the case. It never prescribes a fixed tool sequence.
- **A skill never adds a capability.** `required_capabilities` is a
  precondition: the skill is offered only if every listed tool is visible to
  the run. Nothing in a manifest can make a tool visible.
- **References are background, not sources.** They are never cited, and their
  numbers never go into an answer.
- **Drafts are never offered.** A skill written from a run's experience lands
  as `draft`; making it `active` is a reviewed change.
- **Every change is a release.** Any edit changes the skill's content hash,
  which runs record. Promote a new version only after the paired ablation on its
  `eval_set` (no skill / static / dynamic, same model, tools, cases and budget)
  shows gain on its positive cases with no rise in unsupported claims, false
  reassurance, or false triggers on its negative cases.

## Arms

| Arm | How to select | What the run records |
|---|---|---|
| off | default | nothing |
| static | `TOXAGENT_SCIENTIFIC_SKILLS_STATIC=1` | every offered skill as read |
| dynamic | `TOXAGENT_FLAG_SCIENTIFIC_SKILLS_V1=1` | offered skills; each read and reference read, with its hash |

The record is in the run's decision state (`GET
/v1/sessions/{id}/runs/{run_id}/decision-state`, field `skills`).
