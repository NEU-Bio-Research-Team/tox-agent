# Study report — predictor-error-2026-09-30-budget900

Process description only; quality is graded by the lab.

| System | Arm | ok / error / pending / not run (of expected) | Failed attempts by class (all attempts) | Model(s) | Median s per turn | Usage (reported) |
|---|---|---|---|---|---|---|
| `A_predictor_template` | A | 8 / 0 / 0 / 0 (8) | — | — | — | — |
| `B_anthropic_snapshot` | B | 8 / 0 / 0 / 0 (8) | — | claude-opus-5, claude-opus-5-5 | 34.1 | cache_creation_input_tokens=28136, cache_read_input_tokens=2172, cost_usd=0.655425, input_tokens=28, output_tokens=18242 |
| `D_toxagent_investigator` | D | 8 / 0 / 0 / 0 (8) | — | gpt-5.6-luna | 188.1 | tokens_cache_read=940032, tokens_cache_write=0, tokens_input=255623, tokens_output=53867, tokens_reasoning=9293, tokens_total=1.25882e+06 |
| `P_anthropic_bare` | P | 8 / 0 / 0 / 0 (8) | — | claude-opus-5, claude-opus-5-5 | 37.1 | cache_creation_input_tokens=11375, cache_read_input_tokens=0, cost_usd=0.520384, input_tokens=30, output_tokens=17624 |

## ToxAgent arms: process

| System | Turns | Fallback answers | Median tool calls / turn | Stop reasons |
|---|---|---|---|---|
| `D_toxagent_investigator` | 8 | 0 | 19.0 | budget_exhausted: 1, insufficient_evidence: 6, sufficient: 1 |

## Skill triggering — `D_toxagent_investigator`

| Skill | Cases read | Trigger precision | Trigger recall | False triggers (negative cases) | Misses |
|---|---|---|---|---|---|
| `assess-conflicting-evidence` | 0 | None | None | — | — |
| `critique-case` | 8 | 0.25 | 1.0 | — | — |
| `interpret-model-attribution` | 0 | None | None | — | — |
| `assemble-report-context` | 0 | None | None | — | — |
| `compose-scientific-report` | 0 | None | None | — | — |
| `explain-predictor-results` | 0 | None | None | — | — |
| `preflight-report-draft` | 0 | None | None | — | — |
| `research-toxicology-evidence` | 0 | None | None | — | — |
