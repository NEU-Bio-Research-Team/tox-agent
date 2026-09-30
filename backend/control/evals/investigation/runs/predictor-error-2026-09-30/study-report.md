# Study report — predictor-error-2026-09-30

Process description only; quality is graded by the lab.

| System | Arm | ok / error / pending / not run (of expected) | Failed attempts by class (all attempts) | Model(s) | Median s per turn | Usage (reported) |
|---|---|---|---|---|---|---|
| `A_predictor_template` | A | 8 / 0 / 0 / 0 (8) | — | — | — | — |
| `B_anthropic_snapshot` | B | 8 / 0 / 0 / 0 (8) | — | claude-opus-5, claude-opus-5-5 | 29.0 | cache_creation_input_tokens=27952, cache_read_input_tokens=0, cost_usd=0.591262, input_tokens=26, output_tokens=15829 |
| `B_openai_snapshot` | B | 1 / 0 / 0 / 7 (8) | — | gpt-5.6-sol | 111.1 | cached_input_tokens=48128, input_tokens=69268, output_tokens=1744, reasoning_output_tokens=1001 |
| `D_toxagent_investigator` | D | 2 / 0 / 0 / 6 (8) | other: 1 | gpt-5.6-luna | 146.8 | tokens_cache_read=96768, tokens_cache_write=0, tokens_input=75066, tokens_output=8802, tokens_reasoning=1705, tokens_total=182341 |
| `P_anthropic_bare` | P | 8 / 0 / 0 / 0 (8) | — | claude-opus-5, claude-opus-5-5 | 29.7 | cache_creation_input_tokens=10584, cache_read_input_tokens=0, cost_usd=0.388431, input_tokens=422, output_tokens=12682 |
| `P_openai_bare` | P | 1 / 0 / 0 / 7 (8) | — | gpt-5.6-sol | 140.0 | cached_input_tokens=115712, input_tokens=151039, output_tokens=2295, reasoning_output_tokens=1426 |

## ToxAgent arms: process

| System | Turns | Fallback answers | Median tool calls / turn | Stop reasons |
|---|---|---|---|---|
| `D_toxagent_investigator` | 2 | 0 | 9.5 | insufficient_evidence: 2 |

## Skill triggering — `D_toxagent_investigator`

| Skill | Cases read | Trigger precision | Trigger recall | False triggers (negative cases) | Misses |
|---|---|---|---|---|---|
| `assess-conflicting-evidence` | 0 | None | None | — | — |
| `critique-case` | 2 | 0.5 | 1.0 | — | — |
| `interpret-model-attribution` | 0 | None | None | — | — |
| `assemble-report-context` | 0 | None | None | — | — |
| `compose-scientific-report` | 0 | None | None | — | — |
| `explain-predictor-results` | 0 | None | None | — | — |
| `preflight-report-draft` | 0 | None | None | — | — |
| `research-toxicology-evidence` | 0 | None | None | — | — |
