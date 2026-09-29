# Study report — pilot-2026-09-25

Process description only; quality is graded by the lab.

| System | Arm | ok / error / pending / not run (of expected) | Failed attempts by class (all attempts) | Model(s) | Median s per turn | Usage (reported) |
|---|---|---|---|---|---|---|
| `A_predictor_template` | A | 10 / 0 / 0 / 0 (10) | — | — | — | — |
| `B_anthropic_snapshot` | B | 10 / 0 / 0 / 0 (10) | provider_refusal: 1, quota_or_capacity: 8 | claude-opus-5, claude-opus-5-5 | 67.3 | cache_creation_input_tokens=48927, cache_read_input_tokens=14985, cost_usd=1.04992, input_tokens=52, output_tokens=26094 |
| `B_google_snapshot` | B | 1 / 9 / 0 / 0 (10) | quota_or_capacity: 9 | gemini-3.6-flash | 15.9 | duration_ms=15859 |
| `B_openai_snapshot` | B | 10 / 0 / 0 / 0 (10) | — | gpt-6-sol | 64.9 | cached_input_tokens=1.98554e+06, input_tokens=2.38641e+06, output_tokens=21482, reasoning_output_tokens=11498 |
| `C_toxagent_current` | C | 10 / 0 / 0 / 0 (10) | — | gpt-5.6-luna | 78.1 | tokens_cache_read=327168, tokens_cache_write=0, tokens_input=253260, tokens_output=24730, tokens_reasoning=7019, tokens_total=612177 |
| `D0_toxagent_case_no_skills` | D-ablation | 10 / 0 / 0 / 0 (10) | — | gpt-5.6-luna | 172.8 | tokens_cache_read=1.13459e+06, tokens_cache_write=0, tokens_input=385859, tokens_output=58578, tokens_reasoning=10310, tokens_total=1.58934e+06 |
| `D_toxagent_investigator` | D | 10 / 0 / 0 / 0 (10) | — | gpt-5.6-luna | 176.4 | tokens_cache_read=1.152e+06, tokens_cache_write=0, tokens_input=400215, tokens_output=66476, tokens_reasoning=12783, tokens_total=1.63147e+06 |
| `Ds_toxagent_case_static_skills` | D-ablation | 10 / 0 / 0 / 0 (10) | — | gpt-5.6-luna | 184.3 | tokens_cache_read=1.48378e+06, tokens_cache_write=0, tokens_input=452323, tokens_output=71068, tokens_reasoning=12365, tokens_total=2.01953e+06 |
| `P_anthropic_bare` | P | 9 / 1 / 0 / 0 (10) | other: 1, provider_refusal: 4, quota_or_capacity: 8 | claude-opus-5, claude-opus-5-5 | 36.4 | cache_creation_input_tokens=18262, cache_read_input_tokens=0, cost_usd=0.612086, input_tokens=40, output_tokens=19217 |
| `P_google_bare` | P | 0 / 10 / 0 / 0 (10) | quota_or_capacity: 10 | — | — | — |
| `P_openai_bare` | P | 10 / 0 / 0 / 0 (10) | — | gpt-6-sol | 82.9 | cached_input_tokens=2.36122e+06, input_tokens=2.83121e+06, output_tokens=23920, reasoning_output_tokens=13025 |

## ToxAgent arms: process

| System | Turns | Fallback answers | Median tool calls / turn | Stop reasons |
|---|---|---|---|---|
| `C_toxagent_current` | 11 | 1 | 8 | blocked: 1, insufficient_evidence: 10 |
| `D0_toxagent_case_no_skills` | 11 | 0 | 14 | budget_exhausted: 2, insufficient_evidence: 6, sufficient: 3 |
| `D_toxagent_investigator` | 11 | 0 | 13 | budget_exhausted: 1, insufficient_evidence: 6, sufficient: 4 |
| `Ds_toxagent_case_static_skills` | 11 | 3 | 12 | blocked: 3, budget_exhausted: 2, insufficient_evidence: 5, sufficient: 1 |

## Skill triggering — `D_toxagent_investigator`

| Skill | Cases read | Trigger precision | Trigger recall | False triggers (negative cases) | Misses |
|---|---|---|---|---|---|
| `assess-conflicting-evidence` | 2 | 1.0 | 1.0 | — | — |
| `critique-case` | 9 | 0.444 | 1.0 | — | — |
| `interpret-model-attribution` | 1 | 1.0 | 1.0 | — | — |

## Errors (latest attempt)

- `B_google_snapshot` / inv-02-terfenadine-anonymous: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 27.57532199s.
- `B_google_snapshot` / inv-03-fexofenadine-negative-control: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 27.369364914s.
- `B_google_snapshot` / inv-04-cisapride-conflicting-assays: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 43.166143052s.
- `B_google_snapshot` / inv-05-astemizole-inhouse-contradicts: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 26.38969043s.
- `B_google_snapshot` / inv-06-aspirin-numeric-lookup: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 48.894600029s.
- `B_google_snapshot` / inv-07-dofetilide-attribution: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 11.388082509s.
- `B_google_snapshot` / inv-08-moxifloxacin-exposure: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 32.556596653s.
- `B_google_snapshot` / inv-09-bortezomib-applicability: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 53.559651265s.
- `B_google_snapshot` / inv-10-bisphenol-a-tox21-scope: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 11.454443965s.
- `P_anthropic_bare` / inv-07-dofetilide-attribution: RuntimeError: claude exited 1: API Error: Opus 5's safeguards flagged this message (https://www.anthropic.com/legal/aup). This sometimes happens with safe, normal conversations. Claude Code can't respond to this message with Opus 5.

Try rephrasing the request in a new session or change your model.

Learn more: https://support.claude.com/en/articles/16049681

Details: `[reasoning_extraction]`

Request ID: req_011CfQVN4eps9uDNKHaE1L6x

Message ID: msg_011CfQVN77fKk18vZgYWe8DD
- `P_google_bare` / inv-01-terfenadine-named-go-nogo: RuntimeError: Gemini API failed with HTTP 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- `P_google_bare` / inv-02-terfenadine-anonymous: RuntimeError: Gemini API failed with HTTP 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- `P_google_bare` / inv-03-fexofenadine-negative-control: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 34.286126284s.
- `P_google_bare` / inv-04-cisapride-conflicting-assays: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 37.992994061s.
- `P_google_bare` / inv-05-astemizole-inhouse-contradicts: RuntimeError: Gemini API failed with HTTP 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- `P_google_bare` / inv-06-aspirin-numeric-lookup: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 37.769374827s.
- `P_google_bare` / inv-07-dofetilide-attribution: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 59.825265863s.
- `P_google_bare` / inv-08-moxifloxacin-exposure: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 22.291296217s.
- `P_google_bare` / inv-09-bortezomib-applicability: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 42.680955038s.
- `P_google_bare` / inv-10-bisphenol-a-tox21-scope: RuntimeError: Gemini API failed with HTTP 429: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
Please retry in 4.725693298s.
