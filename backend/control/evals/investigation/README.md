# Comparison study: ToxAgent against general platforms, graded blind by a lab

Implements RETHINK §5.4–§5.5 (`docs/RETHINK_TOXAGENT_AGENTIC_RESEARCH_EVALUATION_VI.md`)
as decided on 2026-09-25: the "SME" arms are general platforms (OpenAI, Anthropic,
Google models) answering the same cases as ToxAgent; a chemistry lab grades later.
Nothing in this directory judges quality. It runs, logs and blinds.

**Result label (RETHINK §5.2): `published-data-transfer`.** Cases are built from
published outcomes and turned into ToxAgent questions. The numbers this study
produces are never an `external-native` benchmark score and are not compared
with any leaderboard.

## Systems

| Id | RETHINK arm | What it is |
|---|---|---|
| `A_predictor_template` | A | ToxPred output through a fixed template; no language model |
| `C_toxagent_current` | C | ToxAgent as shipped (no case, no skills) |
| `D_toxagent_investigator` | D | Case-based investigator, skills loaded on demand |
| `D0_toxagent_case_no_skills` | D ablation | Case, no skills |
| `Ds_toxagent_case_static_skills` | D ablation | Case, every skill composed into the prompt |
| `P_openai_bare` / `B_openai_snapshot` | P / B | OpenAI model via `codex exec`, question only / with the ToxPred snapshot |
| `P_anthropic_bare` / `B_anthropic_snapshot` | P / B | Anthropic model via `claude -p`, no tools |
| `P_google_bare` / `B_google_snapshot` | P / B | Google Gemini, answered out of process (manual adapter) |

Each ToxAgent arm is a separate control-plane deployment with its own flags;
the runner reads `/v1/system/effective-product` and refuses to record an arm
against a deployment whose flags or skill mode do not match (`systems.py`).

Platform arms answer from the model's own knowledge: no web search, no tools,
an empty working directory. ToxAgent arms have their product affordances
(predictor, literature search, case, skills). That asymmetry is the question
being studied, and it is recorded, not hidden.

## Protocol

1. **Cases are fixed first.** `case_specs.json` is hand-written; `cases.py
   --build` resolves structures from PubChem (never typed from memory) and
   writes `cases/`. Each case records its spec hash; the study manifest
   records the case-set hash. Reference outcomes are for graders only and are
   marked `pending_lab_verification`.
2. **Same input for every system.** Every platform receives the same neutral
   preamble (`prompts.py`, hashed), the case text, and in `B_*` arms the
   predictor snapshot the template arm renders from, byte for byte. Multi-turn
   cases are stitched into one transcript per turn for stateless clients.
   ToxAgent receives the same message text; on case arms a turn's structured
   context is also filed through the case API, as the UI would.
3. **Everything is logged** (`record.py`): per (case, system, trial) the exact
   prompt sent, the raw output, the model id the platform *reported*, timings,
   usage and cost when reported, and for ToxAgent the full trace (runs, tool
   calls, answers, decision states with the skill record, the case, its event
   log and every dossier). Records are append-only. The manifest adds git
   commit, host, adapter versions, Codex configuration (non-secret keys only),
   effective-product hashes and denominators.
4. **Blinding** (`packet.py`): responses shuffled per case with a recorded
   seed, product/vendor/model names and internal ids replaced; the unblinding
   key is written outside the packet. Missing responses are counted, never
   silently dropped.
5. **Grading** (`rubric.json`): seven separate 0–3 dimensions (two allow NA)
   and seven critical-error flags. No total score.
6. **Analysis** (`scorecard.py`): per system and dimension, the mean over
   cases with a case-bootstrap interval; critical-error rates; paired
   differences on shared cases; inter-rater agreement when two graders overlap.

## Running

```bash
# 1. cases (once; network)
python -m evals.investigation.cases --build
# 2. one control plane per ToxAgent arm (flags per arm), then:
TOXAGENT_STUDY_TOKEN=... python -m evals.investigation.run --study pilot-2026-09 \
  --systems A_predictor_template,C_toxagent_current,D_toxagent_investigator,P_openai_bare,B_openai_snapshot,P_anthropic_bare,B_anthropic_snapshot,P_google_bare,B_google_snapshot \
  --toxagent C_toxagent_current=http://127.0.0.1:8011 \
  --toxagent D_toxagent_investigator=http://127.0.0.1:8012 \
  --snapshot-from http://127.0.0.1:8011
# 3. manual arms: answer runs/<study>/manual/**/turn<k>.prompt.md, write
#    turn<k>.response.md and turn<k>.meta.json, re-run step 2 until nothing is pending
# 4. packet for the lab
python -m evals.investigation.packet --study pilot-2026-09 --packet-id lab-1
# 5. after grading
python -m evals.investigation.scorecard --study pilot-2026-09 --packet-id lab-1 \
  --grades a.csv b.csv --compare D_toxagent_investigator:C_toxagent_current
```

`turn<k>.meta.json` must contain `model_id_resolved` (as the platform reported
it), `answered_at` and `channel`; the runner refuses a response without them.

The unblinding key (`runs/<study>/keys/`) is git-ignored so nobody reading the
repository can unblind a packet. It is not a single point of loss: the same
study records, systems and seed rebuild the same assignment (checked on
`pilot-2026-09-25` / `lab-1`: identical key and identical case files).

A platform that declines to answer (a provider safety filter, not quota) is
retried once like any error; every attempt stays in `records.jsonl`, the report
counts refusals separately from quota and transport failures, and a response
still refused after the retry is left out of the packet with its reason in the
key.

## Known limits

- Blinding is best effort: style can identify a system.
- The pilot is small (10 cases). Intervals will be wide; that is the honest
  result, not a reason to add a total score.
- The Codex CLI keeps its own agent instructions and the configured provider
  may be a proxy; the manifest records the configured and reported model.
- A consumer chat product (ChatGPT, Gemini app) may behave differently from
  its model through a CLI or API. Use the manual adapter with the consumer
  product as `channel` if that is the comparison wanted.
