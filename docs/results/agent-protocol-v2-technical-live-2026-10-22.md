# Protocol v2: all four arms on the full technical set, live on openai/gpt-6-luna

**Date:** 2026-10-22 · **Branch:** `feat/agent-benchmark-master-plan` · **Plan:**
`ToxAgent_Agent_Implementation_and_Benchmark_Master_Plan_2026-10-21.md` §13–§14, §17.4 step 4

> **This is engineering conformance on synthetic fixtures, not scientific performance.**
> Every case in `cases_protocol_v2_technical/` uses invented compounds, numbers and sources,
> and every gold file is `synthetic_engineering_gold`. Grades come from two LLM graders, not
> SMEs. The scorer itself prints "not publishable as a scientific result". Read the numbers as
> evidence about the machinery and the agent design on this set, never about a molecule.

## 1. Summary

- **What ran.** The four protocol v2 arms -- A2 template, B2 fixed workflow, C2 generic ReAct,
  D2 case-based investigator -- on the 12 technical cases (T1–T6 × toxicity-only /
  activity-guided, 15 turns), one trial each. B2, C2 and D2 ran as local deployments of commit
  `b4538cd` on `openai/gpt-6-luna` through OpenCode 1.17.11, budget `investigator-pilot-v2`,
  release policy `strict-reviewed-v1`; A2 renders the pinned replay record and uses no model.
  (The earlier laptop smoke ran three arms on one case.) Preflight passed: one replay file, one evidence file, identical
  effective budgets and contracts, and only the declared treatment flags differ. 48/48
  primary attempts completed without an infrastructure error.
- **Task Success** (assigned cases; range = the two graders):

  | Arm | All 12 | Scientific (T1–T5, 10) | Stress (T6, 2) |
  |---|---|---|---|
  | A2 template | 3 | 2 | 1 |
  | B2 fixed workflow | 1–3 | 0–2 | 1 |
  | C2 generic ReAct | 7–9 | 5–7 | 2 |
  | D2 investigator | 6 | 5 | 1 |

  D2 − B2, paired: **+0.25 to +0.42**. The interval cannot be read: the 12 cases form only
  **two dependency clusters** (one compound each), so the cluster bootstrap has two units.
  C2 does at least as well as D2 here, at 60% of D2's tokens and 70% of its latency.
- **Four mechanisms account for most failures**, and each is a product or arm defect rather
  than a property of the model (section 5):
  1. the **semantic reviewer blocks correct inferential claims** ("the measured 0.3 µM and the
     model's non-blocker label disagree" is judged *partially supported* because no single
     source states the disagreement), so the answer a T3 case exists for is withheld;
  2. **a value the researcher supplies cannot be cited**, so no arm can use a measured pIC50
     or compute an exposure margin: **T5 (update) is 0/2 for every arm**;
  3. the **deterministic fallback reads the session's active analysis, not the run's bound
     subject**, and tells the researcher to "run an analysis first" while one is bound;
  4. **arm B cannot retrieve on toxicity-only cases**: a frozen analysis carries no ligand
     names, so relevance rejects every record as `compound_mismatch` and B then says the
     evidence does not exist. B2 is therefore a weak baseline and the D2 − B2 delta overstates
     D's design (plan §18, first risk row).
- **Two measurement defects were found by running the study and fixed before scoring**
  (section 3): grader references named limitation codes without saying what they mean
  (`f776bbd`), and the runner recorded every withheld result as an empty reply (or only the
  fallback sentence) although the product had shown the reader the confirmed facts
  (`6edddd0`; 10 of the 45 model-arm turns -- 6 empty, 4 fallback-only -- re-read from the
  deployments' transcripts, no arm rerun).
- **Stability** (4 cases declared in advance, repeated once): A2 and B2 repeated every
  outcome; C2 3–4 of 4; D2 3 of 4, and D2's T3 failure turned into a published, correct answer
  in the repeat -- release outcomes vary run to run (section 4.6).

## 2. Set-up

| Item | Value |
|---|---|
| Agent code | `b4538cd` (frozen worktree; every arm a local uvicorn + SQLite deployment) |
| Evaluator code | packet and scorer at `6edddd0` (`f776bbd` legend, `6edddd0` adapter fix) |
| Model | `openai/gpt-6-luna`, OAuth subscription quota, declared as a custom model in a benchmark-only OpenCode server on port 4097; OpenCode reports `gpt-6-luna` for every call. The served model behind the OAuth endpoint was not independently verified. |
| Runtime | OpenCode 1.17.11, agent surface checked deny-all except the ToxAgent MCP namespace |
| Budget | `investigator-pilot-v2`: 360 s, 20 tool calls in total with 6 reserved for submissions, one semantic repair |
| Release | `strict-reviewed-v1` for B2/C2/D2 (A2 has no model) |
| Inputs | `cases_protocol_v2_technical` (replay `2bf45336…`, evidence `306bf2d9…`), frozen full text, no predictor or live provider calls |
| Treatments | B2 `fixed_workflow_v1`; D2 `scientific_case_v1` + `scientific_skills_v1`; C2 neither |
| Execution | `tech-gpt6luna-20261022b`, one case at a time, arms interleaved per case, 2026-10-22 08:55–09:31 (+08:00) |
| Graders | Claude Opus 5.5 (operated the study; **not blind**) and Codex `gpt-6.1-sol`, reasoning ultra (blind, read-only packet). Codex is from the same vendor as the system under test. |

### Infrastructure log (all attempts are reported)

- A first study, `tech-gpt6luna-20261022`, was started with `--parallel 3 --trials 2` on the
  same 7.8 GB WSL laptop that also ran the user's Docker stack and a second OpenCode server.
  It recorded 14 attempts (4 infrastructure errors: three client read timeouts and one
  `sqlite3 database is locked` HTTP 500) before WSL restarted under memory pressure. It is
  kept as a diagnostic and never scored (`ABORTED.md`).
- In the rerun, the benchmark's OpenCode server grew by about **200 MB per agent run**
  (291 MB → 1.6 GB after the first ten recorded attempts, seven of them model runs). It was
  restarted between cases whenever it passed 900 MB (the run log records each restart with its
  size); a memory watchdog sampling every 20 s never reached its 800 MB floor
  (`.data/benchmarks/watchdog.log`).
- To switch to that per-case regime, the runner was stopped at a case boundary. One B2
  attempt on `inv-a03` had been submitted by the runner in the same second; it was cancelled
  (12 tool calls, 9,733 tokens, 22 s, read back from the deployment), is listed with its usage
  in `interrupted-attempts.jsonl`, and its case was attempted again as the primary.

## 3. Measurement defects found and fixed before scoring

| Defect | Effect | Fix | Handling |
|---|---|---|---|
| Grader references listed `L7, L8` and forbidden inferences as bare codes; the packet never defined them | Two graders could not grade limitations consistently | `f776bbd`: every reference prints each code's meaning (§4.6 of the proposal, in English); the README carries the taxonomy | Packet `pk-main-1` superseded before grading (`SUPERSEDED.md`); grading used `pk-main-2` |
| The replay adapter matched assistant messages on `message.run_id`, which the API's `Message` does not have | Every non-published run was recorded with an empty reply ("no response was produced") or only the fallback sentence, though the reader had been shown the server's confirmed facts and the reason | `6edddd0`: match through the parts that name the run (`result_ref`, `dossier_ref`, `answer_ref`); adapter v2 | `evals.investigation.recapture` re-read 10 turns from the deployments' persisted transcripts into the derived study `tech-gpt6luna-20261022b-r1`; no arm was run again, the source study is untouched, every amended turn says so in its `meta`. Both graders re-graded the 10 responses (Codex blind); one changed outcome (D2 `inv-a01`, now a success for both). |

The four-arm e2e test had not caught the adapter defect because none of its runs was
withheld; `tests/unit/test_replay_adapter_messages.py` now holds it.

## 4. Results (study `tech-gpt6luna-20261022b-r1`, packet `pk-r1`)

Graders agreed on Task Success for 44 of 48 responses (91.7%, κ = 0.83). The four
disagreements are pending adjudication, not resolved here. Three are responses that reached
the reference conclusion but missed a required limitation or a secondary element; the fourth
(B2 `t04`) asked for the right exposure, but Codex judged its claim that the material held no
prediction a factual error. Codex failed all four and Claude passed them. Section 4.1 therefore gives both readings; the scorer's two-grader
table counts a disagreement as not a success.

### 4.1 Task Success by partition

| Partition | A2 | B2 | C2 | D2 |
|---|---|---|---|---|
| activity-guided (5 scientific) | 1 | 0–1 | 2–3 | 2 |
| toxicity-only (5 scientific) | 1 | 0–1 | 3–4 | 3 |
| T1 read/explain (2) | 2 | 0–1 | 1–2 | 2 |
| T2 extraction (2) | 0 | 0 | 2 | 2 |
| T3 conflict (2) | 0 | 0 | 1–2 | 0 |
| T4 missing input (2) | 0 | 0–1 | 1 | 1 |
| T5 update across turns (2) | 0 | 0 | 0 | 0 |
| T6 stress (2): failed stage / injected instruction | 1 / 0 | 1 / 0 | 1 / 1 | 0 / 1 |

### 4.2 D2 − B2 (primary, paired)

| Reading | D only | B only | Both | Neither | Delta |
|---|---|---|---|---|---|
| Claude | 4 | 1 | 2 | 5 | +0.25 |
| Codex | 6 | 1 | 0 | 5 | +0.42 |

Bootstrap intervals are printed by the scorer ([0.00, 0.50] and [0.17, 0.67]) but rest on two
dependency clusters and should not be quoted. Because B2 fails every toxicity-only case that
needs a source or the analysis (`t01`–`t03`, `t05`, `t06`) for defects of its own (F3, F4),
this delta is not evidence for D's design.

### 4.3 Mandatory information, limitations, critical errors (two-grader card)

| Arm | Facts met / owed | Limitation required recall | Stated precision* | All required | Any critical error |
|---|---|---|---|---|---|
| A2 | 4/9 | 53% | 30% | 42% | 1/12 (`predictor_error_uncaught`) |
| B2 | 3/9 | 41% | 58% | 42% | 4/12 (Codex: `major_factual_error` ×4 -- "the material holds no evidence/prediction"; `endpoint_confusion` ×1) |
| C2 | 6/9 | 76% | 57% | 67% | 0/12 |
| D2 | 6/9 | 53% | 45% | 58% | 2/12 (`predictor_error_uncaught` on a withheld T3, `major_factual_error`) |

\* The synthetic gold lists no optional limitation codes, so a correct caveat the case did not
require counts against precision; precision is a lower bound here. Critical errors are counted
when either grader marks one. Claude marked only A2's pass-through of the wrong prediction;
Codex also marked as factual errors the answers that said the material held no prediction or
no evidence when it did (B2 ×4, D2 `t05`), and D2's withheld `t03`, whose released facts were
only the wrong model call.

### 4.4 Release and review behaviour (15 turns per arm)

| Arm | Published | Withheld (partial facts) | Of which reviewer-blocked | Review rounds | Repairs | Verdicts not "supported" |
|---|---|---|---|---|---|---|
| B2 | 11 | 4 (2 fallback) | 1 | 16 | 3 | 5/26 |
| C2 | 13 | 2 (both fallback) | 0 | 19 | 6 | 8/43 |
| D2 | 11 | 4 | 4 | 24 | 9 | 16/79 |

No repair was removal-only. D2 read a skill 16 times and updated its case 14 times; neither
appears to have helped on this set.

### 4.5 Cost and latency (every attempt)

| Arm | Median s/turn | Median s/case | Tokens (all 12 cases) | API-list equivalent† |
|---|---|---|---|---|
| A2 | 0 | 0 | 0 | 0 |
| B2 | 21.3 | 25.4 | 545 k (+9.7 k interrupted) | ≈ $0.04 |
| C2 | 48.8 | 49.8 | 1.32 M | ≈ $0.07 |
| D2 | 69.6 | 68.7 | 2.21 M | ≈ $0.10 |

† The runs used an OAuth subscription quota, so OpenCode reports no cost. The equivalent
uses the GPT-6 Luna prices listed in OpenCode's model catalogue for the routes that sell it
(OpenRouter, LLM Gateway; input $0.10, cached $0.01, output $0.50 per million tokens); it is an order of magnitude, not a bill. Usage was reported for every model
turn; none is unknown.

### 4.6 Stability

Declared before the main results were read (`STABILITY_SUBSET_DECLARED.md`): `inv-a01`,
`inv-a02`, `inv-t02`, `inv-t03`, repeated once as study `stab-gpt6luna-20261022` on the same
deployments (all 16 attempts completed; no withheld reply needed re-reading). Graders agreed
on 14/16. Same outcome in both runs, per arm (Claude / Codex reading):

| Arm | Same outcome | What moved |
|---|---|---|
| A2 | 4/4 / 4/4 | nothing (deterministic) |
| B2 | 4/4 / 4/4 | `t02`: fallback → a published "no evidence" answer; still a failure |
| C2 | 4/4 / 3/4 | `a01`: Codex failed it in the main run for a missing interval caveat, passed it in the repeat |
| D2 | 3/4 / 3/4 | `t03`: withheld by the reviewer in the main run, **published and correct** in the repeat; `a01` also moved from withheld to published (a success both times) |

Release outcomes are the least stable part: of D2's four cases here, two were withheld in the
main run and none in the repeat. On this evidence D2's 0/2 on T3 is partly the luck of one
review round, which is F1's cost made visible. Four cases and one repeat are a check, not a
reliability estimate.

## 5. Why the arms failed: four product and arm defects

**F1 -- The reviewer blocks correct inferences (D2 most).** Under `strict-reviewed-v1` a claim
judged `partially_supported` blocks release after one repair. The reviewer (the same model)
judges claim-by-source and often marks as partial a claim that draws the inference the task
asks for: "the hERG record is about another target, so it does not contradict the EGFR
estimate" (D2, `inv-a03`), "the model's non-blocker label and the measured 0.3 µM disagree"
(D2, `inv-t03`). It is also inconsistent: C2's claim on `inv-a03` that the sources describe
"distinct targets and measurements" was judged *supported* -- the same judgement, stated
without the word "contradict". Citation granularity adds more: "the interval is 6.8 to 7.4"
cites one endpoint; "the decision is review because the interval crosses" cites the reason
code but not the decision field. All four of D2's withheld turns are this. In three the reader
got confirmed facts but not the answer; in `inv-a01` the confirmed facts happened to answer
both questions, and both graders passed it. Gold-relevant consequence: D2 scores 0/2 on T3, the task
group the agent design is meant to win. *Fix direction:* review relations (comparison,
conflict, scope) against the cited set as a whole, with a `derived` claim kind that names its
inputs; allow multi-field citation; measure the reviewer's false-block rate on true claims as
§14.5 asks.

**F2 -- A value the researcher supplies cannot be cited (every arm, T5 = 0/2).** A numeric
claim must name an `observation_id` and `field_path`; researcher context (`add_context`, or a
value in the message) is not an observation. One arm said so outright: "this interface cannot
validate a derived margin against researcher-supplied exposure as a sourced numeric claim".
The measured pIC50 6.2 and the free Cmax 0.01 µM therefore never become claims, and the 30-fold
margin is never stated. This blocks user story 4 on the product path even for D2, whose case
holds the value. *Fix direction:* persist researcher-supplied values as user-sourced
observations (source class `user_supplied`, never independent evidence) that claims and the
margin tool can cite.

**F3 -- The fallback ignores the run's subject.** `SubmitAnswer._build_fallback` reads
`session.active_analysis_id`. A run bound to a frozen toxicity analysis has none, so the
fallback says "this session has no recorded prediction … please run an analysis first"
(B2 `t01`, `t02`; C2 `t05` turn 1) -- false, and the wrong instruction. *Fix:* build the
fallback from the run's subject binding, as inventory and report already do (#44).

**F4 -- Arm B cannot retrieve for a toxicity-only subject.** B2's rule-built search for a
frozen analysis finds the right records, but relevance rejects all of them as
`compound_mismatch`: the analysis carries a SMILES and no ligand names, unlike a screening's
`ligand_names` (`t02`, `t03`, `t05`). On `t06` B2 does not search at all. B2 then says the
material holds no evidence or no prediction; Codex graded that a factual error on `t03`, `t04`,
`t05` and `t06`. C2 and D2 pass only because the model typed the name from
the question. *Fix:* carry the frozen analysis's ligand names into the binding, as screenings
do, then re-run B and ask the owner to review it before any D − B claim (#50, plan §18).

Smaller observations:

- Rendered claim values are spliced into prose raw: "(False)", "(review)", a trailing
  "blocker", "IC50 of The patch-clamp source reports …". Readability, not correctness.
- B2 published "Table 1 reports an IC50 for TX-101" without the 12 nM the question asked for,
  in both the main run and the repeat.
- The injected instruction in `t06` was never followed by any arm.
- A2 (no model) is the floor it should be: it answers what the frozen artifacts state (T1, the
  failed stage) and nothing that needs a source.

## 6. What this does and does not establish

It establishes that the protocol runs end to end on a live model: preflight, interleaved arms,
primary attempts, packets, two graders, scorer, freeze, all with every attempt and defect on
record. It established -- by failing -- that two measurement paths needed fixing, and it
located four defects worth fixing before any pilot.

It does not establish that D is better or worse than B or C. N = 12 synthetic cases in two
dependency clusters, one trial, LLM graders (one not blind, one same-vendor), and a B arm with
a known retrieval defect. The pilot needs SME gold for 24 cases, a fixed B reviewed by its
owner, human graders, and a dependency-aware sample.

## 7. Recommended next steps

1. Fix F3 (small, clear) and F4 (ligand names on the frozen analysis), then have the
   scientific owner review B (#50).
2. Design the F1 change with the release-policy owner (#47) and add the §14.5 control "reviewer
   blocks a true claim" to the evaluator controls with real examples from this run.
3. Implement F2 (user-sourced observations) -- it is the US4 path.
4. Re-run this technical set on the same model after 1–3, two trials, before spending SME time.
5. For laptop runs: keep `--parallel 1` on SQLite deployments and restart OpenCode between
   cases (or move the deployments to Postgres).

## 8. Reproduce

Code: agent at `b4538cd`, evaluator at `6edddd0`. Artefacts under `.data/benchmarks/`
(ignored by git): the source and derived studies, packets, keys, scorecards, a
`freeze-bundle.json` in each scored study, the aborted study and the stability study; grading
sheets under `~/.cache/toxagent-bench/grading/`. The numbers in this report were checked
against those files by a separate read-only Codex pass (`gpt-6.1-sol`); the discrepancies it
found were corrected after being verified against the data.

```bash
cd backend/control
python scripts/agent_benchmark.py preflight --case-set evals/investigation/cases_protocol_v2_technical \
  --arm B2_fixed_workflow=http://127.0.0.1:8021 --arm C2_generic_react=http://127.0.0.1:8022 \
  --arm D2_investigator=http://127.0.0.1:8023 --token <token>
python scripts/agent_benchmark.py run --study tech-gpt6luna-20261022b --case-set evals/investigation/cases_protocol_v2_technical \
  --arm ... --trials 1 --parallel 1 --cases <one case at a time>
python -m evals.investigation.recapture --study tech-gpt6luna-20261022b --derived tech-gpt6luna-20261022b-r1 --arm ... --token <token>
python scripts/agent_benchmark.py packet --study tech-gpt6luna-20261022b-r1 --case-set <set> --packet-id pk-r1
python scripts/agent_benchmark.py score --study tech-gpt6luna-20261022b-r1 --case-set <set> --packet-id pk-r1 \
  --grades grades_claude_pk-r1.csv grades_codex_pk-r1.csv --compare D2_investigator:B2_fixed_workflow
python scripts/agent_benchmark.py freeze --study tech-gpt6luna-20261022b-r1 --case-set <set>
```
