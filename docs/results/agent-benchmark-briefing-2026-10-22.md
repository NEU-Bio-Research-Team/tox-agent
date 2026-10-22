# ToxAgent agent benchmark: status briefing, 2026-10-22

Branch `feat/agent-benchmark-master-plan`. Full technical report of the main run:
[`agent-protocol-v2-technical-live-2026-10-22.md`](agent-protocol-v2-technical-live-2026-10-22.md).

> **Read every number here as engineering evidence, not science.** All cases are synthetic
> (fictional compounds, invented values, `synthetic_engineering_gold`), graders are LLMs, and no
> SME has audited anything yet. The numbers say whether the machinery and the agent design behave;
> they say nothing about a molecule.

## 1. Summary

1. **The last engineering item of milestone M1 (the research UI) is done.** It adds a passage
   viewer, a case board "what changed / what was kept" view, an audit lineage view and
   browser tests of the four user stories.
2. **First live benchmark of all four agent arms**, on `openai/gpt-6-luna` (OpenCode, OAuth),
   12 technical cases plus a 4-case stability repeat. C2 (generic ReAct) was best on this set;
   D2 (case-based investigator) did not beat it and cost more. Most failures trace to **four
   product/arm defects**, not to the model.
3. **Two new measurements were added to the protocol**:
   - hallucination, as atomic-claim precision;
   - **predictor adjudication**: does the agent know when the predictor is right, and does it
     corroborate or challenge the predictor accordingly?
   Hallucination has preliminary results. The adjudication study was cut short when the
   provider quota ran out and is re-running.
4. **Running the benchmark found and fixed three measurement defects** before anything was
   scored:
   - the grading packet named limitation codes without defining them;
   - the runner recorded every withheld result as an empty reply;
   - the old gold points two evidence spans at the wrong table row (recorded, not yet fixed).

## 2. What was built today

| Item | What it does | Where |
|---|---|---|
| Passage viewer | Opens the exact table row/cell a claim quotes: header, cited cell, comparators as written, the frozen capture; says why a text was not read | `GET /evidence/{id}/passages`, `EvidencePassages.tsx` |
| Case diff | What a turn changed and what it kept, replayed from the case log; frozen artifacts compared by hash | `GET /cases/{id}/runs/{rid}/changes`, `CaseChanges.tsx` |
| Audit lineage | Every draft, each review round's verdicts, what a repair removed, what the reader did not get | `GET /runs/{rid}/lineage`, `StageLineage.tsx` |
| Browser e2e | The four user stories plus a phone/keyboard check, on fixtures typed by the API schema | `frontend/e2e/stories.spec.ts` (15/15 browser tests pass) |
| Hallucination metric | Atomic claims labelled against what the system was given | `evals/investigation/claims.py` |
| Predictor adjudication | Stance vs. verified label, correction / uncaught-error / false-alarm rates, lift over the predictor | `evals/investigation/adjudication.py` |
| Adjudication case set | 12 synthetic cases (see section 5) | `evals/investigation/cases_predictor_adjudication_technical/` |

Commits:

| Commit | Change |
|---|---|
| `b4538cd` | UI read models and components |
| `7fad75f` | Browser stories |
| `f776bbd` | Code legend in grader references |
| `6edddd0` | Withheld-reply capture fix and the recapture tool |
| `427cba6` | Main-run report |
| `8de426f` | Hallucination and adjudication metrics |
| `235ec69` | Claim convention |

## 3. Main benchmark: the four arms on the technical set

**Set-up.**
- **Arms:**
  - A2: a template, no model.
  - B2: a fixed workflow.
  - C2: a generic ReAct agent.
  - D2: the case-based investigator, which reads skills on demand.
- **What every arm shares:** the same model (`openai/gpt-6-luna`), budget (360 s, 20 tool
  calls), release policy (`strict-reviewed-v1`, where a semantic reviewer must clear claims)
  and frozen inputs. Preflight verified all of this.
- **Run:** 12 cases (T1–T6 × toxicity / activity), one trial; 48/48 attempts completed.
- **Grading:** two graders, Claude (not blind) and Codex `gpt-6.1-sol` (blind). Agreement
  44/48, κ = 0.83.

**Task Success** (range across the two graders):

| Arm | All 12 | Scientific (T1–T5) | Stress (T6) |
|---|---|---|---|
| A2 template | 3 | 2/10 | 1/2 |
| B2 fixed workflow | 1–3 | 0–2/10 | 1/2 |
| C2 generic ReAct | **7–9** | 5–7/10 | 2/2 |
| D2 investigator | 6 | 5/10 | 1/2 |

**Comparisons.**
- **D2 vs B2** (paired): +0.25 to +0.42. Not interpretable: the 12 cases form only 2 dependency
  clusters, and B2 has a retrieval defect of its own (F4 below).
- **C2 vs D2:** C2 matches or beats D2 at 60% of D2's tokens and 70% of its latency.
- **By task group, for every arm:** T5 (updates across turns) is 0/2, and D2 is 0/2 on T3
  (conflicts).

**Cost (12 cases).**

| Arm | Median time per turn | Tokens | API-price equivalent |
|---|---|---|---|
| B2 | 21 s | 0.55 M | ≈ $0.04 |
| C2 | 49 s | 1.32 M | ≈ $0.07 |
| D2 | 70 s | 2.21 M | ≈ $0.10 |

**Stability** (4 cases, declared in advance, repeated once):
- A2 and B2 repeated every outcome.
- C2 repeated 3–4 of 4.
- D2 repeated 3 of 4: its T3 failure became a correct, published answer in the repeat. Whether
  an answer gets published varies from run to run.

## 4. Why the arms failed: four defects

| | Defect | Effect | Fix direction |
|---|---|---|---|
| **F1** | The semantic reviewer (the same model) blocks **correct inferential claims**, e.g. "the model's non-blocker call and the measured 0.3 µM disagree", because no single source states the inference. It is also inconsistent: the same judgement worded differently passed. | All 4 of D2's withheld answers; D2 0/2 on T3 | Review relations against the cited set as a whole; add a "derived claim" kind; measure the reviewer's false-block rate |
| **F2** | A value the **researcher supplies cannot be cited** (it is not a stored observation) | T5 = 0/2 for every arm; blocks user story 4 | Store researcher values as user-sourced observations |
| **F3** | The fallback looks at the session's active analysis, not the run's bound subject | It tells the user "run an analysis first" while one is bound | Build the fallback from the run's subject |
| **F4** | A frozen toxicity analysis carries no ligand name, so arm B's relevance check rejects every record | B2 fails every toxicity case that needs evidence, then says the evidence does not exist | Carry ligand names into the binding; re-run B; owner review of B |

## 5. New measurement 1: hallucination (atomic claims)

**Method.**
- **Claims:** each final reply is split into atomic factual claims.
- **Labels:** each claim is labelled against what that system was given (the frozen artifacts
  plus the corpus, printed as `sources.md`), never against the gold:
  - `supported`: the sources state it;
  - `unsupported`: invented;
  - `contradicted`: the sources say otherwise;
  - `not_checkable`: a caveat or recommendation.
- **Replies with no claim** are counted apart, never as 100%.
- **Two graders** are kept separate, and their agreement is measured per reply.

**Preliminary results on the main run.** Grader: Claude only, not blind; the second grader is
pending.

| Arm | Claim precision | Replies with ≥1 false claim | Written by the model | Written by the server |
|---|---|---|---|---|
| A2 template | 100% (108/108) | 0/12 | 0 | 0 |
| B2 fixed workflow | 79% (30/38) | 6/11 | 6 | 2 |
| C2 generic ReAct | 96% (44/46) | 2/12 | 1 | 1 |
| D2 investigator | 96% (44/46) | 2/12 | 2 | 0 |

**Reading.**
- **No invented values, assays or sources.** The hard validator forces every number to cite a
  stored field, so this kind of hallucination cannot pass.
- **What remains is "denying what is there"**: "the material has no prediction / no source / no
  measured IC50" when it does. This is mostly B2, a consequence of F4.
- **Three false claims are the server's fallback text** (F3), not the model.

## 6. New measurement 2: predictor adjudication

**Question.** Can the agent tell, from the evidence it can reach, whether the predictor's call
is right?
- It should **corroborate** a right call.
- It should **challenge** a wrong call, with the evidence that shows it.
- It should say **insufficient** when the corpus cannot decide.

Agreeing by default or contradicting at random both count as failures.

**Case set.** 12 synthetic cases, each with its own compound and sources, so there are 12
dependency clusters.
- **Verified labels** follow declared rules: hERG blocker if patch-clamp IC50 < 10 µM;
  "active" if measured pIC50 ≥ 7.0.
- **Comparators** are kept as written ("> 30").
- **Distractors:** another compound on the same target, the same compound on another target,
  and a mutant.

| Expected stance | hERG call (toxicity) | Activity estimate vs threshold 7.0 |
|---|---|---|
| Corroborate (predictor right) | 2 | 3 |
| Challenge (predictor wrong) | 3 (one decided only by an abstract) | 2 |
| Insufficient (corpus cannot decide) | 1 | 1 |

**Metrics.** Kept separate; no composite score.
- stance accuracy;
- correction rate (wrong calls challenged);
- **uncaught-error rate** (wrong calls corroborated, the critical one);
- false-alarm rate (right calls challenged);
- unsupported corroboration where the corpus cannot decide;
- the agent's final-label accuracy against the predictor's on the same cases (lift), with
  coverage;
- the share of positions that cite the deciding source.

**Status.**
- The first run stopped at 11:07 when the shared ChatGPT/OpenAI quota ran out. It is kept as an
  aborted diagnostic and not scored.
- Only one case finished on all arms. On it, C2 and D2 were both withheld by the reviewer
  before stating the comparison with the measurement. If this repeats, F1 will dominate the
  adjudication results. This is a hypothesis, not a result.
- A fresh study restarts automatically when the quota returns (about 12:12), probing the model
  before every case so an outage cannot burn attempts.

## 7. Process and infrastructure notes (every attempt is reported)

- **First main study aborted** on the 7.8 GB WSL laptop: it ran 3 cases in parallel against
  SQLite deployments alongside the Docker stack. Kept as a diagnostic. The rerun went one case
  at a time, with niced processes and a memory watchdog.
- **The benchmark's OpenCode server grows about 200 MB per run**, so it was restarted between
  cases.
- **One attempt was cut by the operator** at a case boundary; it was recorded with its usage
  and re-attempted.
- **The adjudication study was aborted on provider quota**, as described in section 6.
- **Measurement fixes before scoring**, both version-controlled:
  - grader references now define their codes;
  - withheld replies are recaptured from the product's stored transcript (10 of 45 turns,
    without rerunning any arm).
- **A Codex audit** of the main report found discrepancies; they were verified against the
  data and corrected.

## 8. Next steps

1. Finish the adjudication study, grade it with two graders (one non-Claude), and update the
   reports with the hallucination results from both graders.
2. Fix F3 (small) and F4. Then ask the scientific owner to review arm B before any D-vs-B
   claim.
3. Design the F1 change with the release-policy owner, and add the control "reviewer blocks a
   true claim" to the evaluator.
4. Implement F2: it is the path for user story 4.
5. Re-run with two trials after the fixes. Only then spend SME time on the 24-case scientific
   pilot (real sources, SME gold, human graders).

## 9. Limits of what this shows

- **N is small:** 12 cases per set, one trial.
- **Gold is synthetic** and graders are LLMs: one grader is not blind, and one is from the same
  vendor as the model under test.
- **B2 has a known defect** (F4).
- The numbers establish that the protocol runs end to end, and they locate defects worth fixing.
  They do **not** establish whether D is better than B or C.
