# Evaluation design

Written for reviewers assessing what this system claims. For the commands, see
[../agent-layer/benchmark.md](../agent-layer/benchmark.md).

## The comparison

Five arms, each a separate deployment of the same control plane, differing only
in declared flags. B, C, D and E share the model, the budget, the output
schema, the hard validator, the semantic reviewer, the release policy and the
frozen inputs.

| Arm | What it is | Role |
|---|---|---|
| A2 | Code renders the frozen record through a fixed template; no model | Floor |
| B2 | Fixed workflow: the server reads, retrieves and cross-checks by rule; one synthesis turn writes | Is an agent needed? |
| C2 | Generic agent choosing its own actions, same tools and gates, no case, no skills | **Primary control** |
| D2 | Persistent case, scope-aware evidence relations, skills on demand | Treatment |
| E2 | D2 plus the goal contract, the four routes, the case view, procedures | Treatment |

**Primary: D (or E) vs C.** A generic agent with the same tools is the
comparison that tests the domain design. D vs B is secondary.

The weakness here is explicit: on the one live run, C2 did at least as well as
D2 at 60% of the tokens, and B2 could not retrieve at all on toxicity-only
cases because a frozen analysis carries no ligand name — so the D−B delta
overstated D. Both are reported in the 2026-10-22 protocol v2 live report (kept with the
other internal measurement reports). Raising C to a tool-matched generic agent with code execution is open work.

## Fairness is enforced, not asserted

Before a study runs, each deployment is asked what it is
(`/v1/system/effective-product`), and the group is refused when a shared field
differs, a mandatory pin is missing, or **a flag differs between two arms
without being declared a treatment**. An undeclared variable is a confound.

This caught a real error: one arm's flag brought a live ChEMBL provider, and
the "frozen" benchmark reached the network — the lookup then recognised a
case's invented compound as a real drug. The live activity lookup is now a
shared field.

## Case sets

| Set | What it is | What its numbers mean |
|---|---|---|
| `cases_protocol_v2_technical` | 12 synthetic cases, invented compounds and sources, gold marked `synthetic_engineering_gold` | **Machinery only.** Never evidence about a molecule |
| `cases_predictor_error` | The predictor is wrong; does the system notice? | Adjudication behaviour |
| Pilot (24 cases, two cohorts) | Real sources | **Needs SME-verified gold before any number means anything** |

A v3 case keeps its question, its frozen inputs and its gold in **separate
files**, each hash-pinned, and an audit holds them apart — so a run cannot be
given the answer with the question. The goal contract has no parameter through
which gold could reach the model, and a unit test asserts the parameter set.

Cases are grouped into **dependency clusters** (compound, document,
measurement, source). The technical set's 12 cases are two clusters, so a
cluster bootstrap over it has two units and its interval cannot be read. This
is a property of the set, not a result, and the card reports the cluster count.

## What is measured

**Task success**, per assigned case and trial, with a multi-turn case counting
as one unit, reported by task group.

**Paired deltas** with a cluster bootstrap, for the comparisons locked in
advance.

**Mandatory information and limitations.** Did the answer state the facts the
gold requires, and declare the limitations its sources require. A gap statement
can be the correct conclusion: "a margin needs the free Cmax, which this
session does not hold" is a pass when the gold says so.

**Critical errors**, counted separately and never averaged away:
fabricated or wrong citation, false reassurance, unsupported verdict, invented
value, attribution presented as mechanism, endpoint confusion, an uncaught
predictor error, major factual error.

**Hallucination as atomic-claim precision**: supported ÷ (supported +
unsupported + contradicted), with `not_checkable` claims excluded and replies
that make no checkable claim counted apart — never as 100%. A false claim is
attributed to the model's own text or to server fallback text, because those
are different defects.

**Predictor adjudication**, the narrower question that justifies an
investigating agent on top of a predictor: stance accuracy, correction rate,
**uncaught-error rate**, false-alarm rate, unsupported corroboration, lift with
coverage, and grounded share. Agreeing by default and contradicting at random
both count as failures, which is why these are not one number.

**Cost and routing**: tokens, cost and latency per arm and per phase; routes
per arm with their median turn time and tool calls. Routing *errors* are
reported as "not measured", because no gold file states the route a turn should
have taken.

## Grading

Two graders, kept separate, with their agreement measured per reply. The packet
is blinded; the unblinding key is git-ignored, and is reproducible from the
study, the systems and the seed rather than being a single point of loss.

Known limitation on every number so far: the graders have been language models,
one of them operated by the same person who ran the study and therefore **not
blind**, and in one case from the same vendor as the system under test. A
chemistry lab grading the pilot is the step that has not happened.

## Reading rules

- **No total score.** Each dimension is its own measurement.
- **A withheld answer stays in every denominator**, with stance `none`. Not
  answering is an outcome.
- **A2's 100% claim precision is true by construction** — it has no model and
  only re-renders what it was given. It validates the measure; it is not an
  achievement.
- **An "undetermined" final label counts as wrong in lift**, with coverage
  reported beside it, so an arm cannot win by never committing.
- **A measurement defect found mid-study is fixed and reported**, and the
  affected results are re-derived from the recorded transcripts rather than
  quietly rerun. Two such defects are recorded in the briefing.
- **Scripted eval packs are CI evidence, not release evidence.** The runner
  labels them.
