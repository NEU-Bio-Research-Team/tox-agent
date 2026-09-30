# What the predictor-error runs of 2026-09-30 found

Three runs of the eight `predictor-error-v1` cases, on the same stack, the same
day. Nothing here is a quality grade — the chemistry lab grades the blinded
packet. What follows is process evidence and transcript reading, and every
claim below names where to check it.

| Study | Condition | What it isolates |
|---|---|---|
| `predictor-error-2026-09-30` | Europe PMC returning 503 | Behaviour with the research tool gone |
| `predictor-error-2026-09-30-full` | Research working, 300 s budget (the product default) | Completion rate as shipped |
| `predictor-error-2026-09-30-budget900` | Research working, 900 s budget | Capability when time is not the constraint |

The platform arms (Claude with and without the ToxPred snapshot) answer from
their own knowledge and were unaffected by the outage; their records live in
the first two studies.

## 1. At the default 300 s budget, most investigations do not finish

With the research provider working, `D_toxagent_investigator` completed 2 of 5
cases before the run was cut; three consecutive cases ended with
`turns [0] did not complete`. Raising the budget to 900 s changed that to 7 of
7 completed, 0 errors, on the same cases and the same stack.

Europe PMC was answering in 1.3–2.5 s throughout, so this is not provider
latency. It is the investigation itself needing more steps than the default
allows. Note the direction: **the better retrieval works, the more likely the
run is cut**, because a search that returns records leads to more steps than
one that fails fast. The pilot of 2026-09-25 reported few cap hits, which is
consistent with retrieval having contributed little there.

`TOXAGENT_RUN_DEADLINE_S` / `TOXAGENT_TURN_DEADLINE_S` default to 300
([config.py](../../src/toxagent/platform/config.py)).

## 2. Time was not what stopped it from noticing

At 900 s the runs finish, and the answers are better than at 300 s — but on
the cases built around a predictor error, they still do not reach the finding.
Counting the terms an answer that noticed would be expected to use (a keyword
scan, `signal_scan.py`, explicitly not a grade):

| Case | Claude + snapshot | Claude bare | ToxAgent D (900 s) |
|---|---|---|---|
| sotalol, named | 6 | 3 | 0 |
| sotalol, anonymous | 1 | 1 | 0 |
| moxifloxacin | 4 | 0 | 1 |
| catechin | 3 | 3 | 2 |
| uric acid | 2 | 1 | 0 |
| 4,4′-biphenol | 4 | 2 | 4 |
| succinic acid *(control)* | 1 | 1 | 0 |
| fexofenadine *(control)* | 2 | 2 | 1 |

Totals over the eight cases: Claude with the snapshot 23, Claude bare 13,
ToxAgent 8. ToxAgent scores 0 on the "passed the prediction through" column in
every case, so this is not a system that repeats a wrong prediction
confidently. It is a system that says little of substance either way.

The exception is worth reading: on 4,4′-biphenol it matches Claude. It lists
the ten active assays, says plainly "do not turn the pattern into an aggregate
toxicity score or interpret the number of active assays as severity", and asks
for orthogonal concentration-response testing "with cytotoxicity and
assay-interference controls" — the right mechanism for exactly this kind of
overcall. What it still does not do is say that ten of twelve is implausible.
It handles the pattern correctly without ever doubting it.

## 2b. It knows it is short of evidence

The process report for the 900 s study records the investigator's own stop
reasons over the eight cases: `insufficient_evidence` 6, `sufficient` 1,
`budget_exhausted` 1. So in six of eight cases the system concluded, correctly,
that it did not have the evidence it needed — and answered anyway, with the
shortfall stated in the limitations. The self-assessment is right; what is
missing is any path from "I do not have the evidence" to getting it.

Median 188 s per turn and 19 tool calls per turn, against Claude's 34-37 s.
The work is being done; it is not landing on the compound.

## 3. What the transcripts show instead

**Method discipline is good.** On uric acid it states that the twelve Tox21
assays are independent signals and "not an aggregate toxicity score", asks for
orthogonal concentration-response confirmation with viability and
chemical-interference controls, and refuses to clear the compound from a
screen. That is the right shape of answer.

**Compound knowledge is absent.** The same answer treats an endogenous purine
end product, present in human plasma at hundreds of µM, as an unknown material
whose "intended route, dose, duration, formulation" must be established. On
sotalol it never reaches the fact that the compound's therapeutic mechanism is
IKr/hERG blockade. On moxifloxacin it never reaches the fact that the compound
is the standard ICH E14 positive control, which is what the question hinges on.

**The retrieval brings back the neighbourhood, not the compound.** For
moxifloxacin, 11 records were accepted — and they are thorough-QT studies of
*other* drugs (Milvexian, Givinostat, CIN-102, Ribitol …), exactly the studies
in which moxifloxacin serves as the positive control. The relevant fact was
adjacent to what was retrieved and was not extracted. The answer then states
that the search "did not provide accepted compound-specific independent
confirmation" although 11 records had been accepted — worth a closer look at
how acceptance is reported to the drafting step.

For sotalol at 300 s, 25 records came back and all 25 were rejected as
`compound_mismatch`; none of the 25 contained the string "sotalol". The query
the agent built was `sotalol hERG blocker electrophysiology` — four terms that
Europe PMC ANDs together, narrowing to 219 records whose ranking favours broad
reviews. `sotalol AND hERG` returns 664 and puts *The Methanesulfonamide
Group: Bright and Dark Sides of hERG Potassium Channel Inhibition* on the first
page — sotalol's own chemical class.
[europepmc.py:96](../../src/toxagent/research/providers/europepmc.py) passes the
query through unchanged, so query construction is entirely the model's, with
nothing holding the compound as a required term.

## 4. Losing the tool lowered what it could verify, not what it concluded

In the outage study, `inv-11` stated plainly that the search had failed and
that no external confirmation was established — and then called the
below-threshold score "a reassuring model signal" and advised keeping sotalol
as a comparator. With retrieval working, the same case became "unresolved, not
dismissed". The honesty about the failure was there in both; the conclusion
only moderated when the tool worked.

## 5. A result that needs more cases before it means anything

On sotalol the **anonymous** case produced the more cautious answer ("hold a
development decision") than the **named** one. If that survives more cases and
the lab's grading, it would suggest a name can produce unearned familiarity.
One pair is not evidence.

## What to look at, in order

1. **Query construction** — hold the compound name (or an identifier) as a
   required term, and prefer fewer, better-chosen terms over four ANDed ones.
   This is the cheapest change and it gates everything downstream: without the
   right records, no amount of reasoning or time recovers the finding.
2. **The default time budget** — 300 s does not complete an investigation that
   actually retrieves. Either raise it or make the run degrade to a partial
   answer rather than an error.
3. **Extraction from adjacent records** — a TQT study of another drug names its
   positive control in the methods. Something has to read that.
4. **How acceptance is reported to drafting** — one answer reported no accepted
   evidence while holding 11 accepted records.
5. **What happens after `insufficient_evidence`** — six of eight runs ended
   there. A run that knows it is short could re-query differently instead of
   drafting around the gap.

None of this is the lab's verdict, and none of it should be quoted as one.
