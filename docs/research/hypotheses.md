# H1–H3

Written for reviewers assessing what this system claims.

Each hypothesis names a mechanism, the controls it is varied against, the
metrics it is judged on, and what would falsify it. In every case the arms
share the model, the tools, the budget, the output schema, the validator, the
reviewer and the frozen inputs; only the mechanism under study differs.

The common identity guards — a measurement about another target is not direct
evidence about this one — stay on for **every** arm. Removing them would
produce a baseline that fails for a reason unrelated to the hypothesis.

---

## H1 — Measurement semantics

**Claim.** Separating identity, representation and numeric constraint reduces
false contradictions and the misapplication of evidence.

**Mechanism.** A declared value becomes the set of values it permits, in
canonical units, with the conversion recorded. Three questions that were one
string comparison become three:

| Layer | Question | Example of the failure it prevents |
|---|---|---|
| Identity | Are these the same measurement? | A hERG IC50 cited against an EGFR estimate |
| Representation | Is this the same number written differently? | 0.3 µM and 300 nM read as two different results |
| Numeric | Do these two numbers agree? | `> 30 µM` and `< 10 µM` read as a contradiction, or as agreement |

Two deliberate refusals. An unknown unit is unknown, not a wildcard: two
unrecognised unit strings that happen to match are still not interconvertible.
And two disjoint values are **not** a contradiction unless the caller declares
the fold difference that counts as a real one, because how far apart two
measurements of one quantity may legitimately sit is a property of the assay,
not of the arithmetic.

**Controls.** Paired cases that hold the measurement fixed and vary how it is
*written* or *blocked*: unit equivalence (nM/µM), exact vs censored, same
ligand on a different target or variant, missing assay context, and duplicate
records of one experimental origin. Then an ablation that removes the
representation layer while keeping tools and knowledge identical.

**Metrics.** False-contradiction rate; correct refusals to compare; how often
an undetermined comparison is reported as undetermined rather than resolved to
one side.

**What would falsify it.** The ablated arm matching the full arm on paired
cases — the layer would then be adding cost without changing conclusions. Or
the full arm refusing comparisons that are in fact valid, trading false
contradictions for false incomparability.

**"Undetermined" is a correct answer** on the cases built for it, and the gold
says so. An arm that always commits is not thereby better.

---

## H2 — Revision with dependencies

**Claim.** Structured dependencies update the right claims, preserve the
unrelated ones, and cost less than re-synthesising from the transcript.

**Mechanism.** A derived claim records its inputs as references, its rule and
its assumptions. When a premise moves, the graph is walked transitively, and a
claim is marked for reassessment only when *every* justification lost a
premise — one standing route means the claim itself is not in doubt.

Nothing is deleted. The earlier answer was computed from those inputs, and the
next answer has to be able to say what changed.

This is scoped truth maintenance, not a full TMS: the scope is what a server
can check — numeric and identity rules. A dependency cycle is reported as a
finding, not raised as an error, because a graph that is not sound still has to
produce an answer.

**Controls.** Generic ReAct with the full transcript; ReAct with a good
summary; the previous investigator without dependencies; the current one. Same
observations, same guidance content.

**Metrics.** Required-update accuracy (did the claims that had to change,
change); preservation accuracy (did the unrelated ones survive); stale-claim
rate; task success; total cost.

**What would falsify it.** Transcript-based ReAct matching it on both update
and preservation. The interesting failure is asymmetric: marking everything
scores perfectly on required-update and fails preservation, and the two must
therefore be reported together.

**The case gold must include the hard case**: a claim whose premise changed but
which has a second, still-standing justification. An implementation that marks
it is wrong, and one that silently keeps it without recording the lost route is
also wrong.

---

## H3 — Gap-driven acquisition

**Claim.** Explicit information needs spend the budget on sources that could
change the answer.

**Mechanism.** Seven typed gaps, and an action that must state which gap it
closes. A run that stops says which gap it stopped on, so "insufficient" is
attributable rather than a shrug.

**Controls.** The adaptive policy against a fixed acquisition schedule, over
the same case and the same procedures, with evidence-given separated from
frozen retrieval.

**Metrics.** Acquisition cost; decisive-evidence recall; task success.
Explicitly **not** the number of papers read, which rewards reading more.

**What would falsify it.** A fixed schedule reaching the same decisive evidence
at the same or lower cost. On a small corpus this is likely, and that is worth
reporting rather than hiding: the mechanism's value should scale with how much
there is to not read.

**Status: the weakest of the three.** The gap vocabulary exists and the
contract declares the gap for one calculation. Ranking candidate actions by
what they would discriminate is not implemented, so H3 cannot currently be
tested as stated.

---

## Engineering controls, not contributions

Context compilation, batching, caching and durable execution are measured for
operational effect only. They are not scientific reasoning results. If one of
them is turned on, the baseline arms get it too — otherwise the comparison
measures the optimisation rather than the design.
