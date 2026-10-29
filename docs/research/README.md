# The research design

Written for reviewers assessing what this system claims and whether the
evidence supports it. For how to build and run it, see
[../agent-layer/](../agent-layer/).

## The problem

A toxicology question is answered from measurements that were made under
conditions, by predictors with domains of applicability, and from exposures
only the researcher knows. The failures that matter are not fluency failures:

- **Two measurements of one quantity read as a contradiction**, because one is
  in µM and the other in nM, or one is a bound (`> 30 µM`) and the other a
  point value.
- **Two reports of one experiment counted as two independent sources**,
  because a database record restates the paper it came from.
- **A measurement about one target applied to another**, because both
  mention the compound.
- **A corrected input leaving stale conclusions behind**, because nothing
  recorded which conclusion was computed from it.
- **"No evidence found" standing for four different situations**: nothing was
  searched, nothing relevant exists, a provider was down, or only the
  researcher has the input.

Each one produces an answer that reads as confident and is wrong in a way the
reader cannot see. The design question is what mechanism makes those failures
detectable and attributable, rather than how to make the prose better.

## The claim being tested

That a scientific investigator built on three mechanisms — measurement
semantics, dependency-tracked revision, and gap-driven acquisition — answers
these questions better than a generic agent with the same tools, the same
model and the same gates.

**The primary comparison is D vs C**: the domain design against a strong
generic agent. D vs B (a fixed workflow) is secondary, and asks the narrower
question of whether an agent is needed at all. Locking a weak C and declaring D
the winner would produce a number nobody can use.

**Novelty is not claimed as proven.** Truth maintenance, adaptive retrieval and
tool orchestration all have precedents. What may be contributed is the
formalisation for *measurement* evidence, the specific coordination of these
mechanisms, and controlled evidence that they matter. See
[hypotheses.md](hypotheses.md).

## How to check it

| Document | Contains |
|---|---|
| [hypotheses.md](hypotheses.md) | H1–H3: the mechanism, the controls, the metrics, what would falsify each |
| [evaluation.md](evaluation.md) | Arms, case sets, metrics, blinding, and the reading rules |
| [status.md](status.md) | What is implemented, what has been measured, and what is still open |

The measurement reports themselves are kept internally, under `docs/results/`
in the development repository, each naming the commit it ran on.

## What is true today, stated plainly

The mechanisms of H1 and H2 are implemented and run end to end. H3's mechanism
is partial. **No hypothesis has been tested on scientifically meaningful
data.** Everything measured so far used synthetic engineering fixtures with
invented compounds and numbers, one trial per case, graded by language models
rather than subject-matter experts. Those runs are evidence that the machinery
works. They are not evidence about toxicology, and the scorer says so in its
own output.

## The standards this holds itself to

- **A dimension is a measurement; there is no total score.** Precision and
  coverage trade against each other, and one number hides which was given up.
- **Agreeing by default and doubting everything are both failures.** An agent
  that endorses every predictor call catches no errors; one that challenges
  every call catches them all and attacks the correct calls too. They are
  measured separately, because a single score hides both.
- **A withheld answer stays in the denominator.** Not answering is an outcome,
  not an absence of one.
- **Gold is declared, not assumed.** Every case says whether its gold is
  SME-verified or a synthetic engineering fixture.
- **Targets are locked before the test split is seen**, and the results and
  protocols of earlier studies are kept as their own history rather than
  revised after the fact.
