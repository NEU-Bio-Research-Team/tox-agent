# Status

Written for reviewers assessing what this system claims. As of 2026-10-30, on
`main` at `e5275df`.

## Summary

The agent layer runs end to end: a question is routed, given a contract and a
case view, investigated with typed tools, validated field by field, reviewed
claim by claim, and released or withheld with a recorded reason. The mechanisms
of H1 and H2 exist and work in a live run. **No hypothesis has been tested on
scientifically meaningful data**, and the semantic reviewer still withholds
correct answers often enough to be the main obstacle to measuring anything.

## Mechanisms

| Mechanism | State | Note |
|---|---|---|
| Measurement semantics (H1) | Implemented | Three layers separated; unit and comparator no longer read as identity differences |
| Researcher values as citable versions | Implemented | A value the researcher states is citable, versioned, and declared unverified |
| Claim dependencies (H2) | Implemented | Transitive; a second standing justification keeps the claim; nothing deleted. Currently reachable through one calculation (the exposure margin), because that is the only one the server performs |
| Typed stops and gaps | Implemented | Eight stop reasons, seven gaps; provider failure separated from insufficient evidence |
| Goal contract, case view | Implemented | Contract cannot be handed gold; view carries refs and counts what it withheld |
| Four serving routes | Implemented | The route selects the capability profile, so a read can do less |
| Scientific procedures | Implemented | Five procedures behind one allowlisted tool |
| Verification per claim type (§9.1) | **Partial** | The reviewer is shown a computed value's formula and inputs; verifying a derived number by recomputation instead of by the reviewer is not done |
| Gap-driven action selection (H3) | **Partial** | Vocabulary and one calculation's requirements exist; ranking actions by what they would discriminate does not |
| Workers, caching, model routing, durable resume | Not implemented | Deliberately deferred until traces show the benefit |

## What has been measured

**2026-10-22, four arms, 12 synthetic technical cases, live on
`openai/gpt-6-luna`.** Reported in full in the protocol v2 live report and the
benchmark briefing of that date, kept internally under `docs/results/`.

- 48/48 primary attempts completed without infrastructure error.
- Task success (two graders): A2 3, B2 1–3, C2 7–9, D2 6, of 12.
- **C2 matched or beat D2** at ~60% of the tokens. On this set, the domain
  design showed no advantage.
- Four product or arm defects, not model properties, accounted for most
  failures: the reviewer blocking correct inferential claims; a
  researcher-supplied value being uncitable (T5 was 0/2 for every arm); the
  fallback reading the session's active analysis instead of the run's subject;
  and arm B being unable to retrieve on toxicity-only cases.
- Two *measurement* defects were found by running the study and fixed before
  scoring.

**2026-10-29/30, arm E, same 12 cases, several runs.** Not a graded study; a
check that the new layer runs and a source of defects. Recorded here because
the fixes it produced are in `main`, not because its outcomes are results.

- No infrastructure errors in the final run; no live provider calls.
- The four defects above are fixed. T5 went from 0/2 for every arm to a
  published margin of 30 computed from the researcher's exposure and recorded
  as a derived claim.
- Routes were taken as intended; the read route used 4–6 tool calls against a
  median of 10 for investigate.
- Eight defects were found only by running it live, including two that would
  have invalidated a study: an arm reaching live ChEMBL on a frozen case set,
  and a refused submission recording no reason.

**All of this used invented compounds and numbers, one trial per case, and
language-model graders — one of them not blind.** It is evidence about
machinery.

## Open, in the order it matters

1. **The semantic reviewer withholds correct claims.** It is the main cause of
   withheld answers, and it limits every measurement taken through the release
   gate. The remaining work is verifying derived numbers by recomputation
   rather than by judgement, and measuring the reviewer's own false-accept and
   false-block rates.
2. **SME-verified gold for the pilot.** Until it exists, no number is about
   toxicology.
3. **A tool-matched generic control.** C2 must be raised to a generic agent
   with code execution, dev-tuned to parity, with a scientific owner
   confirming it — the same standard B was held to.
4. **H3 cannot be tested as stated** until action selection names the gap it
   discriminates.
5. **H1 and H2 need their case sets.** The paired axes and the four controls
   are declared; the cases, with gold that includes "undetermined is correct"
   and the second-justification case, are not written.
6. **Routing errors are unmeasured**: no gold states the route a turn should
   take.
7. **Procedures were never chosen by the model** in any live run. A tool the
   model does not reach for cannot help, and the reason is not yet known.
8. **Statistical power.** The technical set is two dependency clusters. A real
   comparison needs enough clusters for its interval to mean something.

## What would change the conclusion

If, with SME gold and a tool-matched C, D or E does not beat C on the paired
cases H1 and H2 are built for, then the domain design does not pay for itself
at this scale, and that is the finding. The design is written so that it can
produce that answer: the mechanisms are flagged, the arms are
configuration-only, and the ablations are declared in advance.
