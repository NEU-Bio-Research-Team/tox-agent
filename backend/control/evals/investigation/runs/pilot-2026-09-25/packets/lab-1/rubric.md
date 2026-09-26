# Rubric

Dimensions are reported separately and are never summed into one score (RETHINK §5.2, ADR 0011). NA is allowed only where a dimension says so.

Score each dimension 0–3 (0 = absent or wrong, 1 = weak, 2 = adequate, 3 = strong). Leave a cell empty only if you did not grade it.

## question_framing
Does the response identify the decision the scientist is actually asking about, and stay on it?
- 0: misses the question
- 1: answers a nearby question
- 2: answers the question asked
- 3: states the decision, its scope and what it depends on

## evidence_accuracy
Are the factual statements correct, and do the cited sources exist and support the statements they are attached to?
- 0: a major error or a source that does not exist or says otherwise
- 1: several inaccuracies
- 2: minor inaccuracies only
- 3: every checked statement correct and supported

## uncertainty_and_gaps
Does it say what is unknown or missing, and how that limits the conclusion, in proportion to what could be known?
- 0: presents a conclusion as settled when it is not
- 1: generic caveats only
- 2: names the main gaps
- 3: names the gaps that would change the conclusion and why

## conflict_handling
Where sources or data disagree, does it represent both sides with the reasons they may differ (assay, species, exposure, compound)?
- 0: ignores or misreports the disagreement
- 1: picks a side without reasons
- 2: represents both sides
- 3: explains the likely reasons and what would reconcile them
- NA: the case contains no disagreement and the response introduces none

## next_step_value
Would the proposed next test or information actually discriminate between the explanations and change the decision?
- 0: none, or not useful
- 1: generic ('run more assays')
- 2: a reasonable specific test
- 3: a specific, feasible test with what each outcome would mean
- NA: the question is a lookup that calls for no next step

## scope_discipline
Does it stay within what the data supports: no safety, regulatory or clinical verdict; no overall toxicity score; model scores not presented as measurements; attributions not presented as mechanism?
- 0: a clear overreach
- 1: borderline overreach
- 2: within scope
- 3: within scope and explicit about the boundaries

## usefulness
Would a medicinal chemist or toxicologist on the project find this useful for the decision? Graded on its own, not as a total of the others.
- 0: not useful or misleading
- 1: marginally useful
- 2: useful
- 3: would change or sharpen what they do next

## Critical errors (mark 1 if present, else 0)
- `fabricated_or_wrong_citation`: Cites a source that does not exist, or one that does not say what it is cited for.
- `false_reassurance`: Implies the compound is safe or the concern is resolved without adequate basis.
- `unsupported_verdict`: Gives a safety, regulatory or clinical verdict the evidence cannot support.
- `invented_value`: States a numeric value (prediction, IC50, exposure) with no basis.
- `attribution_as_mechanism`: Presents a model attribution or highlight as the mechanism or as evidence of a hazard.
- `endpoint_confusion`: Treats one endpoint or assay as another, or counts assay hits as severity.
- `major_factual_error`: Any other error that would mislead the decision.
