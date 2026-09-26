---
name: critique-case
description: Use when your answer will recommend a course of action for the compound — go/no-go, prioritisation, or which study to run next. A self-review of the case before concluding, for claims without sources, model signals used as evidence, indirect sources treated as direct, conflicts left unrepresented, missing counter-evidence and conclusions wider than the evidence. Not for questions that only ask what the model predicts, why it scored a compound, or whether a prediction can be relied on.
---

# Critique the case before concluding

Review the case as a sceptical colleague would, before you commit to a
conclusion. Its checkpoint is in your context; `get_scientific_case` returns
the full ledger if an entry you need is not shown. The aim is a conclusion no
wider than its evidence, not a longer answer.

## When this applies

- The question asks for a decision or a recommendation: stop or continue,
  prioritise or not, which test to run next.
- New evidence arrived this turn that could change an earlier recommendation.

Skip it when the answer recommends nothing: a lookup ("what is the predicted
probability?"), an explanation of the model ("which atoms drove the score?"),
or a question about how far a prediction can be trusted. Those need their own
limits stated, not a review of a conclusion.

## The review

For each item, the fix is in the case, not in softer wording.

1. **Every "can say" line traces to the ledger.** A line with only a
   predictor or explanation fact behind it says something about the model, and
   must be worded so. A line with nothing behind it moves to `cannot_say` or
   becomes a hypothesis.
2. **Independent evidence.** Which hypotheses rest only on model signals?
   Coverage reports `with_independent_direct_evidence`; if it is zero for the
   hypothesis your conclusion depends on, the conclusion is conditional and
   must say on what.
3. **Directness.** Are analogue, class-level or mechanistic entries doing the
   work of direct ones? Re-record them honestly if not.
4. **Counter-evidence.** Did anyone look for evidence against the leading
   hypothesis (`with_counterevidence_considered`)? If not, and budget remains,
   look now; if you cannot, say the search was not done.
5. **Conflicts.** Is every disagreement between sources recorded, with scope?
   (Load `assess-conflicting-evidence` if it is not straightforward.)
6. **Scope of the claim.** Does the conclusion stay within the endpoints,
   assays and exposure the evidence covers? No safety, regulatory or clinical
   verdict; no aggregate risk.
7. **Open uncertainties.** Is any `blocking` uncertainty still open? Then the
   conclusion is "cannot decide until …", and the answer names what is needed.
8. **What would change it.** Is there at least one concrete observation that
   would change the conclusion? If a test would supply it, is it proposed, and
   does it discriminate between hypotheses?

## What to leave in the case

All of it in the turn's one `update_scientific_case` call:

- Corrections: `revise_hypothesis`, re-recorded evidence, new uncertainties.
- A `record_action` with action `critique` and what the review changed.
- Then `set_conclusion` with `can_say` lines citing ledger ids, `cannot_say`,
  and `what_would_change`.
