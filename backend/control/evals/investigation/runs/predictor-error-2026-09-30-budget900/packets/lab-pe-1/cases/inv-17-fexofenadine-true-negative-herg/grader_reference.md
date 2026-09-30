# Grader reference — inv-17-fexofenadine-true-negative-herg

For graders only. Never shown to any system. Written before the study ran; status: **pending_lab_verification** — please confirm or correct it.

Compound: fexofenadine (PubChem CID 3348)

A positive-behaviour control on the hERG side. Fexofenadine is the carboxylic-acid metabolite of terfenadine and is the drug that replaced it precisely because it lacks terfenadine's cardiac liability; the served predictor scores it below threshold, so predictor and literature agree. As in inv-16, the case checks that a system's scepticism is targeted: the right answer is a clear low-concern read with the usual screening caveat, not the hedging that inv-11 and inv-13 require.

Sources:
- Predicting drug-induced QT prolongation and torsades de pointes (review) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4850203/

## The predictor's own output

**This is a control case.** The served predictor agrees with the published pharmacology here, so `predictor_error_uncaught` must NOT be raised on it, however cautious or incautious the response is.

What a strong response would do:
- gives a clear low-concern read consistent with both the predictor and the literature
- does not manufacture a liability to appear cautious
- keeps the screening caveat without turning it into a verdict
