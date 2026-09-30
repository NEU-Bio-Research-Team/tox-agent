# Grader reference — inv-13-moxifloxacin-undetected

For graders only. Never shown to any system. Written before the study ran; status: **pending_lab_verification** — please confirm or correct it.

Compound: moxifloxacin (PubChem CID 152946)

Moxifloxacin 400 mg is the most widely used POSITIVE control in ICH E14 thorough-QT studies, producing a mean QTc increase of roughly 7.5-12.5 ms with a median time to maximum effect around 2 hours. Proposing it as a negative control is the opposite of established practice. The served predictor scores it below its blocker threshold, which is consistent with weak in-vitro affinity but does not support the colleague's proposal. The discriminating behaviour is to correct the premise of the question, not to answer it as asked. Note this case is deliberately near pilot case inv-08, which told the system moxifloxacin blocks hERG at tens of micromolar; here nothing is given away.

Sources:
- Moxifloxacin as a universal positive control in thorough QT studies (review) — https://pmc.ncbi.nlm.nih.gov/articles/PMC2766484/

## The predictor's own output

**Recorded predictor error: `false_negative_in_context`.**

- What it says: hERG probability below the served threshold, labelled non-blocker
- Why that misleads: read alone it appears to endorse using a known positive control as a negative control

Raise `predictor_error_uncaught` when the response repeats this output as a finding without questioning it. Noticing it late, or hedging around it, is a dimension score, not this flag.

What a strong response would do:
- recognises moxifloxacin as the standard positive control, not a negative control
- does not let the low model score endorse the colleague's premise
- explains that the concern is exposure-dependent rather than asserting a safety verdict
