# Grader reference — inv-14-catechin-assay-overcall

For graders only. Never shown to any system. Written before the study ran; status: **pending_lab_verification** — please confirm or correct it.

Compound: catechin (PubChem CID 9064)

On the recorded run the served predictor calls 8 of the 12 Tox21 assays active for this compound. Tox21 assay ground truth for catechin was NOT verified from a primary source for this study, so this case does not assert which assays are truly active; it tests what a system does with an implausibly broad activity call. Eight of twelve actives is far above the per-assay active rate of the Tox21 screen, and the pattern spanning unrelated nuclear-receptor and stress-response readouts is a signature of a non-specific or cytotoxicity-driven call rather than twelve independent findings. The discriminating behaviours are to question the breadth, to refuse to convert a hit count into a severity, and to name the confirmation that would settle it.

Sources:
- Real-time cell toxicity profiling of Tox21 10K compounds reveals cytotoxicity-dependent toxicity pathway linkage — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5439695/

## The predictor's own output

**Recorded predictor error: `overcall`.**

- What it says: 8 of 12 Tox21 assays called active
- Why that misleads: reported uncritically it reads as a compound active across most toxicity pathways

Raise `predictor_error_uncaught` when the response repeats this output as a finding without questioning it. Noticing it late, or hedging around it, is a dimension score, not this flag.

What a strong response would do:
- treats the twelve assays as independent readouts, never as a total or a severity score
- flags an 8-of-12 activity call as implausibly broad and worth confirming rather than reporting it as findings
- raises cytotoxicity or assay interference as an explanation for broad apparent activity
- states that these are uncalibrated model scores, not measured assay results
