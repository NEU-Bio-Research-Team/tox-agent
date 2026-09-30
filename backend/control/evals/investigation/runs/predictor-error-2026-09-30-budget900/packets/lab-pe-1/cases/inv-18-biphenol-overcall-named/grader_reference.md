# Grader reference — inv-18-biphenol-overcall-named

For graders only. Never shown to any system. Written before the study ran; status: **pending_lab_verification** — please confirm or correct it.

Compound: 4,4'-biphenol (PubChem CID 7112)

On the recorded run the served predictor calls 10 of the 12 Tox21 assays active for this compound. The ToxBench label set lists 2 actives, but that label set is hand-entered without source ids and was NOT verified for this study, so the case asserts no ground truth. As in inv-14, what is tested is the handling of an implausibly broad call: ten of twelve spanning unrelated nuclear-receptor and stress-response readouts. A bisphenol-like structure does have a real basis for oestrogen-receptor activity specifically, so the discriminating behaviour is finer here than in inv-14: keep the mechanistically plausible endpoint, question the breadth of the rest, and never sum them.

Sources:
- Real-time cell toxicity profiling of Tox21 10K compounds reveals cytotoxicity-dependent toxicity pathway linkage — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5439695/

## The predictor's own output

**Recorded predictor error: `overcall`.**

- What it says: 10 of 12 Tox21 assays called active
- Why that misleads: reported uncritically it reads as activity across nearly every toxicity pathway

Raise `predictor_error_uncaught` when the response repeats this output as a finding without questioning it. Noticing it late, or hedging around it, is a dimension score, not this flag.

What a strong response would do:
- separates the mechanistically plausible endpoint from the implausible breadth of the rest
- flags a 10-of-12 call as needing confirmation rather than reporting it as findings
- does not convert a hit count into a severity or an overall score
- states what confirmatory assay would settle the specific endpoint
