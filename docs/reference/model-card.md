# Model card

The served predictor provides hERG and twelve Tox21 assay measurements. The
historical hERG scaffold-split AUROC is 0.837 and Tox21 macro AUROC is 0.759;
hERG ECE is 0.12. These are model-performance measurements, not clinical risk
estimates. Inputs outside the model domain, uncertain OCR, thresholds and
calibration require expert review.

Atom and token highlights come from the served model's `grad_x_input_v2`
attribution. It is deterministic and does not change with how a SMILES is
written, but on the 42-molecule golden panel of 2026-09-09 it was **not
measurably more faithful than deleting random atoms** (hERG 15/34, Tox21 NR-AR
14/35 wins; [XAI benchmark](../../backend/predictor/evals/xai/README.md)). A
highlight is what the model's gradient points at, never a mechanism or evidence
of a hazard; every attribution the agent reads carries this verdict, and other
targets or methods are stated as not measured
([capability matrix](capability-matrix.md)).

ToxAgent is not a medical device and must not be used as the sole basis for a
medical, safety, regulatory or chemical-handling decision. ClinTox is declared
unavailable because the release lacks its required tokenizer.
