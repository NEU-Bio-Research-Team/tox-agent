# Model card

The served predictor provides hERG and twelve Tox21 assay measurements. The
historical hERG scaffold-split AUROC is 0.837 and Tox21 macro AUROC is 0.759;
hERG ECE is 0.12. These are model-performance measurements, not clinical risk
estimates. Inputs outside the model domain, uncertain OCR, thresholds and
calibration require expert review.

ToxAgent is not a medical device and must not be used as the sole basis for a
medical, safety, regulatory or chemical-handling decision. ClinTox is declared
unavailable because the release lacks its required tokenizer.
