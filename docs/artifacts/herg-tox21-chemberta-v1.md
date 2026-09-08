# Artifact card — herg-tox21-chemberta-v1

- Capabilities: hERG blockade and twelve independent Tox21 assay activities.
- Weights: `models/pretrained_2head_herg_chemberta_model/best_model.pt`, SHA-256
  `c851e81541f8975f66589879ba9bd35c3068c3fbd57417bb7939214183f62690`.
- Tokenizer: vendored and hash-pinned in `artifacts/predictor-manifest.yaml`.
- Backbone configuration: vendored; serving performs no model download.
- Policy: hERG and per-assay thresholds are artifact-owned policy-v1 values.
- Calibration: no separately admitted calibration artifact; served must not
  describe raw probabilities as calibrated.
- Applicability: `element_rules_v1` is a chemical support guard, not proof of
  distributional membership. ECFP/embedding AD remains release-gated.
- Uncertainty: no admitted conformal artifact in v1.
- Attribution: logit-targeted signed gradient × input and integrated gradients
  are available under the XAI interface; neither establishes causality.
- Unsupported: clinical safety decisions, incidence estimates, dose-specific
  risk, aggregate toxicity scores and ClinTox substitution.

The benchmark report and immutable split are under `benchmarks/results/` and
`benchmarks/manifests/eval-split-v1.json`.
