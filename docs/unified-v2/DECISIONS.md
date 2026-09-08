# Unified v2 decisions

- ToxPred is the only quantitative authority; no endpoint fallback or global
  toxicity score exists.
- Provider output ownership is typed and never flattened across models.
- `toxpred-provenance-v2` separates request artifact context from per-sample
  tokenization/truncation facts.
- Calibration exposure and label-policy changes are separate releases.
- Element rules are a chemical support guard, not a learned OOD detector.
- Case state and kernel transitions are durable product state; chain-of-thought
  is neither requested nor stored.
- Semantic capabilities hide closed tool sequences from planners.
- Numeric values, transforms and identifiers are compiled server-side.
- Runtime, model connection and auth mode are separate audit dimensions.
- ClinTox v1 stays unavailable until its exact tokenizer is recovered; a
  retrained replacement must be called v2.
- Codex and DSH stay experimental until authenticated roundtrip, isolation,
  cancellation/recovery and full eval gates pass.
