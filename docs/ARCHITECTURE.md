# Architecture

`frontend` is the sole published service. It proxies browser `/v1` traffic to
`toxagent-control`; the control plane owns workflow, authentication and data,
and calls private `toxpred` and `toxocr` services. PostgreSQL is private and
persists product state. The default deterministic runtime makes the prediction
and OCR path self-contained; external OpenCode is an optional overlay.

The predictor verifies the artifact declarations in
`artifacts/predictor-manifest.yaml`. OCR weights are mounted read-only and
verified before MolScribe loads them. No request triggers model downloading.
