# Architecture

`frontend` is the sole published service. It proxies browser `/v1` traffic to
the control plane in `backend/control`; the control plane owns workflow,
authentication and data, and calls the private `backend/predictor` (`toxpred`)
and `backend/ocr` (`toxocr`) services. PostgreSQL is private and persists
product state.

The default Compose stack is **predictor-only**: prediction and OCR are
self-contained and no agent runtime is started. Session intents that need an
agent (research, report, attribution) are unavailable in that mode and say so
— see [`OPERATIONS.md`](OPERATIONS.md) for the agent-enabled stack.

The predictor verifies the artifact declarations in
`backend/predictor/registry/predictor-manifest.yaml`, overridable with
`TOXPRED_MANIFEST`. OCR weights are mounted read-only and verified before
MolScribe loads them. No request triggers model downloading.
