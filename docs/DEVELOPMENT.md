# Development

Customer operation uses Docker Compose only. Developer workflows may use the
service-specific Python environments and frontend tooling, but they are not a
Quick Start. Run focused tests instead of the full suite on constrained hosts:

Run each service from its own directory. `backend/predictor/tests` and
`backend/control/tests` are both packages named `tests`, so collecting them in
one pytest invocation makes the second shadow the first and the run ends in
collection errors (I25) — this is why there is no single command here.

```bash
(cd backend/predictor && python -m pytest tests/unit tests/contract -q)
(cd backend/control && python -m pytest tests -q -m 'not live_predictor and not live_runtime and not live_evidence and not postgres')
(cd backend/ocr && PYTHONPATH=src python -m pytest tests -q)
(cd frontend && npm test)
```

Do not add generated models, `.env`, `.artifacts`, databases, logs or node
modules to commits. Training, benchmark and experimental material belongs in
the internal development history rather than a customer delivery snapshot.
