# Development

Customer operation uses Docker Compose only. Developer workflows may use the
service-specific Python environments and frontend tooling, but they are not a
Quick Start. Run focused tests instead of the full suite on constrained hosts:

```bash
python -m pytest tests/unit -q
(cd toxagent-control && python -m pytest tests -q -m 'not live_predictor and not live_runtime and not live_evidence')
(cd frontend && npm test -- --run)
```

Do not add generated models, `.env`, `.artifacts`, databases, logs or node
modules to commits. Training, benchmark and experimental material belongs in
the internal development history rather than a customer delivery snapshot.
