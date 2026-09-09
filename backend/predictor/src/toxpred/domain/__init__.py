"""Pure domain layer: value objects and policy. Standard library only.

No module in this package may import FastAPI, torch, RDKit, transformers,
Google/Firebase SDKs or anything under `agents`, `services` or `model_server`.
`tests/unit/test_import_boundaries.py` enforces this.
"""
