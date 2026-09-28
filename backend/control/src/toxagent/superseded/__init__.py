"""The scientific agent kernel ADR 0011 superseded, with its answer compiler, capability
registry and budget.

Not a live path: decision support runs through harness/gateway.py with
domain/decision_state.py. Nothing live imports this package (enforced by
tests/unit/test_eval_paired.py); it stays until the Wave 4 paired benchmark
and a separate retirement of its tables, then it is deleted as a whole.
"""
