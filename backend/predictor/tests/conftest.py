"""Shared handling of the tests that need real model weights.

Weights live outside Git, so on a developer machine that has provisioned them
these tests run and on a bare CI runner they cannot. That is a legitimate
skip — but only when it is declared, named and countable. Three tests in
`tests/unit` needed artifacts without saying so and simply failed on a runner
that had none, which is indistinguishable from a real regression.

Two rules, matching the golden suite's:

- `@pytest.mark.needs_artifacts` skips with the missing path in the reason.
- `TOXPRED_REQUIRE_ARTIFACTS=1` turns that skip into a failure. CI's
  artifact-provisioning job sets it, so a job that was *supposed* to have
  weights cannot pass by skipping everything that would have used them.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

#: backend/predictor/tests/conftest.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = Path(os.getenv("MODELS_ROOT") or (REPO_ROOT / ".data" / "models"))


def artifacts_present() -> bool:
    return MODELS_ROOT.is_dir() and any(MODELS_ROOT.iterdir())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_artifacts: needs model weights under MODELS_ROOT; skipped when absent "
        "unless TOXPRED_REQUIRE_ARTIFACTS=1",
    )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if artifacts_present():
        return
    required = os.getenv("TOXPRED_REQUIRE_ARTIFACTS") == "1"
    marked = [item for item in items if item.get_closest_marker("needs_artifacts")]
    if required and marked:
        raise pytest.UsageError(
            f"TOXPRED_REQUIRE_ARTIFACTS=1 but no model artifacts are present at "
            f"{MODELS_ROOT}; {len(marked)} test(s) would have been skipped."
        )
    skip = pytest.mark.skip(reason=f"no model artifacts under {MODELS_ROOT}")
    for item in marked:
        item.add_marker(skip)
