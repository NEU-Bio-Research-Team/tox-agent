"""The documented way to test this package has to keep existing (P2-1..P2-4).

None of these assert behaviour of the product. They assert that a new
contributor, or a CI job, can run the suite at all — which is the thing the
audit found was not true, and the kind of thing that rots silently because no
feature depends on it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PACKAGE_ROOT / "deploy" / "Dockerfile"
MAKEFILE = PACKAGE_ROOT / "Makefile"
EVALS_README = PACKAGE_ROOT / "evals" / "README.md"


def _instructions(text: str) -> str:
    """The Dockerfile with its comments removed.

    The comments here explain what the stages are *for*, and quote the old
    ``COPY . /app/`` they replaced — so a check that reads the raw text finds
    the very string it is asserting the absence of.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _instructions(DOCKERFILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


# --- P2-3: one command, from nothing ----------------------------------------


def test_the_makefile_offers_a_single_test_target(makefile: str) -> None:
    assert re.search(r"^test:", makefile, re.MULTILINE)
    assert "pytest tests" in makefile
    assert "evals.runner --runtime scripted --trials 3" in makefile


def test_the_test_target_builds_its_own_environment(makefile: str) -> None:
    """It must not depend on whichever workstation virtualenv happens to have
    aiosqlite installed — that is exactly what made the suite unrunnable for
    anyone who had not set one up by hand."""
    assert "$(PYTHON) -m venv" in makefile
    assert "pip install -e '.[dev]'" in makefile
    # Every recipe goes through the venv's own interpreter.
    for line in makefile.splitlines():
        recipe = line.lstrip("\t")
        if line.startswith("\t") and not recipe.startswith("#") and " pytest" in line:
            assert "$(BIN)/" in line, line


# --- P2-3/P2-4: the image can run the suite, and does not ship it ------------


def test_the_image_has_a_test_stage(dockerfile: str) -> None:
    assert "AS test" in dockerfile
    assert "'.[postgres,dev]'" in dockerfile
    for directory in ("tests", "evals", "scripts"):
        assert f"COPY {directory} /app/{directory}" in dockerfile


def test_the_runtime_stage_ships_no_tests_and_no_dev_dependencies(
    dockerfile: str,
) -> None:
    runtime = dockerfile.split("FROM base AS runtime", 1)[1]
    for forbidden in ("COPY tests", "COPY evals", "pytest", "aiosqlite", "[dev]"):
        assert forbidden not in runtime, f"the runtime stage must not carry {forbidden!r}"


def test_the_base_stage_copies_a_named_list_rather_than_everything(
    dockerfile: str,
) -> None:
    """``COPY . /app/`` put the eval corpus and the fixtures into the deployed
    image, and nothing said so."""
    base = dockerfile.split("FROM base AS test", 1)[0]
    assert "COPY . /app/" not in base
    for needed in ("src", "migrations", "deploy"):
        assert f"COPY {needed} /app/{needed}" in base


def test_the_runtime_stage_is_last_so_a_plain_build_ships_it(dockerfile: str) -> None:
    stages = re.findall(r"^FROM .* AS (\w+)", dockerfile, re.MULTILINE)
    assert stages[-1] == "runtime", stages


# --- P2-1: the eval docs describe the driver that exists --------------------


def test_the_eval_readme_documents_the_remote_driver() -> None:
    text = EVALS_README.read_text(encoding="utf-8")
    assert "RemoteHTTPDriver" in text
    assert "--runtime opencode" in text
    assert "not written yet" not in text, (
        "the remote driver exists in runner.py; the README said it did not"
    )


def test_the_eval_readme_points_at_the_one_command() -> None:
    text = EVALS_README.read_text(encoding="utf-8")
    assert "make -C backend/control test" in text


def test_the_remote_driver_the_readme_names_is_importable() -> None:
    from evals.runner import RemoteHTTPDriver

    assert hasattr(RemoteHTTPDriver, "run")
