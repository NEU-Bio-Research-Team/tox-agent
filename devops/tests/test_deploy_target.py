"""The branch-to-target mapping, asserted without a cloud project.

I30's real danger was not that the workflows referenced paths the relocation
removed — that only made them fail. It was that both triggered on pushes to
`agent_test` and both deployed to the one live target, so a test branch
published to production. The runbook described a staging service; nothing in
the pipeline did.

These are the dry-run target assertions that closing criterion asks for. What
they cannot establish is anything about a real deployment: no staging smoke
ran, no rollback to a previous digest was attempted, and no cloud project was
contacted from this repository.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "devops" / "scripts"))

from deploy_target import ENVIRONMENTS, SERVICES, NoTarget, resolve  # noqa: E402


def test_the_test_branch_does_not_reach_production():
    for service in SERVICES:
        staging = resolve("refs/heads/agent_test", service)
        production = resolve("refs/heads/main", service)
        assert staging.environment == "staging"
        assert production.environment == "production"
        assert staging.cloud_run_service != production.cloud_run_service, (
            f"{service}: a push to agent_test would land on the production service"
        )


def test_an_unrecognised_ref_resolves_to_nothing_rather_than_a_default():
    for ref in ("refs/heads/docs/harness-master-plan", "refs/tags/v1.0.0", "refs/pull/7/merge"):
        with pytest.raises(NoTarget):
            resolve(ref, "control")


def test_production_promotes_rather_than_builds():
    """What production runs must be the artifact staging gated, not a rebuild
    of the same source that may not produce the same image."""
    assert all(not resolve("refs/heads/main", s).builds for s in SERVICES)
    assert all(resolve("refs/heads/agent_test", s).builds for s in SERVICES)


def test_every_service_names_a_dockerfile_that_exists():
    """The previous workflow built `model_server/Dockerfile`, which the
    relocation removed. A target naming a path that is not in the tree is the
    failure this check exists for."""
    for service, definition in SERVICES.items():
        dockerfile = ROOT / definition["dockerfile"]
        assert dockerfile.is_file(), f"{service} names {dockerfile}, which is not in the tree"
        context = ROOT / definition["context"]
        assert context.is_dir(), f"{service} names build context {context}, which is not a directory"


def test_an_unknown_service_is_refused():
    with pytest.raises(NoTarget):
        resolve("refs/heads/main", "model_server")


def test_the_cli_refuses_a_non_deploying_ref_with_a_nonzero_status():
    """The workflow runs this as a step; a wrong ref has to stop the job."""
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "devops" / "scripts" / "deploy_target.py"),
            "--ref", "refs/heads/somebodys-branch", "--service", "control",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "refusing to deploy" in result.stderr


def test_the_deploy_workflow_resolves_its_target_before_deploying():
    """Read the workflow, because the mapping is only useful if it is used."""
    yaml = pytest.importorskip("yaml")
    workflow_path = ROOT / ".github" / "workflows" / "deploy.yml"
    assert workflow_path.is_file(), "the deploy workflow is missing"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    triggers = workflow[True] if True in workflow else workflow["on"]
    branches = set(triggers.get("push", {}).get("branches", []))
    assert branches <= {ref.rsplit("/", 1)[-1] for ref in ENVIRONMENTS}, (
        f"the workflow deploys from {branches}, which the target map does not cover"
    )

    for job in workflow["jobs"].values():
        if "environment" not in job:
            continue
        steps = job.get("steps", [])
        assert any(
            "deploy_target.py" in str(step.get("run", "")) for step in steps
        ), "a deploying job must resolve its target first"
