"""Where the two directories a deployment must persist actually are.

I16: the Compose stack gave PostgreSQL a volume and gave the attachment store
and the secret store none, so recreating the control container destroyed every
uploaded image and every stored AI profile credential while the database kept
the rows that reference them — a connection that still reads as configured
with no key behind it, an attachment row with no bytes.

The secrets path was derived at its two call sites as
`object_store_dir.parent / "model-secrets"`, which is why it was easy to miss:
it was not a setting anyone could see or set. These check that it is one now,
that it still resolves to the old location for a deployment that has keys
there, and that the two directories cannot silently become the same one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from toxagent.config import Settings


def _settings(monkeypatch, **env: str) -> Settings:
    for key in ("TOXAGENT_OBJECT_STORE_DIR", "TOXAGENT_SECRETS_DIR"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TOXAGENT_CAPABILITY_SECRET", "x" * 32)
    return Settings.from_env()


def test_both_directories_can_be_pointed_at_a_mounted_volume(monkeypatch):
    settings = _settings(
        monkeypatch,
        TOXAGENT_OBJECT_STORE_DIR="/var/lib/toxagent/attachments",
        TOXAGENT_SECRETS_DIR="/var/lib/toxagent/model-secrets",
    )
    assert settings.object_store_dir == Path("/var/lib/toxagent/attachments")
    assert settings.secrets_dir == Path("/var/lib/toxagent/model-secrets")


def test_an_existing_deployment_keeps_reading_the_keys_it_already_has(monkeypatch):
    """Only the object store dir was ever configurable, and the secrets sat
    beside it. A deployment that set just that one must not lose its keys."""
    settings = _settings(monkeypatch, TOXAGENT_OBJECT_STORE_DIR="/srv/state/attachments")
    assert settings.secrets_dir == Path("/srv/state/model-secrets")


def test_the_two_stores_do_not_share_a_directory_by_default(monkeypatch):
    settings = _settings(monkeypatch)
    assert settings.secrets_dir != settings.object_store_dir


def test_the_secret_store_creates_its_directory_closed(tmp_path):
    """A credential store that inherits a world-readable directory is a
    credential store in name only."""
    from toxagent.connections.secrets import FilesystemSecretStore

    root = tmp_path / "model-secrets"
    store = FilesystemSecretStore(root)
    reference = store.put("user-1", "sk-not-a-real-key")

    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / reference).stat().st_mode & 0o777 == 0o600
    assert store.get(reference) == "sk-not-a-real-key"


def test_the_compose_stack_mounts_a_volume_over_both_directories():
    """The bug was in the deployment, so the check has to be too.

    Reading the file rather than asserting against a parsed service graph
    keeps this honest about what a reviewer would see; a stack that sets the
    paths but forgets the volume is exactly the state this issue described.
    """
    yaml = pytest.importorskip("yaml")
    compose = (
        Path(__file__).resolve().parents[4] / "devops" / "compose" / "compose.yaml"
    )
    if not compose.exists():
        pytest.skip(f"the workspace compose stack is not at {compose}")
    stack = yaml.safe_load(compose.read_text(encoding="utf-8"))
    control = stack["services"]["toxagent-control"]

    mounts = [str(v).split(":") for v in control.get("volumes", [])]
    named = {parts[0]: parts[1] for parts in mounts if len(parts) >= 2}
    assert named, "the control service persists files and has no volume"

    for variable in ("TOXAGENT_OBJECT_STORE_DIR", "TOXAGENT_SECRETS_DIR"):
        configured = Path(control["environment"][variable])
        assert any(
            configured == Path(target) or Path(target) in configured.parents
            for target in named.values()
        ), f"{variable} is {configured}, which no volume covers"

    assert set(named) <= set(stack.get("volumes") or {}), "the mount names an undeclared volume"
