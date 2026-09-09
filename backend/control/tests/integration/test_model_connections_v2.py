from datetime import datetime, timezone

import pytest

from toxagent.connections.probe import ProbeResult, Support
from toxagent.connections.secrets import FilesystemSecretStore
from toxagent.connections.service import ModelConnectionService
from toxagent.domain.runtime import AuthMode

pytestmark = pytest.mark.anyio


class Probe:
    async def probe(self, connection, credential):
        assert credential == "private-key"
        return ProbeResult(
            streaming=Support.SUPPORTED,
            structured_output=Support.SUPPORTED,
            tool_calls=Support.SUPPORTED,
            detail="stubbed",
        )


async def test_connection_crud_persists_only_secret_reference(db, tmp_path):
    secrets = FilesystemSecretStore(tmp_path / "secrets")
    service = ModelConnectionService(db, secrets, Probe())
    created = await service.create(
        owner_id="owner", provider_id="openai_compatible", model_id="model-a",
        auth_mode=AuthMode.API_KEY, base_url="https://model.invalid/v1",
        credential="private-key",
    )
    assert "private-key" not in str(created.public_dict())
    tested = await service.test(created.id, owner_id="owner")
    assert tested.status.value == "ready"
    # Context size stays unknown: nothing in the probe measures or retrieves
    # it, and a number here would be a guess a run could rely on.
    assert tested.capabilities.context_size is None
    assert tested.capabilities.tool_calls is True
    assert len(await service.list(owner_id="owner")) == 1
    assert await service.delete(created.id, owner_id="owner") is True
    assert not any((tmp_path / "secrets").iterdir())
