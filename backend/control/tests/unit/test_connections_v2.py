import os

from toxagent.connections.model import ModelConnection
from toxagent.connections.secrets import FilesystemSecretStore
from toxagent.domain.runtime import AuthMode


def test_secret_store_returns_reference_and_enforces_permissions(tmp_path):
    store = FilesystemSecretStore(tmp_path / "secrets")
    ref = store.put("owner", "do-not-log-me")
    assert "do-not-log-me" not in ref
    assert store.get(ref) == "do-not-log-me"
    assert os.stat(tmp_path / "secrets").st_mode & 0o777 == 0o700
    assert os.stat(tmp_path / "secrets" / ref).st_mode & 0o777 == 0o600


def test_connection_public_shape_never_contains_credential(tmp_path):
    store = FilesystemSecretStore(tmp_path)
    ref = store.put("owner", "key")
    from datetime import datetime, timezone
    connection = ModelConnection.create(
        owner_id="owner", provider_id="openai", model_id="model", auth_mode=AuthMode.API_KEY,
        credential_ref=ref, base_url=None, now=datetime.now(timezone.utc),
    )
    assert "credential_ref" not in connection.public_dict()
    assert connection.public_dict()["has_credential"] is True
