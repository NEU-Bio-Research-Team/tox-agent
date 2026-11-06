"""User-supplied AI credentials: how they are stored, and how they reach a runtime.

Two failure classes this guards, both learned from G1_Web_Design's
``ai_connections`` and from I12 here:

- a stored credential readable as plaintext from the volume, or reusable by
  another owner if its file is copied under another reference;
- a credential handed to the runtime in a form OpenCode 1.17.11 ignores — a
  custom endpoint's key sent through ``/auth`` reached the endpoint with no
  Authorization header at all when this was checked live — so the run bills
  nothing, or someone else.
"""
from __future__ import annotations

import json

import pytest
from cryptography.fernet import Fernet

from toxagent.connections import providers
from toxagent.connections.runtime_install import (
    PROVIDER_KEY_ENV,
    HostCredential,
    install_for,
    parse_oauth,
)
from toxagent.connections.secrets import (
    EncryptedSecretStore,
    FilesystemSecretStore,
    SecretUnavailable,
    build_secret_store,
    read_for_owner,
)
from toxagent.domain.errors import RuntimeProtocolError, RuntimeUnavailable

OAUTH = {"type": "oauth", "refresh": "r-1", "access": "a-1", "expires": 1_900_000_000_000}


# -- storage ----------------------------------------------------------------


def test_an_encrypted_secret_is_not_on_disk_in_the_clear(tmp_path):
    store = EncryptedSecretStore(tmp_path, Fernet.generate_key().decode())
    reference = store.put("owner-a", "sk-very-private")
    raw = (tmp_path / reference).read_text()
    assert "sk-very-private" not in raw
    assert "owner-a" not in raw
    assert store.get(reference) == "sk-very-private"
    assert oct((tmp_path / reference).stat().st_mode & 0o777) == "0o600"


def test_a_secret_is_bound_to_the_owner_it_was_written_for(tmp_path):
    store = EncryptedSecretStore(tmp_path, Fernet.generate_key().decode())
    reference = store.put("owner-a", "sk-a")
    assert read_for_owner(store, reference, "owner-a") == "sk-a"
    with pytest.raises(SecretUnavailable):
        read_for_owner(store, reference, "owner-b")


def test_a_secret_written_under_another_key_does_not_decrypt(tmp_path):
    first = EncryptedSecretStore(tmp_path, Fernet.generate_key().decode())
    reference = first.put("owner-a", "sk-a")
    second = EncryptedSecretStore(tmp_path, Fernet.generate_key().decode())
    with pytest.raises(SecretUnavailable):
        second.get(reference)


def test_a_tampered_token_is_refused_without_echoing_it(tmp_path):
    store = EncryptedSecretStore(tmp_path, Fernet.generate_key().decode())
    reference = store.put("owner-a", "sk-a")
    path = tmp_path / reference
    path.write_text(path.read_text()[:-4] + "AAAA")
    with pytest.raises(SecretUnavailable) as excinfo:
        store.get(reference)
    assert "sk-a" not in str(excinfo.value)


def test_replace_overwrites_in_place_and_never_creates(tmp_path):
    store = EncryptedSecretStore(tmp_path, Fernet.generate_key().decode())
    reference = store.put("owner-a", json.dumps(OAUTH))
    refreshed = json.dumps({**OAUTH, "access": "a-2", "refresh": "r-2"})
    store.replace(reference, "owner-a", refreshed)
    assert json.loads(store.get(reference))["refresh"] == "r-2"
    assert sorted(p.name for p in tmp_path.iterdir()) == [reference]
    with pytest.raises(FileNotFoundError):
        store.replace("secret_" + "0" * 32, "owner-a", "x")


def test_an_invalid_key_is_refused_at_start_up(tmp_path):
    with pytest.raises(ValueError, match="Fernet"):
        EncryptedSecretStore(tmp_path, "not-a-key")


def test_without_a_key_the_plaintext_store_is_used(tmp_path):
    assert type(build_secret_store(tmp_path)) is FilesystemSecretStore
    key = Fernet.generate_key().decode()
    assert type(build_secret_store(tmp_path / "e", key)) is EncryptedSecretStore


def test_production_requires_an_encryption_key(monkeypatch):
    from toxagent.platform.config import SecuritySettings

    monkeypatch.setenv("TOXAGENT_ENV", "production")
    monkeypatch.setenv("TOXAGENT_CAPABILITY_SECRET", "x" * 40)
    monkeypatch.setenv("TOXAGENT_OIDC_ISSUER", "https://issuer.example")
    monkeypatch.setenv("TOXAGENT_OIDC_AUDIENCE", "toxagent")
    monkeypatch.setenv("TOXAGENT_OIDC_JWKS_URL", "https://issuer.example/jwks")
    monkeypatch.setenv("TOXAGENT_EGRESS_POLICY", "hosted")
    monkeypatch.delenv("TOXAGENT_STATIC_TOKENS", raising=False)
    monkeypatch.delenv("TOXAGENT_SECRET_ENCRYPTION_KEY", raising=False)
    with pytest.raises(ValueError, match="TOXAGENT_SECRET_ENCRYPTION_KEY"):
        SecuritySettings.from_env()
    monkeypatch.setenv("TOXAGENT_SECRET_ENCRYPTION_KEY", Fernet.generate_key().decode())
    assert SecuritySettings.from_env().secret_encryption_key


# -- presentation to OpenCode -------------------------------------------------


def _credential(**overrides) -> HostCredential:
    values = dict(
        connection_id="con_" + "a" * 32, owner_id="owner-a", provider_id="openai",
        model_id="gpt-x", auth_mode="api_key", base_url=None, secret="sk-a",
    )
    values.update(overrides)
    return HostCredential(**values)


@pytest.mark.parametrize("provider_id", ["openai", "anthropic", "openrouter", "deepseek"])
def test_a_catalog_provider_takes_its_key_through_opencode_auth(provider_id):
    install = install_for(_credential(provider_id=provider_id))
    runtime = providers.get(provider_id).runtime_provider_id
    assert install.runtime_provider_id == runtime
    assert install.auth == {"type": "api", "key": "sk-a"}
    assert install.env == {}
    # The model is declared, so a model the pinned binary predates still resolves.
    assert install.config["provider"][runtime]["models"] == {"gpt-x": {"name": "gpt-x"}}
    assert "sk-a" not in json.dumps(install.config)


def test_a_custom_endpoint_reads_its_key_from_the_host_environment():
    install = install_for(
        _credential(provider_id="openai_compatible", base_url="http://127.0.0.1:8000/v1")
    )
    custom = install.config["provider"][providers.CUSTOM_RUNTIME_PROVIDER]
    assert install.runtime_provider_id == providers.CUSTOM_RUNTIME_PROVIDER
    assert custom["options"] == {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "{env:" + PROVIDER_KEY_ENV + "}",
    }
    assert install.env == {PROVIDER_KEY_ENV: "sk-a"}
    assert install.auth is None
    assert "sk-a" not in json.dumps(install.config)


def test_a_keyless_local_endpoint_gets_no_key_at_all():
    install = install_for(
        _credential(provider_id="openai_compatible", base_url="http://127.0.0.1:11434/v1",
                    auth_mode="none", secret=None)
    )
    options = install.config["provider"][providers.CUSTOM_RUNTIME_PROVIDER]["options"]
    assert "apiKey" not in options and install.env == {} and install.auth is None


def test_a_catalog_provider_behind_a_gateway_keeps_its_sdk_and_moves():
    install = install_for(_credential(base_url="https://gateway.example/v1"))
    assert install.runtime_provider_id == "openai"
    assert install.config["provider"]["openai"]["options"]["baseURL"] == "https://gateway.example/v1"
    assert install.env == {PROVIDER_KEY_ENV: "sk-a"}


def test_a_chatgpt_sign_in_installs_the_oauth_pair_on_openai():
    install = install_for(_credential(auth_mode="chatgpt_subscription", secret=json.dumps(OAUTH)))
    assert install.runtime_provider_id == "openai"
    assert install.auth == OAUTH


def test_a_chatgpt_sign_in_cannot_back_another_provider():
    with pytest.raises(RuntimeProtocolError):
        install_for(_credential(provider_id="anthropic", auth_mode="chatgpt_subscription",
                                secret=json.dumps(OAUTH)))


@pytest.mark.parametrize(
    "secret",
    [
        "not json",
        json.dumps({**OAUTH, "type": "api"}),
        json.dumps({k: v for k, v in OAUTH.items() if k != "refresh"}),
        json.dumps({**OAUTH, "expires": "soon"}),
        json.dumps({**OAUTH, "extra": "field"}),
    ],
)
def test_a_malformed_sign_in_is_refused_with_a_reconnect_message(secret):
    with pytest.raises(RuntimeUnavailable, match="reconnect"):
        parse_oauth(secret)


def test_a_fingerprint_changes_with_the_secret_and_never_contains_it():
    first = _credential()
    assert "sk-a" not in first.fingerprint()
    assert first.fingerprint() != _credential(secret="sk-b").fingerprint()
    assert first.fingerprint() != _credential(model_id="gpt-y").fingerprint()
    assert "sk-a" not in repr(first)


def test_turning_encryption_on_migrates_a_plaintext_key_on_first_use(tmp_path):
    plain = FilesystemSecretStore(tmp_path)
    reference = plain.put("owner-a", "sk-legacy")
    store = EncryptedSecretStore(tmp_path, Fernet.generate_key().decode())
    assert read_for_owner(store, reference, "owner-a") == "sk-legacy"
    assert "sk-legacy" not in (tmp_path / reference).read_text()
    assert store.owner_of(reference) == "owner-a"
    # Now bound: nobody else can read it.
    with pytest.raises(SecretUnavailable):
        read_for_owner(store, reference, "owner-b")
