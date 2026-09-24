"""Who the control plane believes a user is (K12).

The defect this replaces: `build_auth` ended in
`JwtAuth(secret=settings.capability_secret)`. That is the key a run's
capability tokens are signed with, verifying neither issuer nor audience,
deciding the identity of a *product user* — two trust domains sharing one key,
in exactly the deployment shape (`production`, where the static-token shim is
refused) that had ruled out the obviously unsafe option.

Real capability tokens happened not to get through, because they carry an
`aud` claim and PyJWT rejects an unexpected audience. That is a coincidence of
validation order, not a boundary: any HS256 token signed with that key,
carrying any `sub` and any `roles` and no `aud`, authenticated as that user.
The first test below is that token.
"""
from __future__ import annotations

import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from toxagent.api.auth import JwksAuth, JwtAuth, StaticTokenAuth, build_auth
from toxagent.config import SecuritySettings
from toxagent.domain.errors import Unauthenticated

pytestmark = pytest.mark.anyio

ISSUER = "https://idp.example.com"
AUDIENCE = "toxagent-api"
JWKS_URL = "https://idp.example.com/.well-known/jwks.json"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwks_of(*keys: tuple[str, object]) -> dict:
    return {
        "keys": [
            {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key())),
             "kid": kid, "use": "sig", "alg": "RS256"}
            for kid, key in keys
        ]
    }


def token_from(key, kid: str, **overrides) -> str:
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": "user-1",
        "exp": int(time.time()) + 300, "roles": ["analyst"],
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


def request_with(token: str) -> httpx.Request:
    class _Request:
        headers = {"authorization": f"Bearer {token}"}

    return _Request()


def serving(document: dict, *, calls: list | None = None) -> httpx.AsyncClient:
    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(200, json=document)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def verifier(client: httpx.AsyncClient, **kwargs) -> JwksAuth:
    return JwksAuth(
        jwks_url=JWKS_URL, issuer=ISSUER, audience=AUDIENCE, client=client, **kwargs
    )


# --------------------------------------------------------- the original hole


async def test_a_token_signed_with_the_capability_secret_is_no_longer_a_user():
    """The reproduction. `capability_secret` is now no key for user auth,
    because the only user verifier a production deployment may have takes
    public keys from a provider and never a secret this server holds."""
    settings = SecuritySettings(capability_secret="a-capability-signing-secret" * 2)
    with pytest.raises(ValueError) as excinfo:
        build_auth(settings)
    assert "TOXAGENT_OIDC_ISSUER" in str(excinfo.value)


def test_production_will_not_start_without_an_identity_provider(monkeypatch):
    """The check that makes the above unavoidable rather than merely available."""
    monkeypatch.setenv("TOXAGENT_ENV", "production")
    monkeypatch.setenv("TOXAGENT_CAPABILITY_SECRET", "s" * 40)
    monkeypatch.setenv("TOXAGENT_EGRESS_POLICY", "hosted")
    monkeypatch.delenv("TOXAGENT_STATIC_TOKENS", raising=False)
    for name in ("TOXAGENT_OIDC_ISSUER", "TOXAGENT_OIDC_AUDIENCE", "TOXAGENT_OIDC_JWKS_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="identity provider"):
        SecuritySettings.from_env()


def test_half_configured_oidc_is_refused(monkeypatch):
    """Missing an audience is worse than missing everything: the verifier then
    accepts a token the same provider minted for a different service."""
    monkeypatch.setenv("TOXAGENT_OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("TOXAGENT_OIDC_JWKS_URL", JWKS_URL)
    monkeypatch.delenv("TOXAGENT_OIDC_AUDIENCE", raising=False)
    with pytest.raises(ValueError, match="TOXAGENT_OIDC_AUDIENCE"):
        SecuritySettings.from_env()


def test_a_key_set_must_be_fetched_over_https(monkeypatch):
    monkeypatch.setenv("TOXAGENT_OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("TOXAGENT_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("TOXAGENT_OIDC_JWKS_URL", "http://idp.example.com/jwks.json")
    with pytest.raises(ValueError, match="https"):
        SecuritySettings.from_env()


def test_oidc_is_chosen_over_the_development_shim():
    """Both configured is a misconfiguration, and the safe reading of it is
    the provider — not the token list someone left in an env file."""
    settings = SecuritySettings(
        oidc_issuer=ISSUER, oidc_audience=AUDIENCE, oidc_jwks_url=JWKS_URL,
        static_tokens=("dev:someone:owner",),
    )
    assert isinstance(build_auth(settings), JwksAuth)
    assert isinstance(build_auth(SecuritySettings(static_tokens=("dev:x:owner",))), StaticTokenAuth)


# ------------------------------------------------------------- verification


async def test_a_provider_signed_token_authenticates_with_its_roles():
    key = rsa_key()
    async with serving(jwks_of(("k1", key))) as client:
        actor = await verifier(client).authenticate(request_with(token_from(key, "k1")))
    assert actor.subject_id == "user-1"
    assert actor.roles == frozenset({"analyst"})


async def test_scopes_as_a_space_delimited_string_are_read_as_roles():
    key = rsa_key()
    async with serving(jwks_of(("k1", key))) as client:
        actor = await verifier(client).authenticate(
            request_with(token_from(key, "k1", roles="analyst owner"))
        )
    assert actor.roles == frozenset({"analyst", "owner"})


async def test_a_token_for_another_service_is_refused():
    """Signed by the right provider, with a real user in it, and still not a
    token for this API."""
    key = rsa_key()
    async with serving(jwks_of(("k1", key))) as client:
        with pytest.raises(Unauthenticated):
            await verifier(client).authenticate(
                request_with(token_from(key, "k1", aud="some-other-service"))
            )


async def test_a_token_from_another_issuer_is_refused():
    key = rsa_key()
    async with serving(jwks_of(("k1", key))) as client:
        with pytest.raises(Unauthenticated):
            await verifier(client).authenticate(
                request_with(token_from(key, "k1", iss="https://attacker.example.com"))
            )


async def test_a_token_signed_by_a_key_the_provider_does_not_publish_is_refused():
    published, forged = rsa_key(), rsa_key()
    async with serving(jwks_of(("k1", published))) as client:
        with pytest.raises(Unauthenticated, match="does not publish"):
            await verifier(client).authenticate(request_with(token_from(forged, "k9")))


async def test_a_token_signed_with_the_right_kid_but_the_wrong_key_is_refused():
    """The kid names a key; it does not prove one."""
    published, forged = rsa_key(), rsa_key()
    async with serving(jwks_of(("k1", published))) as client:
        with pytest.raises(Unauthenticated):
            await verifier(client).authenticate(request_with(token_from(forged, "k1")))


async def test_an_expired_token_is_refused():
    key = rsa_key()
    async with serving(jwks_of(("k1", key))) as client:
        with pytest.raises(Unauthenticated):
            await verifier(client).authenticate(
                request_with(token_from(key, "k1", exp=int(time.time()) - 1))
            )


async def test_a_token_with_no_key_id_is_refused():
    key = rsa_key()
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "user-1", "exp": int(time.time()) + 300},
        key, algorithm="RS256",
    )
    async with serving(jwks_of(("k1", key))) as client:
        with pytest.raises(Unauthenticated, match="no key id"):
            await verifier(client).authenticate(request_with(token))


async def test_an_unreachable_provider_denies_rather_than_admits():
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    key = rsa_key()
    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(Unauthenticated, match="key set is unavailable"):
            await verifier(client).authenticate(request_with(token_from(key, "k1")))


# ----------------------------------------------------------------- rotation


async def test_the_key_set_is_cached_rather_than_fetched_per_request():
    key = rsa_key()
    calls: list[str] = []
    async with serving(jwks_of(("k1", key)), calls=calls) as client:
        auth = verifier(client)
        for _ in range(3):
            await auth.authenticate(request_with(token_from(key, "k1")))
    assert len(calls) == 1


async def test_a_rotated_key_is_picked_up_without_waiting_out_the_cache():
    """A provider rotates without announcing it. An unknown kid is a cache
    miss, not an outage until the TTL expires."""
    old, new = rsa_key(), rsa_key()
    document = jwks_of(("old", old))
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=document)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = verifier(client, cache_ttl_s=3600)
        await auth.authenticate(request_with(token_from(old, "old")))
        assert len(calls) == 1

        document = jwks_of(("old", old), ("new", new))
        actor = await auth.authenticate(request_with(token_from(new, "new")))
        assert actor.subject_id == "user-1"
        assert len(calls) == 2, "an unknown kid must refetch inside the TTL"


async def test_an_unknown_kid_refetches_once_and_then_refuses():
    """Otherwise a made-up kid is a way to drive traffic at the provider."""
    key = rsa_key()
    calls: list[str] = []
    async with serving(jwks_of(("k1", key)), calls=calls) as client:
        auth = verifier(client)
        for _ in range(4):
            with pytest.raises(Unauthenticated):
                await auth.authenticate(request_with(token_from(key, "made-up")))
    assert len(calls) == 4, "one refetch per attempt, not a retry loop inside one"


# ------------------------------------------------- the symmetric verifier


async def test_the_symmetric_verifier_still_binds_issuer_and_audience():
    """`JwtAuth` remains, for a deployment running its own issuer. Given an
    audience it enforces one, which the old call site never passed."""
    secret = "a-dedicated-user-signing-secret-not-the-capability-one"
    auth = JwtAuth(secret=secret, audience=AUDIENCE, issuer=ISSUER)
    good = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "user-1", "exp": int(time.time()) + 300},
        secret, algorithm="HS256",
    )
    assert (await auth.authenticate(request_with(good))).subject_id == "user-1"

    for wrong in (
        {"iss": ISSUER, "aud": "elsewhere", "sub": "user-1", "exp": int(time.time()) + 300},
        {"iss": "https://attacker.example.com", "aud": AUDIENCE, "sub": "user-1",
         "exp": int(time.time()) + 300},
    ):
        token = jwt.encode(wrong, secret, algorithm="HS256")
        with pytest.raises(Unauthenticated):
            await auth.authenticate(request_with(token))
