"""Where a user-supplied base URL may send the server.

I15: `base_url` comes from whoever can create a connection, and the probe
POSTs to it from inside the control plane, with the control plane's network
position. On a hosted deployment that reaches the metadata service, a database
admin port, or anything else private — none of which the user could reach
themselves.

The policy is deployment-dependent on purpose, and both halves are tested,
because getting either backwards breaks something real: refusing loopback on a
self-hosted box makes Ollama unusable, and allowing it on a hosted one is the
vulnerability.

No DNS is performed here. `resolve` is stubbed so these assertions are about
the *rule*, not about what a name happens to resolve to on the machine running
the tests.
"""
from __future__ import annotations

import pytest

from toxagent.connections import network
from toxagent.connections.network import (
    BlockedDestination,
    Destination,
    EgressPolicy,
    check,
)


@pytest.fixture
def resolves_to(monkeypatch):
    """Pin what a hostname resolves to, without touching DNS."""

    def _install(*addresses: str):
        def fake(url: str) -> Destination:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return Destination(
                parsed.hostname or "", parsed.port or 443, parsed.scheme, tuple(addresses)
            )

        monkeypatch.setattr(network, "resolve", fake)

    return _install


PRIVATE = [
    ("127.0.0.1", "loopback"),
    ("10.0.0.5", "RFC1918 class A"),
    ("172.16.4.2", "RFC1918 class B"),
    ("192.168.1.10", "RFC1918 class C"),
    ("169.254.169.254", "the cloud metadata service"),
    ("::1", "IPv6 loopback"),
    ("fd00::1", "IPv6 unique-local"),
    ("fe80::1", "IPv6 link-local"),
    ("::ffff:169.254.169.254", "the metadata service as a v4-mapped v6 address"),
    ("0.0.0.0", "the unspecified address"),
]


@pytest.mark.parametrize("address,description", PRIVATE, ids=[d for _, d in PRIVATE])
def test_hosted_refuses_a_private_destination(resolves_to, address, description):
    resolves_to(address)
    with pytest.raises(BlockedDestination):
        check("https://model.example.com/v1", EgressPolicy.HOSTED)


@pytest.mark.parametrize("address,description", PRIVATE, ids=[d for _, d in PRIVATE])
def test_local_allows_a_private_destination(resolves_to, address, description):
    """A self-hosted model on loopback or the LAN is the normal case."""
    resolves_to(address)
    assert check("http://localhost:11434/v1", EgressPolicy.LOCAL).addresses == (address,)


def test_hosted_allows_a_public_destination(resolves_to):
    resolves_to("140.82.121.4")
    assert check("https://api.openai.com/v1", EgressPolicy.HOSTED).host == "api.openai.com"


def test_a_name_resolving_to_both_public_and_private_is_refused(resolves_to):
    """DNS rebinding: one public answer is not a licence to connect.

    Checking "any address is public" would let an attacker publish A records
    for both and race the resolution.
    """
    resolves_to("140.82.121.4", "169.254.169.254")
    with pytest.raises(BlockedDestination):
        check("https://rebind.example.com/v1", EgressPolicy.HOSTED)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/1", "ftp://x/", "/v1"])
def test_only_http_and_https_are_callable(url):
    with pytest.raises(BlockedDestination, match="http"):
        check(url, EgressPolicy.HOSTED)


def test_a_url_with_no_host_is_refused():
    with pytest.raises(BlockedDestination):
        check("https:///v1", EgressPolicy.HOSTED)


def test_a_blocked_message_does_not_disclose_the_resolved_address(resolves_to):
    """Naming the internal address a hostname resolved to is a small scan."""
    resolves_to("10.1.2.3")
    with pytest.raises(BlockedDestination) as excinfo:
        check("https://internal.example.com/v1", EgressPolicy.HOSTED)
    assert "10.1.2.3" not in str(excinfo.value)
