"""Where the server is allowed to send a user-supplied URL.

I15: `base_url` comes from whoever can create a connection, and the capability
probe POSTs to it from inside the control plane. On a hosted deployment that
is a request forgery primitive — the metadata service, a database admin port,
anything else on the private network — reached with the server's own network
position rather than the user's.

The policy is deliberately deployment-dependent, because the same behaviour is
correct in one deployment and a vulnerability in another:

- **local** (single-user self-hosting): a private address is the *normal* case
  — Ollama on 127.0.0.1, vLLM on the LAN. Allowed, because there is no
  boundary here to cross; the user already has this machine.
- **hosted** (multi-user): a private address is never a legitimate model
  endpoint, and the request must be refused before it is made.

Resolution happens before the request and again on every redirect, because a
name that resolved to a public address once can resolve to a private one the
next time (DNS rebinding). The address that was checked is the address that
gets connected to.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse


class EgressPolicy(str, Enum):
    #: Self-hosted, single user. Private destinations are expected.
    LOCAL = "local"
    #: Multi-tenant. Only public destinations.
    HOSTED = "hosted"


class BlockedDestination(ValueError):
    """A URL this deployment will not let the server call.

    The message names the rule, never the resolved address of an internal
    host — telling a caller *which* private address their hostname resolved
    to is itself a small scan.
    """


@dataclass(frozen=True, slots=True)
class Destination:
    host: str
    port: int
    scheme: str
    addresses: tuple[str, ...]


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    # `is_global` is False for private, loopback, link-local, multicast,
    # reserved and unspecified, in both v4 and v6 — including the v4-mapped
    # v6 forms (::ffff:169.254.169.254) that a v4-only check would miss.
    if not ip.is_global:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return _is_public(str(ip.ipv4_mapped))
    return True


def resolve(url: str) -> Destination:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise BlockedDestination(
            f"only http and https URLs may be called, not {parsed.scheme or 'a relative URL'!r}"
        )
    if not parsed.hostname:
        raise BlockedDestination("the URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedDestination(f"{parsed.hostname} could not be resolved") from exc
    addresses = tuple(sorted({info[4][0] for info in infos}))
    if not addresses:
        raise BlockedDestination(f"{parsed.hostname} resolved to no addresses")
    return Destination(parsed.hostname, port, parsed.scheme, addresses)


def check(url: str, policy: EgressPolicy) -> Destination:
    """Raise unless this deployment may call `url`.

    Under HOSTED, *every* resolved address must be public: a name that
    resolves to both a public and a private address is a rebinding attempt,
    not a multi-homed service worth accommodating.
    """
    destination = resolve(url)
    if policy is EgressPolicy.LOCAL:
        return destination
    if not all(_is_public(address) for address in destination.addresses):
        raise BlockedDestination(
            f"{destination.host} resolves to an address this deployment will not call. "
            "Hosted deployments may only reach public model endpoints."
        )
    return destination
