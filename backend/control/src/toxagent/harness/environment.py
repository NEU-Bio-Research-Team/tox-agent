"""Explicit child-runtime environment construction."""
from __future__ import annotations

from typing import Mapping

SAFE_KEYS = frozenset({
    "PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
})
FORBIDDEN_FRAGMENTS = ("DATABASE", "PREDICTOR", "SECRET", "TOKEN", "PASSWORD", "PRIVATE_KEY")


def runtime_environment(source: Mapping[str, str], *, additions: Mapping[str, str] | None = None,
                        allow: frozenset[str] = SAFE_KEYS) -> dict[str, str]:
    env = {key: value for key, value in source.items() if key in allow}
    for key, value in (additions or {}).items():
        upper = key.upper()
        if any(fragment in upper for fragment in FORBIDDEN_FRAGMENTS):
            raise ValueError(f"refusing sensitive runtime environment key {key!r}")
        env[key] = value
    return env
