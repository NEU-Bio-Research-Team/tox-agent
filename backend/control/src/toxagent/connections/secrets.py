"""Secret references; credentials never enter product rows or transcripts.

Two stores share one interface:

- ``FilesystemSecretStore`` keeps each secret in its own 0600 file under a
  0700 directory. It is the local-development store and what a deployment
  without an encryption key falls back to.
- ``EncryptedSecretStore`` writes the same files, but each holds a Fernet
  token whose authenticated plaintext names the owner the secret belongs to.
  A token copied under another reference, or read back for another owner,
  does not decrypt into a usable credential — the same binding G1's
  ``ai_connections`` uses, so a row moved between accounts cannot reassign a
  key.

``replace`` exists for one reason: a ChatGPT subscription credential is an
OAuth token pair that the runtime refreshes. The refreshed pair must overwrite
the stored one under the same reference, or the next run starts from a
refresh token the provider has already rotated away.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class SecretStore(Protocol):
    def put(self, owner_id: str, secret: str) -> str: ...
    def get(self, reference: str) -> str: ...
    def replace(self, reference: str, owner_id: str, secret: str) -> None: ...
    def delete(self, reference: str) -> None: ...


class SecretUnavailable(RuntimeError):
    """A stored secret exists but cannot be read back as this owner's.

    The message never carries the token or the plaintext.
    """


class FilesystemSecretStore:
    """Local-development store with a closed directory and opaque references."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._root, 0o700)

    def _path(self, reference: str) -> Path:
        if not reference.startswith("secret_") or any(c not in "0123456789abcdef" for c in reference[7:]):
            raise ValueError("invalid secret reference")
        path = (self._root / reference).resolve()
        if path.parent != self._root:
            raise ValueError("invalid secret reference")
        return path

    def _encode(self, owner_id: str, secret: str) -> str:
        return secret

    def _decode(self, stored: str) -> str:
        return stored

    def put(self, owner_id: str, secret: str) -> str:
        if not owner_id or not secret:
            raise ValueError("owner and non-empty secret are required")
        reference = f"secret_{uuid4().hex}"
        path = self._path(reference)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(self._encode(owner_id, secret))
        return reference

    def get(self, reference: str) -> str:
        return self._decode(self._path(reference).read_text(encoding="utf-8"))

    def replace(self, reference: str, owner_id: str, secret: str) -> None:
        """Overwrite an existing secret atomically, never create one."""
        if not owner_id or not secret:
            raise ValueError("owner and non-empty secret are required")
        path = self._path(reference)
        if not path.exists():
            raise FileNotFoundError("secret reference does not exist")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self._encode(owner_id, secret))
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def delete(self, reference: str) -> None:
        try:
            self._path(reference).unlink()
        except FileNotFoundError:
            pass

    def owner_of(self, reference: str) -> str | None:
        """Who a secret was written for, when the store records it."""
        return None


#: Every Fernet token starts with its version byte, base64url-encoded.
_FERNET_PREFIX = "gAAAAA"


class EncryptedSecretStore(FilesystemSecretStore):
    """Fernet-encrypted secrets, each bound to the owner it was written for.

    The key is a deployment secret (``TOXAGENT_SECRET_ENCRYPTION_KEY``), the
    same on every replica, and never in Git. Losing it means users re-enter
    their keys; it is not recoverable from the store.
    """

    def __init__(self, root: Path, key: str) -> None:
        from cryptography.fernet import Fernet

        try:
            self._fernet = Fernet(key.encode("ascii"))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "TOXAGENT_SECRET_ENCRYPTION_KEY is not a valid Fernet key"
            ) from exc
        super().__init__(root)

    def _encode(self, owner_id: str, secret: str) -> str:
        payload = json.dumps({"owner": owner_id, "secret": secret}, separators=(",", ":"))
        return self._fernet.encrypt(payload.encode("utf-8")).decode("ascii")

    def _open(self, stored: str) -> dict:
        from cryptography.fernet import InvalidToken

        try:
            payload = json.loads(self._fernet.decrypt(stored.encode("ascii")))
        except (InvalidToken, ValueError, UnicodeError) as exc:
            raise SecretUnavailable("a stored credential cannot be decrypted") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("secret"), str):
            raise SecretUnavailable("a stored credential has an invalid shape")
        return payload

    def _decode(self, stored: str) -> str:
        return self._open(stored)["secret"]

    def owner_of(self, reference: str) -> str | None:
        owner = self._open(self._path(reference).read_text(encoding="utf-8")).get("owner")
        return owner if isinstance(owner, str) else None

    def get_for(self, reference: str, owner_id: str) -> str:
        """Read a secret only if it was written for ``owner_id``.

        A file written by the plaintext store before this deployment had a key
        is read once more as plaintext and re-encrypted for the owner whose row
        points at it (the caller has already proved that ownership), so turning
        encryption on does not make every user re-enter their keys.
        """
        stored = self._path(reference).read_text(encoding="utf-8")
        if not stored.startswith(_FERNET_PREFIX):
            if not stored:
                raise SecretUnavailable("a stored credential is empty")
            self.replace(reference, owner_id, stored)
            return stored
        payload = self._open(stored)
        if payload.get("owner") != owner_id:
            raise SecretUnavailable("a stored credential belongs to another owner")
        return payload["secret"]


def build_secret_store(root: Path, encryption_key: str = "") -> FilesystemSecretStore:
    """The store a deployment's settings ask for."""
    if encryption_key:
        return EncryptedSecretStore(root, encryption_key)
    return FilesystemSecretStore(root)


def read_for_owner(store: SecretStore, reference: str, owner_id: str) -> str:
    """Read a secret, proving ownership when the store can.

    The plaintext store cannot prove it, and relies on the row's owner check
    that every caller already performs.
    """
    getter = getattr(store, "get_for", None)
    if callable(getter):
        return getter(reference, owner_id)
    return store.get(reference)
