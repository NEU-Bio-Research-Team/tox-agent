"""Secret references; credentials never enter product rows or transcripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class SecretStore(Protocol):
    def put(self, owner_id: str, secret: str) -> str: ...
    def get(self, reference: str) -> str: ...
    def delete(self, reference: str) -> None: ...


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

    def put(self, owner_id: str, secret: str) -> str:
        if not owner_id or not secret:
            raise ValueError("owner and non-empty secret are required")
        reference = f"secret_{uuid4().hex}"
        path = self._path(reference)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secret)
        return reference

    def get(self, reference: str) -> str:
        return self._path(reference).read_text(encoding="utf-8")

    def delete(self, reference: str) -> None:
        try:
            self._path(reference).unlink()
        except FileNotFoundError:
            pass
