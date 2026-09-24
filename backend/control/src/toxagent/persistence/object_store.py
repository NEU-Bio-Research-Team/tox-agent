"""Object store interface (plan sections 13.1/17, remaining-plan W4-06).

Byte storage is a boundary like the predictor or the runtime host: the
application depends on this ``Protocol`` only, never a cloud SDK directly.
``AttachmentStore`` (interfaces.py) persists the *metadata* row an
``Attachment`` is (owner, media type, hash, retention class) — this module
owns the *bytes* that row's ``object_uri`` points at. Neither implementation
here is meant for production: ``InMemoryObjectStore`` is for unit tests,
``FilesystemObjectStore`` for local/dev runs and integration tests that need
bytes to actually survive a process restart. A GCS adapter (deployment here
is GCP) is real future work, deliberately not built yet — writing one against
no real bucket/credentials would be exactly the kind of "code no one can
verify" this project's own discipline (progress log section 4.8, DSH) refuses
to check in.

``ObjectRef`` is opaque on purpose: a ``key`` string, never a URL or a
credential. Turning one into something directly fetchable is
``signed_read_ref``'s job alone, and remaining-plan W4-09 restricts even that
to an authorised (auditor-role) reader — nothing here hands raw storage
access to a model or an ordinary API caller.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


class ObjectNotFound(Exception):
    """No object exists at this ref — already deleted, or never written."""


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """An opaque handle into the object store. Two refs are the same object
    iff their ``key`` is equal; nothing about the key's shape is a contract
    an application caller may rely on (a real adapter may prefix it with a
    bucket path, a hash-sharded directory, or anything else it needs)."""

    key: str


@runtime_checkable
class ObjectStore(Protocol):
    async def put(self, key: str, data: bytes, *, content_type: str) -> ObjectRef:
        """Write ``data`` at ``key``, overwriting whatever was there before —
        callers that want immutability get it by using a content-addressed or
        otherwise unique key, not from a guarantee this method makes."""
        ...

    async def get(self, ref: ObjectRef) -> bytes:
        """Raises ``ObjectNotFound`` if nothing exists at ``ref``."""
        ...

    async def delete(self, ref: ObjectRef) -> None:
        """Idempotent — deleting an already-absent object is not an error,
        the same discipline remaining-plan W4-10's TTL cleanup needs (a
        cleanup pass that already partially succeeded must be safe to
        retry)."""
        ...

    async def signed_read_ref(self, ref: ObjectRef, *, ttl_s: int) -> str:
        """A time-limited, directly-fetchable URL for this object. Raises
        ``ObjectNotFound`` if nothing exists at ``ref``. Callers decide who
        is authorised to ask for one — this method itself enforces no ACL,
        the same way a store never decides who may call ``get``."""
        ...


class InMemoryObjectStore:
    """For unit tests: no filesystem, no cleanup between test runs beyond
    whatever the test's own fixture teardown does. Bytes live only as long as
    this instance does."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}

    async def put(self, key: str, data: bytes, *, content_type: str) -> ObjectRef:
        self._objects[key] = (data, content_type)
        return ObjectRef(key=key)

    async def get(self, ref: ObjectRef) -> bytes:
        try:
            return self._objects[ref.key][0]
        except KeyError:
            raise ObjectNotFound(ref.key) from None

    async def delete(self, ref: ObjectRef) -> None:
        self._objects.pop(ref.key, None)

    async def signed_read_ref(self, ref: ObjectRef, *, ttl_s: int) -> str:
        if ref.key not in self._objects:
            raise ObjectNotFound(ref.key)
        # Not a real signed URL — there is nothing on a real network to sign
        # a link to. Shaped like one only so a caller exercising the
        # contract (e.g. "does this look like a URL, is it not the bare key")
        # sees something realistic.
        return f"memory://{ref.key}?ttl={ttl_s}"

    def object_count(self) -> int:
        """Test-only introspection — not part of the ObjectStore protocol."""
        return len(self._objects)


class FilesystemObjectStore:
    """For local/dev runs and integration tests: bytes survive a process
    restart, unlike ``InMemoryObjectStore``. Not a production adapter — no
    encryption at rest, no access control beyond the host's own filesystem
    permissions, and ``signed_read_ref`` returns a ``file://`` URI that is
    not actually time-limited (there is no server to expire it), clearly
    distinguishable from a real signed URL as a reminder not to treat it as
    one."""

    def __init__(self, base_dir: Path | str) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # A key may carry "/" as a namespacing convention (e.g.
        # "evidence/2026/09/06/<hash>"); resolve it under base_dir and refuse
        # anything that would escape it — a key is caller-controlled data,
        # never a trusted filesystem path.
        candidate = (self._base_dir / key).resolve()
        if self._base_dir.resolve() not in candidate.parents and candidate != self._base_dir.resolve():
            raise ValueError(f"object key {key!r} escapes the store's base directory")
        return candidate

    async def put(self, key: str, data: bytes, *, content_type: str) -> ObjectRef:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        (path.parent / f".{path.name}.content-type").write_text(content_type)
        return ObjectRef(key=key)

    async def get(self, ref: ObjectRef) -> bytes:
        path = self._path_for(ref.key)
        if not path.is_file():
            raise ObjectNotFound(ref.key)
        return path.read_bytes()

    async def delete(self, ref: ObjectRef) -> None:
        path = self._path_for(ref.key)
        path.unlink(missing_ok=True)
        content_type_path = path.parent / f".{path.name}.content-type"
        content_type_path.unlink(missing_ok=True)

    async def signed_read_ref(self, ref: ObjectRef, *, ttl_s: int) -> str:
        path = self._path_for(ref.key)
        if not path.is_file():
            raise ObjectNotFound(ref.key)
        return path.as_uri()

    def close(self) -> None:
        """Test-only teardown helper — removes the entire base directory.
        Not part of the ObjectStore protocol; a real deployment never deletes
        its whole bucket on shutdown."""
        shutil.rmtree(self._base_dir, ignore_errors=True)
