"""ObjectStore contract (plan sections 13.1/17, remaining-plan W4-06).

Both implementations must satisfy the exact same behaviour — parametrized so
a future GCS adapter (not built yet; see persistence/object_store.py's
module docstring for why) has a ready-made contract suite to run against,
the same discipline already applied to the predictor client and the
OpenCode adapter.
"""
from __future__ import annotations

import pytest

from toxagent.persistence.object_store import (
    FilesystemObjectStore,
    InMemoryObjectStore,
    ObjectNotFound,
    ObjectRef,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "filesystem"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryObjectStore()
        return
    fs_store = FilesystemObjectStore(tmp_path / "objects")
    yield fs_store
    fs_store.close()


async def test_a_put_object_is_readable_back_byte_for_byte(store):
    ref = await store.put("evidence/one", b"hello world", content_type="text/plain")
    assert await store.get(ref) == b"hello world"


async def test_getting_a_ref_that_was_never_written_raises_not_found(store):
    with pytest.raises(ObjectNotFound):
        await store.get(ObjectRef(key="never/written"))


async def test_put_overwrites_whatever_was_at_the_same_key(store):
    ref = await store.put("k", b"first", content_type="text/plain")
    ref2 = await store.put("k", b"second", content_type="text/plain")
    assert ref == ref2
    assert await store.get(ref) == b"second"


async def test_delete_removes_the_object(store):
    ref = await store.put("k", b"data", content_type="text/plain")
    await store.delete(ref)
    with pytest.raises(ObjectNotFound):
        await store.get(ref)


async def test_deleting_an_absent_object_is_not_an_error(store):
    """Idempotent — remaining-plan W4-10's TTL cleanup must be safe to retry
    after a partial failure, so a second delete of the same ref cannot
    raise."""
    await store.delete(ObjectRef(key="was/never/here"))


async def test_signed_read_ref_of_a_missing_object_raises_not_found(store):
    with pytest.raises(ObjectNotFound):
        await store.signed_read_ref(ObjectRef(key="missing"), ttl_s=60)


async def test_signed_read_ref_of_a_real_object_looks_like_a_url(store):
    ref = await store.put("k", b"data", content_type="application/octet-stream")
    signed = await store.signed_read_ref(ref, ttl_s=60)
    assert "://" in signed


async def test_a_key_with_slashes_round_trips(store):
    """Namespacing convention (e.g. "evidence/<year>/<month>/<day>/<hash>")
    must not be mistaken for a filesystem path escape."""
    ref = await store.put("a/b/c/d", b"nested", content_type="text/plain")
    assert await store.get(ref) == b"nested"


async def test_a_key_cannot_escape_the_filesystem_stores_base_directory(tmp_path):
    """Filesystem-specific: a key is caller-controlled data, never a trusted
    path — ".." must not be able to write outside base_dir."""
    fs_store = FilesystemObjectStore(tmp_path / "objects")
    try:
        with pytest.raises(ValueError):
            await fs_store.put("../../escaped", b"data", content_type="text/plain")
    finally:
        fs_store.close()


async def test_filesystem_store_survives_a_fresh_instance_pointed_at_the_same_directory(tmp_path):
    """The point of this adapter over InMemoryObjectStore: bytes outlive the
    process that wrote them, standing in for a real process restart."""
    base_dir = tmp_path / "objects"
    first = FilesystemObjectStore(base_dir)
    ref = await first.put("k", b"persisted", content_type="text/plain")

    second = FilesystemObjectStore(base_dir)
    try:
        assert await second.get(ref) == b"persisted"
    finally:
        second.close()
