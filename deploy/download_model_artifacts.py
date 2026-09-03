from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.getenv("MODELS_ROOT") or (PROJECT_ROOT / "models")).expanduser()
MODEL_ARTIFACTS_URI = str(os.getenv("MODEL_ARTIFACTS_URI") or "").strip()
FORCE_DOWNLOAD = str(os.getenv("MODEL_ARTIFACTS_FORCE_DOWNLOAD") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _parse_storage_uri(uri: str) -> Tuple[str, str, str]:
    if "://" not in uri:
        raise ValueError(f"Invalid artifact URI: {uri}")
    scheme, remainder = uri.split("://", 1)
    bucket, _, prefix = remainder.partition("/")
    if not scheme or not bucket:
        raise ValueError(f"Invalid artifact URI: {uri}")
    return scheme.lower(), bucket, prefix.strip("/")


def _has_local_artifacts(root: Path) -> bool:
    if not root.exists():
        return False
    for child in root.iterdir():
        if child.name.startswith("."):
            continue
        return True
    return False


def _destination_path(root: Path, object_name: str, prefix: str) -> Optional[Path]:
    normalized_name = object_name.strip("/")
    if not normalized_name:
        return None
    normalized_prefix = prefix.strip("/")
    if normalized_prefix:
        prefix_with_slash = normalized_prefix + "/"
        if normalized_name == normalized_prefix:
            return None
        if normalized_name.startswith(prefix_with_slash):
            relative_name = normalized_name[len(prefix_with_slash) :]
        else:
            relative_name = Path(normalized_name).name
    else:
        relative_name = normalized_name
    if not relative_name:
        return None
    return root / relative_name


def _should_skip_download(destination: Path, expected_size: Optional[int]) -> bool:
    if FORCE_DOWNLOAD or not destination.exists():
        return False
    if expected_size is None:
        return True
    try:
        return destination.stat().st_size == int(expected_size)
    except OSError:
        return False


def _download_from_gcs(bucket: str, prefix: str, root: Path) -> int:
    from google.cloud import storage

    client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
    downloaded = 0
    for blob in client.list_blobs(bucket, prefix=prefix or None):
        if blob.name.endswith("/"):
            continue
        destination = _destination_path(root, blob.name, prefix)
        if destination is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _should_skip_download(destination, blob.size):
            print(f"[model-artifacts] cached: {destination}", flush=True)
            continue
        print(f"[model-artifacts] downloading gs://{bucket}/{blob.name} -> {destination}", flush=True)
        blob.download_to_filename(str(destination))
        downloaded += 1
    return downloaded


def _iter_s3_objects(client: any, bucket: str, prefix: str) -> Iterable[dict]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            yield item


def _download_from_s3(bucket: str, prefix: str, root: Path) -> int:
    import boto3

    client = boto3.client("s3")
    downloaded = 0
    for item in _iter_s3_objects(client, bucket, prefix):
        key = str(item.get("Key") or "")
        if key.endswith("/"):
            continue
        destination = _destination_path(root, key, prefix)
        if destination is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _should_skip_download(destination, item.get("Size")):
            print(f"[model-artifacts] cached: {destination}", flush=True)
            continue
        print(f"[model-artifacts] downloading s3://{bucket}/{key} -> {destination}", flush=True)
        client.download_file(bucket, key, str(destination))
        downloaded += 1
    return downloaded


def main() -> int:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    if not MODEL_ARTIFACTS_URI:
        if _has_local_artifacts(MODELS_ROOT):
            print(f"[model-artifacts] using local artifacts at {MODELS_ROOT}", flush=True)
        else:
            print("[model-artifacts] MODEL_ARTIFACTS_URI not set; skipping remote sync", flush=True)
        return 0

    try:
        scheme, bucket, prefix = _parse_storage_uri(MODEL_ARTIFACTS_URI)
    except Exception as exc:
        print(f"[model-artifacts] invalid MODEL_ARTIFACTS_URI: {exc}", flush=True)
        return 0

    try:
        if scheme == "gs":
            downloaded = _download_from_gcs(bucket, prefix, MODELS_ROOT)
        elif scheme == "s3":
            downloaded = _download_from_s3(bucket, prefix, MODELS_ROOT)
        else:
            print(f"[model-artifacts] unsupported URI scheme: {scheme}", flush=True)
            return 0
    except Exception as exc:
        print(f"[model-artifacts] remote sync failed: {type(exc).__name__}: {exc}", flush=True)
        return 0

    if downloaded == 0:
        print(f"[model-artifacts] no files downloaded from {MODEL_ARTIFACTS_URI}", flush=True)
    else:
        print(f"[model-artifacts] downloaded {downloaded} files into {MODELS_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())