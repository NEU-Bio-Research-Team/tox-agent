from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable, List, Sequence, Tuple, TypeVar

try:
    from google import genai
except Exception:
    genai = None

try:
    from .local_llm_client import build_local_llm_client_from_env, is_local_llm_configured
except Exception:
    build_local_llm_client_from_env = None
    is_local_llm_configured = None


LOG = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_VERTEX_LOCATIONS: Tuple[str, ...] = (
    "global",
    "us-central1",
    "us-east1",
    "us-west1",
    "europe-west1",
    "europe-west4",
    "asia-southeast1",
)


def dedupe_strings(values: Sequence[str | None]) -> List[str]:
    seen: List[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(item.lower() == lowered for item in seen):
            continue
        seen.append(cleaned)
    return seen


def _resolve_project_id() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or ""
    ).strip()


def ordered_vertex_locations(preferred: str | None = None) -> List[str]:
    configured = dedupe_strings(
        item.strip()
        for item in os.getenv("GENAI_VERTEX_LOCATIONS", "").split(",")
        if item.strip()
    )
    if not configured:
        configured = list(DEFAULT_VERTEX_LOCATIONS)

    preferred_location = (
        preferred
        or os.getenv("GEMINI_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or os.getenv("GOOGLE_CLOUD_ZONE")
        or "global"
    ).strip()

    return dedupe_strings([preferred_location, *configured])


def build_genai_client_candidates(location_override: str | None = None) -> List[Tuple[Any, str]]:
    candidates: List[Tuple[Any, str]] = []
    local_only = str(os.getenv("LOCAL_LLM_ONLY", "")).strip().lower() in {"1", "true", "yes"}

    if build_local_llm_client_from_env is not None:
        if is_local_llm_configured is None or is_local_llm_configured():
            local_client = build_local_llm_client_from_env()
            if local_client is not None:
                provider = getattr(local_client, "provider", "local")
                candidates.append((local_client, f"local:{provider}"))
                if local_only:
                    return candidates

    if genai is None:
        return candidates

    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if api_key:
        try:
            candidates.append((genai.Client(api_key=api_key), "api_key"))
            return candidates
        except Exception as exc:
            LOG.warning("GenAI API key client init failed: %s", exc)

    project_id = _resolve_project_id()
    if project_id:
        for location in ordered_vertex_locations(location_override):
            try:
                candidates.append(
                    (
                        genai.Client(
                            vertexai=True,
                            project=project_id,
                            location=location,
                        ),
                        f"vertex_adc:{location}",
                    )
                )
            except Exception as exc:
                LOG.warning("GenAI Vertex client init failed for location=%s: %s", location, exc)

    if candidates:
        return candidates

    try:
        candidates.append((genai.Client(), "default"))
        return candidates
    except Exception as exc:
        LOG.warning("Fallback GenAI client init failed: %s", exc)
        return candidates


def is_resource_exhausted_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(
        token in text
        for token in (
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "too many requests",
            "serverbusy",
            "server busy",
            "pugrest.serverbusy",
        )
    )


def is_model_unavailable_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        ("404" in text or "not found" in text)
        and any(
            token in text
            for token in (
                "model",
                "publisher",
                "location",
                "vertex",
            )
        )
    )


def is_transient_genai_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        is_resource_exhausted_error(exc)
        or "503" in text
        or "unavailable" in text
        or "deadline exceeded" in text
        or "timed out" in text
        or "timeout" in text
    )


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    base_delay_seconds: float = 1.25,
) -> T:
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries - 1 or not is_transient_genai_error(exc):
                raise

            delay = min(base_delay_seconds * (2**attempt) + random.uniform(0.0, 0.5), 10.0)
            LOG.warning(
                "Transient GenAI error on attempt %s/%s; retrying in %.2fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("genai_retry_exhausted")