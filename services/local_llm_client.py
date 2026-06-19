from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import httpx


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_prompt(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(contents)


def _json_instruction(response_schema: Any) -> str:
    if response_schema:
        try:
            schema_text = json.dumps(response_schema, ensure_ascii=True)
        except Exception:
            schema_text = "{}"
        return "Return valid JSON only. Schema: " + schema_text
    return "Return valid JSON only."


def _apply_json_instruction(prompt: str, response_mime_type: Optional[str], response_schema: Any) -> str:
    if (response_mime_type or "").strip().lower() == "application/json":
        return prompt + "\n\n" + _json_instruction(response_schema)
    return prompt


@dataclass
class LocalLLMResponse:
    text: str


class _LocalModelProxy:
    def __init__(self, client: "LocalLLMClient") -> None:
        self._client = client

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> LocalLLMResponse:
        return self._client._generate(model=model, contents=contents, config=config)

    def generate_content_stream(self, *, model: str, contents: Any, config: Any = None) -> Generator[LocalLLMResponse, None, None]:
        response = self._client._generate(model=model, contents=contents, config=config)
        yield response


class LocalLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        provider: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = (provider or "ollama").strip().lower()
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.models = _LocalModelProxy(self)

    def _generate(self, *, model: str, contents: Any, config: Any = None) -> LocalLLMResponse:
        prompt = _normalize_prompt(contents)
        response_mime_type = _get_config_value(config, "response_mime_type")
        response_schema = _get_config_value(config, "response_schema")
        temperature = _get_config_value(config, "temperature")
        max_output_tokens = _get_config_value(config, "max_output_tokens")
        if max_output_tokens is None:
            max_output_tokens = _get_config_value(config, "max_tokens")

        prompt = _apply_json_instruction(prompt, response_mime_type, response_schema)

        if self.provider == "ollama":
            return self._call_ollama(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type=response_mime_type,
            )
        if self.provider in {"openai", "openai-compatible", "lmstudio", "vllm"}:
            return self._call_openai_compatible(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type=response_mime_type,
            )

        raise ValueError(f"Unsupported LOCAL_LLM_PROVIDER: {self.provider}")

    def _call_ollama(
        self,
        *,
        model: str,
        prompt: str,
        temperature: Any,
        max_output_tokens: Any,
        response_mime_type: Optional[str],
    ) -> LocalLLMResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_output_tokens is not None:
            options["num_predict"] = int(max_output_tokens)
        if options:
            payload["options"] = options
        if (response_mime_type or "").strip().lower() == "application/json":
            payload["format"] = "json"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

        return LocalLLMResponse(text=str(data.get("response") or "").strip())

    def _call_openai_compatible(
        self,
        *,
        model: str,
        prompt: str,
        temperature: Any,
        max_output_tokens: Any,
        response_mime_type: Optional[str],
    ) -> LocalLLMResponse:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)
        if (response_mime_type or "").strip().lower() == "application/json":
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code == 400 and "response_format" in payload:
                payload.pop("response_format", None)
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()

        message = None
        try:
            message = data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            message = None
        if message is None:
            message = data.get("choices", [{}])[0].get("text")

        return LocalLLMResponse(text=str(message or "").strip())


def is_local_llm_configured() -> bool:
    return bool(
        (os.getenv("LOCAL_LLM_BASE_URL") or "").strip()
        or (os.getenv("LOCAL_LLM_PROVIDER") or "").strip()
        or (os.getenv("LOCAL_LLM_MODEL") or "").strip()
    )


def _default_base_url(provider: str) -> str:
    if provider == "ollama":
        return "http://127.0.0.1:11434"
    return "http://127.0.0.1:1234"


def build_local_llm_client_from_env() -> Optional[LocalLLMClient]:
    if not is_local_llm_configured():
        return None

    provider = (os.getenv("LOCAL_LLM_PROVIDER") or "ollama").strip().lower()
    base_url = (os.getenv("LOCAL_LLM_BASE_URL") or "").strip()
    if not base_url:
        base_url = _default_base_url(provider)

    timeout_raw = (os.getenv("LOCAL_LLM_TIMEOUT") or "90").strip()
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 90.0

    if not base_url:
        return None

    return LocalLLMClient(
        base_url=base_url,
        provider=provider,
        api_key=(os.getenv("LOCAL_LLM_API_KEY") or "").strip() or None,
        timeout_seconds=timeout_seconds,
    )
