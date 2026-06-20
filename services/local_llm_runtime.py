from __future__ import annotations

import logging
import os
import json
import requests
from typing import Any, Dict, List, Generator, Optional, Sequence

LOG = logging.getLogger(__name__)

# Default configurations for local serving layer (vLLM)
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1").strip()
LOCAL_LLM_MODEL_FAST = os.getenv("LOCAL_LLM_MODEL_FAST", "model-fast").strip()
LOCAL_LLM_MODEL_PRO = os.getenv("LOCAL_LLM_MODEL_PRO", "model-reasoning").strip()

class LocalResponseChunk:
    """Mock Gemini response chunk for streaming support."""
    def __init__(self, text: str):
        self.text = text

class LocalResponse:
    """Mock Gemini response object for standard completions."""
    def __init__(self, text: str, function_calls: list = None):
        self.text = text
        self._function_calls = function_calls or []

    @property
    def function_calls(self):
        return self._function_calls

    def get_function_calls(self):
        return self._function_calls

def _get_gemini_client() -> Any:
    """Initialize a real google-genai client candidate."""
    try:
        from google import genai
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if api_key:
            return genai.Client(api_key=api_key)
        
        project_id = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
            or ""
        ).strip()
        if project_id:
            location = os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
            return genai.Client(vertexai=True, project=project_id, location=location)
        
        return genai.Client()
    except Exception as exc:
        LOG.warning("Failed to initialize Gemini fallback client: %s", exc)
        return None

class LocalLLMModels:
    """Mock client.models layer translating Gemini SDK calls to local OpenAI API."""
    def __init__(self):
        pass

    def _resolve_model(self, model_name: str) -> str:
        """Map Gemini model names or fast/pro abstract roles to local model names."""
        name = str(model_name or "").strip().lower()
        if not name:
            return LOCAL_LLM_MODEL_FAST

        if "fast" in name or "flash" in name or name in ("gemini-2.5-flash", "gemini-2.0-flash"):
            return LOCAL_LLM_MODEL_FAST
        if "pro" in name or "reasoning" in name or "writer" in name or "molrag" in name or name in ("gemini-2.5-pro", "gemini-2.0-pro", "mistral"):
            return LOCAL_LLM_MODEL_PRO

        if model_name == os.getenv("GEMINI_MODEL"):
            return LOCAL_LLM_MODEL_FAST

        return model_name

    def _convert_contents_to_messages(
        self,
        contents: Any,
        system_instruction: Any = None
    ) -> List[Dict[str, str]]:
        """Convert Gemini-style contents/prompts to OpenAI-style messages list."""
        messages: List[Dict[str, str]] = []

        if system_instruction:
            system_text = ""
            if isinstance(system_instruction, str):
                system_text = system_instruction
            elif hasattr(system_instruction, "text"):
                system_text = getattr(system_instruction, "text")
            elif isinstance(system_instruction, list):
                system_text = " ".join(
                    getattr(part, "text", str(part))
                    for part in system_instruction
                )
            elif isinstance(system_instruction, dict):
                parts = system_instruction.get("parts", [])
                if isinstance(parts, list):
                    system_text = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in parts
                    )
                else:
                    system_text = system_instruction.get("text", str(system_instruction))

            if system_text:
                messages.append({"role": "system", "content": system_text})

        if not contents:
            return messages

        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
            return messages

        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if "role" in item and "content" in item:
                        messages.append(item)
                    elif "parts" in item:
                        role = item.get("role", "user")
                        if role == "model":
                            role = "assistant"
                        parts = item.get("parts")
                        parts_text = ""
                        if isinstance(parts, list):
                            parts_text = " ".join(
                                p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in parts
                            )
                        elif isinstance(parts, str):
                            parts_text = parts
                        messages.append({"role": role, "content": parts_text})
                elif hasattr(item, "role") and hasattr(item, "parts"):
                    role = getattr(item, "role", "user")
                    if role == "model":
                        role = "assistant"
                    parts = getattr(item, "parts", [])
                    parts_text = ""
                    if isinstance(parts, list):
                        parts_text = " ".join(
                            getattr(p, "text", str(p))
                            for p in parts
                        )
                    else:
                        parts_text = str(parts)
                    messages.append({"role": role, "content": parts_text})
                elif hasattr(item, "text"):
                    messages.append({"role": "user", "content": getattr(item, "text")})
                else:
                    messages.append({"role": "user", "content": str(item)})

        return messages

    def generate_content(
        self,
        model: str,
        contents: Any,
        config: Any = None
    ) -> LocalResponse:
        """Simulate generate_content by forwarding request to local vLLM server, with optional Gemini fallback."""
        llm_runtime = (os.getenv("LLM_RUNTIME") or "gemini").strip().lower()
        
        # In auto mode, try local first, fallback on error
        if llm_runtime in ("local", "auto"):
            try:
                resolved_model = self._resolve_model(model)
                
                temperature = 0.2
                response_mime_type = None
                response_schema = None
                system_instruction = None

                if config:
                    if isinstance(config, dict):
                        temperature = config.get("temperature", temperature)
                        response_mime_type = config.get("response_mime_type")
                        response_schema = config.get("response_schema")
                        system_instruction = config.get("system_instruction")
                    else:
                        temperature = getattr(config, "temperature", temperature) or temperature
                        response_mime_type = getattr(config, "response_mime_type", None)
                        response_schema = getattr(config, "response_schema", None)
                        system_instruction = getattr(config, "system_instruction", None)

                messages = self._convert_contents_to_messages(contents, system_instruction)

                payload: Dict[str, Any] = {
                    "model": resolved_model,
                    "messages": messages,
                    "temperature": float(temperature)
                }

                if response_schema:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response_schema",
                            "strict": True,
                            "schema": response_schema
                        }
                    }
                elif response_mime_type == "application/json":
                    payload["response_format"] = {"type": "json_object"}

                url = f"{LOCAL_LLM_BASE_URL}/chat/completions"
                LOG.info("Calling local LLM at %s for model=%s", url, resolved_model)

                resp = requests.post(url, json=payload, timeout=240.0)
                resp.raise_for_status()
                data = resp.json()
                generated_text = data["choices"][0]["message"]["content"]
                return LocalResponse(text=generated_text)
            except Exception as exc:
                if llm_runtime == "auto":
                    LOG.warning("Local LLM failed; falling back to Gemini API... Error: %s", exc)
                else:
                    LOG.error("Local LLM request failed: %s", exc)
                    raise exc

        # Gemini API fallback
        client = _get_gemini_client()
        if client is None:
            raise RuntimeError("Gemini fallback client not initialized")
        
        LOG.info("Delegating call to Gemini API for model=%s", model)
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

    def generate_content_stream(
        self,
        model: str,
        contents: Any,
        config: Any = None
    ) -> Generator[LocalResponseChunk, None, None]:
        """Simulate generate_content_stream for local vLLM server, with optional Gemini fallback."""
        llm_runtime = (os.getenv("LLM_RUNTIME") or "gemini").strip().lower()
        
        if llm_runtime in ("local", "auto"):
            try:
                resolved_model = self._resolve_model(model)
                
                temperature = 0.2
                system_instruction = None

                if config:
                    if isinstance(config, dict):
                        temperature = config.get("temperature", temperature)
                        system_instruction = config.get("system_instruction")
                    else:
                        temperature = getattr(config, "temperature", temperature) or temperature
                        system_instruction = getattr(config, "system_instruction", None)

                messages = self._convert_contents_to_messages(contents, system_instruction)

                payload = {
                    "model": resolved_model,
                    "messages": messages,
                    "temperature": float(temperature),
                    "stream": True
                }

                url = f"{LOCAL_LLM_BASE_URL}/chat/completions"
                LOG.info("Calling streaming local LLM at %s for model=%s", url, resolved_model)

                resp = requests.post(url, json=payload, stream=True, timeout=240.0)
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            delta = chunk_data["choices"][0]["delta"]
                            if "content" in delta:
                                yield LocalResponseChunk(text=delta["content"])
                        except Exception:
                            continue
                return
            except Exception as exc:
                if llm_runtime == "auto":
                    LOG.warning("Local streaming LLM failed; falling back to Gemini API... Error: %s", exc)
                else:
                    LOG.error("Local streaming LLM failed: %s", exc)
                    raise exc

        # Gemini API fallback
        client = _get_gemini_client()
        if client is None:
            raise RuntimeError("Gemini fallback client not initialized")
        
        LOG.info("Delegating stream to Gemini API for model=%s", model)
        yield from client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config
        )

class LocalLLMClient:
    """Mock google.genai.Client wrapping local OpenAI compatible API server."""
    def __init__(self):
        self.models = LocalLLMModels()

def build_local_client() -> LocalLLMClient:
    """Build and return local GenAI client candidate."""
    return LocalLLMClient()
