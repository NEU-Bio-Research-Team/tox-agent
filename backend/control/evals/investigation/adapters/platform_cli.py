"""General platforms through their command-line clients, with no tools.

* OpenAI models through ``codex exec``: read-only sandbox, an empty temporary
  working directory, no session persisted. Codex keeps its own agent
  instructions (the CLI cannot replace them); the neutral preamble leads the
  prompt. The model and provider come from the user's Codex configuration and
  are recorded as configured and as reported in the event stream.
* Anthropic models through ``claude -p``: every built-in tool disabled, no MCP
  servers, no settings files, no session persisted, the neutral preamble as the
  system prompt, run in an empty temporary directory. The model ids are the
  ones the CLI's JSON result reports it used.

Both arms answer from the model's own knowledge. That is a stated property of
these arms, not an accident: a researcher's chat window may search the web,
but a search whose results nobody logged cannot be graded or reproduced.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from evals.investigation import prompts
from evals.investigation.adapters import AdapterResult
from evals.investigation.record import STATUS_ERROR, STATUS_OK, StudyStore, TurnRecord, now_iso
from evals.investigation.systems import SystemSpec

TIMEOUT_S = 900.0


async def _run_process(command: list[str], stdin: str, *, cwd: Path, timeout: float = TIMEOUT_S):
    process = await asyncio.create_subprocess_exec(
        *command, cwd=str(cwd), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin.encode("utf-8")), timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError(f"{command[0]} did not finish in {timeout:.0f}s") from None
    return process.returncode, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


async def _version(binary: str) -> str:
    if shutil.which(binary) is None:
        return "not installed"
    try:
        code, out, err = await _run_process([binary, "--version"], "", cwd=Path(tempfile.gettempdir()),
                                            timeout=30)
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({type(exc).__name__})"
    return (out or err).strip().splitlines()[0] if (out or err).strip() else "unknown"


def codex_configuration(path: Path | None = None) -> dict[str, Any]:
    """The non-secret top-level Codex settings that decide which model answers."""
    path = path or Path.home() / ".codex" / "config.toml"
    if not path.exists():
        return {}
    wanted = {"model", "provider", "model_provider", "model_reasoning_effort", "service_tier"}
    found: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if line.startswith("["):
            break  # top-level keys only; tables hold endpoints and credentials
        match = re.match(r'^\s*([A-Za-z_]+)\s*=\s*"?([^"#]*)"?', line)
        if match and match.group(1) in wanted:
            found[match.group(1)] = match.group(2).strip()
    return found


class _PlatformAdapter:
    binary = ""

    def __init__(self) -> None:
        self._version: str | None = None

    async def prepare(self) -> None:
        self._version = await _version(self.binary)

    def describe(self) -> dict[str, Any]:
        return {"adapter": type(self).__name__, "binary": self.binary, "version": self._version,
                "prompt_version": prompts.PROMPT_VERSION, "preamble_sha256": prompts.preamble_sha256()}

    async def ask(self, prompt: str, workdir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    async def run(self, case: dict[str, Any], spec: SystemSpec, *, trial: int,
                  snapshot: dict[str, Any] | None, store: StudyStore) -> AdapterResult:
        if spec.uses_snapshot and not snapshot:
            return AdapterResult(status=STATUS_ERROR, turns=[],
                                 error="this arm needs the predictor snapshot and none was produced")
        turns: list[TurnRecord] = []
        responses: list[str] = []
        models: list[dict[str, Any]] = []
        usage_total: dict[str, float] = {}
        raw: list[dict[str, Any]] = []
        for index, turn in enumerate(case["turns"]):
            prompt = prompts.render(case["turns"], index, responses,
                                    snapshot=snapshot if spec.uses_snapshot else None)
            started, clock = now_iso(), time.monotonic()
            with tempfile.TemporaryDirectory(prefix="toxagent-study-") as workdir:
                try:
                    text, model, extra = await self.ask(prompt, Path(workdir))
                except Exception as exc:  # noqa: BLE001 - recorded
                    store.write_raw(spec.system_id, case["case_id"], trial, "raw.json", raw)
                    return AdapterResult(status=STATUS_ERROR, turns=turns,
                                         error=f"{type(exc).__name__}: {exc}")
            duration = round(time.monotonic() - clock, 3)
            responses.append(text)
            models.append(model)
            raw.append({"turn": index, "prompt": prompt, **extra})
            for key, value in (extra.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage_total[key] = usage_total.get(key, 0) + value
            turns.append(TurnRecord(index=index, user_text=turn["text"], sent_text=prompt,
                                    response_text=text, started_at=started, ended_at=now_iso(),
                                    duration_s=duration, meta={"model": model}))
        artifact = store.write_raw(spec.system_id, case["case_id"], trial, "raw.json", raw)
        return AdapterResult(
            status=STATUS_OK, turns=turns, final_text=responses[-1] if responses else "",
            model={"adapter_version": self._version, **_merge_models(models)},
            usage={"status": "reported" if usage_total else "unknown", **usage_total},
            artifacts={"raw": artifact},
        )


def _merge_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = sorted({m.get("model_id_resolved") for m in models if m.get("model_id_resolved")})
    merged = dict(models[0]) if models else {}
    merged["model_id_resolved"] = resolved[0] if len(resolved) == 1 else (resolved or None)
    return merged


class CodexCLIAdapter(_PlatformAdapter):
    binary = "codex"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "configuration": codex_configuration(),
                "tool_policy": "read-only sandbox in an empty temporary directory; ephemeral"}

    async def ask(self, prompt: str, workdir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
        last = workdir / "last-message.txt"
        command = [
            "codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
            "--sandbox", "read-only", "--color", "never", "--json",
            "--cd", str(workdir), "--output-last-message", str(last), "-",
        ]
        code, stdout, stderr = await _run_process(command, prompt, cwd=workdir)
        events = [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]
        if code != 0 or not last.exists():
            raise RuntimeError(f"codex exited {code}: {stderr.strip()[-500:]}")
        text = last.read_text(encoding="utf-8").strip()
        configured = codex_configuration()
        reported = sorted({
            str(value) for event in events for key, value in _walk(event)
            if key in ("model", "model_id", "model_slug") and isinstance(value, str)
        })
        usage: dict[str, float] = {}
        for event in events:
            for key, value in _walk(event):
                if key in ("input_tokens", "output_tokens", "cached_input_tokens",
                           "reasoning_output_tokens") and isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        model = {
            "provider": configured.get("provider") or configured.get("model_provider"),
            "model_id_requested": None,
            "model_id_configured": configured.get("model"),
            "model_id_reported": reported or None,
            "model_id_resolved": (reported[0] if len(reported) == 1 else configured.get("model")),
            "reasoning_effort": configured.get("model_reasoning_effort"),
        }
        return text, model, {"events": events, "stderr_tail": stderr[-2000:], "usage": usage,
                             "exit_code": code}


class ClaudeCLIAdapter(_PlatformAdapter):
    binary = "claude"

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self._model = model

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model_requested": self._model,
                "tool_policy": "--tools '' --strict-mcp-config --setting-sources '' in an empty "
                               "temporary directory; no session persisted"}

    async def ask(self, prompt: str, workdir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
        command = [
            "claude", "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
            "--setting-sources", "", "--no-session-persistence",
            "--system-prompt", prompts.PREAMBLE,
        ]
        if self._model:
            command += ["--model", self._model]
        code, stdout, stderr = await _run_process(command, prompt, cwd=workdir)
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude exited {code} without JSON: {stderr.strip()[-500:]}") from None
        if code != 0 or result.get("is_error"):
            raise RuntimeError(f"claude exited {code}: {str(result.get('result'))[:500]}")
        models = sorted((result.get("modelUsage") or {}).keys())
        usage = {k: v for k, v in (result.get("usage") or {}).items() if isinstance(v, (int, float))}
        if isinstance(result.get("total_cost_usd"), (int, float)):
            usage["cost_usd"] = result["total_cost_usd"]
        model = {
            "provider": "anthropic",
            "model_id_requested": self._model,
            "model_id_reported": models or None,
            # Several ids when the CLI also ran a small helper model; the
            # resolved one is the requested family, else the costliest.
            "model_id_resolved": _main_model(result.get("modelUsage") or {}, self._model),
        }
        return (str(result.get("result") or "").strip(), model,
                {"result": result, "stderr_tail": stderr[-2000:], "usage": usage, "exit_code": code})


def _main_model(model_usage: dict[str, Any], requested: str | None = None) -> str | None:
    """The model that answered, from the CLI's per-model usage.

    The claude CLI can run a small helper model beside the main one (seen in the
    2026-09-25 smoke test: a Haiku call beside Opus for a one-word reply, with
    more output tokens than the main model). So output volume is not the test:
    a model of the requested family wins, else the one that cost the most.
    """
    if not model_usage:
        return None
    def cost(model: str) -> float:
        return float((model_usage[model] or {}).get("costUSD", 0) or 0)

    def output(model: str) -> int:
        return int((model_usage[model] or {}).get("outputTokens", 0) or 0)

    if requested:
        matching = [m for m in model_usage if requested.lower() in m.lower()]
        if matching:
            # Several ids can match the alias. Seen in the 2026-09-25 pilot:
            # claude-opus-5-5 only wrote a prompt cache (0 output tokens) and
            # claude-opus-5 wrote the whole answer. Whoever wrote the output
            # answered; cost breaks a tie.
            return max(matching, key=lambda m: (output(m), cost(m)))
    return max(model_usage, key=cost)


def _walk(value: Any):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key, inner
            yield from _walk(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk(inner)


class GeminiMCPAdapter(_PlatformAdapter):
    """Google models through an MCP server that exposes a ``review`` tool.

    The operator's own Gemini bridge (configured for Claude Code) is started
    over stdio with the same command and environment it runs with there; this
    adapter never reads a credential. The neutral preamble goes in as the
    system prompt, as for the claude CLI. A transient 503/429 is retried with a
    pause and the attempts are recorded; the model id is the one the bridge
    reports.
    """

    binary = "python3"

    def __init__(self, command: list[str], env: dict[str, str] | None = None, *,
                 model: str | None = None, attempts: int = 6, pause_s: float = 45.0) -> None:
        super().__init__()
        self._command = list(command)
        self._env = dict(env or {})
        self._model = model
        self._attempts = attempts
        self._pause_s = pause_s

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "channel": "MCP stdio bridge, tool 'review'",
                "server": Path(self._command[-1]).name if self._command else None,
                "server_env_keys": sorted(self._env), "model_requested": self._model,
                "tool_policy": "the bridge's plain generation; no tools"}

    async def _call(self, prompt: str) -> dict[str, Any]:
        import os

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self._command[0], args=self._command[1:],
                                       env={**os.environ, **self._env})
        arguments: dict[str, Any] = {"prompt": prompt, "system": prompts.PREAMBLE}
        if self._model:
            arguments["model"] = self._model
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("review", arguments)
        text = "".join(getattr(block, "text", "") for block in result.content)
        data = json.loads(text) if text.strip().startswith("{") else {"error": text}
        if result.isError or "error" in data:
            raise RuntimeError(str(data.get("error") or text)[:500])
        return data

    async def ask(self, prompt: str, workdir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
        del workdir
        errors: list[str] = []
        for attempt in range(1, self._attempts + 1):
            try:
                data = await self._call(prompt)
                break
            except RuntimeError as exc:
                errors.append(str(exc))
                transient = any(code in str(exc) for code in ("HTTP 503", "HTTP 429", "HTTP 500"))
                if not transient or attempt == self._attempts:
                    raise
                await asyncio.sleep(self._pause_s)
        model = {"provider": "google", "model_id_requested": self._model,
                 "model_id_resolved": data.get("model"), "backend": data.get("backend")}
        extra = {"bridge_response": {k: v for k, v in data.items() if k != "response"},
                 "retried_errors": errors,
                 "usage": {"duration_ms": data["duration_ms"]} if isinstance(data.get("duration_ms"), (int, float)) else {}}
        return str(data.get("response") or "").strip(), model, extra
