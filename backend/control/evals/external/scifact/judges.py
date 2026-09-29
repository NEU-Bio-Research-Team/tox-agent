"""Verifiers for the SciFact label + rationale step.

A judge reads one claim and one abstract (sentences numbered from 0) and
returns SUPPORT, CONTRADICT or NOT_ENOUGH_INFO with the evidence sentences.
The prompt is fixed and hashed into the run manifest.

What a judge here measures is *that model as a stand-alone verifier*. It is
not the ToxAgent product: the product's evidence-relation step runs inside an
answer, with its own tools and validator. A number from this module is
reported under the model's name, never as "ToxAgent on SciFact".
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evals.external.scifact.data import Claim, Document
from evals.external.scifact.metrics import LABELS, NEI
from evals.investigation.adapters import platform_cli
from evals.investigation.cases import sha256

JUDGE_SYSTEM = "You verify scientific claims against research abstracts. Answer only with JSON."

JUDGE_TEMPLATE = """Claim: {claim}

Abstract (sentences are numbered from 0):
Title: {title}
{sentences}

Does the abstract SUPPORT the claim, CONTRADICT it, or give NOT_ENOUGH_INFO to decide? If it supports or contradicts the claim, list the numbers of the sentences (at most 3) that are the evidence.

Answer with one JSON object and nothing else:
{{"label": "SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO", "sentences": [numbers]}}"""

PROMPT_VERSION = "scifact-judge-v1"


def render(claim: Claim, document: Document) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(document.sentences))
    return JUDGE_TEMPLATE.format(claim=claim.claim, title=document.title, sentences=numbered)


def template_sha256() -> str:
    return sha256(JUDGE_SYSTEM + "\n\n" + JUDGE_TEMPLATE)


@dataclass
class Judgment:
    label: str
    sentences: list[int]
    raw_text: str
    model: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    #: Why the output was not usable as returned; the judgment then counts as NEI.
    parse_error: str | None = None


def parse(text: str, n_sentences: int) -> tuple[str, list[int], str | None]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return NEI, [], "no JSON object in the output"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return NEI, [], f"invalid JSON: {exc}"
    label = str(data.get("label", "")).strip().upper()
    if label not in (*LABELS, NEI):
        return NEI, [], f"unknown label {label!r}"
    sentences: list[int] = []
    for value in data.get("sentences") or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < n_sentences and index not in sentences:
            sentences.append(index)
    return label, sentences, None


class Judge(Protocol):
    name: str

    def describe(self) -> dict[str, Any]: ...

    async def judge(self, claim: Claim, document: Document) -> Judgment: ...


class ClaudeJudge:
    def __init__(self, model: str = "opus") -> None:
        self.model = model
        self.name = f"anthropic:{model}"

    def describe(self) -> dict[str, Any]:
        return {"judge": "claude_cli", "model_requested": self.model,
                "tool_policy": "no tools, no MCP, no settings, empty working directory"}

    async def judge(self, claim: Claim, document: Document) -> Judgment:
        prompt = render(claim, document)
        command = ["claude", "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                   "--setting-sources", "", "--no-session-persistence",
                   "--system-prompt", JUDGE_SYSTEM, "--model", self.model]
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="scifact-") as workdir:
            code, stdout, stderr = await platform_cli._run_process(command, prompt, cwd=Path(workdir))
        result = json.loads(stdout) if stdout.strip().startswith("{") else {}
        if code != 0 or not result or result.get("is_error"):
            raise RuntimeError(f"claude exited {code}: {stderr[-300:] or str(result)[:300]}")
        text = str(result.get("result") or "")
        label, sentences, error = parse(text, len(document.sentences))
        usage = {k: v for k, v in (result.get("usage") or {}).items() if isinstance(v, (int, float))}
        if isinstance(result.get("total_cost_usd"), (int, float)):
            usage["cost_usd"] = result["total_cost_usd"]
        return Judgment(label, sentences, text, usage=usage,
                        model={"model_id_resolved": platform_cli._main_model(result.get("modelUsage") or {}, self.model),
                               "model_id_reported": sorted((result.get("modelUsage") or {}).keys()),
                               "model_usage": result.get("modelUsage") or {}},
                        duration_s=round(time.monotonic() - started, 3), parse_error=error)


class CodexJudge:
    name = "openai:codex"

    def describe(self) -> dict[str, Any]:
        return {"judge": "codex_cli", "configuration": platform_cli.codex_configuration(),
                "tool_policy": "read-only sandbox, empty working directory, ephemeral"}

    async def judge(self, claim: Claim, document: Document) -> Judgment:
        prompt = JUDGE_SYSTEM + "\n\n" + render(claim, document)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="scifact-") as workdir:
            text, model, extra = await platform_cli.CodexCLIAdapter().ask(prompt, Path(workdir))
        label, sentences, error = parse(text, len(document.sentences))
        return Judgment(label, sentences, text, model=model, usage=extra.get("usage") or {},
                        duration_s=round(time.monotonic() - started, 3), parse_error=error)


def make_judge(spec: str) -> Judge:
    kind, _, arg = spec.partition(":")
    if kind == "claude":
        return ClaudeJudge(arg or "opus")
    if kind == "codex":
        return CodexJudge()
    raise ValueError(f"unknown judge {spec!r}; use claude[:model] or codex")
