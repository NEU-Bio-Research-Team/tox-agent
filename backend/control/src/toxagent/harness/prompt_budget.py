"""What is actually in the prompt, by component (WS10 / PR-16).

The audit measured a minimal hERG report at 18,149 input tokens on its first
turn and 28,408 by the end, against a target of 8,000 — and a Q&A turn at
around 3,800 against a target of 2,500. What it could not say is *where* those
tokens went, so there was nothing to cut but guesses.

This module splits a dispatch into the five things it is made of:

* **policy prefix** — the static, versioned product role, invariants and
  answer-format rules. Identical for every turn of a profile, which is what
  makes it cacheable;
* **tool schemas** — the MCP surface, in a stable order;
* **pinned facts** — this run's references and checkpoint;
* **history** — recent conversation, append-only;
* **user message** — the turn itself.

Two consequences fall out of measuring it this way rather than measuring one
total.

**Cache-ability is visible.** A provider cache hits on a stable prefix. If the
policy prefix and the tool schemas hash the same across turns, the cacheable
fraction is real; if a tool description changed because a flag flipped, the
hash moves and the reason is in the manifest rather than in a latency graph
nobody can explain.

**A budget can name the offender.** "This profile is 4k over" is actionable;
"the report used 28,408 tokens" is not.

The token count is an estimate and says so. A real tokenizer belongs to the
provider and differs per model; carrying one here would add a dependency to be
wrong in a different way. What this needs to be is *stable* — the same text
always costs the same, so a diff between two manifests means the prompt
changed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Components, in the order they appear in a dispatch. Order matters: a
#: provider cache is a prefix cache, so anything that varies per run has to sit
#: after everything that does not.
COMPONENTS = (
    "policy_prefix",
    "tool_schemas",
    "pinned_facts",
    "history",
    "user_message",
)

#: The components a provider can cache, if they are byte-identical to last
#: time. Everything after these varies per run by construction.
CACHEABLE_PREFIX = ("policy_prefix", "tool_schemas")

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)

#: Rough characters-per-token for the mixed English/Vietnamese/JSON this
#: product sends. Calibrated against the audit's own measurements rather than
#: against a general corpus: the report profile's 18,149 reported tokens over
#: its prompt length is what this has to approximate.
_CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """A stable, deterministic estimate. Never presented as a provider count.

    Two estimators averaged, because each is wrong in a different direction:
    character count over-counts dense JSON, and word count under-counts it.
    """
    if not text:
        return 0
    by_chars = len(text) / _CHARS_PER_TOKEN
    by_words = len(_WORD.findall(text)) * 1.3
    return max(1, round((by_chars + by_words) / 2))


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def render_tool_schemas(schemas: Sequence[Mapping[str, Any]]) -> str:
    """The tool surface as the runtime will serialise it, in a stable order.

    Sorted by name and with sorted keys: a schema whose dict ordering changed
    between two processes is not a changed surface, and a hash that moved for
    that reason would make every cache-hit investigation start with a false
    lead.
    """
    ordered = sorted(schemas, key=lambda schema: str(schema.get("name") or ""))
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class ComponentCost:
    name: str
    characters: int
    tokens: int
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.name,
            "characters": self.characters,
            "tokens": self.tokens,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """One dispatch, measured. Attached to the run manifest, not to a log line."""

    profile: str
    components: tuple[ComponentCost, ...]
    #: What this profile is allowed, from ``PromptTargets``. ``None`` when no
    #: target has been agreed — which is a fact about the programme, not a
    #: licence to spend.
    target_tokens: int | None = None
    #: Extra facts a caller wants in the manifest (model id, flag state).
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(item.tokens for item in self.components)

    @property
    def cacheable_tokens(self) -> int:
        return sum(
            item.tokens for item in self.components if item.name in CACHEABLE_PREFIX
        )

    @property
    def cacheable_fraction(self) -> float:
        total = self.total_tokens
        return 0.0 if total == 0 else self.cacheable_tokens / total

    @property
    def prefix_hash(self) -> str:
        """One hash over the cacheable prefix.

        This is the number to watch: if it moves between two turns of the same
        profile, no provider cache can hit, and the reason is a prompt change
        somebody made.
        """
        material = "|".join(
            item.content_sha256
            for item in self.components
            if item.name in CACHEABLE_PREFIX
        )
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    @property
    def over_budget_by(self) -> int:
        """Tokens over target; zero when within it or when no target exists."""
        if self.target_tokens is None:
            return 0
        return max(0, self.total_tokens - self.target_tokens)

    @property
    def largest_component(self) -> ComponentCost | None:
        """The thing to cut first. A budget that cannot name the offender is
        a number, not a tool."""
        return max(self.components, key=lambda item: item.tokens, default=None)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "total_tokens": self.total_tokens,
            "target_tokens": self.target_tokens,
            "over_budget_by": self.over_budget_by,
            "cacheable_tokens": self.cacheable_tokens,
            "cacheable_fraction": round(self.cacheable_fraction, 4),
            "prefix_sha256": self.prefix_hash,
            "components": [item.to_dict() for item in self.components],
            "largest_component": (
                self.largest_component.name if self.largest_component else None
            ),
            **dict(self.context),
        }


#: Per-profile targets from the remediation plan's KPI table. These are targets
#: to measure against, not thresholds that refuse a dispatch: a turn refused for
#: being 200 tokens over budget would be a worse product than a turn that costs
#: 200 tokens more.
PROMPT_TARGETS: Mapping[str, int] = {
    "report_qa": 2_500,
    "analysis": 2_500,
    "evidence_research": 2_500,
    "report_build": 8_000,
    # ADS plan section 8.1/W3 (ADR 0010): the conversational target, same as
    # report_qa/evidence_research/analysis — this profile carries more pinned
    # references and a search-policy block those never needed, so it is the
    # one most likely to actually trip `over_budget_by`; W8 tunes the number,
    # this PR only makes sure it is measured at all.
    "decision_support": 2_500,
}


def measure(
    *,
    profile: str,
    policy_prefix: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    pinned_facts: str = "",
    history: str = "",
    user_message: str = "",
    context: Mapping[str, Any] | None = None,
) -> PromptBudget:
    """Measure one dispatch. Pure; safe to call on every turn."""
    rendered = {
        "policy_prefix": policy_prefix,
        "tool_schemas": render_tool_schemas(tool_schemas),
        "pinned_facts": pinned_facts,
        "history": history,
        "user_message": user_message,
    }
    return PromptBudget(
        profile=profile,
        components=tuple(
            ComponentCost(
                name=name,
                characters=len(rendered[name]),
                tokens=estimate_tokens(rendered[name]),
                content_sha256=content_hash(rendered[name]),
            )
            for name in COMPONENTS
        ),
        target_tokens=PROMPT_TARGETS.get(profile),
        context=dict(context or {}),
    )


def split_system_prompt(prompt: str) -> tuple[str, str, str]:
    """Split a built system prompt into (policy prefix, pinned facts, history).

    ``build_system_prompt`` already emits its sections in a fixed order with a
    blank-line separator, and the two run-varying sections announce themselves.
    Splitting on those markers is how an existing prompt gets measured without
    rebuilding the builder around the measurement — and if the markers ever
    move, the split degrades to "it is all policy prefix", which shows up as a
    prefix hash that changes every turn rather than as a silently wrong number.
    """
    pinned_marker = "\n\nPinned references:\n"
    history_marker = "\n\nRecent conversation:\n"

    history = ""
    head = prompt
    if history_marker in head:
        head, history = head.split(history_marker, 1)

    pinned = ""
    if pinned_marker in head:
        head, pinned = head.split(pinned_marker, 1)

    return head, pinned, history
