"""Which runtime agent an intent actually runs as, and under what step cap.

P0-1 of the 2026-09-13 audit: the report profile declared the agent
``toxagent-report`` and ``maxSteps: 64``, the gateway wrote a 64-step budget
into the run manifest, and the adapter sent ``agent: toxagent`` — a profile
whose checked-in cap is 32. The run's audit trail claimed a budget the runtime
never saw, and a report could be cut off at 32 steps before it ever reached
``submit_report_draft``.

Two numbers have to be told apart for that to be impossible again:

**requested** is what the control plane asked for (``RuntimeSettings``'s
per-intent caps). It is a product decision, and it belongs in the audit row.

**effective** is what the runtime will really enforce. For OpenCode V1 that is
the checked-in agent profile's static ``maxSteps``, because V1's
``prompt_async`` body has no step field at all — so this module reads the
profile JSON that gets deployed rather than restating its number in Python,
where the two could drift apart silently.

When the two differ the run is still dispatched, but the difference is an
explicit, recorded warning rather than a discrepancy someone finds in an audit
six weeks later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from ..domain.errors import RuntimeProtocolError

#: The agent profile files this product ships, by the intent they serve. Paths
#: are relative to ``ProfileSettings.profiles_dir``.
QA_PROFILE_PATH = "opencode/toxagent.json"
REPORT_PROFILE_PATH = "report_build/profile.json"


@dataclass(frozen=True, slots=True)
class RuntimeProfileSpec:
    """One intent's runtime binding, resolved from config and profile files."""

    #: The product intent this serves (``Intent.value``).
    intent: str
    #: The capability profile the MCP tool registry filters on.
    capability_profile: str
    #: The named agent the adapter must ask the runtime to run.
    runtime_agent_name: str
    #: What the control plane asked for.
    requested_step_cap: int
    #: What the deployed agent profile actually enforces; ``None`` when the
    #: profile could not be read, which is a deployment fault, not a zero.
    effective_step_cap: int | None
    #: Where ``effective_step_cap`` was read from, for the audit row.
    profile_source: str
    #: Runtime kinds this spec is valid for. A spec resolved for OpenCode says
    #: so rather than being assumed to describe every adapter.
    runtime_kind: str = "opencode"

    @property
    def cap_is_honoured(self) -> bool:
        return self.effective_step_cap is not None and (
            self.effective_step_cap >= self.requested_step_cap
        )

    @property
    def discrepancy(self) -> str:
        """Empty when there is nothing to warn about."""
        if self.effective_step_cap is None:
            return (
                f"the deployed agent profile for {self.runtime_agent_name!r} could not be "
                f"read from {self.profile_source}, so the effective step cap is unknown"
            )
        if self.effective_step_cap < self.requested_step_cap:
            return (
                f"{self.intent} asked for {self.requested_step_cap} steps but agent "
                f"{self.runtime_agent_name!r} enforces {self.effective_step_cap} "
                f"({self.profile_source})"
            )
        return ""

    def to_manifest(self) -> dict[str, object]:
        """What the run's runtime binding records. Both numbers, always."""
        return {
            "runtime_agent_name": self.runtime_agent_name,
            "capability_profile": self.capability_profile,
            "requested_step_cap": self.requested_step_cap,
            "effective_step_cap": self.effective_step_cap,
            "step_cap_honoured": self.cap_is_honoured,
            "profile_source": self.profile_source,
        }


def _agent_entries(path: Path) -> Mapping[str, Mapping[str, object]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    agents = document.get("agent")
    if not isinstance(agents, dict):
        return {}
    return {
        name: entry for name, entry in agents.items() if isinstance(entry, dict)
    }


@lru_cache(maxsize=32)
def _profile_agents(path_str: str) -> tuple[tuple[str, int | None], ...]:
    """``((agent_name, max_steps), ...)`` for one profile file.

    Cached on the path: these files ship with the package and do not change
    under a running process, and every dispatch would otherwise re-read them.
    """
    entries = _agent_entries(Path(path_str))
    resolved: list[tuple[str, int | None]] = []
    for name, entry in entries.items():
        # ``maxSteps`` is the V1 field. Newer OpenCode documents call it
        # ``steps``; accept either so a controlled upgrade does not need this
        # module changed on the same day the binary is re-pinned.
        raw = entry.get("maxSteps", entry.get("steps"))
        cap = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else None
        resolved.append((name, cap))
    return tuple(sorted(resolved))


def clear_profile_cache() -> None:
    """Drop the parsed-profile cache. Tests that write a profile file call it."""
    _profile_agents.cache_clear()


class RuntimeProfileRegistry:
    """Resolves an intent to the agent and caps a runtime will really use."""

    def __init__(
        self,
        *,
        profiles_dir: Path | None,
        agent_name: str,
        report_agent_name: str,
        max_steps_qa: int,
        max_steps_research: int,
        max_steps_report: int,
        runtime_kind: str = "opencode",
        select_named_agents: bool = True,
    ) -> None:
        self._profiles_dir = profiles_dir
        self._agent_name = agent_name
        self._report_agent_name = report_agent_name
        self._caps = {
            "build_report": max_steps_report,
            "evidence_research": max_steps_research,
        }
        self._default_cap = max_steps_qa
        self._runtime_kind = runtime_kind
        #: Off, every intent is dispatched to ``agent_name`` — the behaviour
        #: that shipped before this module. The flag exists so a deployment
        #: whose runtime host has not been given the report profile yet can
        #: roll back without a code change.
        self._select_named_agents = select_named_agents

    def _path(self, relative: str) -> Path | None:
        if self._profiles_dir is None:
            return None
        return self._profiles_dir / relative

    def _cap_for_agent(self, relative: str, agent_name: str) -> tuple[int | None, str]:
        path = self._path(relative)
        if path is None:
            return None, f"<profiles_dir unset>/{relative}"
        for name, cap in _profile_agents(str(path)):
            if name == agent_name:
                return cap, relative
        return None, relative

    def agent_name_for_intent(self, intent: str) -> str:
        if not self._select_named_agents:
            return self._agent_name
        return self._report_agent_name if intent == "build_report" else self._agent_name

    def resolve(self, intent: str, *, capability_profile: str) -> RuntimeProfileSpec:
        agent = self.agent_name_for_intent(intent)
        relative = (
            REPORT_PROFILE_PATH if agent == self._report_agent_name else QA_PROFILE_PATH
        )
        effective, source = self._cap_for_agent(relative, agent)
        return RuntimeProfileSpec(
            intent=intent,
            capability_profile=capability_profile,
            runtime_agent_name=agent,
            requested_step_cap=self._caps.get(intent, self._default_cap),
            effective_step_cap=effective,
            profile_source=source,
            runtime_kind=self._runtime_kind,
        )

    def required_agent_names(self) -> tuple[str, ...]:
        """Every agent a healthy deployment of this configuration must expose."""
        if not self._select_named_agents:
            return (self._agent_name,)
        return (self._agent_name, self._report_agent_name)

    def agent_for_capability(self, capability: str) -> str | None:
        """Which agent a product capability needs, or ``None`` if it needs no
        runtime agent of its own."""
        if capability == "build_report":
            return self.agent_name_for_intent("build_report")
        return None


def registry_from_settings(
    runtime_settings,
    profile_settings,
    *,
    select_named_agents: bool = True,
) -> RuntimeProfileRegistry:
    """The one construction path, so no caller invents its own defaults."""
    profiles_dir = getattr(profile_settings, "profiles_dir", None)
    report_agent = getattr(runtime_settings, "report_agent_name", "") or "toxagent-report"
    return RuntimeProfileRegistry(
        profiles_dir=Path(profiles_dir) if profiles_dir else None,
        agent_name=runtime_settings.agent_name,
        report_agent_name=report_agent,
        max_steps_qa=runtime_settings.max_steps_qa,
        max_steps_research=runtime_settings.max_steps_research,
        max_steps_report=runtime_settings.max_steps_report,
        runtime_kind=runtime_settings.kind,
        select_named_agents=select_named_agents,
    )


def assert_agent_declared(profiles_dir: Path, relative: str, agent_name: str) -> int:
    """Read a shipped profile's cap, raising rather than returning a guess.

    Used by the deployment check: a profile file that no longer declares the
    agent the adapter is about to ask for is a configuration error we want at
    start-up, not at the 32nd step of a report build.
    """
    for name, cap in _profile_agents(str(profiles_dir / relative)):
        if name == agent_name and cap is not None:
            return cap
    raise RuntimeProtocolError(
        f"agent profile {relative} does not declare {agent_name!r} with a step cap"
    )
