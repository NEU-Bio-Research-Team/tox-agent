from __future__ import annotations

from typing import Any, List, Optional

ADK_AVAILABLE = True

try:
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
except Exception:
    ADK_AVAILABLE = False

    class _BaseAgent:
        def __init__(self, name: str, description: str = "", **kwargs: Any) -> None:
            self.name = name
            self.description = description
            self.kwargs = kwargs

    class LlmAgent(_BaseAgent):
        def __init__(
            self,
            name: str,
            model: Optional[str] = None,
            description: str = "",
            instruction: str = "",
            tools: Optional[List[Any]] = None,
            output_key: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(name=name, description=description, **kwargs)
            self.model = model
            self.instruction = instruction
            self.tools = tools or []
            self.output_key = output_key

    class ParallelAgent(_BaseAgent):
        def __init__(
            self,
            name: str,
            sub_agents: Optional[List[Any]] = None,
            description: str = "",
            **kwargs: Any,
        ) -> None:
            super().__init__(name=name, description=description, **kwargs)
            self.sub_agents = sub_agents or []

    class SequentialAgent(_BaseAgent):
        def __init__(
            self,
            name: str,
            sub_agents: Optional[List[Any]] = None,
            description: str = "",
            **kwargs: Any,
        ) -> None:
            super().__init__(name=name, description=description, **kwargs)
            self.sub_agents = sub_agents or []
