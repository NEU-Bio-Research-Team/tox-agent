"""planning_agent.py — ToxAgent PlanningAgent

Translates a free-text user request into a structured, validated JSON task plan
(AgentPlan) that describes which agents to run and in what order.

v1 role: logging / audit / UI display.
Schema is forward-compatible with a future executor-driven orchestration mode.

Architecture decision (approved):
  Option A — standalone run_planning() called before run_orchestrator_flow().
  No ADK agent-graph changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .adk_compat import LlmAgent

logger = logging.getLogger(__name__)

PLANNING_MODEL = os.getenv(
    "AGENT_MODEL_FAST",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)

# ---------------------------------------------------------------------------
# Agent registry (Enum — easy to extend with new agents)
# ---------------------------------------------------------------------------


class AgentName(str, Enum):
    """Registry of all agents available to the PlanningAgent.

    Using str+Enum instead of Literal[...] so new agents can be registered
    without changing the Pydantic model signature.
    """

    ScreeningAgent  = "ScreeningAgent"
    ResearchAgent   = "ResearchAgent"
    EvidenceQAAgent = "EvidenceQAAgent"
    WriterAgent     = "WriterAgent"
    ReflectionAgent = "ReflectionAgent"


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------


class PlanTask(BaseModel):
    """A single atomic task assigned to one agent.

    Fields
    ------
    id          : Unique task identifier within the plan (e.g. "task_1").
    agent       : Which registered agent executes this task.
    objective   : One-sentence description of what the agent must accomplish.
    depends_on  : List of task IDs that must complete before this task starts.
    rationale   : Why this agent was chosen — for debug, audit, and UI tooltips.
    """

    id: str = Field(..., min_length=1)
    agent: AgentName
    objective: str = Field(..., min_length=5)
    depends_on: List[str] = Field(default_factory=list)
    rationale: str = Field(default="")

    @field_validator("id")
    @classmethod
    def _id_no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("Task id must not contain spaces")
        return v


class AgentPlan(BaseModel):
    """The complete, validated task execution plan for a user request.

    Fields
    ------
    goal        : Human-readable description of what the plan achieves.
    complexity  : LOW / MEDIUM / HIGH — drives pipeline depth and cost.
    confidence  : Planner confidence in this plan (0.0 – 1.0).
    tasks       : Ordered list of PlanTask objects forming a DAG.
    """

    goal: str = Field(..., min_length=5)
    complexity: str = Field(default="MEDIUM")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    tasks: List[PlanTask] = Field(..., min_length=1)

    @field_validator("complexity")
    @classmethod
    def _complexity_valid(cls, v: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"complexity must be one of {allowed}, got {v!r}")
        return upper

    @model_validator(mode="after")
    def _validate_dag(self) -> "AgentPlan":
        """Ensure dependency references are valid and the graph is acyclic."""
        task_ids = {t.id for t in self.tasks}

        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    raise ValueError(
                        f"Task {task.id!r} depends on unknown task {dep!r}"
                    )

        # Topological sort (Kahn's algorithm) — raises on cycle
        in_degree: Dict[str, int] = {t.id: 0 for t in self.tasks}
        adjacency: Dict[str, List[str]] = {t.id: [] for t in self.tasks}
        for task in self.tasks:
            for dep in task.depends_on:
                adjacency[dep].append(task.id)
                in_degree[task.id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self.tasks):
            raise ValueError("AgentPlan contains a dependency cycle")

        # WriterAgent or ReflectionAgent must be the terminal task
        agent_names = [t.agent for t in self.tasks]
        terminal_agents = {AgentName.WriterAgent, AgentName.ReflectionAgent}
        if not any(a in terminal_agents for a in agent_names):
            raise ValueError(
                "AgentPlan must include a WriterAgent or ReflectionAgent as terminal task"
            )

        return self


# ---------------------------------------------------------------------------
# Complexity classification (deterministic, no LLM)
# ---------------------------------------------------------------------------

# Keywords that signal the request needs deeper evidence analysis (HIGH)
_HIGH_COMPLEXITY_PATTERNS = re.compile(
    r"\b("
    r"evidence|literature|paper|study|studies|research|publication|cite|citation|"
    r"compare|comparison|versus|vs\.?|analog|analogue|similar molecule|"
    r"uncertain|low confidence|borderline|ambiguous|"
    r"carcinogen|genotoxic|mutagenic|hepatotox|nephrotox|"
    r"clinical trial|in vivo|in vitro|assay hit|bioassay"
    r")\b",
    re.IGNORECASE,
)

# Keywords that signal explanation / mechanism — MEDIUM complexity
_MEDIUM_COMPLEXITY_PATTERNS = re.compile(
    r"\b("
    r"mechanism|mechanis[mt]|mode of action|moa|pathway|pathway|"
    r"explain|why|how|reason|cause|effect|"
    r"report|summary|summarize|analyze|analyse"
    r")\b",
    re.IGNORECASE,
)

# Keywords that signal a standard screening request (LOW)
_LOW_COMPLEXITY_PATTERNS = re.compile(
    r"\b("
    r"toxicity|toxic|herg|cardiac|qt|qtc|"
    r"safe|safety|risk|predict|screen|score|check"
    r")\b",
    re.IGNORECASE,
)


def _classify_complexity(user_request: str) -> str:
    """Rule-based complexity classification — no LLM calls.

    Returns
    -------
    'HIGH'   if the request requires evidence analysis, comparative reasoning,
             or specific deep-tox signals (carcinogens, clinical trials, etc.)
    'MEDIUM' if the request requires mechanism explanation, analysis, or a report
    'LOW'    for simple toxicity prediction / screening requests
    """
    text = user_request.strip()

    high_hits = len(_HIGH_COMPLEXITY_PATTERNS.findall(text))
    medium_hits = len(_MEDIUM_COMPLEXITY_PATTERNS.findall(text))
    low_hits = len(_LOW_COMPLEXITY_PATTERNS.findall(text))

    # HIGH wins if there are >=2 high-complexity signals,
    # OR any high signal combined with a medium signal (e.g. mechanism + evidence)
    if high_hits >= 2 or (high_hits >= 1 and medium_hits >= 1):
        return "HIGH"

    # MEDIUM for mechanism/explanation requests, or single high-complexity signal
    if medium_hits >= 1 or high_hits == 1:
        return "MEDIUM"

    # LOW for plain screening requests
    if low_hits >= 1:
        return "LOW"

    # Default to MEDIUM when uncertain (safe choice)
    return "MEDIUM"


# ---------------------------------------------------------------------------
# Deterministic plan builder (fast-path — no LLM cost)
# ---------------------------------------------------------------------------

_COMPLEXITY_CONFIDENCE = {
    "LOW": 0.97,
    "MEDIUM": 0.93,
    "HIGH": 0.88,
}


def _build_deterministic_plan(
    user_request: str,
    complexity: str,
    compound_hint: Optional[str] = None,
) -> AgentPlan:
    """Build a plan using rules — zero LLM tokens consumed.

    Complexity → Pipeline:
      LOW    → Screening → Writer
      MEDIUM → Screening → Research → Writer
      HIGH   → Screening → Research → EvidenceQA → Writer
    """
    compound = compound_hint or "the compound"
    tasks: List[PlanTask] = []

    # --- Task 1: ScreeningAgent (always first, no dependencies) ---
    tasks.append(
        PlanTask(
            id="screening",
            agent=AgentName.ScreeningAgent,
            objective=(
                f"Run toxicity screening for {compound}: predict clinical toxicity "
                f"(hERG, ClinTox) and mechanistic toxicity (Tox21 assay hits)."
            ),
            depends_on=[],
            rationale=(
                "ScreeningAgent provides the primary ML-based toxicity signal "
                "and is always the first step regardless of complexity."
            ),
        )
    )

    if complexity in ("MEDIUM", "HIGH"):
        # --- Task 2: ResearchAgent ---
        tasks.append(
            PlanTask(
                id="research",
                agent=AgentName.ResearchAgent,
                objective=(
                    f"Retrieve PubChem compound info, PubMed literature, and "
                    f"bioassay data for {compound}. Synthesize key toxicology findings."
                ),
                depends_on=["screening"],
                rationale=(
                    "ResearchAgent grounds ML predictions in published evidence "
                    "and is needed for MEDIUM/HIGH complexity requests."
                ),
            )
        )

    if complexity == "HIGH":
        # --- Task 3: EvidenceQAAgent (conditional, HIGH only) ---
        tasks.append(
            PlanTask(
                id="evidence_qa",
                agent=AgentName.EvidenceQAAgent,
                objective=(
                    f"Quality-gate the retrieved literature for {compound}: "
                    f"deduplicate, score relevance, detect unsupported claims, "
                    f"and assess overall evidence confidence."
                ),
                depends_on=["research"],
                rationale=(
                    "EvidenceQAAgent is inserted for HIGH-complexity requests "
                    "(mechanism, comparative analysis, uncertain predictions) "
                    "to prevent low-quality evidence from entering the report."
                ),
            )
        )

    # --- Final task: WriterAgent ---
    last_dep = (
        "evidence_qa"
        if complexity == "HIGH"
        else ("research" if complexity == "MEDIUM" else "screening")
    )
    tasks.append(
        PlanTask(
            id="report",
            agent=AgentName.WriterAgent,
            objective=(
                f"Generate the final toxicity analysis report for {compound}, "
                f"integrating screening results"
                + (", research evidence" if complexity in ("MEDIUM", "HIGH") else "")
                + (", and QA-validated citations" if complexity == "HIGH" else "")
                + "."
            ),
            depends_on=[last_dep],
            rationale=(
                "WriterAgent synthesizes all upstream outputs into a coherent, "
                "structured report before the quality review step."
            ),
        )
    )

    # --- Terminal task: ReflectionAgent ---
    tasks.append(
        PlanTask(
            id="reflection",
            agent=AgentName.ReflectionAgent,
            objective=(
                f"Critically evaluate the completed toxicity report for {compound} "
                f"as an independent scientific reviewer: score evidence support, "
                f"detect unsupported claims, assess OOD risk, and recommend "
                f"corrective actions if needed."
            ),
            depends_on=["report"],
            rationale=(
                "ReflectionAgent is the final QA gate — it validates the report "
                "quality and flags any scientific concerns before the result is "
                "returned to the user."
            ),
        )
    )

    pipeline_desc = {
        "LOW": "Screening -> Writer -> Reflection",
        "MEDIUM": "Screening -> Research -> Writer -> Reflection",
        "HIGH": "Screening -> Research -> EvidenceQA -> Writer -> Reflection",
    }[complexity]

    return AgentPlan(
        goal=f"Analyze toxicity of {compound} ({pipeline_desc})",
        complexity=complexity,
        confidence=_COMPLEXITY_CONFIDENCE[complexity],
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# LLM fallback: parse and validate LLM-generated plan JSON
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)


def _extract_json_from_text(text: str) -> str:
    """Strip markdown code fences if the LLM wrapped the JSON."""
    match = _JSON_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _parse_llm_plan(raw_text: str, complexity: str, compound_hint: Optional[str]) -> AgentPlan:
    """Parse and validate a JSON plan returned by the LLM.

    Falls back to the deterministic plan if parsing or validation fails.
    """
    try:
        json_str = _extract_json_from_text(raw_text)
        data = json.loads(json_str)

        # Normalise task list — inject rationale if absent, normalise agent name
        tasks_raw = data.get("tasks", [])
        tasks: List[PlanTask] = []
        for t in tasks_raw:
            # Map plain string agent names to enum values defensively
            agent_raw = t.get("agent", "")
            try:
                agent = AgentName(agent_raw)
            except ValueError:
                logger.warning("LLM returned unknown agent %r; skipping task", agent_raw)
                continue

            tasks.append(
                PlanTask(
                    id=t.get("id", f"task_{len(tasks)+1}"),
                    agent=agent,
                    objective=t.get("objective", ""),
                    depends_on=t.get("depends_on", []),
                    rationale=t.get("rationale", "LLM-generated task."),
                )
            )

        plan = AgentPlan(
            goal=data.get("goal", f"Analyze toxicity of {compound_hint or 'the compound'}"),
            complexity=data.get("complexity", complexity).upper(),
            confidence=float(data.get("confidence", _COMPLEXITY_CONFIDENCE.get(complexity, 0.8))),
            tasks=tasks,
        )
        return plan

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM plan parse/validation failed (%s); falling back to deterministic plan",
            exc,
        )
        return _build_deterministic_plan(
            user_request="",
            complexity=complexity,
            compound_hint=compound_hint,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_planning(
    user_request: str,
    compound_hint: Optional[str] = None,
    force_llm: bool = False,
) -> AgentPlan:
    """Produce an AgentPlan for the given user request.

    Strategy
    --------
    1. Classify complexity deterministically (no LLM).
    2. For LOW and MEDIUM requests — build plan deterministically (fast-path).
    3. For HIGH complexity (or force_llm=True) — invoke planning_agent LlmAgent;
       validate the result; fall back to deterministic if validation fails.

    Parameters
    ----------
    user_request  : Raw free-text from the user.
    compound_hint : Compound name or SMILES extracted upstream (improves plan text).
    force_llm     : Always use LLM planning regardless of complexity.

    Returns
    -------
    AgentPlan — always valid (never raises; falls back to deterministic on error).
    """
    if not user_request or not user_request.strip():
        logger.warning("run_planning: empty user_request; returning minimal LOW plan")
        return _build_deterministic_plan("", "LOW", compound_hint)

    complexity = _classify_complexity(user_request)
    logger.info("run_planning: request=%r  complexity=%s", user_request[:80], complexity)

    use_llm = force_llm or (complexity == "HIGH")

    if not use_llm:
        plan = _build_deterministic_plan(user_request, complexity, compound_hint)
        logger.info(
            "run_planning: deterministic plan  tasks=%d  confidence=%.2f",
            len(plan.tasks),
            plan.confidence,
        )
        return plan

    # --- LLM fallback for HIGH complexity ---
    try:
        from google.adk.runners import InMemoryRunner  # type: ignore[import]

        runner = InMemoryRunner(agent=planning_agent)
        session = runner.session_service.create_session(app_name="tox_planning", user_id="planner")
        result_events = list(
            runner.run(
                user_id="planner",
                session_id=session.id,
                new_message={"role": "user", "parts": [{"text": user_request}]},
            )
        )
        raw_text = ""
        for event in result_events:
            content = getattr(event, "content", None)
            if content and hasattr(content, "parts"):
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        raw_text += part.text

        if raw_text.strip():
            plan = _parse_llm_plan(raw_text, complexity, compound_hint)
            logger.info(
                "run_planning: LLM plan  tasks=%d  confidence=%.2f",
                len(plan.tasks),
                plan.confidence,
            )
            return plan

        logger.warning("run_planning: LLM returned empty output; using deterministic fallback")

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "run_planning: LLM invocation failed (%s); using deterministic fallback", exc
        )

    return _build_deterministic_plan(user_request, complexity, compound_hint)


# ---------------------------------------------------------------------------
# planning_agent LlmAgent — used for LLM fallback path
# ---------------------------------------------------------------------------

_PLANNING_INSTRUCTION = """You are PlanningAgent for ToxAgent.

Your responsibility is to transform a user request into an executable multi-agent plan.

Available agents:

1. ScreeningAgent
   * Toxicity prediction
   * Risk scoring
   * hERG assessment
   * Tox21 inference
   * ClinTox inference

2. ResearchAgent
   * Scientific evidence retrieval
   * PubChem lookup
   * PubMed retrieval
   * Similar molecule retrieval

3. EvidenceQAAgent
   * Verify claims
   * Check evidence support
   * Detect unsupported conclusions
   * Assess evidence quality

4. WriterAgent
   * Generate final report
   * Summarize findings
   * Produce recommendations

Rules:
* Break the request into atomic tasks.
* Each task must have exactly one responsible agent.
* Preserve logical ordering.
* Avoid redundant tasks.
* Include dependencies.
* Add a "rationale" field to each task explaining why that agent was chosen.
* Set "complexity" to LOW, MEDIUM, or HIGH based on the request.
* Set "confidence" (0.0–1.0) reflecting how certain you are this plan is correct.

Output ONLY valid JSON.

Schema:
{
  "goal": "...",
  "complexity": "LOW" | "MEDIUM" | "HIGH",
  "confidence": 0.0-1.0,
  "tasks": [
    {
      "id": "task_1",
      "agent": "ScreeningAgent",
      "objective": "...",
      "depends_on": [],
      "rationale": "..."
    }
  ]
}

Example:

User:
Analyze the toxicity of Aspirin and explain possible mechanisms.

Output:

{
  "goal": "Analyze Aspirin toxicity and explain mechanisms",
  "complexity": "HIGH",
  "confidence": 0.92,
  "tasks": [
    {
      "id": "screening",
      "agent": "ScreeningAgent",
      "objective": "Run toxicity screening for Aspirin",
      "depends_on": [],
      "rationale": "Primary ML-based toxicity signal; always runs first."
    },
    {
      "id": "research",
      "agent": "ResearchAgent",
      "objective": "Retrieve mechanism evidence and literature for Aspirin",
      "depends_on": ["screening"],
      "rationale": "Mechanism explanation requires published evidence."
    },
    {
      "id": "evidence_qa",
      "agent": "EvidenceQAAgent",
      "objective": "Validate supporting evidence quality",
      "depends_on": ["research"],
      "rationale": "Mechanism claims need evidence QA to prevent unsupported conclusions."
    },
    {
      "id": "report",
      "agent": "WriterAgent",
      "objective": "Generate final report integrating all findings",
      "depends_on": ["evidence_qa"],
      "rationale": "WriterAgent always runs last to synthesize all outputs."
    }
  ]
}
"""

planning_agent = LlmAgent(
    name="PlanningAgent",
    model=PLANNING_MODEL,
    description=(
        "Transforms a free-text user request into a structured, validated "
        "multi-agent task plan (DAG) with complexity scoring and rationale."
    ),
    instruction=_PLANNING_INSTRUCTION,
    tools=[],
    output_key="plan",
)
