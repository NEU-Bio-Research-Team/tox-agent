---
name: agentic-ai-framework
description: >
  Expert skill for designing, architecting, and implementing Agentic AI systems and multi-agent frameworks.
  Load when the user asks to: design an agentic pipeline, build a multi-agent system, architect an LLM-based autonomous agent,
  implement orchestrator/sub-agent hierarchies, add planning/tool-use/memory/reflection to AI systems, choose between agentic
  frameworks (Google ADK, LangGraph, CrewAI, AutoGen, OpenAI Agents SDK), build domain-specific AI agents (drug discovery,
  healthcare, finance, NLP, research automation), or design agentic workflows with RAG, GNN, explainability, or scientific backends.
  Trigger phrases: "agentic AI", "multi-agent", "agent framework", "orchestrator agent", "sub-agent", "tool use pipeline",
  "autonomous agent", "planning + reflection", "agentic workflow", "agentic pipeline", "agent memory", "agent design".
license: MIT
metadata:
  author: Perplexity Computer
  version: '1.0'
  domain: Agentic AI, Multi-Agent Systems, LLM Engineering, Systems Architecture
  references:
    - "ToxAgent Proposal — GDGoC Hackathon Vietnam 2026 (internal)"
    - "AgentSquare: Automatic LLM Agent Search — arXiv:2410.06153"
    - "Reflexion: Language Agents with Verbal Reinforcement Learning — arXiv:2303.11366"
    - "DynTaskMAS: Dynamic Task Graph-driven MAS — arXiv:2503.07675"
    - "Multi-Agent Collaboration Mechanisms Survey — arXiv:2501.06322"
    - "AGENTSAFE: Governance in Agentic AI — arXiv:2512.03180"
    - "A-MEM: Agentic Memory for LLM Agents — arXiv:2502.12110"
    - "Google ADK Documentation — google.github.io/adk-docs"
---

# Agentic AI Framework Design Skill

## When to Use This Skill

Load this skill when the user's task involves **any** of:

- Designing a multi-agent architecture from scratch (orchestrator + specialist agents)
- Choosing and configuring an agentic framework (Google ADK, LangGraph, CrewAI, AutoGen)
- Implementing core agentic primitives: planning, tool use, memory, reflection loops
- Building domain-specific agent pipelines (drug discovery, healthcare, finance, NLP, research)
- Integrating external models (GNN, BERT, custom ML) as tools inside an agentic system
- Debugging or optimizing existing agentic workflows
- Evaluating agent reliability, observability, and safety
- Writing proposals, technical specs, or implementation roadmaps for agentic systems

---

## Foundational Theory: What Makes a System "Agentic"

### The Agent Formula

```
Agent = Reasoning (LLM) + Tools + Memory + Execution Loop
```

A system is **truly agentic** (not just a chatbot or RAG pipeline) when it exhibits:

| Property | Definition | Failure if missing |
|---|---|---|
| **Autonomous Planning** | Decomposes a high-level goal into sub-tasks without explicit instruction per step | System requires human to spell out every step |
| **Tool Use** | Selects and calls external APIs/functions based on context | System hallucinates facts it could retrieve |
| **Memory** | Persists state across turns and sessions | System "forgets" prior steps or user context |
| **Execution Loop** | Iterates: observe → think → act → observe until done | System runs once and stops |
| **Reflection** | Detects output quality issues and retries with corrected strategy | System returns wrong answers silently |

**Reference**: [AgentSquare (arXiv:2410.06153)](https://arxiv.org/abs/2410.06153) decomposes agent design into exactly four modules: Planning, Reasoning, Tool Use, and Memory — and shows that modular search over these produces SOTA agent configurations.

---

## Core Design Patterns (2025–2026 SOTA)

### Pattern 1: Hierarchical Orchestrator → Sub-Agent

The dominant production pattern. One **Orchestrator Agent** (LLM with high reasoning capacity) receives the user task and routes to **specialist sub-agents**, each with narrow scope and dedicated tools.

```
User Query
    │
    ▼
[Orchestrator Agent]  ←── task decomposition, routing, conflict resolution
    ├── [Specialist A]  → tool_A1, tool_A2
    ├── [Specialist B]  → tool_B1
    └── [Specialist C]  → tool_C1, tool_C2
    │
    ▼
[Aggregator / Report Writer Agent]  → synthesized output
```

**Key design decisions**:
- Orchestrator uses a **high-quality model** (e.g., Gemini 1.5 Pro, GPT-4o) for planning
- Sub-agents use **fast/cheap models** (e.g., Gemini 2.0 Flash) for execution
- Each sub-agent has a **single responsibility** — reduces hallucination
- Orchestrator holds **global state**; sub-agents are stateless per call

**Reference**: [Hybrid Agentic AI and MAS (arXiv:2511.18258)](https://arxiv.org/abs/2511.18258) validates layered architectures with an LLM Planner Agent coordinating specialized sub-agents in industrial settings.

---

### Pattern 2: Parallel Execution

For independent sub-tasks (e.g., database lookup + model inference), run agents concurrently. Reduces end-to-end latency by 40–60%.

```python
# Google ADK: ParallelAgent pattern
from google.adk.agents import ParallelAgent

parallel_block = ParallelAgent(
    name="parallel_research_screen",
    sub_agents=[screening_agent, researcher_agent]
)
```

**Critical constraint**: Only parallelize when sub-tasks share **no write-dependencies** on shared state. Shared state mutations in parallel agents are a common failure mode.

---

### Pattern 3: Reflection Loop (Reflexion Pattern)

After each agent action, a **Reflection Agent** (or self-reflection prompt) evaluates output quality and decides: accept / retry / escalate.

```
Action → Output
          │
          ▼
    [Reflection Agent]
          │
    ┌─────┴────────┐
    │              │
  ACCEPT         RETRY (with corrected strategy)
    │              │
    ▼              └──→ [Agent] (with error context injected)
  Continue
```

**Implementation**: The Reflexion paper ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) shows verbal reflection stored in episodic memory buffers outperforms simple retry loops. Store reflection outputs as structured memory entries.

```python
reflection_prompt = """
Given the agent output: {output}
And the expected properties: {criteria}
Evaluate: Is this output correct, complete, and consistent?
If not, identify the specific failure mode and suggest a corrected approach.
Return: {"verdict": "accept"|"retry", "reason": str, "correction": str}
"""
```

---

### Pattern 4: Tool Registry with Dynamic Dispatch

Do NOT load all tools upfront. Define a **tool registry** and dynamically surface only tools relevant to the current step. Prevents tool confusion (LLM selecting wrong tool when many are available).

```python
TOOL_REGISTRY = {
    "predict_toxicity": predict_toxicity,     # ML model inference
    "explain_molecule": explain_molecule,      # GNNExplainer
    "lookup_pubchem": lookup_pubchem,          # External DB
    "search_literature": search_literature,    # RAG retrieval
}

def get_tools_for_step(step: str) -> list:
    STEP_TOOL_MAP = {
        "screen": ["predict_toxicity"],
        "explain": ["explain_molecule"],
        "research": ["lookup_pubchem", "search_literature"],
    }
    return [TOOL_REGISTRY[t] for t in STEP_TOOL_MAP.get(step, [])]
```

**Reference**: [TPTU (arXiv:2308.03427)](https://arxiv.org/abs/2308.03427) shows that structured tool-step mapping outperforms giving all tools at once.

---

### Pattern 5: Memory Architecture (3-Tier)

```
┌─────────────────────────────────────────────┐
│  TIER 1: Working Memory (Scratchpad)         │
│  In-context state: current task, tool outputs│
│  Scope: single agent call                    │
└──────────────────┬──────────────────────────┘
                   │ persist on session end
┌──────────────────▼──────────────────────────┐
│  TIER 2: Session Memory (Short-term)         │
│  Conversation history, intermediate results  │
│  Scope: single user session                  │
│  Implementation: InMemorySessionService (ADK)│
└──────────────────┬──────────────────────────┘
                   │ index semantically
┌──────────────────▼──────────────────────────┐
│  TIER 3: Long-term Memory (Episodic/Semantic)│
│  Past sessions, user facts, retrieved docs   │
│  Scope: cross-session, persistent            │
│  Implementation: Vertex AI Memory Bank / RAG │
└─────────────────────────────────────────────┘
```

**Reference**: [A-MEM (arXiv:2502.12110)](https://arxiv.org/abs/2502.12110) shows agentic memory with Zettelkasten-style linking outperforms flat retrieval by organizing notes with context, keywords, and inter-memory links.

---

## Framework Selection Matrix

| Criterion | Google ADK | LangGraph | CrewAI | AutoGen (v0.4) | OpenAI Agents SDK |
|---|---|---|---|---|---|
| **Deploy to cloud** | ✅ 1-command (Vertex AI) | ❌ manual | ❌ manual | ❌ Azure only | ❌ manual |
| **Native Gemini** | ✅ built-in | ⚠️ via LiteLLM | ⚠️ via LiteLLM | ✅ via config | ⚠️ OpenAI-first |
| **Multi-agent** | ✅ native hierarchy | ✅ graph nodes | ✅ role-based | ✅ GroupChat | ✅ handoffs |
| **Observability** | ✅ ADK web UI | ✅ LangSmith | ✅ Crew Control Plane | ✅ OpenTelemetry | ⚠️ limited |
| **Learning curve** | Medium (new, fewer examples) | High (low-level) | Low (easiest) | Medium | Low |
| **State management** | Session service | State dict in graph | Agent memory | Conversation history | Thread-based |
| **Best for** | Google Cloud production | Complex DAG workflows | Role-based teams | Dynamic multi-agent chat | OpenAI ecosystem |

**Decision heuristic**:
- Google Cloud + Gemini → **Google ADK**
- Complex non-linear workflows (cycles, branches) → **LangGraph**
- Fast prototyping with role-based teams → **CrewAI**
- Research / dynamic agent spawning → **AutoGen**

---

## Domain-Specific Agent Design: Scientific / Healthcare Pipelines

When integrating **non-LLM ML models** (GNN, BERT, ResNet, etc.) as agent tools:

### Step 1: Wrap the Model as a Microservice

```python
# FastAPI wrapper — expose model as REST endpoint
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(payload: PredictRequest):
    result = model.inference(payload.input)
    return {"output": result, "confidence": result.confidence}
```

Deploy on Cloud Run (serverless, auto-scale, GPU add-on available) or a dedicated container.

### Step 2: Define ADK Tool Function

```python
import httpx

MODEL_URL = "https://your-model-service.run.app"

def predict_from_model(input_data: str) -> dict:
    """
    Call the domain-specific ML model for prediction.
    Returns structured output with prediction and confidence.
    """
    response = httpx.post(f"{MODEL_URL}/predict",
                          json={"input": input_data}, timeout=30)
    response.raise_for_status()
    return response.json()
```

### Step 3: Register Tool with Agent

```python
from google.adk.agents import LlmAgent

specialist_agent = LlmAgent(
    name="domain_specialist",
    model="gemini-2.0-flash",
    tools=[predict_from_model, explain_output, lookup_database],
    instruction="""
    You are a specialist agent for [domain].
    When given an input, call predict_from_model to get the ML prediction,
    then call explain_output for attribution, then lookup_database for
    prior evidence. Return a structured JSON with all findings.
    Never fabricate predictions — always call the tool.
    """
)
```

---

## Orchestrator Design Template

```python
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-1.5-pro",  # Use stronger model for planning
    description="Orchestrates the full analysis pipeline",
    sub_agents=[
        specialist_agent_a,
        specialist_agent_b,
        researcher_agent,
        report_writer_agent,
    ],
    instruction="""
    You are the orchestrating agent. When given a task:
    1. Decompose into subtasks: [screening] → [explanation] → [research] → [report]
    2. Run [screening] and [research] in parallel when possible
    3. After all sub-agents complete, detect any conflicts between results
    4. If a sub-agent fails, log the failure and continue with available results
    5. Delegate final synthesis to report_writer_agent
    6. Always include confidence level and flag uncertain outputs
    
    Never answer from internal knowledge alone — always use the appropriate sub-agent.
    """,
)

session_service = InMemorySessionService()
runner = Runner(agent=orchestrator, session_service=session_service)
```

---

## Repository Structure (Standard Layout)

```
project_name/
├── model_server/           # Domain ML model as REST microservice
│   ├── main.py             # FastAPI: /predict, /explain, /health
│   ├── Dockerfile
│   └── requirements.txt
│
├── agents/                 # Google ADK agent definitions
│   ├── orchestrator.py     # Root orchestrator agent
│   ├── specialist_a.py     # Sub-agent A
│   ├── specialist_b.py     # Sub-agent B
│   ├── researcher.py       # DB/literature lookup agent
│   └── writer.py           # Report synthesis agent
│
├── tools/                  # Tool function definitions
│   ├── model_tools.py      # ML model API calls
│   ├── database_tools.py   # External DB lookups (PubChem, etc.)
│   └── rag_tools.py        # RAG / vector search
│
├── memory/
│   └── memory_config.py    # Memory service setup
│
├── ui/
│   └── app.py              # Streamlit (MVP) or Next.js (prod)
│
└── deploy/
    ├── .env.example
    └── cloudbuild.yaml     # CI/CD for Cloud Run
```

---

## Implementation Workflow

When asked to **design** an agentic system, follow this sequence:

### Phase 1 — Problem Decomposition
1. Identify the end-to-end task: what does the user input → what must be the output?
2. List all **knowledge sources** needed (ML models, databases, APIs, documents)
3. Map each knowledge source to a **specialist agent role**
4. Identify which steps are **sequential** (dependency chain) vs **parallel** (independent)
5. Define the **failure modes** for each step and fallback behavior

### Phase 2 — Architecture Design
1. Draw the agent topology (Orchestrator → Sub-agents → Memory/Tools)
2. Specify the **model** for each agent (reasoning quality vs. cost tradeoff)
3. Define **tool signatures** (input/output types, docstrings — ADK uses docstrings for tool dispatch)
4. Design the **memory layers** (in-context scratchpad, session state, long-term store)
5. Define the **reflection trigger conditions** (when to retry, when to escalate)

### Phase 3 — Implementation Order (MVP-first)
```
Day 1: Domain model → FastAPI microservice (testable independently)
Day 2: Core specialist agents (Screening + Explainer) — most critical path
Day 3: Orchestrator + Researcher agent + end-to-end pipeline
Day 4: UI + cloud deployment (public URL)
Day 5: Reflection loops + memory + edge cases + polish
```

### Phase 4 — Evaluation Criteria
- **Functional correctness**: does the pipeline produce accurate outputs on test cases?
- **Latency**: end-to-end time per query (target: <2 min for complex pipelines)
- **Failure recovery**: what happens when one sub-agent fails or an API is unavailable?
- **Observability**: can you trace which agent made which decision?
- **Hallucination rate**: does the system ever fabricate tool outputs? (must be 0%)

---

## Common Failure Modes and Fixes

| Failure | Root Cause | Fix |
|---|---|---|
| Agent calls wrong tool | Too many tools loaded at once | Dynamic tool dispatch per step |
| Reflection loop diverges | No convergence criterion | Set max_retries=3 + fallback on exceed |
| Sub-agent output ignored by orchestrator | Weak system prompt | Explicit output schema + structured JSON |
| Context window overflow in long pipelines | Passing full history between agents | Summarize per-agent outputs before passing upstream |
| Parallel agents write to shared state | Race condition on mutable dict | Use immutable message passing; aggregate in orchestrator |
| Model server cold start delays | Serverless spin-up latency | Precompute cache for demo molecules; min-instances=1 |
| LLM hallucinates database results | No tool call verification | Require tool call for any factual claim; verify response schema |

---

## Safety and Observability Checklist

Before deploying any agentic system to production:

- [ ] Input validation layer (type checks, domain-specific constraints, e.g., SMILES validation with RDKit)
- [ ] LLM safety filters enabled (built-in on Gemini, custom on others)
- [ ] Per-agent step logging (Cloud Logging, OpenTelemetry, or LangSmith)
- [ ] Structured output schemas enforced (Pydantic models or JSON schema)
- [ ] Max iteration limits on all reflection loops
- [ ] Graceful degradation: each agent returns structured error, not exception
- [ ] HITL (Human-in-the-Loop) hook for high-stakes decisions
- [ ] Rate limit handling + exponential backoff on all external API calls

**Reference**: [AGENTSAFE (arXiv:2512.03180)](https://arxiv.org/abs/2512.03180) provides a full governance framework: plan→act→observe→reflect loop profiling, semantic telemetry, dynamic authorization, and cryptographic provenance tracing.

---

## Key Academic References

| Paper | Contribution | URL |
|---|---|---|
| AgentSquare (2025) | Modular agent search over Planning/Reasoning/Tool/Memory | [arXiv:2410.06153](https://arxiv.org/abs/2410.06153) |
| Reflexion (2023) | Verbal reinforcement via episodic reflection memory | [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) |
| A-MEM (2025) | Zettelkasten agentic memory with semantic linking | [arXiv:2502.12110](https://arxiv.org/abs/2502.12110) |
| DynTaskMAS (2025) | Dynamic task graph for async/parallel MAS | [arXiv:2503.07675](https://arxiv.org/abs/2503.07675) |
| MUA-RL (2025) | RL for multi-turn agentic tool use with simulated users | [arXiv:2508.18669](https://arxiv.org/abs/2508.18669) |
| AGENTSAFE (2025) | Governance framework for LLM-based agentic systems | [arXiv:2512.03180](https://arxiv.org/abs/2512.03180) |
| TRiSM for Agentic AI (2025) | Trust, Risk, Security in multi-agent systems | [arXiv:2506.04133](https://arxiv.org/abs/2506.04133) |
| Hybrid Agentic AI + MAS (2025) | Layered LLM-planner + rule-based edge agents | [arXiv:2511.18258](https://arxiv.org/abs/2511.18258) |
| Multi-Agent Collab Survey (2025) | Comprehensive survey of LLM-based MAS mechanisms | [arXiv:2501.06322](https://arxiv.org/abs/2501.06322) |
| SagaLLM (2025) | Context management + transaction guarantees for MAS | [arXiv:2503.11951](https://arxiv.org/abs/2503.11951) |

---

## Output Format for Design Responses

When producing an agentic framework design, structure the response as:

1. **Problem Decomposition** — inputs, outputs, knowledge sources needed
2. **Agent Topology Diagram** (ASCII or Mermaid) — agents, tools, data flows
3. **Agent Specifications Table** — name, model, tools, input/output schema, failure behavior
4. **Tool Definitions** — Python function signatures with docstrings
5. **Memory Architecture** — which tier for what data
6. **Reflection Strategy** — trigger conditions, max retries, fallback
7. **Framework Recommendation** — which framework and why
8. **Repository Layout** — directory structure
9. **MVP Priority Order** — what to build first to get a working demo
10. **Risk Table** — failure modes, probability, mitigation
