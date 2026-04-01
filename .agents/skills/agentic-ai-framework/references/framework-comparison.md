# Agentic Framework Comparison (2025–2026)

## Quick Reference

| Framework | GitHub Stars (2025) | Best For | Cloud Native |
|---|---|---|---|
| LangGraph | ~12k | Complex DAG workflows, fine-grained state control | No (any cloud) |
| Google ADK | ~7M downloads | Google Cloud + Gemini production systems | Yes (Vertex AI) |
| CrewAI | ~25k | Role-based teams, fast prototyping | No (any cloud) |
| AutoGen v0.4 | ~35k | Dynamic multi-agent conversations, research | Azure-friendly |
| OpenAI Agents SDK | ~8k | OpenAI ecosystem, handoff patterns | No (any cloud) |

## Google ADK Core Concepts

```python
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.adk.runners import Runner
from google.adk.memory import VertexAiMemoryBankService

# Primitive: Single LLM Agent with tools
agent = LlmAgent(
    name="my_agent",
    model="gemini-2.0-flash",         # or "gemini-1.5-pro"
    tools=[tool_fn_1, tool_fn_2],
    sub_agents=[sub_agent_1],          # optional delegation
    instruction="System prompt here",
)

# Sequential pipeline: A → B → C
pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[agent_a, agent_b, agent_c]
)

# Parallel execution: A + B simultaneously
parallel = ParallelAgent(
    name="parallel_block",
    sub_agents=[agent_a, agent_b]
)
```

## ADK Tool Best Practices

Tools are regular Python functions. ADK uses the **docstring** as the tool description for the LLM:

```python
def predict_toxicity(smiles: str) -> dict:
    """
    Predict clinical toxicity probability for a molecular SMILES string.
    
    Args:
        smiles: SMILES string representation of the molecule (e.g., 'CC(=O)Oc1ccccc1C(=O)O')
    
    Returns:
        dict with keys: p_toxic (float 0-1), label (TOXIC/NON_TOXIC), confidence (float)
    """
    # Implementation
    ...
```

**Critical**: Type annotations are required. ADK generates the JSON schema for function calling from the Python type hints.

## LangGraph Core Concepts

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    prediction: dict
    literature: dict
    report: str

def orchestrator_node(state: AgentState) -> AgentState:
    # Orchestration logic
    return state

graph = StateGraph(AgentState)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("screener", screener_node)
graph.add_edge("orchestrator", "screener")
graph.add_conditional_edges("screener", route_fn, {
    "explain": "explainer",
    "done": END
})

compiled = graph.compile()
```

## CrewAI Core Concepts

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Drug Toxicity Researcher",
    goal="Find prior evidence for molecule toxicity",
    backstory="Expert in chemical databases and literature",
    tools=[pubchem_tool, rag_tool],
    llm="gemini/gemini-2.0-flash"
)

research_task = Task(
    description="Look up toxicity evidence for {smiles} in PubChem and literature",
    expected_output="Structured JSON with known_toxicity, similar_compounds, confidence",
    agent=researcher
)

crew = Crew(
    agents=[screener, researcher, writer],
    tasks=[screen_task, research_task, write_task],
    process=Process.sequential,
    memory=True,
    verbose=True
)
```

## Deployment Commands

### Google ADK → Vertex AI Agent Engine
```bash
# Install
pip install google-cloud-aiplatform[agent_engines,adk]>=1.112

# Local test
adk run agents/orchestrator.py

# Deploy to Vertex AI
adk deploy --project YOUR_PROJECT_ID --region asia-southeast1

# Deploy model server to Cloud Run
gcloud run deploy model-api --source model_server/ --region asia-southeast1
```

### Environment Variables
```
GOOGLE_API_KEY=          # from aistudio.google.com
GOOGLE_CLOUD_PROJECT=    # GCP project ID
GOOGLE_CLOUD_LOCATION=   # e.g., asia-southeast1
MODEL_SERVER_URL=        # Cloud Run URL of your ML model API
```
