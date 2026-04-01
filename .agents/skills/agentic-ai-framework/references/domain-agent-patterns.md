# Domain-Specific Agentic Patterns

## Pattern: Scientific ML Model as Agent Tool (ToxAgent Example)

Derived from the ToxAgent GDGoC Hackathon 2026 proposal — a production-grade example
of wrapping a GNN-based ML model (SMILESGNN, AUC-ROC 0.997 on ClinTox) into a
full agentic pipeline using Google ADK.

### Architecture (5-Agent Pipeline)

```
User Input (SMILES or natural language query)
          │
          ▼
[Orchestrator Agent]  ← Gemini 2.0 Flash + Google ADK
    ├── ① [Screening Agent]   → predict_toxicity(smiles)      ─┐ parallel
    ├── ② [Researcher Agent]  → lookup_pubchem, search_RAG    ─┘
    ├── ③ [Explainer Agent]   → explain_molecule (GNNExplainer)
    └── ④ [Report Writer]     → Gemini 1.5 Pro, structured JSON
          │
          ▼
Structured Report: {executive_summary, toxicity_assessment, 
                    structural_alerts, literature_support, recommendations}
```

### Pipeline Step Specifications

| Step | Agent | Model | Tools | Output Schema |
|---|---|---|---|---|
| Validate + Route | Orchestrator | Gemini 2.0 Flash | — | task_plan: dict |
| Toxicity Prediction | Screening | Gemini 2.0 Flash | predict_toxicity | {p_toxic, label, confidence} |
| Structural Explanation | Explainer | Gemini 2.0 Flash | explain_molecule | {top_atoms, top_bonds, heatmap_b64} |
| Literature Research | Researcher | Gemini 2.0 Flash | lookup_pubchem, search_chembl, search_literature | {known_toxicity, similar_compounds, evidence} |
| Report Synthesis | Writer | Gemini 1.5 Pro | — | {summary, assessment, alerts, citations} |

### FastAPI Model Server Template

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rdkit.Chem as Chem

app = FastAPI(title="Domain Model API")

class PredictRequest(BaseModel):
    smiles: str

@app.post("/predict")
async def predict(req: PredictRequest):
    # Validate input
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    # Run domain model
    result = your_model.predict(req.smiles)
    return {
        "p_positive": float(result.probability),
        "label": "POSITIVE" if result.probability > 0.5 else "NEGATIVE",
        "confidence": float(result.confidence)
    }

@app.post("/explain")
async def explain(req: PredictRequest):
    explanation = your_explainer.explain(req.smiles)
    return {
        "top_features": explanation.top_k_features(10),
        "attribution_map": explanation.to_base64_image()
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Dockerfile for Cloud Run

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Pattern: RAG-Augmented Research Agent

```python
from google.adk.agents import LlmAgent
from vertexai.preview.language_models import TextEmbeddingModel
from google.cloud import aiplatform

def search_literature(query: str, top_k: int = 5) -> dict:
    """
    Search indexed scientific literature using semantic vector search.
    Returns relevant passages with source citations.
    
    Args:
        query: Natural language research question
        top_k: Number of results to return (default 5)
    
    Returns:
        dict with keys: passages (list of str), sources (list of str), confidence (str)
    """
    embedder = TextEmbeddingModel.from_pretrained("text-embedding-004")
    query_embedding = embedder.get_embeddings([query])[0].values
    
    index = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name="projects/.../indexEndpoints/..."
    )
    neighbors = index.find_neighbors(
        deployed_index_id="my_index",
        queries=[query_embedding],
        num_neighbors=top_k
    )
    
    return {
        "passages": [n.restricts for n in neighbors[0]],
        "sources": [n.id for n in neighbors[0]],
        "confidence": "high" if len(neighbors[0]) >= 3 else "low"
    }

researcher_agent = LlmAgent(
    name="researcher",
    model="gemini-2.0-flash",
    tools=[search_literature, lookup_external_db],
    instruction="""
    You are a scientific researcher. For any compound or query:
    1. Search literature for prior evidence
    2. Look up external databases for known properties
    3. If no evidence found, return confidence='low' — never fabricate evidence
    4. Always cite your sources
    """
)
```

## Pattern: Conflict Detection in Orchestrator

When a screening model says TOXIC but literature says SAFE (or vice versa), the
orchestrator must detect and flag this conflict.

```python
def detect_conflict(screening_result: dict, research_result: dict) -> dict:
    """
    Detect conflicts between ML prediction and literature evidence.
    Returns conflict analysis with uncertainty flag.
    """
    p_toxic = screening_result.get("p_toxic", 0.5)
    known_toxicity = research_result.get("known_toxicity", "unknown")
    
    # High confidence prediction contradicts known literature
    if p_toxic > 0.8 and known_toxicity == "non-toxic":
        return {
            "conflict_detected": True,
            "type": "model_vs_literature",
            "severity": "high",
            "action": "flag_for_expert_review",
            "message": f"Model predicts TOXIC (p={p_toxic:.2f}) but literature indicates non-toxic. Manual review required."
        }
    elif p_toxic < 0.2 and known_toxicity == "toxic":
        return {
            "conflict_detected": True,
            "type": "model_vs_literature", 
            "severity": "medium",
            "action": "include_uncertainty_warning",
            "message": f"Model predicts NON_TOXIC (p={p_toxic:.2f}) but literature indicates toxic."
        }
    
    return {"conflict_detected": False, "action": "proceed_normally"}
```

## Pattern: Batch Processing Mode

For screening many compounds in parallel:

```python
import asyncio
import httpx

async def batch_predict(smiles_list: list[str], max_concurrent: int = 10) -> list[dict]:
    """Screen multiple SMILES strings concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def predict_one(smiles: str) -> dict:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{MODEL_URL}/predict",
                        json={"smiles": smiles},
                        timeout=30
                    )
                    return {"smiles": smiles, **response.json()}
                except Exception as e:
                    return {"smiles": smiles, "error": str(e), "label": "ERROR"}
    
    return await asyncio.gather(*[predict_one(s) for s in smiles_list])
```
