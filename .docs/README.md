# Tox-Agent: Molecular Toxicity Analysis Platform
`tox-agent` is a high-performance research platform leveraging `Graph Attention Networks (GATv2)` to predict and interpret molecular toxicity. This service provides a production-ready API for bioinformaticians to analyze chemical compounds via molecular graphs, ensuring both predictive accuracy and scientific interpretability.

## Project Overview
This repository demonstrates a "production-ready" approach to bioinformatics software, integrating deep learning models with robust software engineering principles like MVC architecture, Row-Level Security (RLS), and automated deployment pipelines. 

**Key features include:**
- *Toxicity Prediction*: Advanced GNN engine processing SMILES strings into graph embeddings.

- *Explainable AI (XAI)*: Identification of toxicophores and functional groups contributing to toxicity scores.

- *Secure Data Handling*: Multi-tenant data isolation using PostgreSQL RLS and Firebase Authentication.

- *Research-Centric UI*: Interactive dashboards designed for high-throughput screening analysis.

## Technical Architecture
The service is structured around a clean separation of concerns:
- **Model Layer**: Implements the GATv2 architecture for molecular graph processing, trained on public toxicity datasets (e.g., Tox21).
- **Controller Layer**: Handles API requests, orchestrates model inference, and manages data access with strict RLS policies.
- **View Layer**: Provides a React-based frontend for visualizing toxicity predictions and molecular features.
- **Data Layer**: Utilizes PostgreSQL for structured data storage, with schemas designed for efficient querying of chemical properties and prediction results.
- **Security Layer**: Enforces authentication and authorization, ensuring that sensitive chemical data is protected according to best practices.

## Getting Started
To run the service locally, follow the instructions in the [Local Development](#local-development) section below. For deployment and operational guidelines, refer to the [Deployment and Operations](#deployment-and-operations) section. 

### Prerequisites
- Node.js: 20 or newer
- Python: 3.9+ (for GNN Engine dependencies)
- Database: PostgreSQL 15+ (with RLS enabled)

### Local Development
```bash
# Clone the repository
git clone https://github.com/neu-bio-research/tox-agent.git
cd tox-agent

# Install API dependencies
npm install

# Setup Python environment for the AI Engine
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### Run locally

```bash
# Start the API server in development mode
npm run dev

# Launch the AI Inference Engine
python scripts/engine_main.py
```

The service listens on `http://127.0.0.1:3000` by default.

### Useful commands

```bash
npm test
npm run dev
```

### Quick smoke test

```bash
curl http://127.0.0.1:3000/health
curl http://127.0.0.1:3000/v1/services
```

## API overview

### Platform endpoints

- `GET /health`: liveness and build metadata
- `GET /ready`: readiness state of service dependencies
- `GET /metrics`: basic Prometheus-style metrics

### Domain endpoints

- `GET /v1/services`
- `POST /v1/services`
- `GET /v1/services/{serviceId}`
- `GET /v1/services/{serviceId}/checks`
- `POST /v1/services/{serviceId}/checks`

See [openapi.yaml](./openapi.yaml) for the full contract.

## Deployment and operations

- Deployment guide: [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)
- Day-2 operations and incidents: [docs/RUNBOOK.md](./docs/RUNBOOK.md)
- On-call expectations: [docs/ONCALL.md](./docs/ONCALL.md)
- Security posture: [docs/THREAT_MODEL.md](./docs/THREAT_MODEL.md) and [SECURITY.md](./SECURITY.md)

## Repository ownership

The ownership model is documented in [NEU-Bio-Research-Team/tox-agent](https://github.com/NEU-Bio-Research-Team/tox-agent.git). Architecture changes should also include an ADR in [docs/adr](./docs/adr).

## Team contacts

### NEU Bio Research Team
- Team alias: `@neu-bio-research`
**About Team**:
- Backend Lead: `Nguyen Quang Minh` (`11247324@neu.st.edu.vn`)
- Frontend & Data Architect: `Nguyen Quynh` (`11247346@neu.st.edu.vn`)
- Agentic AI Engineer: `Nguyen Ho Nhat Minh` (`11247321@neu.st.edu.vn`)
- GNN Engineer: `Le Duc Minh` (`11247320@neu.st.edu.vn`)

