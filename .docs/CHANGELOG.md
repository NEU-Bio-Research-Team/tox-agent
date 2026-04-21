# Changelog

All notable changes to the Tox-Agent platform are documented in this file.

This changelog follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Explainable AI (XAI) module with GATv2 attention-weight visualization to highlight toxicophores in molecular graphs
- Batch processing support for bulk SMILES uploads via CSV and JSON
- Interactive 3D molecular visualization in the dashboard using UV-sphere rendering
- Enhanced metrics for model drift tracking and prediction confidence intervals

## [1.0.0] - 2026-04-21

### Added

- Initial release of the GATv2-based core GNN engine for toxicity prediction
- Hybrid backend architecture combining a Node.js API with a Python inference engine
- Security infrastructure with PostgreSQL Row-Level Security (RLS) and Firebase Authentication for researcher data isolation
- Standardized MVC-style project organization for models, views, and controllers
- Health and readiness probes for Kubernetes and Cloud Run compatibility
- OpenAPI 3.0 specification for chemical informatics endpoints
- Automated CI/CD pipelines using Google Cloud Build
- Initial release of research, threat-model, and deployment documentation

## [0.5.0-beta] - 2026-03-15

### Added

- Beta testing workflow for Tox-Agent focused on accuracy and interpretability evaluation
- Finalized database schema for molecular storage and researcher-specific history
- Local development kit with Firebase Emulator Suite integration for security testing

### Changed

- Refactored graph construction logic to use RDKit for more robust SMILES sanitization

## [0.1.0-alpha] - 2026-01-20

### Added

- Initial feasibility study based on seizure-detection research that informed the current GNN direction
- Proof-of-concept graph embedding model for small-molecule toxicity analysis

## Maintainers

- Teddy
- Nhat Minh
- Nghia Nguyen

## Organization

- NEU Bio Research Team
