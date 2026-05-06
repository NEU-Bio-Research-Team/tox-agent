# Legacy Compatibility Layer

This directory is kept only to preserve historical import paths such as `src.inference`
and `src.graph_data` for older scripts.

Current source of truth:
- `backend/` contains the active ML and inference modules.
- `agents/` contains the active orchestration layer.
- `services/` contains the active infrastructure integrations.

Do not add new runtime logic here unless you are intentionally creating a compatibility wrapper.