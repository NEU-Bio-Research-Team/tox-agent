# Legacy Workspace Surfaces

This directory contains archived prototypes that were removed from the main product root.

Current production deploy path:
- `frontend/` builds the Firebase Hosting app.
- `model_server/` provides the Cloud Run backend.
- `agents/`, `backend/`, and `services/` contain the active runtime packages.

Archived surfaces:
- `api-local-tester/`: old root-level API playground used for manual localhost checks.
- `landing-page-prototype/`: design prototype that is not part of the production hosting build.
- `streamlit-clintox/`: old Streamlit ClinTox UI kept only for historical reference.

These folders are intentionally excluded from the main deploy path and should not be treated as the source of truth for the current product.