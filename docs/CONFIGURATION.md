# Configuration

Copying `.env.example` manually is supported, but `./bin/toxagent setup` is
the normal path and preserves all existing values.

| Variable | Required | Default | Secret | Restart |
|---|---:|---|---:|---:|
| `FRONTEND_PORT` | no | `8088` | no | yes |
| `POSTGRES_PASSWORD` | yes | generated | yes | yes |
| `TOXAGENT_STATIC_TOKENS` | local only | generated | yes | yes |
| `TOXAGENT_CAPABILITY_SECRET` | yes | generated | yes | yes |
| `MODEL_ARTIFACTS_URI` | no | empty/local models | no | yes |
| `TOXOCR_CHECKPOINT_HOST_PATH` | yes | `.artifacts/toxocr` | no | yes |
| `TOXAGENT_ACCELERATOR` | no | `cpu` | no | yes |

For GPU, use `docker compose -f compose.yaml -f infra/compose/gpu.yaml up` only
after the host NVIDIA runtime has been verified. For an external, pinned
OpenCode deployment, use `infra/compose/external-opencode.yaml` and set its URL,
runtime-owned directory, MCP URL, provider and model ID. This is an advanced
deployment path; it is not required for prediction or OCR.

Production must replace the local static-token mechanism with an approved JWT
issuer and set `TOXAGENT_ENV=production`; static tokens are rejected there.
