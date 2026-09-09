# Configuration

Copying `.env.example` manually is supported, but `./bin/toxagent setup` is
the normal path and preserves all existing values.

| Variable | Required | Default | Secret | Restart |
|---|---:|---|---:|---:|
| `FRONTEND_PORT` | no | `8088` | no | yes |
| `POSTGRES_PASSWORD` | yes | generated | yes | yes |
| `TOXAGENT_STATIC_TOKENS` | local only | generated | yes | yes |
| `TOXAGENT_CAPABILITY_SECRET` | yes | generated | yes | yes |
| `TOXAGENT_OIDC_ISSUER` | production | empty | no | yes |
| `TOXAGENT_OIDC_AUDIENCE` | production | empty | no | yes |
| `TOXAGENT_OIDC_JWKS_URL` | production | empty | no | yes |
| `TOXAGENT_OIDC_JWKS_CACHE_S` | no | `300` | no | yes |
| `TOXAGENT_OIDC_ROLES_CLAIM` | no | `roles` | no | yes |
| `MODEL_ARTIFACTS_URI` | no | empty/local models | no | yes |
| `TOXOCR_CHECKPOINT_HOST_PATH` | yes | `.artifacts/toxocr` | no | yes |
| `TOXAGENT_ACCELERATOR` | no | `cpu` | no | yes |

For GPU, use `docker compose --project-directory . -f devops/compose/compose.yaml -f devops/compose/gpu.yaml up` only
after the host NVIDIA runtime has been verified. For an external, pinned
OpenCode deployment, use `infra/compose/external-opencode.yaml` and set its URL,
runtime-owned directory, MCP URL, provider and model ID. This is an advanced
deployment path; it is not required for prediction or OCR.

Production must replace the local static-token mechanism with an identity
provider and set `TOXAGENT_ENV=production`; static tokens are rejected there,
and so now is starting with no provider at all.

The three `TOXAGENT_OIDC_*` variables are required together. Setting some of
them is refused rather than partially applied: a verifier with an issuer but
no audience accepts a token the same provider minted for a different service,
and one with neither accepts a token from any provider whose key set it
fetches. The JWKS URL must be `https`.

`TOXAGENT_CAPABILITY_SECRET` is not an alternative. It signs the run-scoped
tokens a runtime carries — a separate trust domain, deliberately not
interchangeable with user authentication. Until this change, a deployment with
no identity provider fell back to verifying *user* tokens with that same key
and checked neither issuer nor audience, so anything able to sign with it
could present any subject and any roles. Production now refuses to start in
that state.

Key rotation needs no restart: the key set is cached for
`TOXAGENT_OIDC_JWKS_CACHE_S`, and a token whose `kid` is not in the cache
refetches immediately rather than failing until the cache expires.
