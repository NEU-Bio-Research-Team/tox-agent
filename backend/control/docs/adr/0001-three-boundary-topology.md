# ADR 0001 — Three deployment boundaries

**Status:** accepted · **Date:** 2026-09-04 · **Plan:** §1.3, §4.1

## Decision

Three independent deployment boundaries:

| Boundary | Process | Owns |
|---|---|---|
| `toxpred` | existing predictor, offline after provisioning | canonical SMILES, prediction, threshold, applicability, attribution, model provenance |
| `toxagent-control` | this project | product API, auth, session, router, snapshots, evidence, answers, validation, SSE, tool gateway |
| `agent-runtime-host` | pinned OpenCode or DSH, never public | model-tool loop, provider requests, runtime-local context |

`toxagent-control` reaches `toxpred` only over its versioned HTTP contract
(`/v1/predictions`, `/v1/predictions:batch`, `/v1/attributions`, `/v1/models`,
`/health/*`). It never imports predictor code, checkpoints, or training modules.

## Repository placement (DEC-01)

The plan's default is a separate repository. This project is instead a
**monorepo sibling**, which §1.3 permits on the condition it is genuinely
isolated. The isolation is enforced, not asserted:

- its own `pyproject.toml`, package (`toxagent`), and dependency set;
- its own CI workflow and its own test root;
- `tests/unit/test_boundaries.py` fails the build if `toxagent` imports
  `toxpred`, `backend`, `torch`, `rdkit`, or any model code;
- the predictor's own `scripts/check_no_agent_deps.py` does not scan this
  directory, and nothing here is added to the predictor's scan set — so the
  predictor's "no agent dependency" gate keeps its original meaning.

Extraction to a standalone repository stays a `git subtree split` away because
no import crosses the directory boundary in either direction.

## Consequences

- Predictor releases and control-plane releases version independently; the
  pinned OpenAPI snapshot in `toxagent/predictor/contract_snapshot.json` is the
  compatibility gate between them.
- A predictor outage degrades to typed `predictor_not_ready` /
  `endpoint_unavailable` errors, never to a substituted answer.
