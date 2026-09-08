# toxagent-control

The ToxAgent control plane: an evidence-and-decision-support layer over the
ToxPred predictor. It keeps the predictor's quantitative truth intact, adds
sourced evidence, answers follow-up questions with grounding, and proposes the
next verification step — through an audit trail that can be reconstructed.

It is a **separate deployable** from the predictor. It talks to ToxPred over
`/v1` HTTP only, and it imports no model code. See
[`docs/adr/0001-three-boundary-topology.md`](docs/adr/0001-three-boundary-topology.md).
The same discipline extends to `../toxocr` (image -> SMILES structure
recognition) — see
[`docs/adr/0006-ocr-fourth-boundary.md`](docs/adr/0006-ocr-fourth-boundary.md).

## Layout

```
toxagent/
  api/           product HTTP API, SSE, error envelope
  domain/        session, run, analysis, observation, evidence, answer
  application/   the workflows: analyse, answer, research, submit, cancel
  predictor/     pinned ToxPred client, OpenAPI snapshot, and the toxocr client
  research/      evidence provider interfaces, normalisation, policy
  tools/         typed tool registry, runner, projections, MCP server
  harness/       AgentRuntimeGateway and the runtime adapters
  validation/    numeric, classification, citation, limitation validators
  persistence/   store interfaces and the SQLAlchemy implementation
  streaming/     transactional outbox and SSE dispatch
  telemetry/     traces and metrics
agent_profiles/  pinned OpenCode / DSH agent configuration and prompts
evals/           task set, frozen fixtures, graders, manifests, runner
```

## What is enforced, not merely intended

- A number in an accepted answer equals the predictor field it cites, or the
  answer does not exist (`validation/numeric.py`).
- hERG, Tox21 and ClinTox never substitute for one another, and there is no
  aggregate score in any schema (ADR 0002).
- A denied tool is invisible to the model *and* refused at the transport.
- Losing the runtime loses no product state; recovery opens a new run.

## Running the tests

```
pip install -e 'toxagent-control[dev]'
python -m pytest toxagent-control/tests -q
```

Suites marked `live_predictor`, `live_runtime` or `live_evidence` need a real
dependency and are deselected by default.

## OpenCode V1 runtime host

The only supported app-facing runtime is the pinned OpenCode V1 `1.17.11`
adapter.  Run its management API on loopback/private networking only, with
[`agent_profiles/opencode/toxagent.json`](agent_profiles/opencode/toxagent.json)
as its configuration.  The profile is deny-all and only enables the ToxAgent
MCP namespace.

To enable the adapter, configure the control plane with at least:

```text
TOXAGENT_RUNTIME_KIND=opencode
TOXAGENT_OPENCODE_URL=http://opencode-runtime.internal:4096
TOXAGENT_OPENCODE_VERSION=1.17.11
TOXAGENT_OPENCODE_DIRECTORY=/var/lib/toxagent/opencode-runs
TOXAGENT_MCP_RUNTIME_URL=https://toxagent-control.internal/internal/mcp
TOXAGENT_CAPABILITY_SECRET=<secret-manager value>
```

`TOXAGENT_OPENCODE_DIRECTORY` is a runtime-host-owned base directory. The
adapter derives one isolated child directory per product run, then configures
the remote MCP over OpenCode's private V1 `/mcp` API only after the immutable
runtime binding exists. Its short-lived capability token is sent as the MCP
authorization header; it is never included in the prompt, run manifest, event,
or product database. The runtime-host supervisor must provision the child
directory before it accepts the V1 session and reap it after the adapter closes
the runtime-local session.

## Local Phase 3 stack

For a localhost-only integration test, activate the existing `drug-tox-env`,
choose an already configured OpenCode provider/model, then run the launcher
from the repository root:

```bash
conda activate drug-tox-env
opencode models
TOXAGENT_OPENCODE_MODEL=provider/model ./scripts/run_local_phase3.sh
```

The launcher binds ToxPred (`8080`), OpenCode (`4096`), and the control plane
(`8000`) to loopback, runs the SQLite migration, and enables local-only
workspace provisioning/reaping. It does not accept external traffic. In a
second terminal, run `./scripts/smoke_local_phase3.sh` to submit aspirin SMILES
to the pinned checkpoint and then attempt a grounded report Q&A through
OpenCode MCP. The Q&A consumes the selected LLM provider and its cost/limits
apply.
