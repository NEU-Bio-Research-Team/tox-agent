# ToxAgent documentation

Start at the [root README](../README.md) to install and run the stack. This
directory documents what is inside it.

## The agent layer

How a toxicology question becomes a grounded answer, and how to work on it.

| Document | Read it for |
|---|---|
| [agent-layer/README.md](agent-layer/README.md) | The shape of the layer, and which document answers which question |
| [agent-layer/architecture.md](agent-layer/architecture.md) | The components, the path a turn takes, and where state lives |
| [agent-layer/running.md](agent-layer/running.md) | Running a turn locally, the rollout flags, the test suites |
| [agent-layer/extending.md](agent-layer/extending.md) | Adding a tool, a case operation, a procedure, a stop reason or a flag |
| [agent-layer/benchmark.md](agent-layer/benchmark.md) | Running the protocol v2 arms and reading what comes back |

## The research design

Why the layer is built this way, what is being claimed, and what has been
measured so far.

| Document | Read it for |
|---|---|
| [research/README.md](research/README.md) | The claims, their status, and how to check them |
| [research/hypotheses.md](research/hypotheses.md) | H1–H3: the mechanism each one needs and how it is varied |
| [research/evaluation.md](research/evaluation.md) | Arms, case sets, metrics, and what the numbers may not be read as |
| [research/status.md](research/status.md) | What is implemented, what is measured, what is open |

## Reference and decisions

| Path | What it is |
|---|---|
| [adr/](adr/) | Architecture decision records, in order |
| [reference/configuration.md](reference/configuration.md) | Every environment variable the services read |
| [reference/metrics.md](reference/metrics.md) | The metric names the control plane emits |
| [reference/model-card.md](reference/model-card.md) | The predictor's model card |
| [reference/benchmark-protocol.md](reference/benchmark-protocol.md) | How the predictor benchmark numbers are produced |
| [reference/architecture-inventory.json](reference/architecture-inventory.json) | Generated: intents, profiles, flags, queues, schemas |
| [reference/capability-matrix.md](reference/capability-matrix.md) | Generated: which capability each flag and profile turns on |
| [how-to/operations.md](how-to/operations.md) | Backup, restore, upgrade, incident response |

The two generated files are written by `python -m evals.architecture_inventory
--write` and `python -m evals.capability_matrix --write` from
`backend/control`, and a unit test fails when either is stale.

## Historical records

These are evidence of what was true when written. They are not edited to match
today's tree, and `devops/checks/check_docs.py` exempts them from its path
checks.

- `results/` — dated measurement reports, with the commit each ran on.
- `internal/` — audits and the issue catalogue the regression guards index.

Both stay in the development repository and are not part of a distribution.

Documentation removed on 2026-10-30, available in git history: the tutorial,
the workspace-layout and architecture explanations, the development how-to, the
ADS glossary, the generated model card, and the plans and specs under
`internal/spec/` and `internal/backlog/` — including the 2026-10-28 agent
architecture proposal, which these documents replace as the description of what
exists.
