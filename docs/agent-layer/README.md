# The agent layer

Written for engineers working on this repository.

The agent layer is the part of the control plane that turns a researcher's
toxicology question into an answer whose every number cites the artifact it
came from. It sits between the HTTP API and the model runtime, and it owns the
state the model is not allowed to own.

## What it is not

- **Not the workflow.** Activity screening → policy decision → optional
  toxicity prediction is executed by the server, deterministically. A
  scientific conclusion the agent reaches never overrides a policy decision or
  replaces a prediction.
- **Not a safety verdict.** Every answer carries the limitations its sources
  require. A margin is a ratio of two measurements; whether it is acceptable is
  a judgement about the conditions they were measured under.
- **Not on by default.** Each capability here is behind a rollout flag whose
  default is off. A deployment with every flag off answers from the artifacts
  it already holds, as it did before this layer existed.

## The one-paragraph version

A message is admitted, bound to one subject (a molecule's analysis, or one
activity screening), and a run is created. The server decides which of four
serving routes the question takes, builds a goal contract and a view of the
case, and dispatches one model turn with only the tools that route allows. The
model reads artifacts, retrieves evidence, runs determinate procedures and
writes to the case through typed operations; it never edits state directly. It
submits a draft answer, which a hard validator checks field by field, and then
an independent reviewer turn judges each claim against the sources it cites.
Only what passes is released, and the run records why it stopped.

## Which document

| Question | Document |
|---|---|
| What are the parts, and what happens in what order? | [architecture.md](architecture.md) |
| How do I run one turn and see it work? | [running.md](running.md) |
| How do I add a tool, an operation, a procedure? | [extending.md](extending.md) |
| How do I run the arms and read the results? | [benchmark.md](benchmark.md) |
| Why is it built this way, and what is proven? | [../research/](../research/) |

## Where the code is

Everything below is under `backend/control/src/toxagent/`.

| Path | What lives there |
|---|---|
| `application/investigation/` | The layer's logic: router, goal contract, case view, procedures, case service, claim review, release gate, skills |
| `domain/scientific_case/` | The case: its model, its typed operations, claim dependencies, researcher values |
| `domain/` | Measurement values, evidence relations, observations, stop reasons, gaps, subjects |
| `harness/gateway/` | Turn assembly and dispatch: context, case hooks, release, completion |
| `tools/` | The tool registry, the capability profiles, and each tool definition |
| `validation/` | The answer schemas, the hard validator, the deterministic fallback |
| `platform/flags.py` | Every rollout flag, with its owner and its removal condition |

The benchmark lives beside it, in `backend/control/evals/investigation/`.
