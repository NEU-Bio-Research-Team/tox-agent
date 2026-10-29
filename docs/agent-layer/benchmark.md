# Running the benchmark

Written for engineers working on this repository. For what the numbers mean and
may not be read as, see [../research/evaluation.md](../research/evaluation.md).

The protocol v2 harness lives in `backend/control/evals/investigation/`, with
its own README for the v1 studies that preceded it. One entry point:

```bash
python scripts/agent_benchmark.py preflight --case-set <set> [--arm ID=URL ...]
python scripts/agent_benchmark.py run     --study <id> --case-set <set> --arm ID=URL ...
python scripts/agent_benchmark.py packet  --study <id> --case-set <set> --packet-id <id>
python scripts/agent_benchmark.py score   --study <id> --case-set <set> --packet-id <id> --grades *.csv
python scripts/agent_benchmark.py freeze  --study <id> --case-set <set>
```

## The arms

Each arm is a **separate deployment of this control plane**, differing only in
the flags it declares. `arms/protocol-v2/*.json` are the configs; each
`extends` `common.json`, scalars replace and `required_flags` merge key by key.

| Arm | Adds to the one above | The question it answers |
|---|---|---|
| `A2_template` | No model at all: code renders the frozen record through a fixed template | The floor |
| `B2_fixed_workflow` | `fixed_workflow_v1`: the server reads, retrieves and cross-checks by rule; one synthesis turn writes | Is an agent needed, or is a good pipeline enough? |
| `C2_generic_react` | The model chooses its own actions, same tools and gates, no case, no skills | Is a generic agent enough? |
| `D2_investigator` | `scientific_case_v1`, `scientific_skills_v1` | Does a persistent case and on-demand skills add anything? |
| `E2_contract_investigator` | `investigator_contract_v1`, `scientific_procedures_v1`, `scientific_primitives_v1` | Does the 2026-10-28 architecture add anything over D? |

B, C, D and E share the model, the budget preset, the output schema, the hard
validator, the reviewer, the release policy and the frozen inputs, so a
difference between them comes from the design.

## Fairness, enforced rather than remembered

`preflight` reads each deployment's `/v1/system/effective-product` and refuses
the group when:

- a field every arm must share differs — the backbone, the whole effective
  budget (not its label), the release policy, the pinned corpus, the full-text
  store, the replay file, the live activity lookup;
- a mandatory pin is not reported at all, because agreement on `None` is a
  missing pin rather than fairness;
- **a flag differs between two arms and is not a declared treatment.** An
  undeclared variable is a confound. Adding a flag to an arm therefore means
  adding it to `treatment_fields` in `common.json`, with the other arms
  requiring it off.

This is what catches the mistake that is easy to make and hard to see: in one
run, arm E's `scientific_primitives_v1` brought a live ChEMBL provider, and the
"frozen" benchmark reached the network. `providers.activity_lookup` is now a
shared field, so that arm is refused beside the others.

## Running it on a laptop

The constraint is memory, not CPU. The pattern that works on 8 GB: one arm at a
time, `--parallel 1`, one case per invocation, and restart the model runtime
between cases — it keeps roughly 200 MB per run. Each arm gets a fresh migrated
SQLite database, and the runner records every attempt, including the failed
ones.

```bash
export PYTHONPATH="$PWD/src"
python scripts/agent_benchmark.py run --study <id> --root <out> \
  --case-set evals/investigation/cases_protocol_v2_technical \
  --arm E2_contract_investigator=http://127.0.0.1:8024 \
  --systems E2_contract_investigator --trials 1 --parallel 1 --cases <case-id>
```

A run that fails `runtime_unavailable` is retried once by the server as a
*recovery run*; the adapter follows it, because sending the next turn while the
recovery is still in flight is refused with a 409 and recorded as an
infrastructure error.

## What comes back

`records.jsonl` holds one record per (case, system, trial) attempt, each with
its turns. Per turn: the run id, the bound subject's hash, the route and its
reason codes, the stop reason, the disposition, the review status, whether the
answer was the fallback, the tool calls, the phase usage, and the case revision
the dossier reached. Beside it, `raw/<arm>/<case>/<trial>/trace.json` carries
the full decision state, dossier, answer drafts and the evidence the session
accepted.

`score` produces the scorecard: task success by task group, paired deltas with
a cluster bootstrap, mandatory information, limitations, critical errors, grader
agreement, completeness, retrieval, hallucination as atomic-claim precision,
predictor adjudication, cost per arm and phase, and routing — routes per arm
with the median turn time and tool calls, read apart from task success.

Two things the card will tell you rather than hide: the dimensions have **no
total score**, and routing *errors* are "not measured", because no gold file
states the route a turn should have taken.

## Before trusting a number

- Was it a primary attempt, or a diagnostic rerun? The runner distinguishes
  them and the card counts them separately.
- Did preflight pass, and did the effective products agree?
- How many dependency clusters does the case set have? The technical set's 12
  cases are two compounds, so a bootstrap over clusters has two units and its
  interval cannot be read.
- Is the gold SME-verified, or `synthetic_engineering_gold`? The scorer prints
  the answer, and for the technical set it is the latter.
