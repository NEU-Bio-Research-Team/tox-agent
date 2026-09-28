# Documentation index

Every Markdown file under `docs/`, grouped by what a reader is trying to do
([Diátaxis](https://diataxis.fr/)): learn the product, get a task done, look
something up, or understand why it is built this way. Records of past work and
measurements sit apart from all four.

Three statuses, and they mean different things:

- **Current** — describes the tree as it is now. Commands must run from a clean
  clone. `devops/checks/check_docs.py` enforces that these name no retired
  path and that every relative link resolves.
- **Generated** — written by a script; regenerate rather than hand-edit. A unit
  test fails when the checked-in copy drifts.
- **Historical** — evidence of what was true when it was written: everything
  under `results/` and `internal/`, and the ADRs. Not edited to match today's
  tree; exempt from the retired-path check, never from the link check. A link
  to a document that no longer exists is kept as text, marked `removed:`.

File names are `kebab-case.md`; a Vietnamese document ends in `.vi.md`.

## Tutorials — learning the product

| Document | Status | What it is for |
|---|---|---|
| [`../README.md`](../README.md) | Current | Repository overview and the seven-step setup |
| [`tutorials/getting-started.md`](tutorials/getting-started.md) | Current | First run via Docker Compose |

## How-to guides — getting a task done

| Document | Status | What it is for |
|---|---|---|
| [`how-to/operations.md`](how-to/operations.md) | Current | Running, stopping, backing up and restoring the stack |
| [`how-to/development.md`](how-to/development.md) | Current | Per-service test commands |
| [`how-to/tab-suite-benchmark-setup.md`](how-to/tab-suite-benchmark-setup.md) | Current | Running the TAB-Suite benchmark lanes |

## Reference — looking something up

| Document | Status | What it is for |
|---|---|---|
| [`reference/configuration.md`](reference/configuration.md) | Current | Environment variables |
| [`reference/metrics.md`](reference/metrics.md) | Current | Metrics the control plane exports |
| [`reference/model-card.md`](reference/model-card.md) | Current | Short customer-facing model card |
| [`reference/model-card-generated.md`](reference/model-card-generated.md) | Generated | Full ToxPred model card, with split hash and benchmark commit |
| [`reference/benchmark-protocol.md`](reference/benchmark-protocol.md) | Current | Frozen split and metric definitions |
| [`reference/artifacts/herg-tox21-chemberta-v1.md`](reference/artifacts/herg-tox21-chemberta-v1.md) | Current | Admitted model card |
| [`reference/artifacts/clintox-smilesgnn-v1.md`](reference/artifacts/clintox-smilesgnn-v1.md) | Current | **Blocked** artifact — records why v1 cannot be served (I07) |
| [`reference/ads-glossary.md`](reference/ads-glossary.md) | Current | Adaptive decision support vocabulary |
| [`reference/architecture-inventory.json`](reference/architecture-inventory.json) | Generated | Intents, lanes, flags, profiles, queues, providers, eval packs, graders, schema versions — `python -m evals.architecture_inventory --write` |
| `reference/capability-matrix.md`, `reference/capability-matrix.json` | Generated | Served/blocked models, explainer validation, tool profiles, flags, instruction cost — `python -m evals.capability_matrix --write` |

## Explanation — understanding why

| Document | Status | Scope |
|---|---|---|
| [`explanation/architecture.md`](explanation/architecture.md) | Current | Service boundaries across the whole product |
| [`explanation/workspace-layout.md`](explanation/workspace-layout.md) | Current | Where each service, script and document lives |
| [`explanation/predictor-architecture.md`](explanation/predictor-architecture.md) | Current | **ToxPred only** — the predictor's internal layering |
| [`adr/`](adr/) | Historical | Architecture decision records, ADR 0001–0012, for the whole product |
| [`../backend/control/README.md`](../backend/control/README.md) | Current | Control plane: package layout and dependency order |
| [`../backend/ocr/README.md`](../backend/ocr/README.md) | Current | Structure-recognition service |

[`../frontend/README.md`](../frontend/README.md) covers the frontend: commands and feature layout. See also
[`how-to/development.md`](how-to/development.md).

## Results — dated measurements

| Document | Status |
|---|---|
| `results/tab-suite-live-2026-09-17.md` | Historical — TAB-Suite live results, 2026-09-17 |
| `results/bm1-explainer-benchmark-analysis.md` | Historical — explainer analysis of the pre-refactor monolith |

## Internal documents

**These live in the development repository and are not part of a customer
distribution** — `devops/handoff_allowlist.json` withholds `results/` and
`internal/`, and the names above and below are deliberately not links so a
handed-over clone has no dead ones.

| Document | Status |
|---|---|
| `internal/rethink-agentic-research-evaluation.vi.md` | Current — research and design proposal for the agent's scientific role, external benchmarks, and evaluation protocol; observed-state claims are pinned to commit `ad66022`. |
| `internal/backlog/SCIENTIFIC_INVESTIGATION_BACKLOG.md` | Current — execution backlog and status for the RETHINK proposal, wave by wave |
| `internal/external-benchmarks.md` | Current — which published benchmark runs, which is blocked and on what, and the label each result may carry |
| `internal/spec/WORKSPACE_AND_UI_SIMPLIFICATION_PLAN.md` | Historical — the workspace cleanup and UI simplification plan, with its execution log |
| `internal/spec/WORKSPACE_STRUCTURE_REVIEW.md` | Current — structure review of the code that stays and its phased plan, with an execution log |
| `internal/audit/` | Historical — the 2026-09 system audit and the regression guards it produced (`regression_guards.json` is read by `devops/tests`) |

**Removed 2026-09-27, at the product owner's request:** the process
documentation — the plan, audit, progress, review, runbook, `unified-v2` and
`refactor` documents (40 files). They recorded work that is finished; what they
established lives in the commits that did it, in the ADRs and in the backlog.
`git log --diff-filter=D -- docs/` names them, and `git show <commit>^:<path>`
still reads any one of them. A citation to one of these paths in a code comment
or a task fixture is provenance — what the author read at the time — and was
left as written.

## Test numbers cited anywhere in these documents

A count belongs to the commit, environment and configuration it was measured
on. Do not add counts from different runs together, and do not treat a figure
recorded in a progress log as a statement about a later HEAD. Baselines for a
release carry commit, image digest, artifact hashes and skip reasons.
