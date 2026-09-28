# Workspace structure review and improvement plan

**Date:** 2026-09-28 · **Branch:** `feat/scientific-investigation` · **HEAD:** `e8519c7`
**Status:** in progress — Phases 0 and 1 executed 2026-09-28; see the execution log at the end.

This is the follow-on to
[`WORKSPACE_AND_UI_SIMPLIFICATION_PLAN.md`](WORKSPACE_AND_UI_SIMPLIFICATION_PLAN.md).
That plan removed what no longer belongs to the product (dead roots, old docs,
unused UI code). This one looks at the code that stays and asks whether its
structure is easy to read, easy to change, and consistent across services.

---

## 1. Verdict

The **deployable topology is right**; the **inside of the largest service is
not yet organised for its size**, and the **tooling around the code is uneven**.

| Area | Grade | One-line reason |
|---|---|---|
| Top-level topology (`frontend/`, `backend/{control,predictor,ocr}/`, `devops/`) | Good | One directory per deployable boundary, matching ADR 0001/0006 |
| `backend/predictor/src/toxpred` | Good | 3.8k lines, clear `api → application → domain / scientific` layering |
| `backend/ocr` | Fair | Small and clear, but packaged differently from the other two services |
| `backend/control/src/toxagent` | **Needs work** | 46k lines in 19 packages; features scattered across layers, several 1k–2.4k-line modules, 7 packages the README does not list |
| `backend/control/evals` | **Needs work** | Eval code and ~12 MB of run output share one tree |
| `backend/predictor/research` + `evals/scripts` | Fair | Isolated from the service, but eval scripts still import `legacy_backend` |
| `frontend/src` | Fair | Readable at 17k lines; API types hand-written; no general linter |
| DevOps / CI | Fair | Strong test gates; no Python lint/type check, no lockfiles, scripts in six places |
| Docs | Fair | Indexed and link-checked; two doc roots and mixed naming |

"Good" means: leave it alone. Most of the work below is in `backend/control`,
the evals tree, and tooling.

---

## 2. How this was reviewed

1. **Graphify** (`graphify-out/`, built at `e9638bf6`, one `.graphifyignore`-only
   commit behind HEAD): 11,860 nodes, 30,746 edges, 479 communities. Used for
   orientation — god nodes, community spread, and `graphify query` on how the
   report feature is split.
2. **Direct measurement** of the tree: tracked-file counts per directory
   (`git ls-files`), line counts per package, and an AST pass over every
   `import` in `backend/control/src/toxagent` to find which package depends on
   which.
3. **Reading** each service's `pyproject.toml` / `package.json`, the CI
   workflow, `bin/toxagent`, the existing boundary test
   (`backend/control/tests/unit/test_boundaries.py`) and the docs index.

Every number below was measured on `e8519c7` and belongs to that commit.

---

## 3. Reference patterns from large product organisations

Only the patterns that change a recommendation here are listed.

| Pattern | Where it comes from | What it means for this repo |
|---|---|---|
| **One directory per deployable, grouped by domain** | Uber's Domain-Oriented Microservice Architecture groups services into domains with one gateway each | Already done. Do **not** rename `backend/` or split repos — the cost is high and the gain is nil |
| **Modular monolith with enforced package boundaries** | Shopify's core monolith is split into components with public entry points; Packwerk fails CI on dependency and privacy violations | `toxagent` is a monolith inside one deployable. Give it feature subpackages and make the dependency rules a failing check, not a docstring |
| **Layered / hexagonal architecture checked by a linter** | `import-linter` "layers" contracts: a lower layer may not import a higher one, even indirectly | Extend the existing `test_boundaries.py` (or adopt `import-linter`) so the whole layer order is enforced, not only `domain` and the tool gateway |
| **Schema-first API, generated clients** | Common practice for public APIs: the server's schema is the contract, clients are generated from it | Add `response_model` to control-plane routes, then generate `frontend/src/lib/api` types instead of hand-typing them |
| **Code and experiment output live apart** | ML platforms keep run artifacts in an artifact store (object storage, MLflow/DVC-style), not beside the code | Move eval run output out of `backend/control/evals/` into one results root (or object storage), keep only summaries in git |
| **One toolchain per language, one entry point per repo** | Monorepos standardise formatter, linter, type checker and a top-level task runner | Add `ruff` + a type checker for Python, ESLint for TS, Python lockfiles, and a root `Makefile` that delegates to each service |
| **Documentation by purpose (Diátaxis)** | Tutorials, how-to guides, reference, explanation — kept separate | Split `docs/` by purpose; separate operator docs from research results |

Sources: see §9.

---

## 4. What is already right — keep it

- **Deployable boundaries are directories.** `frontend/`, `backend/control/`,
  `backend/predictor/`, `backend/ocr/`, each with `src/`, `tests/`, `deploy/`
  (Dockerfile + entrypoint). Compose overlays live in `devops/compose/`.
- **`toxagent.domain` is pure.** The import pass found zero imports from
  `domain/` into any other `toxagent` package, and `test_boundaries.py` enforces
  it together with "no model code in the control plane".
- **src-layout and packaging are tested.** CI builds wheels, installs them into
  a clean venv and imports from an unrelated working directory; the control
  image is checked for the agent profiles it will load.
- **Rollout flags carry `owner`, `added_on` and `remove_by`**
  (`backend/control/src/toxagent/flags.py`) — the discipline many teams lack.
- **Workspace hygiene has checkers.** `devops/scripts/check_docs.py` (dead links,
  retired paths) and `check_workspace.py --strict` (stray source roots) run in CI.
- **ADRs exist** (0001–0012) and the docs index records every file's status.
- **The predictor service is small and layered** (`api/`, `application/`,
  `domain/`, `scientific/`), with research code outside the package.

---

## 5. Findings

Each finding has an ID used by the plan in §7.

### 5.1 `backend/control` — the control plane

**C1. Features are scattered across layers.** The package is organised
layer-first (`application/`, `domain/`, `validation/`, `tools/`, …). At 46k
lines this hides features: the report feature alone is **21 files in 6
packages** —

```text
application/  report_dispatch, report_orchestrator, report_stage_handlers,
              report_stages, report_synthesis, report_inputs, submit_report_draft
report/       compiler, compiler_v3, artifact_v3, fact_bundle, figures, renderers
domain/       report.py
validation/   report_validator, report_semantics, report_wire, synthesis_validator,
              synthesis_wire
tools/        definitions/report.py, definitions/report_synthesis.py
harness/      report_profile.py, synthesis_profile.py
```

`application/` is a flat directory of **37 modules, 10.8k lines** spanning at
least six features (report, runs/scheduling, explanation, scientific case,
messaging/answers, skills).

**C2. Layer rules are partly unenforced, and several are broken.**
`test_boundaries.py` checks `domain` purity, "no model imports", and "tools do
not call the runtime gateway". Nothing checks the rest. The first import pass
found the rows marked *(review)*; executing Phase 1, a pass that also follows
imports made inside functions found the rest *(Phase 1)*:

| From | To | Where |
|---|---|---|
| `validation` | `application` | `validation/report_semantics.py:37` imports `application.xai_coverage` *(review)* |
| `report` | `application` | `report/fact_bundle.py:39` imports `application.xai_coverage` *(review)* |
| `report` ↔ `validation` | cycle | `report/compiler*.py`, `renderers.py` import `validation.*`; `validation/synthesis_validator.py:22-23` imports `report.compiler_v3`, `report.fact_bundle` *(review)* |
| `persistence` | `agent` | `persistence/sql/repositories.py`, `persistence/investigations.py` import `agent.kernel.KernelTransition` *(review)* |
| `application` | `tools` | `report_dispatch.py` (`ToolContext`), `run_budget.py`, `skill_catalog.py`, `skill_drafts.py`, `effective_product.py` — mostly imports inside functions, i.e. cycle workarounds *(Phase 1)* |
| `application` | `harness` | `effective_product.py` *(Phase 1)* |

`xai_coverage` is a pure stdlib computation living in the application layer.
`agent/`, `answer/` and `capabilities/` are the kernel ADR 0011 **superseded**:
kept deliberately, unimported by any live module (a test in
`test_eval_paired.py` enforces that), until its tables are retired — so
`persistence → agent` is retired together with it, not moved.

**C3. God modules.** Files large enough that one change touches unrelated code:

| File | Lines | Why it is large |
|---|---|---|
| `persistence/sql/repositories.py` | 2,358 | Every repository in one file |
| `harness/gateway.py` | 1,531 | Runtime gateway plus its helpers |
| `api/routes.py` | 1,464 | 55 routes in one router |
| `domain/report.py` | 1,158 | All report types |
| `domain/scientific_case.py` | 1,141 | Case model, transitions and helpers |
| `validation/report_validator.py` | 1,079 | Every report check |
| `application/submit_report_draft.py` | 987 | One use case |

**C4. Package list drifts from the README; micro-packages.** The README
layout lists 12 packages; the tree has 19. Missing from the README:
`activities/` (54 lines), `agent/` (372), `answer/` (99), `capabilities/` (140),
`connections/` (734), `report/` (3,060), `runtime/` (37). `telemetry/` is an
empty `__init__.py` while `metrics.py` and `observability.py` sit at the package
root. `capabilities/registry.py` and `application/capabilities.py` share a name
with different meanings. `agent/`, `answer/` and `capabilities/` are the
superseded ADR 0011 kernel (C2) — nothing in the package says so except a
docstring; they belong under one clearly named `superseded/` subpackage until
they are deleted.

**C5. Version numbers in module names.** `report/compiler.py` +
`compiler_v3.py` + `artifact_v3.py`; `validation/wire.py` + `wire_v2.py`, next to
`report_wire.py` and `synthesis_wire.py`. Both generations are live (legacy
behind flags), which is fine as a strangler migration, but the file name should
say *role*, and the old path should be visibly marked for deletion on the flag's
`remove_by` date.

**C6. The HTTP contract is untyped.** Only **7 of 55** routes in
`api/routes.py` declare a `response_model`. The frontend therefore hand-types
the contract in `frontend/src/lib/api/types.ts` (1,102 lines). Its provenance
comment cites `docs/spec/TOXAGENT_AGENTIC_LAYER_REBUILD_PLAN_VI.md`, which was
deleted on 2026-09-27. The `openapi:types` npm script exists but nothing uses
its output. Six code comments in total cite deleted `docs/spec|audit|…` paths.

**C7. Eval code and eval output share a tree.** `backend/control/evals/` has
551 tracked files; 290 are run output under `investigation/runs/` (8.7 MB), and
`manifests/` holds another 3.5 MB — with 30+ more `live-b*` run directories
untracked right now. Code (`runner.py`, `graders/`, `packs/`, `schema/`) is
mixed with results, and `.gitignore` handles it path by path. `evals` and
`scripts` are importable only through `pythonpath = ["src", "scripts", "."]` in
`pyproject.toml`; they are not an installable package.

**C8. Tests do not mirror the source.** `tests/unit/` is a flat directory of
95 files. It works, but finding "the tests for `validation/`" means grepping.

### 5.2 `backend/predictor` and `backend/ocr`

**P1. Research still leaks into evals.** `research/legacy_backend/` is imported
by `evals/scripts/{consolidate_results,generate_curves,eval_dualhead_new_models,explainability_benchmark}.py`
and `evals/benchmark/{build_split_manifest,capture_baseline}.py`. `configs/`
holds 21 training configs; one model is served.

**P2. Dependencies declared twice.** `pyproject.toml` and
`deploy/requirements.txt` both list runtime dependencies, and they already
differ (`transformers==4.56.2` is pinned only in the requirements file). The
research tree adds a third environment (`environment.yml`,
`requirements-legacy.txt`, `constraints/`).

**O1. OCR is packaged differently.** `backend/ocr` has `requirements.txt` and no
`pyproject.toml`; its tests run with `PYTHONPATH=src`. The `torch<2` pin is a
good reason for a separate environment, not for a different packaging style.

### 5.3 `frontend`

**F1. Folders by type, not by feature.** `components/ pages/ lib/ hooks/` is
fine at 17k lines, but features are already split across them (report UI in
`components/transcript/report/`, `lib/store/reportProgress.ts` and pages). The
next 10k lines will make this harder.

**F2. No general linter or formatter.** `tsc` and a custom `policy-lint.mjs`
run in CI; there is no ESLint (hooks rules, unused imports) or Prettier.

**F3. No `frontend/README.md`** (the docs index says so explicitly).

**F4. Large page files.** `pages/LandingPage.tsx` (796 lines),
`pages/QuickPredictPage.tsx` (451), `pages/WorkbenchPage.tsx` (429).

### 5.4 DevOps and CI

**D1. Scripts live in six places**: `devops/scripts/`,
`backend/control/scripts/`, `backend/predictor/scripts/`,
`backend/predictor/evals/scripts/`, `backend/predictor/research/scripts/`,
`frontend/scripts/`. Inside `devops/scripts/`, release tooling (`handoff.py`,
`release_manifest.py`, `deploy_target.py`, `restore_drill.sh`) sits next to
eval-lane scripts (`run_tab_suite.sh`, `study_arm.sh`, `tab_lane.sh`) and the
OpenCode bridge (`agent/`).

**D2. No single developer entry point.** `bin/toxagent` is the operator CLI;
`backend/control/Makefile` exists for one service; the frontend uses npm
scripts; predictor and OCR have nothing. CI re-encodes each command by hand.

**D3. No Python lint, format or type check anywhere.** None of the three
`pyproject.toml` files configures `ruff`, `mypy`/`pyright`, `black` or `isort`,
and there is no pre-commit config. `pip-audit` runs with `|| true`.

**D4. No Python lockfiles.** Every service installs from `>=` ranges. For a
product whose selling point is provenance, the image contents are not
reproducible from the tree.

**D5. Handoff allowlist and `.gitignore` carry stale rules.** The allowlist
includes `docs/architecture.md` and `docs/runbooks/**`, which no longer exist;
`.gitignore` repeats the predictor benchmark-results block (lines 78–79 and
118–119). `check_docs.py` is **red at HEAD**: 10 problems, all in
`docs/audit/SYSTEM_ISSUES_VI.md` (8 links to documents removed on 2026-09-27,
2 retired paths) — the CI docs gate would fail on this branch.

### 5.5 Docs

**Doc1. Two doc roots.** Product-wide decisions (ADR 0001 topology, 0006 OCR
boundary, 0011 single control plane) live under `backend/control/docs/adr/`.
`docs/README.md` says "ADR 0001–0011"; 0012 exists.

**Doc2. Mixed naming.** `UPPER_SNAKE.md`, `kebab-case.md`, `_VI` suffixes and
dated names side by side; `MODEL_CARD.md` and `model-card.md` differ only by
case (they are different documents — generated vs summary — but the names do
not say so).

**Doc3. Purposes are mixed in one folder.** Operator how-tos
(`GETTING_STARTED`, `OPERATIONS`), reference (`CONFIGURATION`, generated
matrices), explanation (`ARCHITECTURE`) and research results
(`TAB_SUITE_LIVE_RESULTS_2026-09-17`, `BM1_…`, `RETHINK_…_VI`) sit in the same
directory.

### 5.6 Repository root (local state)

**R1.** Untracked at the root: `CLAUDE.md`, `.claude/`, `grapify_setup.md`
(typo in name), `documents photos/` (space in name), `legacy/`, `data/`.
Local state: `.data/` 2.5 GB, `.cache/` 1.1 GB, two SQLite files in `.data/`,
and a second `.data/` inside `backend/control/`. A2 of the previous plan
already covers the data decisions; the rest needs a commit-or-delete call.

---

## 6. Target structure

### 6.1 Repository

Deployable roots stay where they are. Changes are marked `←`.

```text
tox-agent/
├── README.md  LICENSE  CHANGELOG.md  VERSION  THIRD_PARTY_NOTICES.md
├── AGENTS.md / CLAUDE.md          ← commit one agent-context file (R1)
├── Makefile                       ← dev entry point: make {setup,lint,fmt,typecheck,test} [SERVICE=…] (D2)
├── bin/toxagent                   operator CLI (unchanged)
├── frontend/                      README.md ← (F3)
├── backend/
│   ├── control/                   see 6.2
│   ├── predictor/                 pyproject.toml is the only dependency source ← (P2)
│   └── ocr/                       pyproject.toml ← (O1)
├── evals/                         ← optional later step; see 6.3
├── devops/
│   ├── compose/  cloud/
│   ├── release/                   ← handoff, release_manifest, deploy_target, restore_drill (D1)
│   ├── agent-runtime/             ← OpenCode bridge scripts (D1)
│   ├── checks/                    ← check_docs, check_workspace (D1)
│   └── tests/
├── docs/                          see 6.4
└── .github/  (+ CODEOWNERS, pull_request_template.md — optional)
```

### 6.2 `backend/control/src/toxagent` — layers outside, features inside

A full vertical-slice rewrite (one top-level package per feature) is **not**
recommended: it would touch almost every import and most of the 157 test modules for a
readability gain that a lighter change gets most of. Keep the layers; group by
feature *within* each layer; enforce the order.

```text
toxagent/
  platform/        config.py, flags.py, observability.py, metrics.py   ← was package root + empty telemetry/
  domain/          report/ (split domain/report.py), scientific_case/, run.py, …
                   agent_transition.py   ← KernelTransition moves here (C2)
                   xai_coverage.py       ← pure parts of application/xai_coverage (C2)
  application/
    report/        dispatch, orchestrator, stages, stage_handlers, synthesis, inputs, submit_draft
    runs/          run_scheduler, runs, run_budget, run_envelope, concurrency, queues, startup_reconciliation
    explanation/   explanation, readiness, identity, xai_coverage (orchestration only)
    investigation/ scientific_case_service, claim_review, decision_state_service, skill_catalog, skill_drafts
    conversation/  submit_message, submit_answer, router, intent_matching, sessions, projections
    prediction/    create_analysis, quick_predict, recognize_structure, effective_product, capabilities
  validation/
    answer/        answer_validator, numeric, classification, citations, coverage, claim_resolver, wire
    report/        report_validator, report_semantics, report_wire, synthesis_validator, synthesis_wire
    common/        prohibited_claims, limitations, fallback
  report/          compiler/ { legacy.py (was compiler.py), synthesis.py (was compiler_v3.py) }, artifact, fact_bundle, figures, renderers
  answer/          (merge into report/ or application/conversation — 99 lines)
  agent/ + runtime/ + activities/   ← fold into harness/ and domain/ (C4)
  api/             routes/ { sessions.py, runs.py, reports.py, predict.py, settings.py, … } (C3, C6)
  persistence/sql/ repositories/ { one module per aggregate } (C3)
  harness/  tools/  research/  predictor/  connections/  streaming/   (unchanged)
```

Layer order, as enforced since Phase 1 by `LAYERS` in
`backend/control/tests/unit/test_boundaries.py` (outermost first; a module may
import its own package and lower lines, never a higher line or a different
package on its own line):

```text
worker
api
runtime                          re-export namespace over harness.provider
harness
tools
application
agent                            superseded (ADR 0011)
answer | report
validation
persistence
research | predictor | connections | streaming | activities | capabilities
domain | platform (config, flags, metrics, observability, telemetry)
```

The `report ↔ validation` cycle was resolved by moving the compiled-report gates
(`synthesis_validator.py`) into `report/`, where the type they check lives.

### 6.3 Evals and results

```text
backend/control/evals/          code only: runner, graders, packs, schema, tasks, fixtures
backend/control/evals/results/  ← one ignored output root for every run (C7)
docs/results/ or object storage ← the paired-*.json summaries and signed scorecards that are cited
```

Keep in git only what a document or test cites (summaries, the scorecard, the
frozen fixtures). Full run directories go to the results root, which is
`.gitignore`d as one rule, or to the artifact bucket already used for model
weights. A later, optional step is to lift `backend/control/evals` and
`backend/predictor/evals` into one top-level `evals/` with a `pyproject.toml`,
so it stops depending on `pythonpath` hacks.

### 6.4 Docs (Diátaxis)

```text
docs/
  README.md                 index (keep the status column)
  getting-started/          tutorial: first run
  how-to/                   operations, configuration tasks, model admission, backup/restore
  reference/                configuration variables, capability matrix (generated), model cards, benchmark protocol
  explanation/              architecture, workspace layout, predictor architecture
  adr/                      ← all ADRs, moved from backend/control/docs/adr (Doc1)
  results/                  dated measurement reports (TAB suite, BM1, …)
  internal/                 spec/, backlog/, audit/ — withheld from handoff
```

Use one naming convention (`kebab-case.md`; language suffix `.vi.md` when a
document is Vietnamese). Rename `model-card.md` to
`reference/model-card-generated.md` so the name says what it is.

---

## 7. Plan

Ordered so each phase is small, reversible and green before the next starts.
Moves use `git mv` so history follows the file.

| Phase | Items | Effort | Risk | Done when |
|---|---|---|---|---|
| **0. Guard rails first** | Add `ruff` (lint + format) and `pyright` or `mypy` in *report-only* mode to all three `pyproject.toml`; ESLint + Prettier for the frontend; a root `Makefile` delegating to each service; `pre-commit` config. Make `pip-audit` blocking or record why not (D2, D3, F2) | 1 day | Low | `make lint` runs everywhere; CI job added as non-blocking |
| **1. Enforce the layers** | Extend `test_boundaries.py` with the layer order in §6.2 (or add an `import-linter` layers contract); mark the current violations as known exceptions so the test is green; fix them one by one: move `xai_coverage` to `domain/`, break the `report ↔ validation` cycle, move the application → tools/harness imports (C2) | 1–2 days | Low | New violations fail CI; exception list is empty |
| **2. Group `application/` and `validation/` by feature** | Create the subpackages in §6.2 with `git mv`; keep a re-export in the old module path for one release only if external scripts import it; rename `compiler_v3` → `compiler/synthesis.py`, `compiler` → `compiler/legacy.py` (C1, C5) | 2 days | Medium (import churn) | Unit, integration and e2e suites green; README layout updated |
| **3. Fold micro-packages** | `telemetry/` + `metrics.py` + `observability.py` → `platform/`; `runtime/`, `activities/`, `agent/` into `harness/` / `domain/`; resolve the two `capabilities` (C4) | 1 day | Low | Package list = README list |
| **4. Split god modules** | `api/routes.py` → `api/routes/<resource>.py`; `repositories.py` → one module per aggregate; `domain/report.py` and `domain/scientific_case.py` → subpackages. Pure moves, no behaviour change (C3) | 2–3 days | Medium | No file over ~800 lines in `src/` except generated ones |
| **5. Typed contract** | Add `response_model` to the 48 routes without one, route by route; generate `frontend/src/lib/api/openapi.d.ts` in CI and fail on drift; migrate `types.ts` to re-export generated types (C6) | 3–5 days | Medium | `types.ts` has no hand-written response shapes; drift check in CI |
| **6. Separate eval output** | One ignored results root; commit only cited summaries; move the untracked `live-b*` summaries per A4 of the previous plan (C7) | 1 day | Low | `git ls-files backend/control/evals \| wc -l` roughly halves |
| **7. Service packaging parity** | `backend/ocr/pyproject.toml`; predictor `deploy/requirements.txt` generated from `pyproject.toml`; lockfiles (`uv lock` or `pip-compile`) per service, used by the Dockerfiles (O1, P2, D4) | 1–2 days | Medium (image rebuilds) | Images build from lockfiles; one dependency source per service |
| **8. Scripts and docs** | Regroup `devops/scripts/` (§6.1); move eval-lane scripts next to evals; move ADRs to `docs/adr/`; Diátaxis folders; fix the six dead doc citations in code; clean the allowlist and `.gitignore` (D1, D5, Doc1–3) | 1–2 days | Low | `check_docs.py`, `check_workspace.py --strict`, handoff tests green |
| **9. Frontend by feature** (when it next grows) | `src/features/{workbench,report,quick-predict,settings,auth}/` holding their components, hooks and store slices; `src/shared/` for `ui/`, `lib/`; split `LandingPage.tsx` into sections (F1, F4) | 2 days | Low | vitest + Playwright green |
| **10. Research** | Either move `backend/predictor/research/` + the eval scripts that import `legacy_backend` into a separate research repo, or copy the few functions they need into `evals/` (P1) | 1–2 days | Low | No `legacy_backend` import outside `research/` |

Phases 0–1 are the ones that stop the structure from decaying again; everything
after them is a refactor the guard rails then protect.

### Verification after every phase

```bash
make lint typecheck test                          # once Phase 0 lands
python -m pytest backend/control/tests -q
python -m pytest backend/predictor/tests -q
(cd backend/ocr && PYTHONPATH=src python -m pytest -q tests)
(cd frontend && npm run typecheck && npm test && npm run build)
python devops/scripts/check_docs.py
python devops/scripts/check_workspace.py --strict
python -m pytest devops/tests -q
graphify update . && codegraph sync .
```

---

## 8. What not to do

- **Do not rename the deployable roots** (`backend/` → `services/`, splitting
  into several repositories). CI, Compose, Dockerfiles, the handoff allowlist
  and the ADRs all name these paths; the readability gain is zero.
- **Do not rewrite `toxagent` into a vertical-slice layout in one pass.** Group
  inside the layers first; revisit only if Phase 2 still leaves features hard
  to find.
- **Do not mix moves with behaviour changes.** Every phase above is either a
  pure move/rename or a pure contract change, so a failing test points at one
  cause.
- **Do not turn on blocking lint/type checks before the baseline is clean.**
  Start report-only, fix, then flip to blocking per service.
- **Do not delete legacy code paths ahead of their flag's `remove_by` date.**
  Rename and isolate them now; delete on schedule.

---

## 9. Sources

- Shopify Engineering, *Under Deconstruction: The State of Shopify's Monolith* — https://shopify.engineering/shopify-monolith
- Shopify Engineering, *Enforcing Modularity in Rails Apps with Packwerk* — https://engineering.shopify.com/blogs/engineering/enforcing-modularity-rails-apps-packwerk
- Shopify Engineering, *A Packwerk Retrospective* — https://shopify.engineering/a-packwerk-retrospective
- Uber Engineering, *Introducing Domain-Oriented Microservice Architecture* — https://www.uber.com/us/en/blog/microservice-architecture/
- Import Linter, *Layers contract* — https://import-linter.readthedocs.io/en/latest/contract_types.html
- Diátaxis documentation framework — https://diataxis.fr/

---

## Execution log

### 2026-09-28 — Phases 0 and 1

**Phase 0 — guard rails.**

- `ruff.toml` at the root: one configuration for `backend/` and `devops/`.
  Blocking rules are pyflakes, the syntax-level pycodestyle errors and bugbear
  (`B008` FastAPI idiom, `E741`, `B905` ignored with reasons). The tree is
  clean against them. Import order, pyupgrade and `ruff format` are
  report-only (`make lint-report`); the tree was **not** mass-reformatted.
- Fixes to get there: ~60 unused imports removed; two mid-file imports moved
  to the top (`persistence/sql/database.py`, `repositories.py`); a stray
  duplicated docstring removed from `harness/adapters/__init__.py`; a loop
  variable that shadowed `dataclasses.field` renamed
  (`validation/report_validator.py`); one-line `if` statements split in
  `backend/predictor/scripts/model_admission.py`. Closure-in-loop (`B023`),
  `B007` and `B904` sites were read one by one: all intentional, marked
  `noqa` with the reason. The frozen research-era scripts in
  `backend/predictor/evals/scripts/` were left byte-for-byte (per-file ignore).
- `mypy` configured in both `pyproject.toml` files, report-only
  (control baseline: 122 errors in 32 files).
- `ruff` and `mypy` added to both `dev` extras.
- Frontend: ESLint 9 (`typescript-eslint`, `react-hooks`) via
  `frontend/eslint.config.js`; `npm run lint` runs it and then the policy lint.
  Five findings fixed (an unused prop threaded through three landing sections,
  two stale `eslint-disable` directives).
- Root `Makefile`: `lint`, `lint-report`, `fmt FILES=…`, `typecheck`,
  `test [SERVICE=…]`, `check`.
- `.pre-commit-config.yaml` (optional, ruff + basic hygiene hooks).
- CI: new `lint` job (ruff blocking, mypy `continue-on-error`); the frontend job
  runs `npm run lint` instead of `lint:policy`.
- Handoff allowlist: rules for `Makefile`, `ruff.toml`,
  `.pre-commit-config.yaml`.
- Not done: `pip-audit` still runs with `|| true`.

**Phase 1 — layer order enforced.**

- `LAYERS` + `LAYER_EXCEPTIONS` in
  `backend/control/tests/unit/test_boundaries.py`: every top-level package must
  have a layer; every import (including imports inside functions) must point
  down; every exception must still be real, so the list can only shrink.
  Probed by adding an upward import to `streaming/sse.py`: the test failed on it.
- Violations fixed by moving code (`git mv`, no behaviour change):
  - `application/xai_coverage.py` → `domain/xai_coverage.py` (pure stdlib) —
    removes `report → application` and `validation → application`.
  - `validation/synthesis_validator.py` → `report/synthesis_validator.py` —
    removes the `report ↔ validation` cycle.
  - `ToolContext` → `application/tool_context.py`, re-exported by
    `tools/registry.py` — removes `application/report_dispatch.py → tools`.
  - `application/effective_product.py` → `api/effective_product.py` (it
    describes the assembled product: harness, tools, flags) — removes
    `application → harness` and one `application → tools`.
  - An unused `KernelTransition` import in `persistence/interfaces.py` removed.
- Remaining exceptions (5): `application/{run_budget,skill_catalog,skill_drafts}.py → tools`
  (tool profiles and limits; Phase 2), `persistence/{investigations,sql/repositories}.py → agent`
  (superseded ADR 0011 kernel; retired with its tables).

**Verification.**

| Check | Before | After |
|---|---|---|
| `backend/control` pytest | 1997 passed, 18 skipped, 1 failed | 2201 passed, 18 skipped (the +204 are the new layer tests) |
| `backend/predictor` pytest | — | 251 passed, 5 skipped |
| `backend/ocr` pytest | — | 6 passed |
| `devops/tests` | 2 failed (`test_handoff_allowlist`, pre-existing) | same 2 failed, 94 passed |
| Scripted eval `suite:pr`, 3 trials | — | 6 pass, 54 skipped `needs_agentic_runtime`, exit 0 |
| `ruff check backend devops` | 106 findings | clean |
| Frontend `npm run lint` / `typecheck` | no ESLint | clean / clean |
| `check_docs.py` | 10 problems (D5) | the same 10, none new |

The one control failure, before and after, is
`tests/unit/test_scifact.py::test_tfidf_ranks_the_matching_abstract_first`: the
`.venv-test` environment has no `numpy`, which `evals/external/scifact/retrieval.py`
imports. It was deselected in the "after" run; it is an environment gap, not a
regression.

**Next:** Phase 2 (group `application/` and `validation/` by feature), which
also retires the three `application → tools` exceptions.
