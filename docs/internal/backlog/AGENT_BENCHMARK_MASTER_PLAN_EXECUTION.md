# Agent benchmark master plan — execution log

Source plan: `ToxAgent_Agent_Implementation_and_Benchmark_Master_Plan_2026-10-21.md`
(repo root, untracked), which follows the 2026-10-20 proposal
(`docs/internal/spec/TOXAGENT_AGENT_LAYER_AND_BENCHMARK_PROPOSAL_2026-10-20_VI.md`).

Branch: `feat/agent-benchmark-master-plan`, cut from
`fix/activity-identity-guards` at `025db84`. One commit per planned PR slice
(P01–P12), so the stack can be split into the stacked PRs §16 describes.

Baseline before any change: control suite 2686 passed, 19 skipped.

## Status by slice

| Slice | Issue | State | Notes |
|---|---|---|---|
| P01 subject-binding | #44 | done | see below |
| P02 frozen-full-text | #43 | done | see below |
| P03 subject-evidence | AG-EVIDENCE | done | see below |
| P04 strict-release | #47 | done | see below |
| P05 budget-phases | #46/#53 | done | see below |
| P06 case-inputs | #51 | done (engineering part) | SME gold still needed |
| P07a shared-replay-A | #50 | done | four arms on one case set's frozen inputs |
| P07b strong-B-fairness | #50 | done | effective fairness, attempt isolation, B v3 |
| P08 stories-skills | #46 | done (engineering part) | live-LLM level of two variants is declared, not run |
| P09 research-UI | AG-UI | partly done | subject panel, screening card, release banner; passage viewer and case diff not done |
| P10 scoring-packets | #53 | done | packet v2, scorer v2, cluster-paired D-B |
| P11 pilot-24 | #52 | engineering done | the CLI runs a pilot; the 24 cases need SME gold |
| P12 freeze-agent | #56 | engineering done | `agent_benchmark.py freeze`; the report needs the pilot |

## P01 — subject binding (#44)

Flag `subject_binding_v1` (default off).

- `domain/subject.py`: `RunSubjectBinding` (run-subject-binding-v1). Case key
  `analysis:<id>` (unchanged) or `activity:<40 hex>` digest of ligand, target,
  variant, assay context. Optional enrichment (UniProt, sequence) checked by
  `mismatches`, never part of the key. `TargetIdentity.variant` added, not
  serialised when unset (existing hashes unchanged).
- Resolved at admission (`application/conversation/subject_resolution.py`):
  explicit `subject_ref` on `POST /messages`, the legacy `analysis_id`, a new
  molecule (pending until its snapshot), or the session's only subject. Two
  subjects and no selection is the `subject_ambiguous` clarification.
- Frozen into the run: `RunContext.subject`, run envelope v2 (a v1 worker
  refuses to claim it), `run_configuration_snapshots.subject_binding`
  (migration 0026), `run.subject_bound` event. Recovery runs keep it.
- Tools receive it through `ToolContext.subject` (MCP and local);
  inventory/report summary default to it, never to the session's selection.
- Case: keyed by the binding, stores `subject`; each run records its binding
  hash and artifact refs. `SubjectMismatch` on an incompatible open case.
  With a binding, a case that cannot be opened fails the run
  (`case_unavailable`) instead of silently running without it.
- Ledger: `policy_fact` and `workflow_fact` source classes (not independent).
  Activity estimate → predictor_fact scoped by target and unit `pIC50`, never
  an endpoint. Full EvidenceScope axes kept; quoted spans and server
  provenance kept per entry. Another subject's artifact is left out.
- Replay: a frozen screening may carry its toxicity snapshot
  (`frozen-toxicity-snapshot-v1`); binding makes one session-local copy with
  its prediction observation, hash-checked, never calling the predictor.
- `GET /v1/sessions/{sid}/screenings/{id}`: screening read model with every
  stage stated.
- Effective product: `contracts.run_envelope_version`,
  `contracts.run_subject_binding`.

Tests: `tests/unit/test_subject_binding.py`,
`tests/e2e/test_subject_binding_through_the_product.py`,
`tests/integration/test_subject_binding_queue.py`.

## P02 — frozen full text (#43)

- `research/fulltext_store.py`: `frozen-full-text-v1` store -- each
  document's passages (rows with cells), the outcome (retrieved /
  access_failed), the capture (provider, time, sha256 of the bytes received)
  and the parser version (`jats-passages-v1`). Every document re-hashed on
  load; a sidecar file is pinned by sha256.
- `FrozenFullTextProvider` wraps the snapshot or corpus provider when a store
  is configured (`TOXAGENT_RESEARCH_FULL_TEXT_PATH`/`_SHA256`, or inline in a
  snapshot fixture under `evidence.full_text`). Only then is the provider
  text-capable, so `read_evidence_text` exists on a frozen deployment. A
  record the store does not hold is answered "not held", never fetched; an
  injected snapshot outage reaches the text as transient.
- Live Europe PMC: bounded per-provider cache of definitive answers and
  single-flight per PMCID; an outage is never cached.
- Provenance of every read text is stored (`evidence_full_texts.provenance`,
  migration 0027), shown by `read_evidence_text` (`text_provenance`) and by
  the evidence API (`full_text.provenance`).
- `python -m evals.investigation.capture_full_text`: capture once from
  Europe PMC into a store; transient fetches are reported, not frozen.

Tests: `tests/unit/test_frozen_full_text.py`,
`tests/e2e/test_frozen_full_text_through_the_product.py` (offline table-row
quote validated; captured access failure owes `evidence_full_text_not_read`;
unknown locator refused). Suite: 2732 passed, 19 skipped.

## P03 — subject-aware evidence retrieval (AG-EVIDENCE)

- `research/target_catalog.py` (`benchmark-targets-v1`): KCNH2 (hERG,
  CHEMBL240, live mapping verified), EGFR (CHEMBL203), ABL1 (CHEMBL1862),
  each with the vocabulary a relevance check looks for. An entry whose UniProt
  disagrees is not the target. EGFR/ABL1 live ChEMBL mappings are recorded as
  unverified.
- Relevance over a target (`evidence-relevance-target-1`): ligand on the
  target → direct (with `assessed_scope` ligand + target); ligand on another
  target → contextual; target with another ligand → irrelevant; target not
  in the catalog → uncertain at best; thin metadata → uncertain.
- `search_toxicology_evidence` under `subject_binding_v1` takes
  `screening_id`, defaults to the run's subject, refuses another subject
  (`subject_mismatch`), and passes the same data-scope guard keyed by the
  bound subject. The ligand's identity is the frozen record's
  (`subject_names`, server-held), never the query's words.
- `get_chembl_activities` follows the binding; a screening whose target is
  not hERG is `capability_unavailable`, never a string passed to the hERG
  lookup.
- Answer validation: a relation may not restate the target an activity
  artifact or an assessed record states
  (`evidence_relation_scope_contradicts_source`), and a record about the
  ligand on another target cannot be direct support/contradiction
  (`evidence_relation_direct_out_of_subject`).
- Arm B (`fixed-workflow-b-v2`): activity branch with queries from the
  binding (ligand name, target, readout); REVIEW triggers a search; toxicity
  branch whose endpoint comes from the question and reads the linked
  toxicity analysis; provider-failure branch said as such.
- Evidence API exposes `relevance_assessment`.

Tests: `tests/unit/test_subject_relevance.py`,
`tests/e2e/test_subject_evidence_through_the_product.py`. Suite: 2745 passed.

## P04 — strict publication policy (#47)

`TOXAGENT_RELEASE_POLICY` = `legacy` (default, unchanged) |
`strict-reviewed-v1` | `hard-only-v1`, stated in the effective product with the
hash of its decision table.

- `domain/run_result.py`: `RunResultV1` keeps publication disposition
  (publish / partial_facts / withhold), review status, execution outcome and
  knowledge outcome apart; task success is none of them. The policy is a pure
  table (`decide_publication`).
- One writer: the gateway's publication coordinator records the run result
  (`run_results`, migration 0028) and, only for a published answer, the
  compatible `answer_releases` row and `answer.accepted`. Idempotent on
  run_id. A run that ends without committing (failure, cancel) records a
  withheld result with its typed execution outcome.
- The assistant message of a non-published run carries no candidate prose:
  server-resolved facts, the reason, needed inputs, and a `result_ref` part.
  The answer route serves only the run's public answer.
  `GET /runs/{rid}/result`; drafts carry their phase (initial /
  hard_correction / semantic_repair).
- Case isolation: under a declared policy a run's conclusion is `proposed`
  and is promoted only with its published answer; the last released
  conclusion is kept beside it; checkpoints and report views never present a
  proposal as the case's conclusion; hypothesis statuses set by an
  unpublished run are marked as proposals; runs record their disposition.
- `hard-only-v1`: the declared no-review ablation; publishes the
  hard-validated candidate with review status `not_reviewed_by_design`.

Tests: `tests/unit/test_run_result_policy.py`, `tests/e2e/test_strict_release.py`.
Suite: 2773 passed; `test_worker_quotas_and_drain::test_a_draining_worker...`
failed once under full-suite load and passes in isolation (timing).

## P05 — budget v2 and phase attribution (#46/#53)

- Preset `investigator-pilot-v2` (`total-logical-invocations-v2`): 360 s root
  deadline; 20 tool calls in total -- case, skills, reads, searches,
  candidates and reviewer submissions all count; 6 reserved for the
  submissions (gathering may use 14); searches 3, records 8, full-text pages 4
  within the total; one semantic repair; two hard submissions per phase;
  6 invalid/denied calls then everything but the submissions is refused;
  120 s of the deadline kept for review and repair; token/cost limits
  declared `observed_only`. `investigator-pilot-v1` is unchanged and now
  named `action-read-ceiling-v1`.
- The effective budget states all of it and has a `sha256`, so two arms with
  one preset label but different limits are told apart.
- Tool calls and runtime bindings carry their phase (`investigator`,
  `workflow_acquisition`, `workflow_synthesis`, `hard_correction`,
  `review_initial`, `semantic_repair`, `review_final`; migration 0029); MCP
  calls take it from their binding. Invalid calls are recorded.
- `GET /runs/{rid}` reports `phase_usage` (phase-usage-v1): per phase,
  dispatches, calls by status, tokens (cumulative latest once, deltas added,
  unknown kept unknown), cost by currency (missing cost is unknown).
- The investigator turn ends before the review's reserve; running into it is
  `phase_budget_exhausted`, and the run result says `budget_exhausted`.
- Under v2 a recovery continues the original's deadline and call budget.

Tests: `tests/unit/test_budget_v2_and_phase_usage.py`,
`tests/e2e/test_budget_phases_through_the_product.py`. Suite: 2785 passed.

## P06 — protocol v2 cases, inputs and gold (#51, engineering part)

- Three files per case, three schemas: `investigation-case-v3` (what the case
  measures, `workflow_context` separate from `cohort`, pins of the other two),
  `frozen-case-inputs-v1` (everything a system is given, the same for every
  arm: subject by reference, frozen artifacts with hashes, corpus and
  full-text store, contract versions, turns with `add_context` /
  `attach_evidence_ref` / `switch_subject`, provenance, dependency groups) and
  `case-gold-v1` (never model-facing).
- `case_inputs.py`: loader, audit (shapes, hashes, artifacts against the
  replay file, spans against the frozen text or a declared access failure,
  pairing, answerability, T3 polarity, T6 faults, gold isolation, SAIR
  availability) and `dependency_split` by whole dependency components.
- The replay file may hold frozen toxicity analyses that belong to no
  screening (`analyses`), for toxicity-only cases; each is hash-checked on
  load, and `materialise_frozen_analysis` makes the session-local copy.
- `cases_protocol_v2_technical/` (built by `technical_seed.py`): 12 cases,
  T1-T6 × {toxicity_only, activity_guided}, both T3 polarities, a multi-turn
  update with a target switch, two T6 faults -- fictional compounds and
  invented numbers, labelled `synthetic_engineering_fixture` /
  `synthetic_engineering_gold` on every file. Audits clean.

Not done here (needs the SME): real cases, acceptable conclusions, critical
errors, verified T3 labels and source checks for the 24-case pilot.

Tests: `tests/unit/test_case_inputs.py`. Suite: 2794 passed; the one failure
(`describe()` gained `analyses: 0` for every replay file) is fixed and its
file rerun.

## P07 — the four arms on one case set's frozen inputs (#50)

### P07a: shared inputs, and arm A on them

- `evals/investigation/turn_actions.py` (`turn-actions-text-v1`): one place
  renders a v3 turn into the message every arm receives -- the researcher's
  words plus the facts their `add_context` / `attach_evidence_ref` actions
  carry. `switch_subject` is not text: it is the subject the next message
  names, as a selection in the UI is. A case arm *additionally* files
  `add_context` into the subject's case; nothing reaches D that B and C do
  not see.
- `frozen_analysis` subjects: a message may name a toxicity analysis the
  deployment's replay file holds (`{"kind": "frozen_analysis",
  "analysis_ref": ...}`). The run is bound to a session-local copy, made in
  the run's transaction, hash-checked, never by calling the predictor; the
  binding keeps the origin id and hash beside the local id, which is what
  arms compare. This is how a toxicity-only case has a subject at all.
- `adapters/template_v2.py` (arm A): reads the set's pinned replay file --
  the same bytes the deployments replay -- and renders the estimate, the
  interval, the policy decision, each stage's status and the frozen toxicity
  output, following `switch_subject` because the subject is an input, not a
  judgement. It refuses when a rendered artifact's hash is not the one the
  inputs declare.
- `adapters/toxagent_replay.py` (B, C, D): attaches the case's screenings,
  names its subject per turn, sends the rendered text, files context on a
  case arm, and reads back every run with its result, drafts, decision state,
  dossier, phase usage and binding. The record's text per turn is what the
  researcher saw -- the published answer, or the incomplete-result message
  the policy showed instead.
- `case_inputs.load_case_set`: a set is a directory with `cases/`, `inputs/`,
  `gold/` and a set manifest naming the replay and evidence files with their
  hashes; a pinned file whose bytes differ is refused before any arm runs.
- `run.run_protocol_study`: the protocol v2 runner. Arms may run in several
  invocations (a laptop runs one deployment at a time) and every later one is
  held to the first -- each arm's effective product is kept under
  `products/`, compared with every other arm in the study, and the case set,
  its pins, the turn rendering and the arm order are fingerprinted, so a
  changed study is refused (`study_manifest_mismatch`) instead of silently
  mixing conditions.

### P07b: fairness that compares conditions, not labels

- `systems.fairness_problems` now compares the **whole effective budget**
  (the probe the plan quotes -- 360 s/20 calls against 900 s/50 calls under
  one preset name -- is now refused), the contracts block, every pinned
  provider file, and the prompt/profile hashes; a **mandatory pin** no arm
  reports is a missing pin, not agreement; and a flag that differs between
  two arms and is not one of the protocol's declared `treatment_fields` is a
  confound, not a treatment.
- `product_mismatch` also holds a deployment to `required_product`, so an arm
  of protocol v2 must report `release_policy=strict-reviewed-v1`.
- The arm files now pin `investigator-pilot-v2`, `subject_binding_v1` and the
  strict release policy, and declare the three treatment flags.
- Attempt isolation: a (case, arm, trial) has one **primary** attempt, and
  that is what the packet and the scorer read (`primary_records`). A rerun is
  a `diagnostic_rerun`: kept, costed, reported, never a replacement. Each
  attempt writes its raw files under its own attempt id, and `write_raw`
  refuses to overwrite one.
- Arm B is now `fixed-workflow-b-v3`: it reads only the bound subject's
  screening (another screening of the same ligand is named, not read), and
  takes an exposure or measurement the researcher already stated in the
  conversation instead of asking for it again.
- `providers.research_snapshot` in the effective product gained the
  snapshot file's `sha256`, so two arms on files of one name are told apart.

Tests: `tests/e2e/test_protocol_v2_four_arms.py` (four arms, one case set:
same digests, same sent text, context filed only on the case arm, a second
case opened on a target switch, a wrong budget and a wrong replay file both
refused, a repeat invocation skipping the primary),
`tests/e2e/test_frozen_analysis_subject.py`,
`tests/unit/test_investigation_protocol_arms.py`.

## P08 — the four stories and the skills (#46)

- Three skills added to the catalog: `read-activity-screening` (what an
  estimate, a policy decision and a stage status each are, and what settles a
  REVIEW), `update-case-across-turns` (what new information changes and what
  it must leave untouched; another target is another case) and
  `estimate-exposure-margin` (compute the ratio with the server's tool, ask
  for the exposure instead of assuming one, and never read a ratio as a
  clinical conclusion).
- `tests/e2e/test_user_story_acceptance.py` holds the acceptance matrix:
  every story and control variant names the test that proves it, two tests
  keep the matrix honest (every named test exists; all four stories and their
  variants have rows), and the variants nothing covered yet run there -- a
  failed activity stage, an interval clear of the threshold, an unadmitted
  model, and a measured value that changes the assessment while the estimate,
  interval, decision and stage statuses stay the frozen artifact's.
- Two rows are marked `live_only` and say so: a live model choosing its own
  reads, and a predictor that is genuinely wrong against an SME-verified
  label.
- A failed activity stage now carries `activity_failure_code` in the
  screening-state model view: it was already a citable field path, and a
  model may cite only what it was shown.
- Arm B gained the test its provider-failure branch lacked: a provider that
  cannot be reached is reported as a failure, and an empty corpus as an empty
  result -- a reader can tell them apart.

## P09 — the research workbench (AG-UI), partly done

Done: `listScreenings` / `getScreening` / `getRunResult` and `subject_ref` on
`sendMessage`; `SubjectSelector` (one ligand on two targets is two subjects,
the variant is part of the target, nothing is selected automatically, and a
queued run keeps the subject it was sent with); `ScreeningCard` (estimate,
policy decision and stage statuses in three zones, `not_run`/`failed` in
words with "not a negative result"); `SubjectPanel` above the composer, so
the numbers appear without waiting for a model; `ReleaseBanner` on every
answer and on a withheld result (reviewed / repaired-and-reviewed /
partial-facts / withheld / no-review-by-design / legacy unknown, never
"scientifically correct").

Not done: the evidence passage viewer's table/locator extension, the case
board's "what changed / what was kept" diff, the audit view's stage lineage,
and the browser-level e2e of the four stories.

## P10 — packet v2 and scorer v2 (#53)

- `packet_v2.py`: per case the turns exactly as every arm received them and a
  grader reference built from the gold (reference outcome, acceptable
  conclusions, owed facts, required and optional limitations, forbidden
  inferences, spans, the predictor-error control note, the across-turn
  expectations), with the gold's verification status printed at the top. Per
  response, the result the researcher saw per turn and, folded away, the
  stages behind it, so a repair's effect can be judged. The key stays outside
  the packet; only primary attempts are packed.
- `scorecard_v2.py`: Task Success over every *assigned* case and trial (a
  response that never came is in the denominator), partitioned by workflow
  context, task group and the T6 stress set, with one named cause per
  failure; the paired D - B discordance with a bootstrap over **dependency
  clusters**; mandatory-information recall; limitation recall, stated
  precision and all-required, with NA where a case requires none; critical
  errors and forbidden inferences n/N; retrieval from the traces alone; cost
  over every attempt with unknown usage kept unknown; and a completeness
  block that refuses to call the table publishable while anything is
  ungraded, unadjudicated or not SME-verified.

Tests: `tests/unit/test_scorecard_v2.py` (fourteen hand-calculated controls
from §14.5), `tests/e2e/test_protocol_v2_packet_and_score.py` (a study
becomes a blinded packet and a card; the gold reaches only the grader
reference).

## P11/P12 — the benchmark CLI, and what still needs people (#52/#56)

`backend/control/scripts/agent_benchmark.py` is the single entry point §17.2
asks for: `preflight` (audit the case set, then check each deployment is the
arm it claims, replays the set's own files, and differs from the others in
nothing but a declared treatment), `run`, `packet`, `score` and `freeze`
(the repro bundle: manifest, arms' effective products, case-set pins, file
hashes and the commands, deliberately without packets or keys).

Still needed from people, not code:

- **SME gold for 24 cases** (#51/#52): real cases, acceptable conclusions,
  owed facts and limitations, critical errors, verified T3 labels (a genuinely
  wrong predictor and a correct one with a distractor), and source checks.
  Until then every gold file says `synthetic_engineering_gold` and the
  scorer refuses to call its table publishable.
- **A scientific owner's review of arm B** before test, per #50.
- **Two graders** and the adjudication of their disagreements; the scorer
  reports them and resolves nothing by itself.
- **A live LLM budget**: the four-arm study here runs on the scripted
  runtime. The two `live_only` story rows and §17.4 step 4 need a deployment
  with a model and a declared quota.
- **The live ChEMBL mappings** for EGFR and ABL1 in the target catalog are
  still recorded as unverified.

## Live verification on a laptop (2026-10-22)

Not a pilot and not a result: a smoke test that the protocol runs outside the
test suite, on one technical case (`inv-a01-screening-review-explained`),
against the OpenCode runtime already on this machine (`openai` /
`gpt-5.6-luna`). Each arm ran as its own local deployment, one at a time, which
is how a laptop runs them.

| Arm | Outcome | Tool calls | Tokens (provider-reported) |
|---|---|---|---|
| A2_template | rendered the pinned replay record | none (no model) | none |
| B2_fixed_workflow | published, review passed | 12 by rule (3 searches, 2 records, 2 tables) | 40,282 |
| D2_investigator | published after one repair | 9, one of them a skill read | 90,801 |

What it showed:

- all three arms were given the same artifact (one observed content hash), and
  `preflight` passed only once every arm's deployment matched its arm, replayed
  the set's own files and agreed on the shared fields;
- D read `read-activity-screening` unprompted, and its phases were recorded
  apart: investigator 49,763 tokens, review 8,398, repair 24,391, review again
  8,249;
- the release gate did its job and left something to measure: the reviewer
  judged "the status is review *because* the interval crosses the threshold"
  only partially supported, correctly -- the claim cited the screening state,
  while the reason is in the prioritisation decision -- and the repair narrowed
  the claim instead of re-citing the right artifact. That is a true statement
  lost to a mis-citation, which is what §7.2's required-information retention
  measures;
- `packet` and `score` ran on the live study, and the card refused to call
  itself a scientific result because the gold is synthetic;
- `freeze` recorded both deployments' effective-product hashes;
- the fairness check was exercised against two real deployments: a second arm
  B under the *same* preset name but 900 s / 50 calls was refused, and the
  refusal also caught that the skill catalogue on disk had changed between the
  two arms' runs (`hashes.agent_profiles`), which is a condition change
  mid-study.

Commands, for a repeat:

```bash
cd backend/control
python scripts/agent_benchmark.py preflight --case-set evals/investigation/cases_protocol_v2_technical \
    --arm D2_investigator=http://127.0.0.1:8011 --token <token>
python scripts/agent_benchmark.py run --study <id> --case-set evals/investigation/cases_protocol_v2_technical \
    --arm D2_investigator=http://127.0.0.1:8011 --token <token> --cases <case-id>
python scripts/agent_benchmark.py packet --study <id> --case-set <set> --packet-id <packet>
python scripts/agent_benchmark.py score  --study <id> --case-set <set> --packet-id <packet> \
    --grades <csv> --compare D2_investigator:B2_fixed_workflow
python scripts/agent_benchmark.py freeze --study <id> --case-set <set>
```

A deployment per arm needs: the set's `replay.json` and `evidence.json` as its
pins, `TOXAGENT_RUN_BUDGET_PRESET=investigator-pilot-v2`,
`TOXAGENT_RELEASE_POLICY=strict-reviewed-v1`, the common flags, and only its
own treatment flags (`fixed_workflow_v1` for B; `scientific_case_v1` +
`scientific_skills_v1` for D).
