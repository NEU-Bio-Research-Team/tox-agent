# External benchmarks: what runs, what is blocked, and on what

Status of RETHINK §5.1 (backlog Waves 7 and 10) as of 2026-09-27. The labels
are the ones RETHINK §5.2 defines: `external-native` (benchmark data, split and
scorer unchanged), `published-data-transfer` (published data turned into
ToxAgent questions), `product-regression` (TAB-Suite). No result from one label
is ever reported as another, and none is merged into a single ToxAgent score.

| Benchmark | Measures | Status | Label when run |
|---|---|---|---|
| SciFact | claim verification with rationales | **Adapter runs** (`backend/control/evals/external/scifact`) | `external-native` (full dev) / `external-native-subset` |
| SciFact through the product | the product's own retrieval and evidence-relation step | **Adapter runs, not yet run live** (`evals/external/scifact/product.py`) | `published-data-transfer` |
| BioASQ Task b | biomedical QA: documents, snippets, exact and ideal answers | Blocked: registration | `external-native` only via the official evaluation |
| AstaBench LitQA2-FT, PaperFindingBench | literature agent: search, full text, answer | Blocked: environment, keys, research profile | `external-native` for a declared research profile |
| TDC ADMET hERG, MoleculeNet Tox21 | the predictor | Protocol below; not an agent score | predictor track |
| τ-bench / AgentDojo | multi-turn state, robustness to untrusted tool output | Method borrowed by TAB-Suite; not run | — |

## SciFact (runs)

See `backend/control/evals/external/scifact/README.md`. The official metrics
were ported and checked against the official module (480/480 values equal on
random predictions over the dev split). A judge there is a *model* used as a
stand-alone verifier, reported under the model's name. The first run is recorded
in `docs/backlog/SCIENTIFIC_INVESTIGATION_BACKLOG.md` (Wave 7).

## SciFact through the product (adapter built 2026-09-27, W7-04)

`evals/external/scifact/product.py` measures ToxAgent rather than a model:

1. the pinned release's 5,183 abstracts are written as a
   `research-corpus-v1` file and served by the new **corpus provider**
   (`research/providers/corpus.py`): BM25 over the local corpus, no network, the
   file pinned by SHA-256 and the pin recorded in the effective product. The
   product writes its own queries, so retrieval is the product's, not the
   benchmark's oracle or TF-IDF setting;
2. each claim is asked as a question, and answered on a **subjectless case**
   (`subjectless_research_v1`, W9-08) — the router block that made this
   impossible in Wave 7 is gone, so no compound subset and no fabricated
   molecule subject is needed, and the claim distribution is unchanged;
3. the accepted answer's `evidence_relations` (`answer_draft_v2`) are read back
   through `GET /v1/sessions/{id}/runs/{run_id}/evidence-relations` and mapped
   per abstract: `supports` → SUPPORT, `contradicts` → CONTRADICT, everything
   else absent (NEI). Two propositions disagreeing about one abstract leave it
   unlabelled and are reported, never resolved by a rule.

Only `abstract_label_only` is computed: the product selects no rationale
sentences, so the other three official metrics are **left out rather than
reported as zero**. The run refuses to start against a deployment that is not
serving the pinned corpus or has either flag off, because such a run would
write a file of zeros that looks like a result.

The label is `published-data-transfer` — the task format changed and retrieval
is not the benchmark's. It compares only with `run.py` judges re-run over the
same claim ids (recorded in the manifest as `compare_with`), never with a
published full-split number.

Not yet run live: it needs a deployment with the corpus provider, both flags and
a model runtime. The path is covered end to end on the scripted runtime
(`tests/e2e/test_scifact_through_the_product.py`).

## BioASQ Task b (blocked on registration)

Task 14b asks for relevant articles, snippets, exact answers and "ideal"
paragraph answers, evaluated "both automatically and manually"; data and
submission require participant registration, and the 2026 edition started in
March 2026 ([BioASQ challenges](https://bioasq.org/participate/challenges)).
More than 5,700 training questions with gold answers are available after
registration. What is needed from the product owner: a BioASQ account. Then:
an adapter that sends each question to a declared research profile, maps its
cited evidence records to PubMed ids (documents) and passages (snippets), and
writes the official JSON; the challenge window determines whether the result
is an official evaluation or a post-hoc one on released gold, and the label
says which.

## AstaBench literature tasks (blocked on environment and keys)

AstaBench runs on InspectAI with `uv`; its literature tasks include
`astabench/litqa2_{validation,test}` and `astabench/paper_finder_{validation,test}`;
it needs `ASTA_TOOL_KEY` (literature tools over MCP), `HF_TOKEN` (gated data)
and model provider keys for scoring, and recommends far more memory than this
8 GB host has ([asta-bench](https://github.com/allenai/asta-bench)). A custom
agent plugs in as an Inspect solver. What running it natively requires:

1. a host with the recommended memory, and the keys above;
2. a **research profile** of ToxAgent declared for it: the investigation loop
   with AstaBench's own search tools instead of ToxAgent's closed evidence
   provider (the production profile cannot use them, so its number would not
   describe the deployment — RETHINK §5.1);
3. validation split for development, test split untouched until the end.

The result would describe that research profile, never the production
deployment.

## Predictor track (TDC, MoleculeNet)

Kept apart from every agent result. A comparison with a TDC ADMET leaderboard
requires retraining and evaluating on TDC's own data, label definition, scaffold
split and seeds; the served ChemBERTa checkpoint was trained on its own hERG
set, whose label definition is not assumed to equal TDC's hERG. Before any
comparison: (1) a structure-level overlap check between the checkpoint's
training data and the benchmark test set, (2) the benchmark's split and
metric unchanged, (3) the result reported under the predictor, never as an
agent capability. MoleculeNet Tox21 follows the same rule, and the Tox21
assays stay separate metrics. The current measured performance is in
`docs/model-card.md`.
