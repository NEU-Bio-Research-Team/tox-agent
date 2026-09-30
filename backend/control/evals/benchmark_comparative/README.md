# ToxBench: Comparative Agent Benchmark

Evaluates **hallucination** and **predictive accuracy** across three systems:
ToxAgent (the `POST /v1/predict` predictor, not the agent), ChatGPT and Gemini
(web answers collected by hand). Runbook, caveats and the blind-export flow:
[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).

## This measures the predictor. The agent is measured elsewhere.

`POST /v1/predict` returns structured numbers and runs none of the agent's
reasoning, retrieval or drafting, so nothing in this directory says anything
about the agent. The agent is measured by
[`evals/investigation`](../investigation/PREDICTOR_ERROR_STUDY.md), on cases
built around the predictor's own mistakes.

Two of this benchmark's four dimensions carry over to the agent and were
measured on it (2026-09-30, eight cases, `predictor-error-2026-09-30-budget900`;
full numbers and caveats in
[PREDICTOR_ERROR_FINDINGS.md](../investigation/PREDICTOR_ERROR_FINDINGS.md)):

| Dimension | ChatGPT | Gemini | Predictor | **Agent** |
|---|---|---|---|---|
| `screening_not_safety_assessment` stated | 6.1% | 6.1% | 0% | **100%** |
| `uncalibrated_probability` stated | 32.9% | 0% | 100% | **87.5%** |
| FActScore_tox | N/A | N/A | N/A | **0.855** (47/55 claims carry a source or field ref) |
| Safety gate pass rate | 100% | 100% | 100% | 100% |

FActScore is N/A for every other arm because none of them returns claims: the
web prompts do not ask for them and `/v1/predict` has none. The agent is the
only system this dimension can be computed on at all, which is worth knowing
before reading a table of N/As as a tie.

**hERG accuracy and Tox21 F1 are deliberately not carried over.** The agent
takes those numbers from the predictor, so measuring them again through the
agent would re-measure the predictor behind one more layer. The predictor's own
figures are in [runs/2026-09-30/](runs/2026-09-30/).

## Literature Grounding

| Paper | Venue | Used For |
|---|---|---|
| MedHallu (Pandit et al., 2025) | ACL 2025 | Hallucination taxonomy, difficulty tiers |
| RAGTruth (Niu et al., 2024) | ACL 2024 | Span-level hallucination detection |
| AgentHallu (2026) | arXiv | Tool-use hallucination, trajectory evaluation |
| FActScore (Min et al., 2023) | EMNLP 2023 | Claim-level factual precision |
| SAFE (Google, 2024) | ICML 2024 | Agentic fact verification |
| FACTS Grounding (DeepMind, 2024) | NeurIPS 2024 | Multi-judge grounded evaluation |

## Quick Start

```bash
# 1. Build the dataset (no dependencies needed)
python -m evals.benchmark_comparative.build_dataset

# 2. Run ToxAgent-only (needs a running stack)
python -m evals.benchmark_comparative.runner \
    --systems toxagent --trials 1 \
    --base-url http://127.0.0.1:8000

# 3. Run full comparison (needs API keys in .env)
python -m evals.benchmark_comparative.runner \
    --systems toxagent,gpt,gemini --trials 3
```

## Dataset

95 compounds in 7 groups:

| Group | Count | Purpose |
|---|---|---|
| `herg_positive` | 20 | Known hERG blockers (IC50 < 10µM) |
| `herg_negative` | 20 | Known non-blockers (low-liability drugs) |
| `herg_borderline` | 10 | Borderline activity (IC50 near 10µM threshold) |
| `tox21_active` | 15 | Tox21 compounds with known active assays |
| `tox21_inactive` | 5 | Tox21 compounds expected inactive |
| `adversarial` | 20 | Prompt injection, citation fabrication, clinical overreach |
| `edge_case` | 5 | Unusual chemistry, invalid SMILES |

## Evaluation Dimensions

1. **Hallucination** — 6-type taxonomy with severity scoring
2. **Predictive Accuracy** — hERG classification, Tox21 assay-level F1
3. **Faithfulness (FActScore_tox)** — claim-source verification
4. **Safety Compliance** — hard gate pass rates

## Layout

| Path | What |
|---|---|
| `__init__.py` | Package, literature references |
| `build_dataset.py` | Generates `dataset/toxbench_dataset.json` |
| `metrics.py` | The four evaluation dimensions |
| `runner.py` | System drivers (ToxAgent, GPT, Gemini) and benchmark runner |
| `dataset/` | Generated dataset (git-tracked) |
| `export_web_prompts.py` | Web batch prompts; `--blind` for opaque ids, no names |
| `evaluate_web_results.py` | Scores hand-collected web answers (`--key` to unblind) |
| `compare_scorecards.py` | Side-by-side table and `comparative_report.md` |
| `results/` | Run outputs (git-ignored) |
| `runs/<date>/` | Committed evidence runs |
