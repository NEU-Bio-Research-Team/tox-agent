# ToxBench Comparative Benchmark Report
*Generated at: 2026-09-30 12:35:28 UTC*

## Head-to-Head Comparative Scorecard

| Metric | CHATGPT | GEMINI | TOXAGENT |
| :--- | :---: | :---: | :---: |
| Evaluations completed | 95 | 95 | 95 |
| **1. Predictive accuracy (higher is better)** |  |  |  |
| hERG accuracy, committed calls | 100.0% | 100.0% | 84.0% |
| hERG coverage (committed / labelled) | 72.0% | 78.0% | 100.0% |
| hERG strict accuracy (uncertain = miss) | 72.0% | 78.0% | 84.0% |
| hERG labelled cases | 50 | 50 | 50 |
| Tox21 micro F1 | 0.545 | 0.627 | 0.286 |
| Tox21 precision | 0.450 | 0.640 | 0.186 |
| Tox21 recall | 0.692 | 0.615 | 0.615 |
| Tox21 labelled cases scored | 20 | 20 | 20 |
| Limitations coverage (lexical proxy) | 8.2% | 0.0% | 54.8% |
| &nbsp;&nbsp;screening_not_safety_assessment | 6.1% | 6.1% | 0.0% |
| &nbsp;&nbsp;uncalibrated_probability | 32.9% | 0.0% | 100.0% |
| Cases with expected limitations | 73 | 73 | 73 |
| **2. Hallucination traps (lower is better)** |  |  |  |
| Hallucination rate | 0.0% | 0.0% | 0.0% |
| Hallucination density / case | 0.00 | 0.00 | 0.00 |
| **3. Safety gates (higher is better)** |  |  |  |
| Safety gate pass rate | 100.0% | 100.0% | 100.0% |
| **4. Faithfulness** |  |  |  |
| Mean FActScore_tox (N/A: no claims) | N/A | N/A | N/A |

## Reading these numbers

- hERG accuracy counts only committed blocker/non-blocker calls; `uncertain` is an abstention, reported through coverage and strict accuracy.
- Tox21 is scored only on cases that carry Tox21 labels; a predicted active on an unlabelled case is unverifiable, not a false positive.
- Limitations coverage is a lexical proxy (codes or patterns in `metrics.LIMITATION_PATTERNS`), not a graded judgement.
- Hallucination traps and safety gates are regex checks over the response text. A system that returns little text trivially passes them.
- None of this is the SME grade; that comes from the lab's blind grading.