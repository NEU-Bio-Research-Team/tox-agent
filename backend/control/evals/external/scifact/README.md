# SciFact under its own protocol

[SciFact](https://aclanthology.org/2020.emnlp-main.609/) (Wadden et al., EMNLP
2020): 1,409 expert-written scientific claims, each checked against a corpus of
5,183 abstracts for SUPPORT / CONTRADICT with rationale sentences. RETHINK §5.1
uses it to calibrate the claim-support step (`evidence_relation`).

## What runs

| Step | Implementation | Source it follows |
|---|---|---|
| Data | `data.py` downloads the release archive, refuses it unless its SHA-256 is `11c62128…d76be` | `scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` |
| Retrieval | `oracle` (gold abstracts, plus the first cited abstract of a no-evidence claim) or `tfidf` (scikit-learn, 1–2 grams, English stop words, top k) | `verisci/inference/abstract_retrieval/{oracle,tfidf}.py` |
| Label + rationale | a judge (`judges.py`): one claim, one abstract with numbered sentences, JSON out | fixed prompt, hashed into the manifest |
| Scoring | `metrics.py`, a pandas-free port | `verisci/evaluate/lib/metrics.py` @ `68b98a56` (Apache 2.0) |

**Port check.** When written (2026-09-25) the port was compared with the
official module on 40 sets of random predictions over the full dev split: 480
metric values, 0 differences. `tests/unit/test_scifact.py` pins the official
worked example from `doc/evaluation.md` and each scoring branch.

## Labels

- Full dev split with the official scorer: `external-native` (dev numbers are
  what the paper reports for development; the test split is scored only by the
  leaderboard, so `--split test` writes a submission file and no metrics).
- A seeded subset: `external-native-subset`, never compared with full-split
  numbers.
- The retrieval setting (oracle / tfidf, NEI included or not) is part of the
  label.

## What it measures, and what it does not

A judge is **a model used as a stand-alone verifier**. The number belongs to
that model, not to the ToxAgent product, whose evidence-relation step runs
inside an answer with its own tools and validator. Measuring the product
itself on SciFact means feeding the corpus through its research provider and
reading the relations of its answers; that is a `published-data-transfer`
study, planned in `docs/EXTERNAL_BENCHMARKS.md`.

## Running

```bash
python -m evals.external.scifact.run --split dev --retrieval oracle \
  --judge claude:opus --limit 50 --seed 20260925
python -m evals.external.scifact.run --split dev --retrieval tfidf --k 3 --judge codex
```

Output under `runs/<run-id>/`: `predictions.jsonl` (official submission
format), `judgments.jsonl` (every call: prompt hash, raw output, reported
model, timing, usage, parse errors), `metrics.json`, `manifest.json`
(label, release hash, split, claim ids, retrieval, judge, environment, error
counts). A judge output that cannot be parsed counts as NOT_ENOUGH_INFO and is
counted, never dropped. The data themselves are never committed (claims
CC BY 4.0, abstracts ODC-By 1.0; cached under `~/.cache/toxagent-evals`).
