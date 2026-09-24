# XAI benchmark — does an explanation mean anything?

`benchmark.py` measures three properties of the atom-level attributions
`/v1/explanations` returns. It is a harness plus one measurement, not a claim
that the explanations are good; the numbers below say the opposite in one
respect and it is recorded here rather than left out.

```
PYTHONPATH=src MODELS_ROOT=<weights> python evals/xai/benchmark.py \
  --endpoint herg --out results/herg-<date>.json
```

## What it measures, and why each one

| Property | Question | Failing it means |
|---|---|---|
| Determinism | Two attributions of one input, identical? | The response's `deterministic: true` is false |
| Invariance to spelling | Same molecule typed differently, same attribution? | What a user sees depends on how they typed the molecule |
| Faithfulness | Do the atoms it calls important move the probability more than arbitrary ones? | The highlight is a picture, not an explanation |

Faithfulness is measured by deletion **against a random control**, never as an
absolute number: "the probability moved when we deleted atoms" says nothing on
its own. Both arms draw from the same candidate set — atoms whose individual
removal leaves a single sanitisable molecule — because drug-like molecules are
mostly fused rings and deleting three arbitrary heavy atoms usually fragments
them. A first version without that restriction skipped every molecule in the
panel, which measured ring topology rather than attribution.

## Measured 2026-09-09, `herg-tox21-chemberta-v1`, golden panel (42 molecules)

| | hERG | Tox21 NR-AR |
|---|---:|---:|
| Determinism | **42/42** | **42/42** |
| Invariance to spelling | **40/40** | **40/40** |
| Faithfulness wins | 15/34 | 14/35 |
| Median Δp, top-k deleted | 0.0340 | 0.0383 |
| Median Δp, random deleted | 0.0294 | 0.0449 |

Two molecules could not be re-spelled into a different valid SMILES; six to
seven had too few individually-removable atoms to form two disjoint arms.

### Reading the faithfulness row honestly

**On this panel, with this deletion metric, gradient×input attribution is not
measurably more faithful than deleting arbitrary atoms.** 15 wins out of 34 is
what a coin flip produces; on Tox21 NR-AR the median for the top-k arm is
*lower* than for the control. This is a negative result and it is the reason
the number is reported rather than gated: the script exits non-zero for a
determinism or spelling failure — properties the service claims — and reports
faithfulness without passing judgement, because the threshold at which an
attribution method is good enough is a scientific decision, not a script's.

What it does **not** establish: that the attributions are wrong. Deletion on
molecular graphs is a blunt instrument — removing an atom changes valence and
the model sees an out-of-distribution string — and 42 molecules is a small
panel. What it does establish is that no one may present these highlights as
evidence of causation. That was already the plan's rule; this is the first
measurement supporting it rather than asserting it.

Better instruments, when there is data and a decision to fund them: masking
rather than deletion, sufficiency alongside comprehensiveness, and a panel
with per-endpoint actives. Those are K10's blocked items.

## What running it found

Four of the forty-two panel molecules crashed `/v1/explanations` with an
unhandled `ValueError`: sodium salicylate, diphenhydramine hydrochloride,
cisplatin and ferrocene. All four are written as more than one component, and
the SMILES walk in `token_structure_align.py` had no case for `.` — so it
asked RDKit for a bond between the last atom of one fragment and the first of
the next. Salts are an ordinary way to write a drug. Fixed, with tests.
