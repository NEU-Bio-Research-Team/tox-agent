# What an atom/token attribution can and cannot show

Background for your reasoning, not a source: do not cite this page, and do not
write numbers from it into an answer.

## The served method

The served explainer is signed gradient × input over the model's token
embeddings, projected from SMILES tokens onto atoms and bonds. It is
deterministic, and the same molecule written differently gives the same
attribution. On the measured benchmark panel it was not measurably more
faithful than deleting random atoms, for either measured target; every other
target and method is unmeasured. The deletion benchmark has its own weakness —
deleting atoms produces molecules the model never saw — so this result does not
show the attributions are wrong. It shows nobody may present them as evidence.

## Three different questions

| Question | Can an attribution answer it? |
|---|---|
| Which input tokens moved this model's score? | Yes, with the verdict above attached |
| Does this structural feature cause the toxicity? | No. That needs experimental evidence |
| Is the model's score right? | No. An attribution explains a score; it does not validate it |

## Known ways attributions mislead

- Saliency methods can look plausible while being insensitive to the model's
  weights; sanity checks (randomising the model or the labels) exist for that
  reason ([Adebayo et al., NeurIPS 2018](https://proceedings.neurips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf)).
- Token-to-atom projection spreads importance over atoms that share a token,
  and bonds without a token of their own get a colour derived from their atoms.
- A highlight on a familiar alert reads as confirmation. It is the model
  pattern-matching, which may or may not reflect biology.
