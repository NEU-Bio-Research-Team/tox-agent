# Artifact card — clintox-smilesgnn-v1 (blocked)

The checkpoint is retained for provenance and is not served.

## Why it is blocked

Its exact training tokenizer is absent. The checkpoint's
`smiles_encoder.token_embedding.weight` is `(69, 96)` — a 69-token vocabulary
derived from the ClinTox training corpus — and the other SMILES tokenizers on
disk have 80 tokens. Substituting one would remap every token and produce
confident, meaningless probabilities, so the provider refuses to load and the
registry leaves the endpoint unregistered rather than answering `clintox` with
a different model.

## Admission criteria

These live in the manifest
([`backend/predictor/registry/models/clintox-smilesgnn-v1.yaml`](../../backend/predictor/registry/models/clintox-smilesgnn-v1.yaml)),
under `tokenizer:`, so admission is a check rather than a paragraph someone has
to read and apply. `ClinToxSmilesGnnProvider.admission_report()` reports every
unmet one, not the first.

| Criterion | Declared | State |
|---|---|---|
| Vocabulary size | `vocab_size: 69` | Verified against the checkpoint's embedding matrix |
| Which tensor to read it from | `checkpoint_embedding_key` | Present |
| Tokenizer file hash | `sha256` | **Not recorded** — the training run did not produce one |
| Token-to-id mapping hash | `vocab_sha256` | **Not recorded** |
| Tokenizer file present | `path: tokenizer.pkl` | **Absent** from this repository |

The two hashes are what would distinguish the training artifact from a
different tokenizer that happens to have 69 tokens. While they are null,
nothing is admitted — including a correct file, because there would be no way
to tell that it was correct. Recording them is a manifest change, not a code
change.

## Even fully admitted, v1 does not serve

The v1 inference path depended on the retired backend and is deliberately not
part of the standalone wheel. A provider that met every criterion above would
still refuse, saying so. Restoring the endpoint means a reproducible retrain
released as `clintox-smilesgnn-v2`, with its own provider, tokenizer, weights,
calibration, threshold provenance, untouched-test metrics and manifest shipped
together.

Until then `clintox` is an unavailable capability. hERG and Tox21 are different
endpoints and are not a substitute for it.
