# Artifact card — clintox-smilesgnn-v1 (blocked)

The checkpoint is retained for provenance but is not served. Its exact
training tokenizer is absent; the checkpoint embedding expects a 69-token
mapping, while unrelated tokenizers cannot be substituted safely. A recovered
tokenizer must be verified by vocabulary, token-to-id mapping, special tokens,
embedding dimension and hash. Otherwise the endpoint requires a reproducible
retrain named `clintox-smilesgnn-v2`, with tokenizer, weights, calibration,
threshold, untouched-test metrics and manifest shipped together.
