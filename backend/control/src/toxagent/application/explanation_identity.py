"""What makes two explanations the same explanation (XAI-01).

There used to be two answers to that question. ``create_analysis`` keyed its
checkpoints on ``explanation_checkpoint_key`` — canonical SMILES, endpoint,
task, model id and that model's artifact hashes — while
``application/explanation.py`` keyed its observations on ``idempotency_key``
over *every* model's artifact hashes and wrote them under a different schema
version. Two keys over the same facts means two caches, and a report builder
looking in one of them could not see what the analysis had already paid for: it
recomputed the attribution, or recorded a gap, while a perfectly good payload
sat in the other cache under the other key.

So identity lives here, once, and both pipelines import it. The rule is that
every input which can change the numbers is in the key:

- **what is explained** — canonical SMILES, endpoint, task;
- **what explained it** — model id *and* that model's weights/tokenizer
  hashes, because an id is a name a deployment chooses and a retrain replaces
  the weights behind it without the name moving (K06);
- **how** — the attribution method when one is pinned, and
  ``ATTRIBUTION_ALIGNMENT_VERSION``, which covers the part of the computation
  that is ours rather than the predictor's: mapping token attributions back
  onto atom and bond indices. A change there re-labels every atom in every
  cached payload while leaving the payload byte-identical, which is precisely
  the kind of drift a content hash cannot see.
"""
from __future__ import annotations

import hashlib
from typing import Final

#: Bumped whenever token->atom/bond alignment changes meaning. Part of every
#: cache key: aligning the same attribution differently is a different
#: explanation of the same molecule, and serving the old one would attach last
#: version's numbers to this version's atom indices.
ATTRIBUTION_ALIGNMENT_VERSION: Final = "atom-alignment-v1"

#: Schema version written by both pipelines from XAI-01 onwards. v1 was
#: ``application/explanation.py``'s, v2 was ``create_analysis``'s; the two are
#: still readable (see ``EXPLANATION_SCHEMA_VERSIONS``) because reports already
#: exist that cite them, but nothing writes them any more.
EXPLANATION_SCHEMA_VERSION: Final = "toxpred-explanation-v3"

#: Every schema version an explanation observation may carry, newest first. A
#: reader matches against this set; only ``EXPLANATION_SCHEMA_VERSION`` is
#: written. Matching on the *string* is what let the report builder miss a
#: finished explanation, so readers resolve by target identity and treat the
#: version as a decoding hint rather than as a filter.
EXPLANATION_SCHEMA_VERSIONS: Final[tuple[str, ...]] = (
    "toxpred-explanation-v3",
    "toxpred-explanation-v2",
    "toxpred-explanation-v1",
)


def is_explanation_schema(schema_version: str | None) -> bool:
    return schema_version in EXPLANATION_SCHEMA_VERSIONS


def model_artifact_fingerprint(provenance, model_id: str | None) -> tuple[str, ...]:
    """The artifact hashes belonging to one model, out of a response's provenance.

    ToxPred reports provenance per model — `[{"model_id": ..., "weights_sha256":
    ..., "tokenizer_sha256": ...}]` — which the client flattens to
    `"<model_id>:<field>=<hash>"`. Selecting by prefix keeps the pin precise: a
    retrain of one admitted model invalidates its own explanations and leaves
    the other's alone.

    An empty result is the honest answer when the predictor reported no
    artifacts for this model, and it is a *different* key from any non-empty
    one — so a checkpoint written while the weights were unidentified is never
    served as though it had been pinned to them.
    """
    if not model_id:
        return ()
    prefix = f"{model_id}:"
    return tuple(sorted(
        h for h in getattr(provenance, "artifact_hashes", ()) or () if h.startswith(prefix)
    ))


def explanation_cache_key(
    *,
    canonical_smiles: str,
    endpoint: str,
    task: str | None,
    model_id: str | None,
    artifact_fingerprint: tuple[str, ...] = (),
    method: str | None = None,
    alignment_version: str = ATTRIBUTION_ALIGNMENT_VERSION,
) -> str:
    """Identity of one explanation: what it explains, and what produced it.

    ``method`` is the *pinned* attribution method when a caller has one, not
    the method the predictor happened to use — the latter is only known after
    the call and so cannot take part in the lookup that decides whether to make
    it. Neither pipeline pins one today; both pass ``None``, and the recorded
    payload carries the method that actually ran so a mismatch is visible in
    the audit trail rather than silently cached over.
    """
    # Unit-separated, so no two different tuples of inputs can flatten to the
    # same string: ("herg", "") and ("her", "g") are different questions.
    material = "\u001f".join(
        (
            canonical_smiles,
            endpoint,
            task or "",
            model_id or "",
            method or "",
            alignment_version,
            *artifact_fingerprint,
        )
    )
    return f"sha256:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


#: The name ``create_analysis`` has always used. Same function; kept so the
#: checkpoint call sites and their tests read as they did.
explanation_checkpoint_key = explanation_cache_key
