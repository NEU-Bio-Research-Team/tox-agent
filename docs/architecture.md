# Architecture

ToxPred is a headless toxicity prediction service. One request in, typed
probabilities out, with enough provenance attached to reproduce the answer.

## Layers and the rule between them

```
        api  ──────────►  application  ──────────►  domain
         │                     │                       ▲
         │                     ▼                       │
         └──────────────►  scientific  ────────────────┘
                          (registry, providers, featurisation)
```

```
api          → application → domain, scientific interfaces
application  → domain, scientific interfaces
scientific   → domain, model libraries, RDKit
domain       → standard library only
```

`domain/` imports no FastAPI, no torch, no RDKit, no yaml, and nothing from
`backend/`. `tests/unit/test_import_boundaries.py` enforces this by parsing the
AST, so a violation fails CI without needing the heavy dependencies installed.

`import toxpred` loads no model and opens no socket. That is also a test.

## Request flow

```
POST /v1/predictions
  │
  ├─ pydantic validation ......... unknown fields rejected (422)
  ├─ RDKit resolve ............... canonicalise once; invalid input → 400
  ├─ resolve endpoints ........... unserved endpoint → 503, never a substitute
  ├─ policy snapshot ............. thresholds from the artifact, source recorded
  ├─ registry → provider ......... one call per provider, not per endpoint
  ├─ raw probabilities ........... provider returns floats, never labels
  ├─ apply thresholds ............ label derived in the domain layer
  ├─ applicability ............... element rules, method named in the payload
  └─ provenance .................. artifact SHA-256, policy version, request id
```

Thresholding lives in `domain/policy.py`, not in a provider. A provider that
could bake in an operating point would make the same class of mistake the old
code did — a number applied far from where it was chosen.

## Package map

| Path | Holds | Must not contain |
|---|---|---|
| `toxpred/api/` | FastAPI app, routes, request schemas, error mapping | Model-specific branching |
| `toxpred/application/` | `ToxicityPredictor`, `AttributionService` | HTTP concerns |
| `toxpred/domain/` | Endpoints, thresholds, typed results, molecule | Any third-party import |
| `toxpred/scientific/` | Manifest, registry, providers, featurisation, applicability | Business workflow |
| `backend/` | Model architectures, graph featurisation, tokenizers, training | New serving logic |
| `artifacts/` | Manifest: which models may be served, and their checksums | Weights |
| `benchmarks/` | Frozen split, golden panel, benchmark runner | Generated per-sample dumps |
| `deploy/` | Dockerfile, entrypoint, artifact download, Cloud Run env | Application code |

`backend/` is deliberately not rewritten. It carries the model architectures and
the training code, and providers call into it rather than reimplementing the
science — so a checkpoint keeps loading the way it was trained.

## The three endpoints are separate on purpose

hERG blockade, Tox21 assay activity and ClinTox clinical toxicity are different
measurements from different datasets. Each is a distinct frozen type with its own
field name (`probability_blocker`, `probability_activity`,
`probability_clinical_toxicity`), and `PredictionResult` builds its payload from
those types alone.

This is the defect the refactor exists to fix. The previous implementation took
the hERG head's sigmoid and emitted it as `clinical.p_toxic` with a "clinical"
threshold: an ion-channel liability presented as clinical-trial toxicity, read at
0.35 while the model was calibrated at 0.4133. Contract tests fail if a hERG
value can be serialised under a `clinical` key.

There is no aggregate verdict. Counting hits across chemically unrelated assays
produces a number that reads like severity and is not.

## Artifacts and the registry

A directory existing is not an artifact. `artifacts/predictor-manifest.yaml` declares each
model's files with SHA-256 and size; the registry verifies every one before
loading and reports all problems at once.

- A **required** model that fails to load fails startup.
- An **optional** model that fails removes only its own capability. `/v1/models`
  reports it with the reason.
- A missing capability raises. Nothing is substituted — the failure mode that
  let `DEFAULT_TOX_TYPE_MODEL_KEY = "tox21_ensemble_3_best"` point at a directory
  holding a metrics file and no weights.

Weights are not in the image. `deploy/entrypoint.sh` fetches them from
`MODEL_ARTIFACTS_URI` at container start; a request never triggers a download.

Serving needs no network at all. The ChemBERTa checkpoint carries every backbone
weight, and the architecture config is vendored from a pinned revision under the
artifact, so the backbone is built from a local file rather than resolved against
Hugging Face. The container runs with `HF_HUB_OFFLINE=1` and CI asserts it.

## Thresholds

| Source | Meaning |
|---|---|
| `artifact` | Calibrated on a validation split, shipped with the weights (hERG: 0.4133, Youden-J over 3-fold CV) |
| `manifest_declared` | Chosen operationally, **not** calibrated (ClinTox: 0.35) |
| `request_override` | Supplied by the caller for one request |

Every label carries its threshold and the source. The distinction is the point:
collapsing the first two is how an uncalibrated number came to look calibrated.

## Attribution

Its own endpoint, because it costs a backward pass. Gradient × input-embedding
L2 per token, one head at a time, deterministic. Numbers only — nothing in the
path imports a plotting library, so the runtime image needs none. Attributing the
whole Tox21 endpoint is refused; the twelve assays are independent.

## Related

- [`model-card.md`](model-card.md) — measured performance and limitations
- [`benchmark-protocol.md`](benchmark-protocol.md) — how those numbers are produced
- [`refactor/PREDICTOR_ONLY_STATUS_VI.md`](refactor/PREDICTOR_ONLY_STATUS_VI.md) — what changed and why
