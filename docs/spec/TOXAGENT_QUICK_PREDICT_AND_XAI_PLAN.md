# ToxAgent — Quick Predict, Image Upload & XAI Explainer Plan

> **Date:** 2026-09-06
> **Status:** implementation plan (not a progress report)
> **Scope:** a predictor-only "Analyze" path that bypasses the agent layer,
> image upload usable on that path from day one, and an XAI explainer usable
> immediately after this plan lands.
> **Source contracts:** `TOXAGENT_AGENTIC_LAYER_REBUILD_PLAN_VI.md`,
> `toxpred/`, `toxocr/`, `toxagent-control/`, `frontend/`.

---

## 0. Motivation & non-goals

Today every molecule analysis goes through the control plane's session /
message / run / analysis / observation / outbox machinery, even in the
deterministic Lane D. Users who just want "SMILES in, numbers out" pay for a
session lifecycle they never asked for.

This plan adds a **stateless Quick Predict path**: `frontend → control-plane
thin proxy → ToxPred`, with **no session, no run, no analysis row, no outbox
event, no agent runtime**. It also makes the two things that path is missing
usable at the same time:

* **Image upload** — a stateless OCR proxy so the Quick Predict page accepts a
  photo/drawing, not only a typed SMILES.
* **XAI explainer** — atom-level attribution with a client-rendered highlighted
  2D depiction, built on the model that is actually served (ChemBERTa), not a
  hypothetical graph model.

### Non-goals

* No change to ToxPred's scientific meaning. No aggregate toxicity / safety
  verdict at any layer (ADR 0002). Every number still traces to a predictor
  provenance block.
* No revival of the ClinTox SMILES-GNN provider (its `tokenizer.pkl` is absent
  and gitignored). ClinTox stays `EndpointUnavailable`.
* No true graph GNNExplainer / PGExplainer in this plan — that needs a new
  served graph provider and its own ADR (see §6, "Later").
* The Quick Predict path does **not** persist an audit trail server-side. A
  caller that needs a durable, provenance-stamped record uses the existing
  Lane D analysis flow. Quick Predict returns the provenance in the response
  body and the client keeps it if it wants it.

---

## 1. Baseline facts this plan builds on

| Area | Fact | File |
|---|---|---|
| Predictor endpoints | `herg`, `tox21` served by `herg-tox21-chemberta-v1` (`required: true`); `clintox` `required: false`, unloadable, returns a deterministic "not served by this build" 503 | `artifacts/predictor-manifest.yaml`, `toxpred/application/predictor.py` |
| Predict API | `POST /v1/predictions` `{smiles, endpoints?, threshold_overrides?}` → `{input_smiles, canonical_smiles, predictions, applicability, provenance}` | `toxpred/api/routes.py`, `toxpred/domain/prediction.py` |
| Attribution API | `POST /v1/attributions` `{smiles, endpoint, task?}` → `{status, probability, tokens:[{token,position,importance,relative_importance}], metadata}`. Method `grad_x_embedding_l2_v1` = ‖grad ⊙ input-embedding‖₂ per **ChemBERTa SMILES token**. Numeric only — no plot, no image. | `toxpred/application/attribution.py`, `toxpred/scientific/providers/herg_tox21_chemberta.py` |
| OCR | `toxocr` `POST /v1/structure-recognition` `{mime_type, data_base64}` → `{smiles, canonical_smiles, confidence}`. MolScribe. 400 invalid base64 / oversize, 415 unsupported format, 422 `smiles_not_detected`. | `toxocr/api/routes.py` |
| Control-plane predictor client | `app.state.predictor: PredictorClient` — HTTP→typed errors (`InvalidSmiles`→422, `EndpointUnavailable`→SCI-06, `PredictorNotReady`→503), provenance copied verbatim (SCI-10) | `toxagent-control/toxagent/predictor/client.py`, `api/app.py:156` |
| Control-plane OCR client | `OcrClient` built in `create_app` when `TOXAGENT_OCR_URL` is set, but **not** currently exposed on `app.state` | `toxagent-control/toxagent/predictor/ocr_client.py`, `api/app.py:97` |
| Projection helper | `application/projections.display_projection(snapshot: AnalysisSnapshot)` produces the exact `AnalysisProjection` shape the frontend already renders (adds `measurement`, splits `label`, `required_limitations`, `policy_snapshot`) | `toxagent-control/toxagent/application/projections.py` |
| Threshold-override authorisation | `application/policy.authorise_threshold_overrides(actor, ...)` — expert-role gate | `toxagent-control/toxagent/application/policy.py` |
| Frontend analysis renderer | `AnalysisPanel` takes `AnalysisProjection`; composes `MoleculeDepiction` (smiles-drawer, pure JS), `EndpointCard`, `Tox21AssayTable`, `ApplicabilityChip`, `AttributionPanel` | `frontend/src/components/workbench/AnalysisPanel.tsx` |
| Frontend structure input | `MessageComposer` (typed SMILES + `looksLikeSmiles` heuristic), `ImageUploadDialog` (base64 PNG/JPEG/WebP), `StructureEditorDialog` (openchemlib, lazy) | `frontend/src/components/workbench/` |
| Frontend API types | `HergSection`, `Tox21Section`, `Tox21Assay`, `Applicability`, `EndpointSectionCommon`, `AnalysisProjection` already defined | `frontend/src/lib/api/types.ts` |
| Tox21 tasks (frozen order) | `NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53` | `toxpred/domain/endpoints.py` |

---

## 2. Decisions (resolved here, not deferred)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D-QP-1 | Does Quick Predict require auth? | **Yes** — same `Depends(actor)` bearer token as every other `/v1` route. | An unauthenticated route onto the predictor is a free DoS/enumeration surface. Alpha static token is enough; do not special-case it. |
| D-QP-2 | Direct browser → ToxPred, or proxy through control plane? | **Proxy through control plane.** | ToxPred has no auth, no CORS for browsers, and the deploy topology keeps it on a private network (three-boundary topology, ADR 0001). The proxy also reuses the typed error mapping and provenance discipline already in `PredictorClient`. |
| D-QP-3 | Route namespace | `POST /v1/predict`, `POST /v1/predict:batch`, `POST /v1/predict/recognize`, `POST /v1/predict/explain`, `GET /v1/predict/capabilities`. All **outside** `/v1/sessions/*`. | The path prefix itself signals "no session state". |
| D-QP-4 | Response shape | Return the **`AnalysisProjection` shape** (build an in-memory `AnalysisSnapshot`, run `display_projection`, never persist it). | The frontend already renders this shape verbatim via `AnalysisPanel`; zero new render code, and `rendered_value` canonicalisation (ADR 0005) is preserved. |
| D-QP-5 | Idempotency / retry | **None.** Stateless. Document (as `PredictorClient` already does) that a client retry buys a second predictor forward pass. | An idempotency store is session state by another name. Client owns its retries. |
| D-QP-6 | Abuse control without the session admission guard | A small dedicated limiter on the `predict*` routes: 1 SMILES per single call, batch cap = `PredictorSettings.max_batch_size`, and a per-token in-flight concurrency cap (default 4) returning `429 provider_rate_limited`. | The session path's concurrent-run guard does not cover these routes; without a cap they are a direct amplification path onto ToxPred. |
| D-QP-7 | `threshold_overrides` on Quick Predict | Allowed, but run through `authorise_threshold_overrides(actor, …)` exactly like Lane D. Non-expert → `403 forbidden`. | The operating point is a scientific choice with a role gate; the gate does not move just because there is no session. |
| D-IMG-1 | Image path: stateful reuse of `RecognizeStructure`, or a new stateless proxy? | **New stateless proxy** `POST /v1/predict/recognize` that only calls `OcrClient.recognize` and returns `{smiles, canonical_smiles, confidence}`. | `RecognizeStructure` creates a run + analysis + events. Quick Predict must not. The OCR service call itself is already stateless. |
| D-IMG-2 | Does the proxy persist the uploaded bytes (W4-07 object store)? | **No.** Bytes are decoded, size/mime/magic-byte checked, passed to OCR, discarded. | No object store means no TTL/ACL/cleanup obligation (W4-10, still open). The durable, audited image path remains the session flow. |
| D-IMG-3 | Two-step (recognize → user confirms SMILES → predict) or one-shot (`/v1/predict` accepts an image)? | **Two-step.** `/v1/predict/recognize` returns the SMILES + confidence into an **editable** field; the user confirms; then `/v1/predict`. | OCR is fallible; letting a wrong SMILES silently drive a prediction is the exact failure mode `StructureRecognitionCard` was built to prevent. Also keeps `/v1/predict` single-purpose. |
| D-XAI-1 | What model backs the explainer? | The **served ChemBERTa model**, via the existing `grad_x_embedding_l2_v1` token attribution. No new model. | It is the only served model with `token_attribution`. A graph explainer needs a served graph model, which does not exist. |
| D-XAI-2 | Token importance, atom importance, or both? | **Both.** Keep `tokens[]` verbatim; add `atoms[]` — token importances aggregated onto heavy-atom indices via a deterministic SMILES↔token span alignment computed in ToxPred with RDKit. | Chemists reason about atoms/substructures, not BPE tokens. Keeping `tokens[]` preserves full transparency about how `atoms[]` was derived. |
| D-XAI-3 | Where is the "plot" rendered? | **Client-side.** ToxPred stays numeric-only (rebuild plan §3.2 / D-5). The frontend draws the 2D depiction with per-atom highlight colouring. | Preserves "the predictor renders nothing for the UI". Adds no plotting dependency to the service image. |
| D-XAI-4 | Atom-index contract between BE and FE | `atoms[i].atom_index` is the 0-based index of a heavy atom in **`canonical_smiles`** in RDKit's output order for that exact string. The FE **must depict that same `canonical_smiles`**. The response echoes `canonical_smiles` and an `atom_order_version` string. | Both sides walk the identical string, so indices line up without shipping a coordinate set. A mismatch is a loud version-string check, not a silently wrong highlight. |
| D-XAI-5 | Alignment ambiguity (a token spanning 0 or >1 atoms, ring-closure digits, brackets) | Tokens that map to **no** heavy atom (bond symbols, parentheses, ring digits, stereo marks) contribute their importance to **no** atom and are surfaced under `unmapped_importance` (a scalar) so the UI can show "N% of importance is on bonds/topology, not atoms". A token spanning multiple atoms splits its importance **equally** across them. | Honest about what the mapping can and cannot claim; never invents an atom. |
| D-XAI-6 | tox21 explainer granularity | One assay per call. `endpoint=tox21` requires `task`. Running all 12 is 12 backward passes and is refused. | Cost, and a combined tox21 attribution is scientifically meaningless (assays are independent). |
| D-XAI-7 | Colour semantics on the depiction | A single-hue sequential ramp keyed to `relative_importance` (magnitude only). **No** red/green, no "toxic/safe" colour coding (keeps W5-08). A legend states "gradient×embedding attribution magnitude — not causal, not a mechanism". | The limitation `attribution_not_causality` already exists; the visual must not overclaim. |

---

## 3. Part A — Quick Predict ("Analyze" button)

### 3.1 Backend

**New module** `toxagent-control/toxagent/application/quick_predict.py`

```python
class QuickPredict:
    def __init__(self, predictor: PredictorClient, policy: PolicySettings): ...

    async def execute(
        self, *, actor: Actor, smiles: str,
        endpoints: tuple[str, ...] | None,
        threshold_overrides: Mapping[str, Any] | None,
    ) -> dict:  # AnalysisProjection-shaped, NOT persisted
```

Steps:

1. `resolve_endpoints(endpoints, policy)` — same helper Lane D uses; unknown
   endpoint → `422`.
2. If `threshold_overrides`: `authorise_threshold_overrides(actor, overrides)` —
   non-expert → `Forbidden` (403).
3. `resp = await predictor.predict(smiles, endpoints, threshold_overrides=…)`.
   Typed errors propagate unchanged: `InvalidSmiles` (422), `EndpointUnavailable`
   (SCI-06, 422), `PredictorNotReady` (503), `PredictorProtocolError` (502).
4. Build an **in-memory** `AnalysisSnapshot` from `resp` + `predictor.provenance_of(resp)`
   + `policy_snapshot(actor, …)`. Reuse the snapshot builder factored out of
   `create_analysis.py` (see §3.3). Do not touch the database.
5. `return projections.display_projection(snapshot)` with two extra keys:
   `{"persisted": false, "analysis_id": null}`.

**New routes** in `toxagent-control/toxagent/api/routes.py`

```
POST /v1/predict                 -> 200 AnalysisProjection(persisted=false)
POST /v1/predict:batch           -> 200 {results: [...], errors: [...], count}
GET  /v1/predict/capabilities    -> 200 {served_endpoints, models:[ModelInfo], predictor_id}
```

* `PredictRequest` (`api/schemas.py`, `extra="forbid"`):
  `{ smiles: str(min_length=1), endpoints?: list["herg"|"tox21"|"clintox"],
     threshold_overrides?: {herg?: float, clintox?: float, tox21?: {task: float}},
     include_attribution?: bool = false }`.
  `include_attribution` is a convenience that additionally calls
  `POST /v1/predict/explain` internally per requested endpoint (tox21 skipped
  unless a single task is unambiguous) and attaches `attributions: [...]`.
  Prefer the explicit `/v1/predict/explain` call from the UI; keep this flag
  for API users.
* `capabilities` is a straight proxy of `PredictorClient.models()` +
  `served_endpoints()`.
* All three routes are decorated with the §3.2 limiter.

**Limiter** `toxagent-control/toxagent/api/predict_limits.py`

* Async `Semaphore`-per-principal, `maxsize` from
  `PredictSettings.max_inflight_per_principal` (default 4). Acquire with
  `nowait`; on failure raise `ProviderRateLimited("too many predict calls in flight")`.
* Batch length check against `PredictorSettings.max_batch_size` → `422`.
* This is process-local. Note in the module docstring that a multi-instance
  deployment gets `N × cap`; a global limiter is deferred to W9 abuse controls
  and is out of scope here.

**Wiring** `api/app.py`

* `app.state.quick_predict = QuickPredict(client, settings.policy)`.
* `app.state.ocr = ocr` (also needed by Part B).
* New `PredictSettings` in `config.py` (`TOXAGENT_PREDICT_MAX_INFLIGHT`, …).

**Explicitly NOT added:** no `Run`, no `RunStatus`, no `EventType`, no outbox
write, no `analyses` row, no `observations` row, no SSE. A test asserts the DB
row counts are unchanged across a `/v1/predict` call (§7).

### 3.2 Backend tests

| Layer | Cases |
|---|---|
| Contract (`tests/contract/test_quick_predict.py`, `httpx.MockTransport` predictor) | shape passthrough == `display_projection`; `extra="forbid"`; endpoint filter; `invalid_smiles`→422; clintox→`endpoint_unavailable` (SCI-06, not 503-retryable); `PredictorNotReady`→503; threshold override by non-expert→403; threshold override by expert→applied and reflected in `threshold_source` |
| Limiter (`tests/unit/test_predict_limits.py`) | 5th concurrent call → 429 `provider_rate_limited`; batch over `max_batch_size` → 422; capacity releases after completion |
| Integration (`tests/integration/test_quick_predict_stateless.py`, real app + stub predictor) | `SELECT count(*)` on `runs`, `analyses`, `observations`, `event_outbox` unchanged before/after; response has `persisted=false`, `analysis_id=null` |
| Live smoke (manual, documented in `PROGRESS`) | real ToxPred on `127.0.0.1:8080`, `CCO` → herg+tox21 sections, provenance has real `predictor_version` + artifact hashes |

### 3.3 Refactor prerequisite

Factor the "predictor response → `AnalysisSnapshot`" construction out of
`CreateAnalysis.execute` into a free function
`domain/analysis.py::snapshot_from_prediction(resp, provenance, policy_snapshot, *, now)`.
`CreateAnalysis` keeps calling it, then persists; `QuickPredict` calls it and
does not. This is a pure move with no behaviour change; covered by the existing
`create_analysis` tests plus one new direct test of the function.

### 3.4 Frontend

**API layer**

* `src/lib/api/endpoints.ts`:
  ```ts
  quickPredict(input: QuickPredictRequest): Promise<AnalysisProjection>
  quickPredictCapabilities(): Promise<PredictCapabilities>
  recognizeStructure(input: {mime_type; data_base64}): Promise<RecognizedStructure>   // Part B
  explainPrediction(input: ExplainRequest): Promise<AtomAttribution>                  // Part C
  ```
* `src/lib/api/types.ts`: `QuickPredictRequest`, `PredictCapabilities`,
  `RecognizedStructure`, `ExplainRequest`, `AtomAttribution`. Reuse
  `AnalysisProjection`, `HergSection`, `Tox21Section`, `Applicability`.

**Route & page** — new lazy route chunk (pattern from W5-14).

* `src/pages/QuickPredictPage.tsx` at path `/predict`. No session context, no
  `useSessionEvents`, no React Query session keys.
* Left: an input card —
  * a SMILES text field (reuse `looksLikeSmiles` for inline hinting only);
  * "Vẽ cấu trúc" button → `StructureEditorDialog` (already lazy) → fills the
    SMILES field client-side;
  * "Tải ảnh" button → `ImageUploadDialog` → Part B flow;
  * endpoint checkboxes `herg`, `tox21`; `clintox` rendered **disabled** with a
    tooltip sourced from `quickPredictCapabilities()` (`served_endpoints`);
  * expert-only: threshold override fields (reuse the Lane D override component;
    gate on the same role signal the composer uses);
  * primary button **"Phân tích"** → `quickPredict`.
* Right: results — render `<AnalysisPanel analysis={result} />` directly (it
  already handles `null`, sections, unavailable endpoints, applicability,
  provenance, tox21 table). Pass a no-op `onAskAboutAnalysis` (there is no
  session to ask in) or hide that button behind a prop.
* Errors: `invalid_smiles` → inline field error; `endpoint_unavailable` → per
  endpoint badge; `predictor_not_ready` → retryable banner; `forbidden` on
  overrides → explain the expert gate.
* An "Phân tích nhanh (không lưu vào session)" note near the button so nobody
  expects it in their history.

**Navigation:** link from `LandingPage` and the app header.

**Frontend tests**

* `endpoints` unit (mock `fetch`): body shape, bearer header present.
* `QuickPredictPage` component test: renders `AnalysisPanel` from a fixture
  projection; clintox checkbox disabled; override fields hidden for non-expert.
* Playwright `e2e/quick-predict.spec.ts`: mock `POST /v1/predict`; type `CCO`,
  click Phân tích, assert herg + tox21 cards; **assert no request to
  `/v1/sessions`** was made (route interception count).

---

## 4. Part B — Image upload usable immediately

### 4.1 Backend

**Route** `POST /v1/predict/recognize` in `api/routes.py`

* `RecognizeRequest` (`extra="forbid"`): `{ mime_type: "image/png"|"image/jpeg"|"image/webp", data_base64: str }`.
* Handler:
  1. `if app.state.ocr is None:` → `503` `{code: "capability_unavailable",
     message: "no structure recognition service is configured"}`.
  2. Decode base64 (`InvalidRequest` on failure), enforce
     `OcrSettings.max_image_bytes` (reuse the constant), verify magic bytes
     against `mime_type` with the existing `_matches_declared_image_type`
     helper (lifted from the session route; extract to `api/_image.py` and have
     both call it).
  3. `result = await app.state.ocr.recognize(bytes, mime_type)`.
     `OcrError` → `422 smiles_not_detected`; `OcrUnavailable` → `503
     structure_recognition_unavailable`.
  4. Return `{ smiles, canonical_smiles, confidence }` verbatim. **No persistence.**
* Same §3.2 limiter (OCR forward pass is ~1–2 s CPU; cap in-flight).

**Wiring:** `app.state.ocr = ocr` in `create_app` (currently only passed to
`RecognizeStructure`). Nothing else changes.

### 4.2 Backend tests

* Contract (`tests/contract/test_predict_recognize.py`, stub `OcrClient`):
  happy path passthrough; `OcrError`→422; `OcrUnavailable`→503; oversize→400;
  mime/magic-byte mismatch→400; `app.state.ocr is None`→503
  `capability_unavailable`.
* Integration: DB row counts unchanged.

### 4.3 Frontend

* `ImageUploadDialog` already stages a `{mime_type, data_base64}` + preview.
  On the Quick Predict page, "confirm" calls `recognizeStructure(...)`.
* On success, populate the SMILES field with `canonical_smiles` and show a
  `StructureRecognitionCard`-style strip: thumbnail, recognised SMILES,
  `confidence` as a percentage **only if not null** (reuse the null-vs-unknown
  discipline from `ocr_client.py`), and an **editable** SMILES field with a
  "Phân tích với SMILES này" button. The user can fix the SMILES before
  predicting.
* Errors: `smiles_not_detected` → "Không nhận ra cấu trúc trong ảnh. Thử ảnh
  rõ hơn hoặc nhập SMILES."; `capability_unavailable` → hide the upload button
  entirely (feature-detect via `quickPredictCapabilities()` returning an
  `ocr_available` bool — add that field to the capabilities response).
* Reuse the existing safe-preview code (object URL, revoke on unmount); never
  inline-render bytes from the response.

**Frontend tests**

* Component: recognised SMILES lands in an editable field; confidence hidden
  when null; edit + predict uses the edited value.
* Playwright: mock `recognize` + `predict`; upload the 1×1 PNG fixture from the
  existing e2e; assert the SMILES field is populated and prediction renders.

---

## 5. Part C — XAI explainer usable immediately

### 5.1 Backend — ToxPred

**New: SMILES↔token↔atom alignment** `toxpred/scientific/featurization/token_atom_align.py`

```python
def align_tokens_to_atoms(canonical_smiles: str, tokens: list[str]) -> AtomAlignment
# returns, per token index: the tuple of heavy-atom indices it covers (possibly empty)
# atom indices are RDKit's output order for `canonical_smiles`
```

Implementation:

* `mol = Chem.MolFromSmiles(canonical_smiles)`; the heavy-atom output order for
  this exact string is positional (RDKit writes atoms in the order they appear
  in the string it produced). Walk the string char by char, tracking the
  running heavy-atom counter, and record each atom symbol's `[start, end)` char
  span (handle two-letter elements `Cl`/`Br`, bracket atoms `[nH]`, aromatic
  lowercase, ignoring ring-closure digits, bond symbols, parens, `%NN`,
  stereo `@`, `/\`).
* The tokenizer is a byte-level BPE (`convert_ids_to_tokens` gives pieces with
  the leading-space marker). Re-derive each token's char span in the original
  string by re-running the tokenizer with `return_offsets_mapping=True` (HF
  fast tokenizers support this) instead of string-matching pieces.
* Token span ∩ atom span ≠ ∅ ⇒ token maps to that atom. A token overlapping
  multiple atom spans maps to all of them.
* Deterministic; pure function; unit-testable without a model.

**Extend the provider** `herg_tox21_chemberta.token_attribution`

* Also return `offsets` (from `return_offsets_mapping`) alongside `tokens`.
* No change to the gradient computation.

**New application service** `toxpred/application/explain.py` (wraps
`AttributionService`)

* `explain(smiles, endpoint, task) -> dict`:
  1. `raw = attribution.attribute(...)` (existing).
  2. `mol = resolve(smiles)`; `alignment = align_tokens_to_atoms(mol.canonical_smiles, raw["tokens"])`.
  3. Aggregate: `atom_importance[a] = Σ over tokens t mapping to a of
     (importance_t / |atoms(t)|)`. Tokens mapping to no atom accumulate into
     `unmapped_importance`.
  4. Normalise `relative_importance` over `Σ atom_importance + unmapped_importance`.
  5. Return:
     ```json
     {
       "status": "completed|partial|failed",
       "endpoint": "herg", "task": null,
       "input_smiles": "...", "canonical_smiles": "...",
       "atom_order_version": "rdkit-output-order-v1",
       "probability": 0.37,
       "atoms": [{"atom_index": 0, "symbol": "C", "importance": ..., "relative_importance": ...}, ...],
       "unmapped_importance": 0.12,
       "tokens": [ ...raw tokens... ],
       "method": "grad_x_embedding_l2_v1+token_atom_align_v1",
       "metadata": { "model_id": "...", "deterministic": true, "duration_ms": ..., "note": ... }
     }
     ```
* `status` / `partial` / timeout semantics inherited from `AttributionService`
  unchanged.

**New route** `POST /v1/explanations` in `toxpred/api/routes.py`

* `ExplainRequest`: `{ smiles, endpoint: "herg"|"tox21", task?: str }`.
  `endpoint=tox21` without `task` → `422`. `task` for non-tox21 → `422`.
* Delegates to `app.state.explain`.
* `/v1/attributions` stays as-is (token-only) for backward compatibility.

**ToxPred tests**

* `test_token_atom_align.py` (no model): `CCO` → tokens map to atoms 0,1,2 and
  the `O`; `c1ccccc1` aromatic ring → all six carbons covered, ring digits
  unmapped; `CC(=O)Cl` → `Cl` two-letter handled, `=` and `(` unmapped;
  bracket atom `C[NH3+]` → N mapped, `H3`/`+`/brackets unmapped;
  `Σ relative_importance (atoms) + unmapped == 1.0` within float tol.
* `test_explain_service.py` (stub provider returning fixed token scores):
  aggregation math; multi-atom token split; `partial` on slow attribution.
* `test_explain_route.py`: tox21 without task → 422; herg happy path shape;
  `atom_order_version` present.
* Live smoke: real model, `CC(=O)Oc1ccccc1C(=O)O` (aspirin), herg — eyeball
  that ester/aromatic atoms carry weight and `Σ ≈ 1`.

### 5.2 Backend — control plane

**New route** `POST /v1/predict/explain` in `toxagent-control` — thin proxy to
ToxPred `POST /v1/explanations`.

* `PredictorClient.explain(smiles, endpoint, task)` added next to
  `.attribution(...)`, same timeout budget (`attribution_read_timeout_s`),
  same typed error mapping.
* `ExplainRequest` in `api/schemas.py` (`extra="forbid"`).
* Response passthrough of the ToxPred JSON, plus `attribution_not_causality`
  echoed as a `limitations: ["attribution_not_causality"]` array so the UI
  always has it (mirrors what the grounded-answer path enforces).
* Same §3.2 limiter.
* No persistence.

**Control-plane tests:** contract (mock transport) for passthrough, tox21
task-required 422, endpoint_unavailable, timeout→`partial` note preserved;
integration DB-unchanged assertion.

### 5.3 Frontend

**`AtomHighlightDepiction`** — new component
`src/components/workbench/AtomHighlightDepiction.tsx`

* Props: `{ smiles: string /* MUST be canonical_smiles from the response */,
  atoms: {atom_index; relative_importance}[], size? }`.
* `smiles-drawer` supports a highlight argument: build a
  `Map<atomIndex, color>` where colour = a single-hue sequential ramp
  (`--accent-blue` family) with alpha ∝ `relative_importance / max`. If the
  installed `smiles-drawer` version's highlight support is inadequate, fall
  back to lazy-loading `@rdkit/rdkit` (WASM) **only for this view** (never on
  the workbench first paint) and use
  `MolDraw2D` `drawMoleculeWithHighlights`. Decide during PR-C3 by spike; the
  component's public props do not change either way.
* Assert-in-dev: if `atom_order_version !== "rdkit-output-order-v1"` render a
  "cannot align highlights" fallback (the plain depiction + token list only),
  never a guessed highlight.

**`ExplainPanel`** — `src/components/workbench/ExplainPanel.tsx`

* Endpoint/assay selector (herg, or one tox21 task).
* "Giải thích" button → `explainPrediction(...)`.
* Renders: `AtomHighlightDepiction` + a horizontal bar list of top-k atoms
  (`symbol + index → relative_importance`) + the existing `AttributionPanel`
  token view (collapsed) + `unmapped_importance` shown as "X% trọng số nằm ở
  liên kết/tô-pô, không gán được cho nguyên tử".
* Legend: "Độ lớn attribution gradient×embedding — không phải cơ chế, không
  phải quan hệ nhân quả." Limitation chip `attribution_not_causality` shown at
  content level, not hidden.
* Single-hue ramp only; no red/green (W5-08).
* `status: "partial"` → a "best-effort, vượt ngân sách thời gian" note.
  `status: "failed"` → error state, no depiction highlight.

**Placement:** a collapsible "Giải thích (XAI)" section under each served
endpoint card on the Quick Predict results, and (reused) in the session
`AnalysisPanel` where `AttributionPanel` already lives.

**Frontend tests**

* `token→atom` fixture render: highlights applied to the right atom indices;
  version mismatch → fallback, no highlight.
* `ExplainPanel`: partial/failed states; unmapped importance line; legend
  present; no red/green colour tokens in output.
* Playwright: mock `/v1/predict/explain`; open XAI section on a herg result;
  assert depiction canvas + top-atom bars render.

---

## 6. Sequencing (small, independently reviewable PRs)

| PR | Title | Contents | Gate |
|---|---|---|---|
| **PR-QP1** | control-plane snapshot refactor | Extract `snapshot_from_prediction`; no behaviour change | existing `create_analysis` tests green |
| **PR-QP2** | `POST /v1/predict` + limiter | `QuickPredict`, `PredictRequest`, limiter, `capabilities`; DB-unchanged integration test; live smoke | backend suite + smoke |
| **PR-QP3** | Quick Predict page (SMILES + draw only) | `/predict` route, `quickPredict` client, `AnalysisPanel` reuse, nav links; component + e2e (no `/v1/sessions` call) | frontend suite + e2e + bundle gate |
| **PR-IMG1** | `POST /v1/predict/recognize` | stateless OCR proxy, `app.state.ocr`, shared `_image.py`; contract tests | backend suite |
| **PR-IMG2** | image on Quick Predict page | `ImageUploadDialog` → recognize → editable SMILES → predict; capability feature-detect; component + e2e | frontend suite + e2e |
| **PR-XAI1** | ToxPred token↔atom alignment + `/v1/explanations` | `token_atom_align.py`, `explain.py`, route, `return_offsets_mapping`; unit + route tests + live smoke | toxpred suite + smoke |
| **PR-XAI2** | control-plane `POST /v1/predict/explain` | `PredictorClient.explain`, proxy route, limiter, limitations echo; contract tests | backend suite |
| **PR-XAI3** | `AtomHighlightDepiction` + `ExplainPanel` | highlight rendering (smiles-drawer or RDKit-JS spike), XAI section on Quick Predict + session `AnalysisPanel`; component + e2e | frontend suite + e2e + bundle gate |
| **PR-QP4** | `POST /v1/predict:batch` | batch route + page multi-SMILES input | backend + frontend suites |

Dependency order: QP1 → QP2 → QP3; IMG1 → IMG2 (after QP3); XAI1 → XAI2 → XAI3
(after QP3). QP4 last.

---

## 7. Invariants preserved (checklist for every PR)

* [ ] No aggregate toxicity / safety verdict anywhere (ADR 0002). tox21 stays a
      mapping, never a count or score (SCI-05).
* [ ] Every number in a Quick Predict response carries the predictor
      `provenance` block, copied verbatim (SCI-10).
* [ ] `rendered_value` canonicalisation (ADR 0005) — reused via
      `display_projection`, not re-derived on the client.
* [ ] Invalid SMILES → `422 invalid_smiles`, never a prediction of zero risk
      (SCI-08).
* [ ] ClinTox → `endpoint_unavailable` (SCI-06), never a substitute model.
* [ ] Applicability is presented as rule-based (`method` string surfaced),
      never as a learned OOD score.
* [ ] Quick Predict writes **zero** rows: `runs`, `analyses`, `observations`,
      `event_outbox` counts unchanged (asserted in an integration test per
      route).
* [ ] ToxPred renders no images; it returns numbers only (rebuild plan §3.2 /
      D-5). The 2D highlight is client-rendered.
* [ ] XAI never claims causality/mechanism; `attribution_not_causality` shown
      at content level; magnitude-only single-hue colour, no red/green.
* [ ] XAI atom highlight only rendered when `atom_order_version` matches;
      otherwise a plain fallback, never a guessed alignment.
* [ ] All `predict*` routes require a bearer token and are covered by the
      in-flight limiter.
* [ ] Frontend policy-lint, typecheck, unit/component, Playwright, production
      build and gzip bundle budget all green.

---

## 8. Later (explicitly out of this plan)

* **True graph explainer (GNNExplainer / PGExplainer / SubgraphX).** Requires a
  *served* graph provider. Options: restore `clintox-smilesgnn` (needs
  `tokenizer.pkl`), or add a new GATv2/GIN provider from `backend/` with its
  own artifact manifest entry, calibrated thresholds, and an ADR. Then a real
  subgraph-mask explainer returns `atoms[]` + `bonds[]` natively and this
  plan's token→atom alignment becomes one of two explainer backends behind the
  same `/v1/explanations` contract.
* **Bond-level importance** in the token→atom aligner (currently bonds fall
  into `unmapped_importance`).
* **Server-side persisted Quick Predict history** — deliberately omitted; use
  the Lane D analysis flow when a durable record is needed.
* **Global (multi-instance) predict rate limiting** — folded into W9 abuse
  controls.
* **One-shot `/v1/predict` accepting an image** — rejected in favour of the
  two-step recognise-then-confirm flow (D-IMG-3).
