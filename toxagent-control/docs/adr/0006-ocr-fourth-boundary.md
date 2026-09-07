# ADR 0006 — Structure recognition (OCR) is a fourth deployment boundary

**Status:** accepted · **Date:** 2026-09-05

## Context

Image-based structure input ("upload a picture of a molecule") needs an
optical chemical structure recognition (OCSR) model — image in, SMILES out.
This codebase's own history already tried the alternative: the pre-refactor
agent layer (tag `archive/agent-layer-165319beede5`) imported `molscribe`
straight into its monolithic `model_server/main.py`. The predictor-only
rebuild (`docs/refactor/PREDICTOR_ONLY_STATUS_VI.md`) deliberately removed it
and added a static import-boundary test (`toxpred/tests/unit/
test_import_boundaries.py`) that fails the build if `toxpred/` ever imports
`molscribe` again — OCR is a torch/opencv/timm-heavy dependency chain with
nothing to do with toxicity prediction, and MolScribe's own pin
(`torch>=1.11.0,<2.0`) actively conflicts with the predictor's torch version.

The frontend redesign plan recorded this as closed: D-7 concluded "after MVP,
because the backend no longer has an OCR service" (`TOXAGENT_FRONTEND_
REDESIGN_PLAN_VI.md`). That premise is what this ADR changes.

## Decision

A fourth independent deployment boundary, `toxocr`, alongside the three ADR
0001 already names:

| Boundary | Process | Owns |
|---|---|---|
| `toxpred` | existing predictor | canonical SMILES, prediction, threshold, applicability, attribution |
| `toxocr` | new, this ADR | image -> SMILES (MolScribe), nothing else |
| `toxagent-control` | this project | product API, auth, session, router, snapshots, evidence, answers, validation, SSE, tool gateway |
| `agent-runtime-host` | pinned OpenCode or DSH | model-tool loop, provider requests |

`toxagent-control` reaches `toxocr` only over `POST /v1/structure-recognition`
(`toxagent/predictor/ocr_client.py`). It never imports MolScribe, torchvision,
or any vision model code — the same discipline ADR 0001 already applies to
`toxpred`, extended to a second external prediction-shaped service.

`toxocr` needs its own Python environment, not `toxpred`'s or
`toxagent-control`'s: MolScribe pins `torch<2.0`, both of the others run a
different (or no) torch. See `toxocr/requirements.txt` for the verified
install sequence.

Configuration is pluggable exactly like `ResearchSettings`/evidence_research,
not a hardcoded capability flag: an unset `TOXAGENT_OCR_URL` means
`Intent.STRUCTURE_RECOGNITION` is routed but never dispatched to a runtime —
`SubmitMessage` answers `capability_unavailable` deterministically (see
`application/submit_message.py`, `_CAPABILITY_UNAVAILABLE_MESSAGE`). A
configured deployment instead schedules `RecognizeStructure`
(`application/recognize_structure.py`), which hands a recognised SMILES to
the exact same `CreateAnalysis` pipeline a typed SMILES already goes through
— same validators, same snapshot, same provenance. Recognition failure (no
structure found, service unreachable) completes the run with a conversational
answer, never a queued run that later fails.

## Consequences

- `toxagent`'s own import-boundary test still holds unmodified: nothing in
  `toxpred/` or `toxagent-control/toxagent/` imports MolScribe. `toxocr/` is a
  sibling package, not a dependency of either.
- A `toxocr` outage degrades the same way a `toxpred` outage does: a typed,
  non-substituted answer (`structure_recognition_failed`), never a guess at
  what the image showed.
- Measured end-to-end on CPU (no GPU in this deployment): ~1-2s per image once
  the model is loaded, well inside the general `run_deadline_s`. A separate,
  more generous `structure_recognition_deadline_s` (`config.py`) exists only
  as a margin against a cold load or a contended host — an initial isolated
  measurement showed ~15 minutes under heavy concurrent host load, which is
  why the margin is large rather than tight.
- D-7 in the frontend redesign plan is reversed: image upload is real, wired
  into the composer and the empty-state hero, verified live.
