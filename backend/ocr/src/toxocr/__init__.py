"""ToxOCR: optical chemical structure recognition (image -> SMILES).

A third external boundary alongside the predictor and the runtime (see
toxagent-control's ADR 0001, three-boundary-topology): this is a separate
deployable that toxagent-control only ever talks to over HTTP. Nothing in
toxagent-control or toxpred imports this package or the vision model it wraps.

The recognition model itself (MolScribe, github.com/thomas0809/MolScribe) is
the same one this codebase's pre-refactor agent-layer used successfully —
see docs/refactor/PREDICTOR_ONLY_STATUS_VI.md for why it was pulled out of
that layer's own dependency footprint rather than dropped as unworkable.
"""
from __future__ import annotations

__version__ = "0.1.0"
