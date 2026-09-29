"""The product-owned gateway for one agentic run (plan section 10).

The gateway deliberately does *not* implement an agent loop.  A runtime owns
reasoning and its own local transcript; ToxAgent owns the session, the tool
authorization, observations, accepted answer and every state transition that a
client can observe.  This module is the narrow seam between those two worlds.

Only a validated ``GroundedAnswer`` becomes an assistant message.  Runtime text
deltas are not product truth: persisting them before ``submit_grounded_answer``
would allow an ungrounded number to survive in the transcript even when the
validator correctly refused the final candidate.

``AgentRuntimeGateway`` is one class assembled from mixins, one per concern:
``case`` (scientific case, decision state, claim review), ``context`` (the
prompt context and the AI profile), ``reports`` (report builds and their
completion), ``completion`` (event consumption, usage, committing a turn),
``health`` (runtime health and identity). ``runtime_gateway`` holds the entry
points and the dispatch.
"""
from __future__ import annotations

from ._common import _now, _no_commit, ResolvedProfile, log, _DIAGNOSTIC_DELTA_PREVIEW_CHARS  # noqa: F401
from .runtime_gateway import AgentRuntimeGateway  # noqa: F401
