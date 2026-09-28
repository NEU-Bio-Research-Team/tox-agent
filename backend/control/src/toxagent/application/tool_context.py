"""Who is calling a tool: the context the server injects into every handler.

It lives beside the application services rather than in ``tools/`` because the
application builds it too — a server-initiated report stage runs tools under a
context it constructs itself. ``tools.registry`` re-exports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .policy import Actor


@dataclass(frozen=True)
class ToolContext:
    """Everything a handler is allowed to know about who is calling.

    ``session_id`` and ``run_id`` come from the capability token, never from the
    model's arguments — a tool argument that disagrees with the token loses
    (plan section 8.5).
    """

    session_id: str
    run_id: str
    actor: Actor
    profile: str
    deadline_at: datetime
    language: str = "en"
    #: Immutable run intent carried by the signed capability token. It is
    #: presentation context only; tool authorization remains ``profile``.
    intent: str = ""
    call_id: str = ""
    #: The run's resolved predictor binding: endpoint -> admitted model id.
    #: Injected by the server from the run configuration, exactly like
    #: session_id and run_id, and for the same reason — a tool argument that
    #: let a model choose its own provider would put scientific model
    #: selection in the hands of the thing being explained (I10). No tool
    #: input schema exposes a model field; this is the only way one arrives.
    model_selection: Mapping[str, str] | None = None
