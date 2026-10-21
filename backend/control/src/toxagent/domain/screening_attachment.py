"""A screening attached to a session (#44).

The session holds its own copy of the frozen record it was given, so what a
run read stays readable after the deployment's replay file is replaced, and
two sessions given the same screening each own one attachment. A screening is
one ligand on one target in one assay context: the same ligand on another
target is another screening, attached separately, never a reuse of this one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .ids import SESSION, require_id

SOURCE_REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class ScreeningAttachment:
    session_id: str
    screening_id: str
    record: Mapping[str, Any]
    record_sha256: str
    attached_at: datetime
    source: str = SOURCE_REPLAY

    def __post_init__(self) -> None:
        require_id(self.session_id, SESSION, field="screening_attachment.session_id")
        if not self.screening_id.startswith("scr_"):
            raise ValueError(f"{self.screening_id!r} is not a screening id")
        if self.record_sha256 != self.record.get("content_sha256"):
            raise ValueError("an attachment's hash must be its record's own content_sha256")
