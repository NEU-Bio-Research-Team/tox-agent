"""What was read of an evidence record's full text, and where each piece sits.

An :class:`~toxagent.domain.evidence.EvidenceRecord` is what a search returned:
a title, an abstract, provider metadata. Its full text is a separate fact with
a separate lifecycle -- fetched later, possibly never, possibly refused -- so
it is a separate type rather than more fields on a record whose content hash
was fixed when it was stored.

Two outcomes are kept, and only definitive ones: the text was read
(``full_text_retrieved``, with its passages), or the provider answered that it
cannot be (``access_failed``, with the reason). A provider outage is neither --
it says nothing about the document -- so it is never stored as one, and a later
call asks again.

Passages are external content. The text is data for a model to read and quote,
never instructions (domain/evidence.py), and a claim cites a passage by its
locator, so a quoted number can be checked against the text it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .ids import EVIDENCE, RUN, SESSION, require_id
from .provenance import content_sha256

FULL_TEXT_RETRIEVED = "full_text_retrieved"
ACCESS_FAILED = "access_failed"
STATES = frozenset({FULL_TEXT_RETRIEVED, ACCESS_FAILED})


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    """One located piece of a document (research/fulltext.py's ``Passage``,
    in the domain's own terms)."""

    locator: str
    section: str
    kind: str
    text: str
    truncated: bool = False
    #: For a table row: its cells as written.
    cells: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "section": self.section,
            "kind": self.kind,
            "text": self.text,
            "truncated": self.truncated,
            "cells": list(self.cells),
        }


@dataclass(frozen=True, slots=True)
class EvidenceFullText:
    evidence_id: str
    session_id: str
    run_id: str
    state: str
    fetched_at: datetime
    #: The identifier the text was asked for under (a PMCID for Europe PMC).
    full_text_id: str | None = None
    #: Set when ``state`` is ``access_failed``.
    reason: str | None = None
    passages: tuple[EvidencePassage, ...] = ()
    section_titles: tuple[str, ...] = ()
    #: What the parser's caps left out, by passage kind.
    omitted: Mapping[str, int] = field(default_factory=dict)
    truncated: bool = False
    document_sha256: str = ""

    def __post_init__(self) -> None:
        require_id(self.evidence_id, EVIDENCE, field="evidence_full_text.evidence_id")
        require_id(self.session_id, SESSION, field="evidence_full_text.session_id")
        require_id(self.run_id, RUN, field="evidence_full_text.run_id")
        if self.state not in STATES:
            raise ValueError(f"evidence_full_text.state {self.state!r} is not one of {sorted(STATES)}")
        if self.state == ACCESS_FAILED and not self.reason:
            raise ValueError("a failed text access must record why")
        if self.state == FULL_TEXT_RETRIEVED and not self.passages:
            raise ValueError("a retrieved text must hold at least one passage")

    @classmethod
    def retrieved(
        cls,
        *,
        evidence_id: str,
        session_id: str,
        run_id: str,
        passages: tuple[EvidencePassage, ...],
        section_titles: tuple[str, ...],
        omitted: Mapping[str, int],
        truncated: bool,
        full_text_id: str | None,
        now: datetime,
    ) -> "EvidenceFullText":
        return cls(
            evidence_id=evidence_id, session_id=session_id, run_id=run_id,
            state=FULL_TEXT_RETRIEVED, fetched_at=now, full_text_id=full_text_id,
            passages=passages, section_titles=section_titles, omitted=dict(omitted),
            truncated=truncated,
            document_sha256=content_sha256([p.to_dict() for p in passages]),
        )

    @classmethod
    def failed(
        cls,
        *,
        evidence_id: str,
        session_id: str,
        run_id: str,
        reason: str,
        full_text_id: str | None,
        now: datetime,
    ) -> "EvidenceFullText":
        return cls(
            evidence_id=evidence_id, session_id=session_id, run_id=run_id,
            state=ACCESS_FAILED, fetched_at=now, full_text_id=full_text_id, reason=reason,
        )

    @property
    def was_read(self) -> bool:
        return self.state == FULL_TEXT_RETRIEVED

    def passage(self, locator: str) -> EvidencePassage | None:
        """The passage at ``locator``, including one cell of a table row
        (``table-wrap:1/row:3/cell:2``, cells counted from 1)."""
        found = next((p for p in self.passages if p.locator == locator), None)
        if found is not None:
            return found
        row_locator, marker, index_text = locator.rpartition("/cell:")
        if not marker or not index_text.isdigit():
            return None
        row = next((p for p in self.passages if p.locator == row_locator), None)
        index = int(index_text)
        if row is None or not 1 <= index <= len(row.cells):
            return None
        return EvidencePassage(
            locator=locator, section=row.section, kind="table_cell", text=row.cells[index - 1],
        )
