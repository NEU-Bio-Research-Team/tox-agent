"""Reading an evidence record's full text, once per session (#43).

The read is a fact about the document, so it is stored and reused: a second
call for the same record, in this run or a later one, returns what was read
without asking the provider again. Only a definitive answer is stored -- the
text, or the provider's statement that it has none. An outage is reported to
the caller and forgotten, because it says nothing about the document.

Which passages a caller sees is a separate, pure step (:func:`select`): the
whole text is kept, and a model is handed a bounded page of it, chosen by
locator, kind or section, so a long paper's tables are reachable without its
prose crowding them out.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

from ...domain.evidence import EvidenceRecord, SourceType
from ...domain.evidence_text import (
    ACCESS_FAILED,
    EvidenceFullText,
    EvidencePassage,
)
from ...research.interfaces import FullTextCapableProvider
from ...research.text_access import TextAccess, access_of

#: Kinds a caller can ask for. ``table`` is a table's label, caption and
#: footnotes; its rows are ``table_header``/``table_row``.
PASSAGE_KINDS: tuple[str, ...] = (
    "abstract", "paragraph", "table", "table_header", "table_row",
)


@dataclass(frozen=True)
class TextRead:
    """The outcome of asking for a record's text, as a caller reports it."""

    #: A :class:`~toxagent.research.text_access.TextAccess` value.
    state: str
    full_text: EvidenceFullText | None = None
    reason: str | None = None
    #: True when this call asked the provider; False when the text, or its
    #: refusal, was already stored, or nothing could be asked.
    fetched_now: bool = False
    full_text_id: str | None = None


def is_article(record: EvidenceRecord) -> bool:
    """A record whose text is a document that can be read in full. A ChEMBL
    activity or a regulatory database entry is a structured record: it has no
    body or tables beyond what was stored."""
    return record.source_type is SourceType.ARTICLE


def _passage(item) -> EvidencePassage:
    return EvidencePassage(
        locator=item.locator, section=item.section, kind=item.kind, text=item.text,
        truncated=bool(getattr(item, "truncated", False)),
        cells=tuple(getattr(item, "cells", ()) or ()),
    )


async def read_full_text(
    database,
    provider,
    *,
    session_id: str,
    run_id: str,
    record: EvidenceRecord,
    now: datetime,
) -> TextRead:
    async with database.unit_of_work() as uow:
        stored = await uow.evidence_text.get(record.id, session_id=session_id)
    if stored is not None:
        return TextRead(
            state=stored.state, full_text=stored, reason=stored.reason,
            full_text_id=stored.full_text_id,
        )

    can_fetch = (
        isinstance(provider, FullTextCapableProvider)
        and getattr(provider, "name", None) == record.provider
        and is_article(record)
    )
    if not can_fetch:
        return TextRead(
            state=access_of(record.normalized_facts).value,
            reason=(
                f"this deployment cannot fetch the full text of a {record.provider} "
                f"{record.source_type.value} record; only what the search stored is held"
            ),
        )

    fetch = await provider.fetch_full_text(
        provider_record_id=record.provider_record_id, identifier=record.identifier,
    )
    if fetch.transient:
        return TextRead(
            state=TextAccess.ACCESS_FAILED.value, reason=fetch.reason, fetched_now=True,
            full_text_id=fetch.full_text_id,
        )
    if fetch.retrieved:
        candidate = EvidenceFullText.retrieved(
            evidence_id=record.id, session_id=session_id, run_id=run_id,
            passages=tuple(_passage(p) for p in fetch.passages),
            section_titles=tuple(fetch.section_titles), omitted=fetch.omitted,
            truncated=fetch.truncated, full_text_id=fetch.full_text_id, now=now,
        )
    else:
        candidate = EvidenceFullText.failed(
            evidence_id=record.id, session_id=session_id, run_id=run_id,
            reason=fetch.reason or "the provider returned no text", full_text_id=fetch.full_text_id,
            now=now,
        )
    async with database.unit_of_work() as uow:
        stored = await uow.evidence_text.add_if_absent(candidate)
        await uow.commit()
    return TextRead(
        state=stored.state, full_text=stored, reason=stored.reason, fetched_now=True,
        full_text_id=stored.full_text_id,
    )


@dataclass(frozen=True)
class Selection:
    passages: tuple[EvidencePassage, ...]
    total: int
    offset: int
    next_offset: int | None


def select(
    full_text: EvidenceFullText,
    *,
    locators: Sequence[str] | None = None,
    kinds: Iterable[str] | None = None,
    section: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> Selection:
    """A page of passages. Named ``locators`` come back in the order asked
    (a cell locator resolves to that one cell); otherwise passages are
    filtered by kind and by a case-insensitive section substring, in document
    order."""
    if locators:
        found = [full_text.passage(locator) for locator in locators]
        chosen = [p for p in found if p is not None]
    else:
        wanted_kinds = set(kinds or ())
        needle = (section or "").strip().casefold()
        chosen = [
            p for p in full_text.passages
            if (not wanted_kinds or p.kind in wanted_kinds)
            and (not needle or needle in p.section.casefold())
        ]
    start = max(0, offset)
    page = chosen[start:start + limit]
    end = start + len(page)
    return Selection(
        passages=tuple(page), total=len(chosen), offset=start,
        next_offset=end if end < len(chosen) else None,
    )


def text_not_read(record: EvidenceRecord, full_text: EvidenceFullText | None) -> bool:
    """Whether an article was cited without its text having been read --
    abstract only, refused, or never asked for."""
    if not is_article(record):
        return False
    return full_text is None or full_text.state == ACCESS_FAILED
