"""Full text split into passages that each know where they came from.

An excerpt without a locator cannot be checked. The reviewer asked for the
numbers in a results section or an assay table, and "it says so in the paper"
is not a citable claim -- so every passage carries the section it sits in and
an index within that section, and a claim can name one.

Europe PMC serves JATS XML. Parsing is deliberately shallow: section titles,
their paragraphs, and table captions with their cells. Nothing here tries to
understand the science; it produces located text for a reader to quote.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator

#: A passage longer than this is truncated. Values and sentences survive; a
#: whole methods section pasted into a prompt does not.
MAX_PASSAGE_CHARS = 2_000
#: Enough for a results section and an assay table or two.
MAX_PASSAGES = 120

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Passage:
    """One quotable piece of a document, and where it is."""

    #: Stable within one parse of one document, e.g. ``sec:3/p:2`` or
    #: ``table-wrap:2``. A claim cites this, not an offset into a blob.
    locator: str
    #: The section heading this sits under, or ``"abstract"``/``"table"``.
    section: str
    kind: str
    text: str
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class FullText:
    passages: tuple[Passage, ...]
    section_titles: tuple[str, ...]
    #: True when the document had more passages than the cap allowed.
    truncated: bool = False

    def passage(self, locator: str) -> Passage | None:
        return next((p for p in self.passages if p.locator == locator), None)


def _local(tag: str) -> str:
    """JATS from Europe PMC is usually un-namespaced, but not always."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text_of(element: ET.Element) -> str:
    """All descendant text, flattened. Inline markup carries no meaning here."""
    return _WHITESPACE.sub(" ", "".join(element.itertext())).strip()


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_PASSAGE_CHARS:
        return text, False
    return text[:MAX_PASSAGE_CHARS].rstrip() + " …", True


def _children(element: ET.Element, name: str) -> Iterator[ET.Element]:
    for child in element:
        if _local(child.tag) == name:
            yield child


def _find_first(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local(element.tag) == name:
            return element
    return None


def _section_title(section: ET.Element, *, index: int) -> str:
    for child in _children(section, "title"):
        title = _text_of(child)
        if title:
            return title
    return f"section {index}"


def parse_jats(xml_text: str) -> FullText:
    """Passages from a JATS document. Malformed XML yields no passages."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return FullText(passages=(), section_titles=())

    passages: list[Passage] = []
    titles: list[str] = []
    truncated_document = False

    abstract = _find_first(root, "abstract")
    if abstract is not None:
        text = _text_of(abstract)
        if text:
            clipped, clipped_flag = _clip(text)
            passages.append(
                Passage(
                    locator="abstract",
                    section="abstract",
                    kind="abstract",
                    text=clipped,
                    truncated=clipped_flag,
                )
            )

    body = _find_first(root, "body")
    if body is not None:
        sections = [e for e in body.iter() if _local(e.tag) == "sec"]
        for section_index, section in enumerate(sections, start=1):
            title = _section_title(section, index=section_index)
            titles.append(title)
            for paragraph_index, paragraph in enumerate(
                _children(section, "p"), start=1
            ):
                text = _text_of(paragraph)
                if not text:
                    continue
                if len(passages) >= MAX_PASSAGES:
                    truncated_document = True
                    break
                clipped, clipped_flag = _clip(text)
                passages.append(
                    Passage(
                        locator=f"sec:{section_index}/p:{paragraph_index}",
                        section=title,
                        kind="paragraph",
                        text=clipped,
                        truncated=clipped_flag,
                    )
                )
            if truncated_document:
                break

        for table_index, table in enumerate(
            (e for e in body.iter() if _local(e.tag) == "table-wrap"), start=1
        ):
            if len(passages) >= MAX_PASSAGES:
                truncated_document = True
                break
            text = _text_of(table)
            if not text:
                continue
            clipped, clipped_flag = _clip(text)
            caption = _find_first(table, "caption")
            passages.append(
                Passage(
                    locator=f"table-wrap:{table_index}",
                    section=_text_of(caption) if caption is not None else "table",
                    kind="table",
                    text=clipped,
                    truncated=clipped_flag,
                )
            )

    return FullText(
        passages=tuple(passages),
        section_titles=tuple(titles),
        truncated=truncated_document,
    )
