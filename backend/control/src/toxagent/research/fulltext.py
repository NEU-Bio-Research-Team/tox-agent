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
from dataclasses import dataclass, field
from typing import Iterator, Mapping

#: Bumped when how a document is split into passages changes -- a locator
#: frozen under one version is only guaranteed to name the same passage
#: under the same version (#43).
PARSER_VERSION = "jats-passages-v1"

#: A passage longer than this is truncated. Values and sentences survive; a
#: whole methods section pasted into a prompt does not.
MAX_PASSAGE_CHARS = 2_000
#: Prose and tables are budgeted apart. With one shared cap, a long paper's
#: paragraphs -- which come first in document order -- used up the whole
#: budget, and the assay table a question needed was never read.
MAX_PARAGRAPHS = 120
MAX_TABLES = 20
#: Per table, so one long supplementary table cannot crowd out the others.
MAX_TABLE_ROWS = 60

_WHITESPACE = re.compile(r"\s+")
#: Not part of the prose they sit in: a table or figure nested in a paragraph
#: is read as its own thing, and alt-text repeats what the caption says.
_NOT_PROSE = frozenset({"table-wrap", "fig", "alt-text"})
#: Elements that end one run of text and start another. Their text is
#: separated by a space; inline markup between them is not.
_BLOCKS = frozenset({
    "p", "title", "label", "caption", "fn", "list-item", "def-item", "term", "def",
    "td", "th", "tr", "sec", "disp-quote",
})
_CELL_LOCATOR = re.compile(r"^(?P<row>table-wrap:\d+/row:\d+)/cell:(?P<cell>\d+)$")


@dataclass(frozen=True, slots=True)
class Passage:
    """One quotable piece of a document, and where it is."""

    #: Stable within one parse of one document, e.g. ``sec:3/p:2``,
    #: ``table-wrap:2`` (the table's label, caption and footnotes) or
    #: ``table-wrap:2/row:4``. A claim cites this, not an offset into a blob.
    locator: str
    #: The section heading this sits under, ``"abstract"``, or a table's label.
    section: str
    kind: str
    text: str
    truncated: bool = False
    #: For a table row: its cells as written, so ``.../cell:N`` names one.
    cells: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FullText:
    passages: tuple[Passage, ...]
    section_titles: tuple[str, ...]
    #: True when the document had more than the caps allowed.
    truncated: bool = False
    #: What the caps left out, by passage kind, so a reader knows what was
    #: not read rather than only that something was not.
    omitted: Mapping[str, int] = field(default_factory=dict)

    def passage(self, locator: str) -> Passage | None:
        """The passage at ``locator``, including one cell of a table row
        (``table-wrap:1/row:3/cell:2``, counting cells from 1)."""
        found = next((p for p in self.passages if p.locator == locator), None)
        if found is not None:
            return found
        match = _CELL_LOCATOR.match(locator)
        if match is None:
            return None
        row = next((p for p in self.passages if p.locator == match["row"]), None)
        index = int(match["cell"])
        if row is None or not 1 <= index <= len(row.cells):
            return None
        return Passage(
            locator=locator, section=row.section, kind="table_cell",
            text=row.cells[index - 1],
        )


def _local(tag: str) -> str:
    """JATS from Europe PMC is usually un-namespaced, but not always."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text_of(element: ET.Element, *, skip: frozenset[str] = frozenset()) -> str:
    """Descendant text, flattened, leaving out the elements named in ``skip``.

    Inline markup (``IC<sub>50</sub>``) carries no meaning here, so it joins
    without a space; block elements (:data:`_BLOCKS`) do not, so a footnote's
    label and its text, or two cells, do not run together into one token.
    """
    def pieces(node: ET.Element) -> Iterator[str]:
        if node.text:
            yield node.text
        for child in node:
            name = _local(child.tag)
            if name not in skip:
                block = name in _BLOCKS
                if block:
                    yield " "
                yield from pieces(child)
                if block:
                    yield " "
            if child.tail:
                yield child.tail

    return _WHITESPACE.sub(" ", "".join(pieces(element))).strip()


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


def _rows(table: ET.Element) -> list[tuple[bool, list[ET.Element]]]:
    """Every row of a table in document order, each flagged as a header row
    when it sits in ``thead`` or holds only ``th`` cells."""
    in_head = {
        id(row)
        for head in table.iter() if _local(head.tag) == "thead"
        for row in head.iter() if _local(row.tag) == "tr"
    }
    rows: list[tuple[bool, list[ET.Element]]] = []
    for element in table.iter():
        if _local(element.tag) != "tr":
            continue
        cells = [c for c in element if _local(c.tag) in ("td", "th")]
        if cells:
            is_header = id(element) in in_head or all(_local(c.tag) == "th" for c in cells)
            rows.append((is_header, cells))
    return rows


def _header_columns(cells: list[ET.Element]) -> list[str]:
    """A header row with each spanned heading repeated per column it covers."""
    columns: list[str] = []
    for cell in cells:
        try:
            span = max(1, int(cell.get("colspan") or 1))
        except ValueError:
            span = 1
        columns.extend([_text_of(cell)] * span)
    return columns


def _row_text(cells: tuple[str, ...], header: list[str] | None) -> str:
    """``Compound: 5 | hERG IC50 (µM): 0.9`` when the header lines up with
    the row, so a quoted row keeps what each number is; plain ``|``-separated
    cells when it does not (a spanned or ragged row)."""
    if header is not None and len(header) == len(cells):
        return " | ".join(
            f"{name}: {value}" if name else value
            for name, value in zip(header, cells) if value
        )
    return " | ".join(cells)


def _table_passages(
    table: ET.Element, *, index: int
) -> tuple[list[Passage], int]:
    """A table's caption passage and one passage per row, and how many rows
    the cap left out."""
    locator = f"table-wrap:{index}"
    caption_element = next(_children(table, "caption"), None)
    caption = _text_of(caption_element) if caption_element is not None else ""
    caption_title = next(
        (_text_of(t) for t in _children(caption_element, "title")), ""
    ) if caption_element is not None else ""
    # What the table is called, for every passage cut from it: its label
    # ("Table 2"), else its caption's title, else the start of its caption.
    label = (
        next((_text_of(c) for c in _children(table, "label")), "")
        or caption_title
        or caption[:120]
        or f"table {index}"
    )
    footnotes = " ".join(
        _text_of(foot) for foot in table.iter() if _local(foot.tag) == "table-wrap-foot"
    )
    passages: list[Passage] = []
    context = " ".join(
        part for part in (label if label != caption[:120] else "", caption, footnotes) if part
    )
    clipped, clipped_flag = _clip(context)
    passages.append(
        Passage(locator=locator, section=label, kind="table", text=clipped,
                truncated=clipped_flag)
    )

    header: list[str] | None = None
    rows = _rows(table)
    for row_index, (is_header, elements) in enumerate(rows, start=1):
        if row_index > MAX_TABLE_ROWS:
            return passages, len(rows) - MAX_TABLE_ROWS
        cells = tuple(_text_of(cell) for cell in elements)
        if is_header:
            header = _header_columns(elements)
            text = " | ".join(cells)
        else:
            text = _row_text(cells, header)
        if not text:
            continue
        clipped, clipped_flag = _clip(text)
        passages.append(
            Passage(
                locator=f"{locator}/row:{row_index}",
                section=label,
                kind="table_header" if is_header else "table_row",
                text=clipped,
                truncated=clipped_flag,
                cells=cells,
            )
        )
    return passages, 0


def parse_jats(xml_text: str) -> FullText:
    """Passages from a JATS document. Malformed XML yields no passages."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return FullText(passages=(), section_titles=())

    passages: list[Passage] = []
    titles: list[str] = []
    omitted: dict[str, int] = {}

    abstract = _find_first(root, "abstract")
    if abstract is not None:
        text = _text_of(abstract, skip=_NOT_PROSE)
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

    paragraphs = 0

    def add_paragraph(element: ET.Element, *, locator: str, section: str) -> None:
        nonlocal paragraphs
        text = _text_of(element, skip=_NOT_PROSE)
        if not text:
            return
        if paragraphs >= MAX_PARAGRAPHS:
            omitted["paragraph"] = omitted.get("paragraph", 0) + 1
            return
        paragraphs += 1
        clipped, clipped_flag = _clip(text)
        passages.append(
            Passage(locator=locator, section=section, kind="paragraph",
                    text=clipped, truncated=clipped_flag)
        )

    body = _find_first(root, "body")
    if body is not None:
        # Prose straight under <body>, before or without any <sec>.
        for paragraph_index, paragraph in enumerate(_children(body, "p"), start=1):
            add_paragraph(paragraph, locator=f"body/p:{paragraph_index}", section="body")
        sections = [e for e in body.iter() if _local(e.tag) == "sec"]
        for section_index, section in enumerate(sections, start=1):
            title = _section_title(section, index=section_index)
            titles.append(title)
            for paragraph_index, paragraph in enumerate(
                _children(section, "p"), start=1
            ):
                add_paragraph(
                    paragraph, locator=f"sec:{section_index}/p:{paragraph_index}",
                    section=title,
                )

    # Tables wherever the article keeps them: in the body, or gathered into a
    # <floats-group> after it, which some publishers do.
    tables = [e for e in root.iter() if _local(e.tag) == "table-wrap"]
    for table_index, table in enumerate(tables, start=1):
        if table_index > MAX_TABLES:
            omitted["table"] = len(tables) - MAX_TABLES
            break
        table_passages, rows_left_out = _table_passages(table, index=table_index)
        passages.extend(table_passages)
        if rows_left_out:
            omitted["table_row"] = omitted.get("table_row", 0) + rows_left_out

    return FullText(
        passages=tuple(passages),
        section_titles=tuple(titles),
        truncated=bool(omitted),
        omitted=omitted,
    )
