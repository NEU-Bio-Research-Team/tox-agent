"""Whether a source's text was actually read, and why not.

"We found no evidence", "we only ever saw the abstract" and "the full text
exists and we could not open it" are three different facts. Collapsing them
into one produces the failure the review named: a search that came back with
abstracts gets reported as an absence of evidence, and raising a timeout does
nothing about it because the provider was never asked for the text.

So the state is recorded explicitly, derived from what the provider declared
rather than assumed:

``not_assessed``
    Nothing is known about the text. This is the honest default -- an absent
    signal is not a denial.
``abstract_only``
    An abstract is held, and no full text is declared available.
``full_text_available``
    The provider says the full text can be fetched. Nothing has fetched it.
``full_text_retrieved``
    The text was fetched and its passages are held.
``access_failed``
    A fetch was attempted and did not produce text. The reason is kept.

Open access is a licence, not a location: ``isOpenAccess`` on its own does not
mean this provider can serve the text, so availability keys on whether the
record is in Europe PMC's own full-text store.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

#: Keys this module owns inside an evidence record's ``normalized_facts``.
ACCESS_KEY = "text_access"
SIGNALS_KEY = "full_text_signals"
FAILURE_KEY = "access_failure_reason"
URL_KEY = "full_text_url"
PASSAGE_COUNT_KEY = "full_text_passage_count"


class TextAccess(str, Enum):
    NOT_ASSESSED = "not_assessed"
    ABSTRACT_ONLY = "abstract_only"
    FULL_TEXT_AVAILABLE = "full_text_available"
    FULL_TEXT_RETRIEVED = "full_text_retrieved"
    ACCESS_FAILED = "access_failed"


#: Europe PMC answers these as "Y"/"N". Anything else, including absence, is
#: unknown -- which is not the same as "N".
_TRISTATE_SIGNALS = ("inEPMC", "inPMC", "isOpenAccess", "hasPDF")
#: The signals that mean *this* provider can serve the text.
_AVAILABILITY_SIGNALS = ("inEPMC", "inPMC")


def _tristate(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().upper()
    if text == "Y":
        return True
    if text == "N":
        return False
    return None


def signals_from_record(record: Mapping[str, Any]) -> dict[str, bool | None]:
    return {name: _tristate(record.get(name)) for name in _TRISTATE_SIGNALS}


def _full_text_url(record: Mapping[str, Any]) -> str | None:
    """The provider's own declared full-text link, when it gives one."""
    urls = (record.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    for entry in urls:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("availability") or "").strip().lower() in {"open access", "free"}:
            url = entry.get("url")
            if url:
                return str(url)
    for entry in urls:
        if isinstance(entry, Mapping) and entry.get("url"):
            return str(entry["url"])
    return None


def access_facts(
    record: Mapping[str, Any], *, has_abstract: bool
) -> dict[str, Any]:
    """The access state a search result starts in, and the signals behind it."""
    signals = signals_from_record(record)
    available = any(signals.get(name) is True for name in _AVAILABILITY_SIGNALS)
    if available:
        state = TextAccess.FULL_TEXT_AVAILABLE
    elif has_abstract:
        state = TextAccess.ABSTRACT_ONLY
    else:
        # Metadata only. Not an absence of evidence, and not an abstract.
        state = TextAccess.NOT_ASSESSED
    return {
        ACCESS_KEY: state.value,
        SIGNALS_KEY: signals,
        URL_KEY: _full_text_url(record),
    }


def after_successful_fetch(facts: Mapping[str, Any], *, passage_count: int) -> dict[str, Any]:
    updated = dict(facts)
    updated[ACCESS_KEY] = TextAccess.FULL_TEXT_RETRIEVED.value
    updated[PASSAGE_COUNT_KEY] = passage_count
    updated.pop(FAILURE_KEY, None)
    return updated


def after_failed_fetch(facts: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    """A fetch that produced no text.

    The previous state is not kept: a record whose text was declared available
    and could not be opened is in a worse position than one that never tried,
    and a reader has to be able to tell.
    """
    updated = dict(facts)
    updated[ACCESS_KEY] = TextAccess.ACCESS_FAILED.value
    updated[FAILURE_KEY] = reason
    return updated


def access_of(facts: Mapping[str, Any]) -> TextAccess:
    """The state a stored record is in, defaulting to ``not_assessed``.

    Records written before this module existed carry no key, and that is
    exactly ``not_assessed``: nothing was ever established about their text.
    """
    try:
        return TextAccess(str(facts.get(ACCESS_KEY) or ""))
    except ValueError:
        return TextAccess.NOT_ASSESSED


def full_text_was_read(facts: Mapping[str, Any]) -> bool:
    return access_of(facts) is TextAccess.FULL_TEXT_RETRIEVED
