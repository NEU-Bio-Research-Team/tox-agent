"""Retrieval recall, at record and span level, apart from extraction (#43).

A wrong answer about a value in a paper's table fails in one of several
places, and they call for different fixes: the paper was never found, it was
found and not read, it was read only as an abstract, its text was read and the
row was not quoted, or the row was quoted and the value misread. The last is
extraction, which the grader judges against the answer. Everything before it
is retrieval, and this module measures it from the product trace alone, so a
failure can be placed before or after the source was read.

Gold comes from a v2 case's ``evidence_spans``: each names an ``evidence_ref``
and a ``locator``. An ``evidence_ref`` is a stable identifier of the source --
``doi:``, ``pmid:``, ``pmcid:`` or ``cid:`` as research records carry them, or
``evidence:<id>`` for a frozen-snapshot record. A case-local name that matches
no scheme a run could retrieve is reported as unusable and left out of every
denominator, so a malformed gold reference cannot read as a retrieval failure.

Locators nest: ``table-wrap:1`` covers its rows and ``table-wrap:1/row:3``
covers its cells, so a gold span on a table is met by quoting any of its rows,
and a gold span on a row is not met by quoting the table's caption.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

READ_TOOLS = frozenset({"get_evidence_record", "read_evidence_text", "get_chembl_activities"})
_SCHEMES = ("doi", "pmid", "pmcid", "cid")


def _norm(scheme: str, value: Any) -> str:
    return f"{scheme}:{str(value).strip().lower()}"


def record_keys(evidence: Mapping[str, Any]) -> set[str]:
    """Every key a gold ``evidence_ref`` may use for this record."""
    keys = {_norm("evidence", evidence.get("evidence_id", ""))}
    identifier = evidence.get("identifier") or {}
    for scheme in _SCHEMES:
        if identifier.get(scheme):
            keys.add(_norm(scheme, identifier[scheme]))
    return keys


def _ref_key(ref: str) -> str | None:
    """The record key a gold reference names, or ``None`` when it is not a
    reference a run could ever retrieve (a case-local name)."""
    scheme, _, value = ref.partition(":")
    scheme = scheme.strip().lower()
    if not value.strip() or scheme not in (*_SCHEMES, "evidence"):
        return None
    return _norm(scheme, value)


def within(locator: str, gold: str) -> bool:
    """Whether ``locator`` is ``gold`` or a part of it."""
    return locator == gold or locator.startswith(gold + "/")


def _runs(trace: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    return trace.get("runs") or ()


def retrieval_recall(case: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    """Record- and span-level recall of one run's trace against one case.

    Every rate is reported as ``n``/``of``, never only as a fraction: a case
    with one gold span and a case with ten are not the same evidence.
    """
    gold_spans = [
        (str(span["evidence_ref"]), str(span["locator"]))
        for span in case.get("evidence_spans") or ()
    ]
    if not gold_spans:
        return {"scored": False, "reason": "the case declares no evidence_spans"}

    by_key: dict[str, str] = {}
    full_text: dict[str, str | None] = {}
    for record in trace.get("evidence") or ():
        evidence_id = str(record.get("evidence_id"))
        for key in record_keys(record):
            by_key.setdefault(key, evidence_id)
        full_text[evidence_id] = (record.get("full_text") or {}).get("state")

    read: set[str] = set()
    cited: set[str] = set()
    quoted: set[tuple[str, str]] = set()
    for run_trace in _runs(trace):
        for call in (run_trace.get("run") or {}).get("tool_calls") or ():
            if call.get("tool_name") in READ_TOOLS and call.get("status") == "completed":
                read.update(str(i) for i in call.get("observation_ids") or ())
        for claim in (run_trace.get("answer") or {}).get("claims") or ():
            cited.update(str(i) for i in claim.get("citation_ids") or ())
            for span in claim.get("evidence_spans") or ():
                quoted.add((str(span["evidence_id"]), str(span["locator"])))

    unusable = sorted({ref for ref, _ in gold_spans if _ref_key(ref) is None})
    gold_spans = [(ref, locator) for ref, locator in gold_spans if ref not in unusable]
    refs = sorted({ref for ref, _ in gold_spans})
    matched = {ref: by_key.get(_ref_key(ref)) for ref in refs}
    found = [matched[ref] for ref in refs if matched[ref] is not None]

    def count(predicate) -> dict[str, int]:
        return {"n": sum(1 for item in found if predicate(item)), "of": len(refs)}

    span_rows = []
    for ref, locator in gold_spans:
        evidence_id = matched.get(ref)
        span_rows.append({
            "evidence_ref": ref,
            "locator": locator,
            "retrieved": evidence_id is not None,
            "text_read": evidence_id is not None and full_text.get(evidence_id)
            == "full_text_retrieved",
            "quoted": evidence_id is not None and any(
                eid == evidence_id and within(loc, locator) for eid, loc in quoted
            ),
        })

    def spans(key: str) -> dict[str, int]:
        return {"n": sum(1 for row in span_rows if row[key]), "of": len(span_rows)}

    return {
        "scored": True,
        "records": {
            "retrieved": {"n": len(found), "of": len(refs)},
            "read": count(lambda eid: eid in read),
            "cited": count(lambda eid: eid in cited),
        },
        "spans": {
            "retrieved": spans("retrieved"),
            "text_read": spans("text_read"),
            "quoted": spans("quoted"),
        },
        # Gold references no run could retrieve: left out of every denominator,
        # because they measure the gold, not the run.
        "unusable_refs": unusable,
        "span_rows": span_rows,
    }
