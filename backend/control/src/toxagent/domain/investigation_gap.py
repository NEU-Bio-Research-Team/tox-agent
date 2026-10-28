"""The gaps an investigation can be stuck on (#70, §6.2).

§6.2 lists seven kinds of unresolved requirement, and the point of typing them
is that they are not interchangeable: four are things this product can act on,
one needs the researcher, one needs a provider that is working, and one is
about a question whose subject is not yet pinned down. A run that reports
"insufficient" without saying which of the seven it hit has told the
researcher nothing they can use.

Two of these have live users already: ``PROVIDER_OR_CORPUS`` is what an access
failure records (``decision_state.record_access_failure``), and
``ONLY_RESEARCHER_HAS_IT`` is what the ``needs_user_input`` stop means. The
rest are declared here so the loop that selects among gaps (#70) and the
action ranking that must say which gap it discriminates (#71) share one
vocabulary instead of inventing two.
"""
from __future__ import annotations

from enum import Enum


class Gap(str, Enum):
    """What is missing. One of these, never free text."""

    #: Which ligand, target, variant or assay context the question is about
    #: is not established, so nothing can be compared yet.
    IDENTITY_OR_SCOPE = "identity_or_scope"
    #: No source in reach bears on the requirement.
    NO_RELEVANT_EVIDENCE = "no_relevant_evidence"
    #: A record was found and the span that would settle it has not been read.
    SPAN_NOT_READ = "span_not_read"
    #: The number is there and its unit, comparator or extraction is unsettled.
    MEASUREMENT_UNRESOLVED = "measurement_unresolved"
    #: Two sources disagree and the context that would explain it is unchecked.
    DISAGREEMENT_UNEXPLAINED = "disagreement_unexplained"
    #: An input only the researcher can supply, e.g. an exposure.
    ONLY_RESEARCHER_HAS_IT = "only_researcher_has_it"
    #: The provider or corpus could not supply the data: an access failure, a
    #: breaker open, a corpus that does not carry the record. Deliberately
    #: separate from NO_RELEVANT_EVIDENCE -- §6.3: "Provider failure và thiếu
    #: bằng chứng khoa học là hai kết quả khác nhau."
    PROVIDER_OR_CORPUS = "provider_or_corpus"


#: Gaps no further work by this run can close: it has to say so and stop.
#: Reporting one of these as "no evidence found" is the error §6.3 names.
BLOCKING_GAPS: frozenset[Gap] = frozenset({Gap.ONLY_RESEARCHER_HAS_IT, Gap.PROVIDER_OR_CORPUS})


def is_blocking(gap: Gap | str) -> bool:
    try:
        return Gap(gap) in BLOCKING_GAPS
    except ValueError:
        return False
