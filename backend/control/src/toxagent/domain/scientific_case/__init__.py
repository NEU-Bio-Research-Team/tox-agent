"""ScientificCaseV1: the durable record of one investigation, across turns.

RETHINK §3.1 and §4.4 step 1. ``DecisionSupportStateV1`` records what *one run*
set out to establish; it ends with the run. A researcher's question does not:
they come back with an assay result, the compound's exposure, a second paper.
This is the state that survives — the decision question, the competing
hypotheses and what would refute each, a ledger of evidence for and against, a
ledger of what is still unknown, why the agent did what it did, and the
conclusion it could support so far.

Rules that make it a research record rather than a transcript:

* **Append-only.** A case is the fold of its updates (``replay``). Nothing is
  overwritten; a hypothesis that was supported and later refuted keeps both
  updates in the log, with the run that made each.
* **Evidence is an artifact, never prose.** A ledger entry names a source the
  session really has (an observation, an evidence record, a report, or context
  the user supplied). There is no ``agent_synthesis`` source class here: an
  inference is a hypothesis, not evidence for one (RETHINK §4.3, last
  paragraph).
* **A status needs its evidence.** A hypothesis is ``supported`` or
  ``refuted`` only with a ledger entry of that stance behind it, and a
  conclusion line the case can say must name the entries it rests on.
* **Ref coverage is not quality coverage.** ``coverage`` reports how many
  hypotheses have *any* source separately from how many have direct,
  independent evidence and how many had counter-evidence considered. It is
  never folded into one number.

Ids inside a case are short and server-issued (``h1``, ``e3``, ``u2``); the
model never names one that does not exist. Every transition is a pure function,
so a case can be rebuilt and tested without a database.

Split by role: ``model`` (the case and its entries), ``operations`` (the
closed set of updates and ``apply``/``replay``), ``updates`` (updates derived
from an accepted answer or its citations), ``dossier`` (read models). Every
name is re-exported here.
"""
from __future__ import annotations

from .model import (  # noqa: F401
    SCHEMA_VERSION,
    DOSSIER_SCHEMA_VERSION,
    MAX_HYPOTHESES,
    MAX_EVIDENCE,
    MAX_UNCERTAINTIES,
    MAX_ACTIONS,
    MAX_NEXT_TESTS,
    MAX_CONTEXT,
    MAX_TEXT,
    InvalidCaseUpdate,
    Actor,
    CaseStatus,
    Stance,
    SourceClass,
    REF_KIND,
    INDEPENDENT_SOURCES,
    Directness,
    HypothesisKind,
    HypothesisStatus,
    UncertaintyKind,
    Severity,
    ActionDecision,
    _values,
    Hypothesis,
    EvidenceEntry,
    Uncertainty,
    Action,
    NextTest,
    ContextItem,
    ConclusionLine,
    Conclusion,
    DataScope,
    RunRecord,
    CaseUpdate,
    ScientificCaseV1,
    subject_key,
)
from .operations import (  # noqa: F401
    _text,
    _choice,
    _texts,
    _known_ids,
    _next_local_id,
    parse_ref,
    _scope,
    _open,
    _set_question,
    _set_scope,
    _add_context,
    _add_hypothesis,
    _revise_hypothesis,
    _record_evidence,
    _record_uncertainty,
    _resolve_uncertainty,
    _record_action,
    _propose_next_test,
    _set_conclusion,
    _run_id,
    _attach_run,
    _finish_run,
    _close,
    _OPS,
    MODEL_OPS,
    apply,
    replay,
)
from .updates import (  # noqa: F401
    _STANCE_BY_RELATION,
    _DIRECTNESS_BY_RELATION,
    updates_from_answer,
    updates_from_citations,
)
from .dossier import (  # noqa: F401
    compile_dossier,
    checkpoint_summary,
)
