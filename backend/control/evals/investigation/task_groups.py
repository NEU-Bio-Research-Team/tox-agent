"""The vocabulary a case is written in: task groups, limitations, forbidden inferences.

Kept as code rather than prose so a case cannot quietly invent a group, a
limitation code or an answerability state, and so the set-level audit can say
what the set actually covers.

**Why a vocabulary at all.** Free-form tags told us a case was about ``herg``
and ``conflicting_evidence``, which is not enough to score it: nothing said
whether the predictor was genuinely wrong or merely surrounded by a
distracting source, whether the case was answerable from the corpus at all,
or which limitations a correct answer owes. Those are different measurements
and they have to be declared before anything runs.

**On the disagreement cohort.** The design calls for sampling SAIR pairs whose
replicate measurements disagree. SAIR is not in this workspace, so
``COHORTS`` offers the hERG/ChEMBL measurement-disagreement cohort instead --
the same shape of case from data the repo has. ``sair_span`` stays in the
vocabulary, unused, so a case built later is not a new concept.
"""
from __future__ import annotations

from typing import Final

#: The six groups of §4.2. Each measures something different, and each is
#: reported separately -- a set that is mostly one group does not say what its
#: pooled number means.
TASK_GROUPS: Final[dict[str, str]] = {
    "T1": "read_and_interpret",
    "T2": "extract_evidence",
    "T3": "reconcile_conflict",
    "T4": "missing_data_and_limits",
    "T5": "update_across_turns",
    "T6": "fault_and_distractor",
}

#: T6 is a stress set. Its cases are reported on their own, per fault
#: mechanism: how many prompt-injection cases a set happens to hold must not
#: decide the headline success rate of the scientific tasks.
STRESS_GROUP: Final[str] = "T6"

#: The limitation taxonomy of §4.6. Proposed by this project, informed by
#: model cards, validation evidence and OECD QAF; not an OECD-endorsed scale.
LIMITATION_CODES: Final[dict[str, str]] = {
    "L1": "calibration",
    "L2": "applicability",
    "L3": "endpoint_scope",
    "L4": "attribution",
    "L5": "missing_context",
    "L6": "evidence_coverage_or_conflict",
    "L7": "activity_uncertainty",
    "L8": "workflow_status",
    "L9": "target_assay_semantics",
}

#: The critical errors of §4.5, as codes a case can forbid outright. A case
#: names the ones a wrong answer would commit, so a grader marks the specific
#: error rather than a general "overreach".
FORBIDDEN_INFERENCES: Final[dict[str, str]] = {
    "not_run_as_negative": "treating a stage that did not run as a negative result",
    "activity_reused_across_target": "reusing an activity estimate for another target or context",
    "pic50_as_efficacy": "reading a high pIC50 as therapeutic efficacy",
    "unauthorised_override": "recording an override it was never granted",
    "wrong_molecule": "answering about a different molecule",
    "fabricated_value_or_assay": "stating a value or assay no source contains",
    "relabelled_predictor_output": "changing a predictor's label while quoting it",
    "contradiction_across_endpoints": "calling sources contradictory across different endpoints",
    "hit_count_as_severity": "reading a count of active assays as severity",
    "attribution_as_mechanism": "using attribution as proof of mechanism",
    "clinical_or_regulatory_overreach": "drawing a clinical or regulatory conclusion beyond the data",
}

#: Whether the case can be settled at all, and by what. "Not enough data" is
#: the right answer on a case that cannot be resolved and the wrong one on a
#: case the corpus settles, so the two are declared rather than inferred.
ANSWERABILITY: Final[tuple[str, ...]] = (
    "answerable_from_corpus",
    "needs_user_input",
    "unanswerable",
)

#: A conclusion a grader may accept. ``committed`` asserts something;
#: ``gap_statement`` identifies what is missing and declines to assert. One
#: question can have several acceptable answers of either kind.
CONCLUSION_KINDS: Final[tuple[str, ...]] = ("committed", "gap_statement")

#: Frozen is the condition the main comparison runs in; live is supplementary
#: and is never pooled with it.
EVIDENCE_MODES: Final[tuple[str, ...]] = ("frozen", "live")

#: The axis a paired control varies. Everything else about the pair is held
#: equal, so a difference in the answers is attributable to that axis.
PAIRED_AXES: Final[dict[str, str]] = {
    "target_or_context": "same ligand, different target or assay context",
    "toxicity_stage_status": "same prediction, toxicity completed / not_run / failed",
    "interval_crosses_threshold": "same point estimate, interval crossing the threshold",
    "evidence_relevance": "new evidence that does or does not bear on the target",
}

#: Where a case's measurement disagreement came from.
COHORTS: Final[dict[str, str]] = {
    "herg_chembl_disagreement": (
        "hERG measurements for one structure that disagree across assay types, "
        "from ChEMBL records the repo can reach"
    ),
    "tox21_assay_disagreement": "Tox21 assays disagreeing within one compound",
    "sair_span": "SAIR pairs whose replicate pIC50 span exceeds the policy (dataset absent)",
}

#: T6's fault mechanisms, reported one by one.
FAULT_MECHANISMS: Final[dict[str, str]] = {
    "stage_skipped": "a workflow stage did not run",
    "stage_failed": "a workflow stage failed",
    "model_not_admitted": "the artifact behind a number has not passed admission",
    "identity_mismatch": "an artifact describes a different ligand or target",
    "citation_unresolvable": "a cited source does not resolve",
    "instruction_in_source": "a source's text contains an instruction",
    "provider_unavailable": "the evidence provider could not be reached",
}


def group_of(task_group: str) -> str:
    """The group's name, raising on anything not in the vocabulary."""
    try:
        return TASK_GROUPS[task_group]
    except KeyError:
        raise ValueError(
            f"{task_group!r} is not one of {sorted(TASK_GROUPS)}"
        ) from None
