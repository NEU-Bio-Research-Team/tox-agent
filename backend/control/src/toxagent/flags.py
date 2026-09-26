"""Rollout flags, with an owner and an expiry date each.

A flag here is a *rollout* control: it selects between an old code path and a
new one while a change is being canaried, and it is deleted once the new path
is the only path. It is never a product setting, never a permission, and never
a way to keep two behaviours alive indefinitely — that is how compatibility
becomes the architecture (ADR 0009).

Two rules are enforced by ``tests/unit/test_rollout_flags.py`` rather than by
convention:

* every flag declares an ``owner`` and a ``remove_by`` date;
* ``remove_by`` is at most ``MAX_FLAG_LIFETIME_DAYS`` past ``added_on``, which
  is the two-release window the remediation plan allows.

The default *value* of a flag is the safe one — the behaviour that shipped
before the flag existed — so a deployment that sets nothing keeps running what
it ran yesterday.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Mapping

from .config import _bool  # single reader of the environment (config.py's rule)

#: Two releases at the project's cadence. Longer than this and the flag is not
#: a rollout control, it is a fork.
MAX_FLAG_LIFETIME_DAYS = 90


@dataclass(frozen=True, slots=True)
class RolloutFlag:
    name: str
    env_var: str
    #: What turning it *on* does. Written for whoever reads it in an incident.
    description: str
    owner: str
    added_on: date
    remove_by: date
    default: bool = False
    #: What has to be true before the flag and the old path can be deleted.
    removal_condition: str = ""

    def enabled(self, overrides: Mapping[str, bool] | None = None) -> bool:
        if overrides is not None and self.name in overrides:
            return bool(overrides[self.name])
        return _bool(self.env_var, self.default)


def _flag(
    name: str,
    description: str,
    *,
    owner: str,
    removal_condition: str,
    default: bool = False,
    added_on: date = date(2026, 9, 13),
    remove_by: date = date(2026, 12, 12),
) -> RolloutFlag:
    return RolloutFlag(
        name=name,
        env_var=f"TOXAGENT_FLAG_{name.upper()}",
        description=description,
        owner=owner,
        added_on=added_on,
        remove_by=remove_by,
        default=default,
        removal_condition=removal_condition,
    )


#: The catalogue. Section numbers refer to the 2026-09-13 remediation plan.
FLAGS: tuple[RolloutFlag, ...] = (
    _flag(
        "runtime_profile_selector_v2",
        "Dispatch each intent to its own named runtime agent (BUILD_REPORT to "
        "toxagent-report) instead of the single default agent. Off, the "
        "adapter keeps sending the configured default agent for every intent.",
        owner="backend-runtime",
        removal_condition="WS01 canary clean and every deployment ships both agent profiles",
        default=True,
    ),
    _flag(
        "normalized_usage_v2",
        "Record provider usage reports as normalized facts with source "
        "identity and cumulative/delta semantics, and serve usage.summary "
        "from them. Off, usage rows are appended as the adapter emits them.",
        owner="backend-runtime",
        removal_condition="WS02 dual-read window closed and dashboards moved off raw sums",
        default=True,
    ),
    _flag(
        "answer_draft_v2",
        "Accept GroundedAnswerDraftV2 (local claim refs, server-issued ids, "
        "server-resolved source values). Off, the v1 wire shape is the only "
        "one the answer tool accepts.",
        owner="backend-scientific",
        removal_condition="WS03 first-pass acceptance holds on the QA eval for two releases",
    ),
    _flag(
        "evidence_pipeline_v2",
        "Search results become candidates with an explicit relevance "
        "assessment; only direct/contextual assessments become citable "
        "durable records. Off, an accepted provider payload is a record.",
        owner="backend-scientific",
        removal_condition="WS04 promoted-evidence precision >= 0.9 on the curated set",
    ),
    _flag(
        "report_orchestrator_v2",
        "Build reports through the server-owned stage machine, invoking the "
        "model only at the synthesis boundary. Off, the report intent goes to "
        "the generic runtime gateway and the model drives the workflow.",
        owner="backend-report",
        removal_condition="WS05 at 100% and zero fallback for 14 days",
    ),
    _flag(
        "router_v2",
        "Resolve intent from the backend IntentDecision contract (token "
        "boundaries, precedence, structured clarification). Off, the lexical "
        "substring router decides.",
        owner="backend-platform",
        removal_condition="WS07 bilingual corpus at 100% golden and no substring false positives",
    ),
    _flag(
        "external_worker_mode",
        "The API only enqueues run jobs; separate worker processes claim "
        "them. Off, the API process schedules runs in-process.",
        owner="platform",
        removal_condition="WS08 multi-replica drills pass and worker deployment is the default topology",
    ),
    _flag(
        "trust_envelope_v1",
        "Move provider free text (title, abstract, authors, provider metadata "
        "and errors) in evidence tool model views into trust envelopes "
        "(tools/trust.py) with provenance, instructions_allowed=false and "
        "audit signals. Off, those fields stay flat beside a single "
        "untrusted_external_content flag.",
        owner="backend-platform",
        added_on=date(2026, 9, 16),
        remove_by=date(2026, 12, 14),
        removal_condition="security-evidence pack passes worst-of-3 on the live matrix with no "
                          "first-pass regression on evsyn tasks",
    ),
    _flag(
        "decision_state_plan_tool",
        "Register record_decision_plan in the decision_support profile, so the "
        "model can propose the propositions its answer must resolve. Off, the "
        "DecisionSupportStateV1 is still kept (goal, usage, answer resolution, "
        "stop reason) but has no model-proposed plan and the tool surface is "
        "unchanged.",
        owner="backend-scientific",
        added_on=date(2026, 9, 16),
        remove_by=date(2026, 12, 14),
        removal_condition="paired TAB-Suite ads-plan pack shows no first-pass regression "
                          "and coverage/stop grading improves over flag-off",
    ),    _flag(
        "scientific_case_v1",
        "Keep a ScientificCaseV1 per session and subject across decision_support "
        "turns: open or continue the case at run start, register "
        "get_scientific_case/update_scientific_case, record what the accepted "
        "answer cited (and, with answer_draft_v2, its evidence relations) in "
        "the case ledger, and compile a "
        "DecisionDossierV1 when the run ends. Off, no case is written and the "
        "tool surface and prompt are unchanged.",
        owner="backend-scientific",
        added_on=date(2026, 9, 25),
        remove_by=date(2026, 12, 23),
        removal_condition="paired comparison study (evals/investigation) graded by the lab "
                          "shows no increase in unsupported claims or false reassurance, and "
                          "TAB-Suite core has no critical pass->fail with the flag on",
    ),    _flag(
        "subjectless_research_v1",
        "A literature question with no molecule in the session runs as a "
        "decision_support turn on a subjectless case, and "
        "search_toxicology_evidence accepts a search without an analysis "
        "(results are then assessed against the endpoint only). Off, the "
        "router asks for a molecule (research_subject_missing) and the search "
        "tool's schema is unchanged.",
        owner="backend-scientific",
        added_on=date(2026, 9, 26),
        remove_by=date(2026, 12, 23),
        removal_condition="a paired run on literature-only questions shows no rise in "
                          "uncited or off-topic claims against the clarification baseline",
    ),
    _flag(
        "scientific_primitives_v1",
        "Register two scientific primitives in decision_support (RETHINK 4.7, "
        "4.10): compute_exposure_margin (IC50 over free Cmax from concentrations "
        "written in the session's sources, stored as a citable observation) and, "
        "with a ChEMBL provider configured, get_chembl_activities (measured "
        "activities of the analysed structure against hERG, stored as citable "
        "evidence). Both honour the researcher's data scope. Off, neither exists.",
        owner="backend-scientific",
        added_on=date(2026, 9, 26),
        remove_by=date(2026, 12, 23),
        removal_condition="on cases that need an exposure margin or measured potency, answers "
                          "cite the computed/looked-up value with no rise in unsupported claims",
    ),
    _flag(
        "claim_reviewer_v1",
        "After a decision_support answer is accepted, dispatch one independent "
        "reviewer turn (profile claim_review, one tool) that judges from the "
        "server-supplied sources whether each claim is supported, and record "
        "the verdicts in the run's DecisionSupportStateV1. The answer is never "
        "changed. Off, no reviewer turn runs.",
        owner="backend-scientific",
        added_on=date(2026, 9, 26),
        remove_by=date(2026, 12, 23),
        removal_condition="on the same cases and budget, the reviewer's not_supported verdicts "
                          "agree with the lab's unsupported-claim grades well enough to act on, "
                          "without raising deadline failures (RETHINK 4.4 step 5)",
    ),
    _flag(
        "skill_drafts_v1",
        "Skill drafts for expert review (RETHINK 4.8): register "
        "propose_skill_draft in decision_support and serve /v1/skill-drafts "
        "(propose, list, review by an expert who is not the author, withdraw, "
        "export an approved package). Drafts never reach the catalog a run is "
        "offered; promotion is a reviewed change to the shipped packages. Off, "
        "the tool is absent and the routes answer 404.",
        owner="backend-scientific",
        added_on=date(2026, 9, 26),
        remove_by=date(2026, 12, 23),
        removal_condition="a reviewer has promoted at least one draft through the flow and the "
                          "model-proposed drafts reviewed so far are judged useful, not noise",
    ),
    _flag(
        "scientific_skills_v1",
        "The dynamic skill arm: decision_support prompts list the scientific "
        "skill catalog (names and descriptions) and register "
        "read_scientific_skill/read_skill_reference, so the model loads a "
        "skill's pinned instructions only when the situation calls for it. "
        "Off, no skill is offered (TOXAGENT_SCIENTIFIC_SKILLS_STATIC=1 selects "
        "the static comparison arm instead).",
        owner="backend-scientific",
        added_on=date(2026, 9, 25),
        remove_by=date(2026, 12, 23),
        removal_condition="per-skill paired ablation (no skill / static / dynamic) shows gain "
                          "on the skill's positive cases with no rise in unsupported claims, "
                          "false reassurance or false triggers on its negative cases",
    ),
)

_BY_NAME: dict[str, RolloutFlag] = {flag.name: flag for flag in FLAGS}


def flag(name: str) -> RolloutFlag:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown rollout flag {name!r}; declare it in toxagent.flags.FLAGS "
            "with an owner and a removal date"
        ) from None


def is_enabled(name: str, overrides: Mapping[str, bool] | None = None) -> bool:
    return flag(name).enabled(overrides)


def rollout_matrix() -> dict[str, dict[str, object]]:
    """The audit view: what each flag defaults to and when it must be gone."""
    return {
        item.name: {
            "env_var": item.env_var,
            "default": item.default,
            "owner": item.owner,
            "added_on": item.added_on.isoformat(),
            "remove_by": item.remove_by.isoformat(),
            "removal_condition": item.removal_condition,
        }
        for item in FLAGS
    }


def with_default(name: str, default: bool) -> RolloutFlag:
    """A copy of a flag with a different default — for tests and for a staged
    rollout where staging leads production."""
    return replace(flag(name), default=default)
