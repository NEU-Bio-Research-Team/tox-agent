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
        "get_scientific_case/update_scientific_case, record the accepted "
        "answer's evidence relations in the case ledger, and compile a "
        "DecisionDossierV1 when the run ends. Off, no case is written and the "
        "tool surface and prompt are unchanged.",
        owner="backend-scientific",
        added_on=date(2026, 9, 25),
        remove_by=date(2026, 12, 23),
        removal_condition="paired comparison study (evals/investigation) graded by the lab "
                          "shows no increase in unsupported claims or false reassurance, and "
                          "TAB-Suite core has no critical pass->fail with the flag on",
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
