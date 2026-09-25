"""The systems of the comparison study, named once.

RETHINK §5.4 asks for four systems on the same cases — (A) predictor + template,
(B) an LLM given the predictor snapshot, (C) the current agent, (D) the
case-based investigator — and §5.5 for the skill ablation of D. The product
owner added general platforms as comparison arms (2026-09-25): OpenAI,
Anthropic and Google models, each bare and with the snapshot.

A ToxAgent system is a *deployment* (a control plane with a given flag set),
so its spec states the flags and skill arm the deployment must report through
``/v1/system/effective-product``; the runner refuses to record a ToxAgent arm
against a deployment that does not match, instead of mislabelling it.

``label`` is for the study report only. It never reaches the blinded packet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    rethink_arm: str
    family: str
    adapter: str
    label: str
    uses_snapshot: bool = False
    platform: str | None = None
    required_flags: Mapping[str, bool] = field(default_factory=dict)
    skills_mode: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id, "rethink_arm": self.rethink_arm, "family": self.family,
            "adapter": self.adapter, "label": self.label, "uses_snapshot": self.uses_snapshot,
            "platform": self.platform, "required_flags": dict(self.required_flags),
            "skills_mode": self.skills_mode,
        }


_CASE_OFF = {"scientific_case_v1": False, "scientific_skills_v1": False}

SYSTEMS: dict[str, SystemSpec] = {spec.system_id: spec for spec in (
    SystemSpec("A_predictor_template", "A", "predictor", "predictor_template",
               "ToxPred output rendered by a fixed template; no language model"),
    SystemSpec("C_toxagent_current", "C", "toxagent", "toxagent",
               "ToxAgent decision support as shipped (no case, no skills)",
               required_flags=_CASE_OFF, skills_mode="off"),
    SystemSpec("D_toxagent_investigator", "D", "toxagent", "toxagent",
               "ToxAgent case-based investigator, skills loaded on demand",
               required_flags={"scientific_case_v1": True, "scientific_skills_v1": True,
                               "answer_draft_v2": True},
               skills_mode="dynamic"),
    SystemSpec("D0_toxagent_case_no_skills", "D-ablation", "toxagent", "toxagent",
               "ToxAgent case-based investigator without skills",
               required_flags={"scientific_case_v1": True, "scientific_skills_v1": False,
                               "answer_draft_v2": True},
               skills_mode="off"),
    SystemSpec("Ds_toxagent_case_static_skills", "D-ablation", "toxagent", "toxagent",
               "ToxAgent case-based investigator, every skill composed into the prompt",
               required_flags={"scientific_case_v1": True, "scientific_skills_v1": False,
                               "answer_draft_v2": True},
               skills_mode="static"),
    SystemSpec("P_openai_bare", "P", "platform", "codex_cli",
               "OpenAI model through the codex CLI, question only", platform="openai"),
    SystemSpec("B_openai_snapshot", "B", "platform", "codex_cli",
               "OpenAI model through the codex CLI, with the ToxPred snapshot",
               uses_snapshot=True, platform="openai"),
    SystemSpec("P_anthropic_bare", "P", "platform", "claude_cli",
               "Anthropic model through the claude CLI (no tools), question only",
               platform="anthropic"),
    SystemSpec("B_anthropic_snapshot", "B", "platform", "claude_cli",
               "Anthropic model through the claude CLI (no tools), with the ToxPred snapshot",
               uses_snapshot=True, platform="anthropic"),
    SystemSpec("P_google_bare", "P", "platform", "google",
               "Google Gemini model through the operator's MCP bridge (or answered out of "
               "process with --google-channel manual), question only", platform="google"),
    SystemSpec("B_google_snapshot", "B", "platform", "google",
               "Google Gemini model through the operator's MCP bridge (or answered out of "
               "process with --google-channel manual), with the ToxPred snapshot",
               uses_snapshot=True, platform="google"),
)}


def resolve(names: list[str]) -> list[SystemSpec]:
    unknown = sorted(set(names) - set(SYSTEMS))
    if unknown:
        raise ValueError(f"unknown systems {unknown}; known: {sorted(SYSTEMS)}")
    return [SYSTEMS[name] for name in names]


def product_mismatch(spec: SystemSpec, effective_product: Mapping) -> list[str]:
    """Why a deployment is not the arm it would be recorded as (empty if it is)."""
    problems: list[str] = []
    flags = effective_product.get("flags") or {}
    for name, wanted in spec.required_flags.items():
        actual = (flags.get(name) or {}).get("enabled")
        if actual is not wanted:
            problems.append(f"flag {name} is {actual}, the arm needs {wanted}")
    if spec.skills_mode is not None:
        actual_mode = (effective_product.get("scientific_skills") or {}).get("mode", "off")
        if actual_mode != spec.skills_mode:
            problems.append(f"skills mode is {actual_mode}, the arm needs {spec.skills_mode}")
    return problems
