"""Arm B: a strong fixed workflow on the same tools and gates (#50).

The comparison the study turns on is D (an investigator that chooses what to
do next) against B (a workflow that does the sensible thing in a fixed order).
For that to measure investigation and nothing else, B gets everything D gets
except the choosing: the same tools, read through the same tool runner and
counted against the same budget; the same guidance in its prompt; the same
answer schema, validator and release gate. What differs is who decides the
reads. Here the server does, by rule:

1. read the inventory, and every artifact of the subject -- the analysis
   bundle, each attached screening's state, estimate and decision;
2. decide whether to search at all: not for a lookup of a value, label or
   status the artifacts already hold; yes for a recommendation, a hazard or
   clinical/regulatory context, limited applicability, or an explicit request
   for sources -- the same rules the agent's prompt states;
3. search by fixed query templates, read the top accepted records, and read
   their results tables when the text can be read;
4. note what only the researcher can supply (exposure, for a margin
   question), so the answer asks for it instead of guessing it;
5. hand everything to one synthesis turn whose only tool is
   submit_grounded_answer.

Every step is recorded with a one-line public reason in the run's decision
state, so a grader sees what B did and why without reading a transcript.
The rules are deliberately the obvious ones; they are what dev tuning to
parity with D adjusts, and the scientific owner reviews before test.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from ...application.investigation import decision_state_service
from ...application.runs.budget import decision_support_limit
from ...application.runs.scheduler import RunContext
from ...application.tool_context import ToolContext
from ...domain import decision_state
from ...domain.provenance import content_sha256
from ...platform.flags import is_enabled

#: v2 (AG-EVIDENCE): an activity branch whose queries come from the bound
#: screening's ligand, target and assay context, a toxicity branch whose
#: endpoint comes from the question, and a provider-failure branch -- v1 only
#: searched with an analysis and always for hERG.
FIXED_WORKFLOW_VERSION = "fixed-workflow-b-v2"

#: Records read in full per search, and records whose text is read.
_READS_PER_SEARCH = 3
_TEXT_READS = 2

#: Step 2's rules, as phrases. A keyword list is a crude router, which is the
#: point: it is the fixed baseline the adaptive arm has to beat.
_DECISION = re.compile(
    r"\b(should|develop|candidate|go/no-?go|proceed|deprioriti[sz]e|safe|safety|risk|hazard|"
    r"clinical|regulatory|liability|nên|phát triển|an toàn|rủi ro)\b", re.I)
_SOURCES = re.compile(
    r"\b(literature|paper|papers|study|studies|source|sources|evidence|reported|measured|"
    r"published|cite|citation|verify|tài liệu|bài báo|nguồn|bằng chứng|nghiên cứu)\b", re.I)
_EXPOSURE = re.compile(
    r"\b(exposure|cmax|c_max|dose|dosing|margin|plasma|concentration|liều|phơi nhiễm|nồng độ)\b",
    re.I)

WORKFLOW_SYNTHESIS_INSTRUCTIONS = """\
You are the synthesis step of a fixed workflow. The server has already read \
the artifacts below and run the searches its rules call for; you have no read \
tools. Write the answer from this material only, and submit it with \
submit_grounded_answer: cite observations by observation_id and field_path, \
and evidence by evidence_id (and evidence_spans for a quoted passage). Owe \
every limitation the material lists. If "needs_from_researcher" is not \
empty, say what the researcher must supply and stop there (stop.reason \
needs_user_input where the schema has it) rather than guessing it."""


def fixed_workflow_on() -> bool:
    return is_enabled("fixed_workflow_v1")


class FixedWorkflowMixin:
    """Run a decision_support turn as arm B's fixed workflow."""

    _workflow_prompts: dict[str, str]

    def _workflow_runner(self):
        runner = getattr(self, "_tool_runner", None)
        if runner is None:
            from ...tools.runner import ToolRunner

            policy = self._policy
            runner = ToolRunner(
                self._registry, self._db,
                max_calls_per_run=getattr(policy, "max_tool_calls_per_run", None),
            )
            self._tool_runner = runner
        return runner

    async def _run_fixed_workflow(
        self, context: RunContext, *, system_prompt: str, deadline, commit,
    ) -> None:
        tool_context = ToolContext(
            session_id=context.session_id, run_id=context.run_id, actor=context.actor,
            profile="decision_support", deadline_at=deadline, language=context.language,
            intent=context.intent.value, model_selection=context.model_selection,
            subject=context.subject,
        )
        runner = self._workflow_runner()
        registered = set(self._registry.names())
        steps: list[dict[str, Any]] = []
        material: dict[str, Any] = {"workflow": FIXED_WORKFLOW_VERSION}

        async def call(name: str, arguments: dict[str, Any], reason: str) -> dict[str, Any] | None:
            if name not in registered:
                steps.append({"tool": name, "reason": reason, "status": "not_available"})
                return None
            result = await runner.call(tool_context, name, arguments)
            steps.append({"tool": name, "reason": reason, "status": result.get("status"),
                          "error_code": (result.get("error") or {}).get("code")})
            return result if result.get("status") == "completed" else None

        inventory = await call("get_artifact_inventory", {}, "what this run already holds")
        view = (inventory or {}).get("model_view") or {}
        material["inventory"] = view

        analysis = view.get("analysis") or {}
        if analysis.get("analysis_id"):
            bundle = await call(
                "get_analysis_bundle", {"analysis_id": analysis["analysis_id"]},
                "the prediction this session analysed",
            )
            if bundle:
                material["analysis"] = bundle["model_view"]
        for screening in view.get("screenings") or ():
            sid = screening["screening_id"]
            for tool, key in (("get_screening_state", "state"),
                              ("get_activity_snapshot", "estimate"),
                              ("get_prioritisation_decision", "decision")):
                read = await call(tool, {"screening_id": sid}, f"the screening's {key}")
                if read:
                    material.setdefault("screenings", {}).setdefault(sid, {})[key] = (
                        read["model_view"]
                    )

        question = context.text or ""
        subject = view.get("subject") or {}
        screening = next(
            (s for s in view.get("screenings") or () if s.get("is_subject")), None,
        ) if subject.get("kind") == "activity_screening" else None
        if screening is not None:
            # The activity branch: the subject is a ligand on a target.
            decision = ((material.get("screenings") or {}).get(screening["screening_id"]) or {})
            search, why = _should_search_activity(question, decision)
            steps.append({"tool": None, "reason": why, "status": "decided",
                          "decision": "search" if search else "no_search"})
            if search:
                await self._activity_retrieval(call, material, screening=screening)
            toxicity_id = subject.get("linked_toxicity_analysis_id")
            if toxicity_id and _TOXICITY.search(question):
                bundle = await call("get_analysis_bundle", {"analysis_id": toxicity_id},
                                    "the toxicity analysis the screening's toxicity stage made")
                if bundle:
                    material["toxicity_analysis"] = bundle["model_view"]
                if _DECISION.search(question) or _SOURCES.search(question):
                    await self._fixed_retrieval(call, material, analysis_id=toxicity_id,
                                                question=question)
        else:
            applicability = json.dumps(material.get("analysis") or {})
            search, why = _should_search(question, applicability)
            steps.append({"tool": None, "reason": why, "status": "decided",
                          "decision": "search" if search else "no_search"})
            if search and analysis.get("analysis_id"):
                await self._fixed_retrieval(call, material, analysis_id=analysis["analysis_id"],
                                            question=question)
        failures = [step for step in steps
                    if step.get("tool") == "search_toxicology_evidence"
                    and step.get("status") == "error"
                    and step.get("error_code") in _PROVIDER_FAILURES]
        if failures:
            # The provider-failure branch: said as such, never as "no evidence".
            material["provider_failure"] = {
                "searches_failed": len(failures),
                "note": "the literature provider could not be reached; this is not an "
                        "absence of evidence",
            }
            steps.append({"tool": None, "status": "decided", "decision": "report_provider_failure",
                          "reason": "a search failed at the provider, not for lack of sources"})

        needs = []
        if _EXPOSURE.search(question) and "exposure" not in json.dumps(material).lower():
            needs.append("the free plasma Cmax (or exposure) at the intended dose")
            steps.append({"tool": None, "status": "decided", "decision": "ask_researcher",
                          "reason": "an exposure question, and no exposure in the session"})
        material["needs_from_researcher"] = needs

        await decision_state_service.advance(
            self._db, context.run_id,
            lambda state: decision_state.record_workflow(
                state, {"version": FIXED_WORKFLOW_VERSION, "steps": steps},
            ),
        )
        prompt = (
            system_prompt + "\n\n" + WORKFLOW_SYNTHESIS_INSTRUCTIONS
            + "\n\nMaterial:\n```json\n"
            + json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
            + "\n```"
        )
        # Kept per run: the gateway serves runs concurrently, and the release
        # gate's repair turn for B is this same synthesis step, told what the
        # review blocked.
        self._workflow_prompts[context.run_id] = prompt
        await self._dispatch(
            replace(context, text=question or "Answer the question from the material."),
            system_prompt=prompt, profile="workflow_synthesis", deadline=deadline,
            instructions_hash=content_sha256(WORKFLOW_SYNTHESIS_INSTRUCTIONS),
            has_product=self._has_answer, commit=commit,
        )

    async def _activity_retrieval(self, call, material: dict[str, Any], *,
                                  screening: dict[str, Any]) -> None:
        """Step 3 for a screening: queries from the ligand, the target and the
        assay readout the binding states -- never a hard-coded target."""
        target = screening.get("target") or {}
        target_id = str(target.get("target_id") or "")
        names = [n for n in screening.get("ligand_names") or () if n]
        ligand = names[0] if names else str(screening.get("canonical_smiles") or "")
        readout = str(screening.get("assay_context") or "").split("|")[0]
        templates = [f"{ligand} {target_id}", f"{ligand} {target_id} {readout}".strip(),
                     f"{ligand} {target_id} inhibition"]
        await self._search_and_read(
            call, material, templates=templates,
            arguments={"screening_id": screening["screening_id"]},
        )

    async def _fixed_retrieval(self, call, material: dict[str, Any], *, analysis_id: str,
                               question: str = "") -> None:
        """Step 3: fixed query templates for the endpoint the question names
        (hERG when it names none), the top accepted records read, and their
        tables read when the text can be."""
        endpoint = _endpoint_of(question)
        name = _subject_name(material)
        templates = {
            "herg": [f"{name} hERG", f"{name} hERG IC50 patch clamp", f"{name} QT prolongation"],
            "tox21": [f"{name} Tox21", f"{name} nuclear receptor assay",
                      f"{name} stress response assay"],
            "clintox": [f"{name} clinical toxicity", f"{name} adverse events",
                        f"{name} clinical trial safety"],
        }[endpoint]
        await self._search_and_read(
            call, material, templates=templates,
            arguments={"analysis_id": analysis_id, "endpoint": endpoint},
        )

    async def _search_and_read(self, call, material: dict[str, Any], *, templates: list[str],
                               arguments: dict[str, Any]) -> None:
        found: list[str] = list(material.get("_found") or [])
        for query in templates[: decision_support_limit("max_searches")]:
            result = await call(
                "search_toxicology_evidence",
                {**arguments, "query": query, "limit": 5},
                f"fixed query: {query}",
            )
            for item in ((result or {}).get("model_view") or {}).get("results") or ():
                if item.get("evidence_id") and item["evidence_id"] not in found:
                    found.append(item["evidence_id"])
            if len(found) >= _READS_PER_SEARCH:
                break
        material["_found"] = found
        records = list(material.get("evidence") or [])
        already = {r.get("evidence_id") for r in records}
        for evidence_id in found[: decision_support_limit("max_evidence_reads")][:_READS_PER_SEARCH * 2]:
            if evidence_id in already:
                continue
            read = await call("get_evidence_record", {"evidence_id": evidence_id},
                              "read before citing")
            if read:
                records.append(read["model_view"])
        material["evidence"] = records
        texts = list(material.get("full_text") or [])
        for evidence_id in found[:_TEXT_READS]:
            if any(t.get("evidence_id") == evidence_id for t in texts):
                continue
            read = await call("read_evidence_text",
                              {"evidence_id": evidence_id, "kinds": ["table_row"], "limit": 20},
                              "the results tables, for values the abstract does not give")
            if read:
                texts.append(read["model_view"])
        if texts:
            material["full_text"] = texts


_TOXICITY = re.compile(
    r"\b(toxicity|toxic|herg|qt|cardiac|tox21|clintox|safety|độc|an toàn)\b", re.I)
_PROVIDER_FAILURES = frozenset({"evidence_unavailable", "provider_rate_limited", "tool_timeout"})


def _endpoint_of(question: str) -> str:
    if re.search(r"\btox21\b|nuclear receptor|stress response", question, re.I):
        return "tox21"
    if re.search(r"\bclintox\b|clinical (toxicity|trial)", question, re.I):
        return "clintox"
    return "herg"


def _should_search_activity(question: str, read: dict[str, Any]) -> tuple[bool, str]:
    """Step 2 for a screening. A REVIEW decision is the question the
    literature can settle -- the estimate's interval crosses the threshold --
    so it is searched; so is a recommendation or a request for sources. A
    lookup of the screening's own values is not."""
    decision = (read.get("decision") or {}).get("status")
    if decision == "review":
        return True, "the policy sent it to review: measured activity on this target can settle it"
    if _DECISION.search(question):
        return True, "a recommendation question: the estimate alone does not settle it"
    if _SOURCES.search(question):
        return True, "the question asks for sources or measurements"
    return False, "a lookup: the screening's own artifacts answer it, so no search"


def _should_search(question: str, applicability: str) -> tuple[bool, str]:
    if _DECISION.search(question):
        return True, "a recommendation or hazard question: the prediction alone does not settle it"
    if _SOURCES.search(question):
        return True, "the question asks for sources or measurements"
    if '"limited"' in applicability or '"out_of_domain"' in applicability:
        return True, "applicability is limited or out of domain"
    return False, "a lookup: the run's own artifacts answer it, so no search"


def _subject_name(material: dict[str, Any]) -> str:
    analysis = material.get("analysis") or {}
    text = json.dumps(analysis)
    match = re.search(r'"preferred_name":\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return str((material.get("inventory") or {}).get("analysis", {}).get("canonical_smiles") or "")
