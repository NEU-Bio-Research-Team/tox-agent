"""``run_scientific_procedure``: the determinate work, behind one allowlist (#66).

§7.2 asks for "một allowlisted procedure runner hoặc DSL có kiểu trên
backend" rather than a general shell, and §7.3 asks for the core tool surface
to stay small. So the four procedures of ``application/investigation/
procedures.py`` are one tool with a typed ``procedure`` name, not four tools:
the model picks a procedure that exists or is refused by name, and the tool
list does not grow by one entry per determinate operation.

What this tool does **not** do is as much the point:

* it does not reach a provider -- it works on records the run already read,
  so a repeat is free and a frozen benchmark can replay it;
* it does not write to the case -- a procedure proposes, the investigator
  commits (§6.2 step six), which keeps one writer for the case;
* it does not summarise away the inconvenient half: every result carries the
  criteria it applied, the records it excluded and why, the fields the inputs
  did not state, and its own sub-operation count for the run's budget (§7.2).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application.investigation import procedures
from ...application.investigation import scientific_case_service
from ...domain import scientific_case as sc
from ...domain.errors import InvalidRequest
from ...domain.evidence import EvidenceStatus
from ..registry import ToolContext, ToolDefinition, ToolOutput


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunProcedureInput(_Input):
    procedure: Literal[
        "normalise_measurements", "deduplicate_by_origin", "compare_measurements",
        "check_dependencies", "compare_candidates",
    ] = Field(description="Which determinate operation to run.")
    evidence_ids: list[str] = Field(
        default_factory=list, max_length=24,
        description="Records to work on, already read in this session. "
                    "normalise_measurements and deduplicate_by_origin take any number; "
                    "compare_measurements takes exactly two; check_dependencies takes none.",
    )
    endpoint: str | None = Field(
        default=None, max_length=80,
        description="The endpoint you went looking for, e.g. 'herg'. Passed in rather "
                    "than guessed from the records: a record can be about hERG without "
                    "saying so in a field, and guessing is how an unrelated source "
                    "becomes comparable.",
    )
    criterion: Literal["herg_probability_blocker", "clintox_probability",
                       "predicted_pic50"] | None = Field(
        default=None,
        description="compare_candidates only: the one criterion the question declares. "
                    "Every analysis and screening in this session is a candidate; each row "
                    "is read from that candidate's own artifact. There is no combined score.",
    )
    tolerance_fold: float | None = Field(
        default=None, gt=0,
        description="compare_measurements only: the fold difference beyond which two "
                    "values of one quantity are more than assay spread. Without it, "
                    "disjoint values come back 'undetermined' -- which is the honest "
                    "answer when nobody has declared what counts as a real difference.",
    )


def build(database) -> list[ToolDefinition]:
    async def run_procedure(context: ToolContext, payload: RunProcedureInput) -> ToolOutput:
        spec = procedures.spec_for(payload.procedure)

        if payload.procedure == "compare_candidates":
            if payload.criterion is None:
                raise InvalidRequest(
                    "compare_candidates needs the criterion the question declares; if it "
                    "declares none, ask the researcher which one rather than choosing"
                )
            rows = await _candidate_rows(database, context, payload.criterion)
            result = procedures.compare_candidates(rows, criterion=payload.criterion)
        elif payload.procedure == "check_dependencies":
            async with database.unit_of_work() as uow:
                case = await scientific_case_service.case_for_run(
                    uow, session_id=context.session_id, run_id=context.run_id
                )
                accepted = await uow.evidence.list_for_session(
                    context.session_id, status=EvidenceStatus.ACCEPTED, limit=500,
                )
            if case is None:
                raise InvalidRequest(
                    "this run has no scientific case, so there are no derived claims to "
                    "check; the dependency graph is written by the case, not by a run"
                )
            # Claims cite a researcher value by its version (case_value:key@vN),
            # not by its context item; every version stays citable, so every
            # version is known.
            known = (
                [version.ref for version in sc.value_versions(case)]
                + [f"context:{item.id}" for item in case.context]
                + [entry.id for entry in case.evidence]
                + [entry.source_ref for entry in case.evidence if entry.source_ref]
                + [f"evidence:{record.id}" for record in accepted]
            )
            result = procedures.check_dependencies(case, known_refs=known)
        else:
            records = await _load(database, context, payload.evidence_ids)
            if payload.procedure == "compare_measurements":
                if len(records) != 2:
                    raise InvalidRequest(
                        "compare_measurements takes exactly two evidence_ids: it answers "
                        "whether two records can be set against each other"
                    )
                result = procedures.compare_measurements(
                    records[0], records[1], endpoint=payload.endpoint,
                    tolerance_fold=payload.tolerance_fold,
                )
            elif not records:
                raise InvalidRequest(
                    f"{payload.procedure} needs evidence_ids of records this session has "
                    "already read"
                )
            elif payload.procedure == "normalise_measurements":
                result = procedures.normalise_measurements(records, endpoint=payload.endpoint)
            else:
                result = procedures.deduplicate_by_origin(records, endpoint=payload.endpoint)

        view: dict[str, Any] = result.to_dict()
        return ToolOutput(
            canonical=view,
            model_view=view,
            ui_view=view,
            provenance={"procedure": spec.name, "procedure_version": spec.version,
                        "sub_operations": result.sub_operation_count},
        )

    return [
        ToolDefinition(
            name="run_scientific_procedure",
            title="Run a scientific procedure",
            description=(
                "Determinate work, done by code over records you have already read: "
                "normalise_measurements (canonical value, unit and constraint kind, with "
                "the conversion used); deduplicate_by_origin (one record per experiment -- "
                "two rows of one measurement are not two independent sources); "
                "compare_measurements (two records: whether their scopes are comparable, "
                "then whether their values agree under a tolerance you declare); "
                "check_dependencies (derived claims a changed premise left stale, plus "
                "unresolved references and cycles); compare_candidates (every analysis and "
                "screening in this session on one declared criterion, each from its own "
                "artifact, with citable observation_ids). Every result states the criteria it "
                "applied and the records it excluded with the reason -- read those before "
                "quoting a count. Do not do this arithmetic yourself."
            ),
            input_model=RunProcedureInput,
            handler=run_procedure,
            profiles=frozenset({"decision_support", "evidence_research"}),
            soft_timeout_s=5.0,
            hard_timeout_s=15.0,
            idempotent=True,
            cost_class="cheap",
        ),
    ]


async def _load(
    database, context: ToolContext, evidence_ids: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """The records' normalised facts, in the order asked for.

    A record this session has not accepted is refused by name rather than
    skipped: a procedure that silently worked on three of four records would
    report a count for a set the caller did not ask about.
    """
    records: list[tuple[str, dict[str, Any]]] = []
    async with database.unit_of_work() as uow:
        for evidence_id in dict.fromkeys(evidence_ids):
            record = await uow.evidence.get(evidence_id, session_id=context.session_id)
            if record is None:
                raise InvalidRequest(
                    "no such evidence in this session; read it before running a procedure "
                    f"on it (evidence_id={evidence_id})"
                )
            records.append((evidence_id, dict(record.normalized_facts or {})))
    return records


async def _candidate_rows(database, context: ToolContext, criterion: str) -> list[dict[str, Any]]:
    """One row per subject in the session, read from that subject's own artifact.

    A value is cited through an observation, so a screening's estimate is
    projected into one for this run (as get_activity_snapshot does); an
    analysis's prediction already is one.
    """
    from datetime import datetime, timezone

    from ...application.activity import replay
    from ...domain.observation import ObservationKind

    spec = procedures.CRITERIA[criterion]
    rows: list[dict[str, Any]] = []
    async with database.unit_of_work() as uow:
        analyses = await uow.analyses.list_for_session(context.session_id, limit=50)
        screenings = await uow.screenings.list_for_session(context.session_id)
        for analysis in analyses:
            row: dict[str, Any] = {"subject_key": f"analysis:{analysis.id}"}
            if spec["artifact"] != "prediction":
                row["reason"] = "an analysis has no activity estimate"
            else:
                prediction = next(
                    (o for o in await uow.observations.list_for_analysis(analysis.id)
                     if o.kind is ObservationKind.PREDICTION), None)
                value = _value_at(prediction, spec["field_path"])
                if value is None:
                    row["reason"] = "the analysis has no such prediction"
                else:
                    row.update(value=value, observation_id=prediction.id)
            rows.append(row)
        for attachment in screenings:
            row = {"subject_key": f"screening:{attachment.screening_id}"}
            if spec["artifact"] != "activity_estimate":
                row["reason"] = ("a screening's toxicity prediction is its toxicity analysis, "
                                 "listed as its own candidate when it ran")
                rows.append(row)
                continue
            observation = replay.observation_for(
                attachment.record, "activity_estimate", session_id=context.session_id,
                run_id=context.run_id, now=datetime.now(timezone.utc),
            )
            value = _value_at(observation, spec["field_path"])
            if value is None:
                state = (attachment.record.get("screening_state") or {})
                row["reason"] = f"no activity estimate (screening {state.get('status')})"
            else:
                await uow.observations.add(observation)
                subject = (attachment.record.get("activity_estimate") or {}).get("subject") or {}
                row.update(value=value, observation_id=observation.id,
                           target_id=(subject.get("target") or {}).get("target_id"),
                           assay_context=subject.get("assay_context"))
            rows.append(row)
        await uow.commit()
    return rows


def _value_at(observation, path: str) -> float | None:
    if observation is None:
        return None
    try:
        value = observation.value_at(path)
    except Exception:  # noqa: BLE001 - an absent path is "no value", not a failure
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
