"""``submit_report_synthesis``: the only tool an orchestrated report run can see.

Its input *is* ``ReportSynthesisV3``. There is no wrapper, no report id the
model chooses beyond the one it was given, and no numeric field anywhere in the
schema, so the description can be short: the schema already says what cannot be
submitted (PR-16's second half — the words that described a workflow the model
no longer runs are gone).
"""
from __future__ import annotations

from ...application.report_synthesis import SubmitReportSynthesis
from ...validation.synthesis_wire import ReportSynthesisV3
from ..registry import ToolContext, ToolDefinition, ToolOutput

DESCRIPTION = (
    "Submit the narrative of the report build named in the run context. Write every "
    "value as {{fact_id}} from the fact bundle and list the facts each section, "
    "conclusion and recommendation rests on. Do not write predictor_results, "
    "limitations, references or provenance_appendix: the server compiles them. A "
    "refusal lists typed violations; fix only those and submit once more."
)


def build(database) -> list[ToolDefinition]:
    submit = SubmitReportSynthesis(database)

    async def handler(context: ToolContext, payload: ReportSynthesisV3) -> ToolOutput:
        accepted = await submit.execute(
            session_id=context.session_id, run_id=context.run_id, synthesis=payload
        )
        view = {
            "report_build_id": accepted.report_build_id,
            "accepted": True,
            "content_sha256": accepted.synthesis_sha256,
            "attempt": accepted.attempt,
            "gap_count": accepted.gap_count,
            "next_action": "Stop. The server validates, renders and publishes the report.",
        }
        return ToolOutput(
            canonical=view, model_view=view, ui_view=view,
            provenance={"report_build_id": accepted.report_build_id},
        )

    return [
        ToolDefinition(
            name="submit_report_synthesis",
            title="Submit report synthesis",
            description=DESCRIPTION,
            input_model=ReportSynthesisV3,
            handler=handler,
            profiles=frozenset({"report_synthesis"}),
            soft_timeout_s=20.0,
            hard_timeout_s=60.0,
            idempotent=False,
        )
    ]
