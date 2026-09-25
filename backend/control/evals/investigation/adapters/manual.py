"""Platforms answered out of process: the runner writes the prompt, someone answers.

For a platform with no scriptable client here (Google Gemini in this study), or
when the product owner wants the consumer chat product itself, the runner
writes each turn's exact prompt to a file and waits for the answer file:

    manual/<system>/<case>/t<trial>/turn<k>.prompt.md       (written by the runner)
    manual/<system>/<case>/t<trial>/turn<k>.response.md     (written by the operator)
    manual/<system>/<case>/t<trial>/turn<k>.meta.json       (written by the operator)

``meta.json`` must name the model that answered as the platform reported it
(``model_id_resolved``), when it answered (``answered_at``) and how
(``channel``, e.g. "gemini-review MCP bridge" or "gemini.google.com"); a
response without it is refused rather than recorded under a guessed model.
A multi-turn case advances one turn per runner pass, because turn k's prompt
contains the answer to turn k-1. Until every turn is answered the record is
``pending`` — neither a result nor a failure.
"""
from __future__ import annotations

import json
from typing import Any

from evals.investigation import prompts
from evals.investigation.adapters import AdapterResult
from evals.investigation.record import STATUS_ERROR, STATUS_OK, STATUS_PENDING, StudyStore, TurnRecord
from evals.investigation.systems import SystemSpec

REQUIRED_META = ("model_id_resolved", "answered_at", "channel")


class ManualAdapter:
    def describe(self) -> dict[str, Any]:
        return {"adapter": "manual", "prompt_version": prompts.PROMPT_VERSION,
                "preamble_sha256": prompts.preamble_sha256(),
                "protocol": "prompt file written by the runner; response and meta files by the operator"}

    async def run(self, case: dict[str, Any], spec: SystemSpec, *, trial: int,
                  snapshot: dict[str, Any] | None, store: StudyStore) -> AdapterResult:
        if spec.uses_snapshot and not snapshot:
            return AdapterResult(status=STATUS_ERROR, turns=[],
                                 error="this arm needs the predictor snapshot and none was produced")
        directory = store.root / "manual" / spec.system_id / case["case_id"] / f"t{trial}"
        directory.mkdir(parents=True, exist_ok=True)
        turns: list[TurnRecord] = []
        responses: list[str] = []
        models: list[str] = []
        for index, turn in enumerate(case["turns"]):
            prompt = prompts.render(case["turns"], index, responses,
                                    snapshot=snapshot if spec.uses_snapshot else None)
            prompt_path = directory / f"turn{index}.prompt.md"
            if not prompt_path.exists() or prompt_path.read_text(encoding="utf-8") != prompt:
                prompt_path.write_text(prompt, encoding="utf-8")
            response_path = directory / f"turn{index}.response.md"
            meta_path = directory / f"turn{index}.meta.json"
            if not response_path.exists():
                return AdapterResult(status=STATUS_PENDING, turns=turns,
                                     error=f"waiting for {response_path.relative_to(store.root)}")
            if not meta_path.exists():
                return AdapterResult(status=STATUS_PENDING, turns=turns,
                                     error=f"waiting for {meta_path.relative_to(store.root)}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            missing = [key for key in REQUIRED_META if not meta.get(key)]
            if missing:
                return AdapterResult(status=STATUS_ERROR, turns=turns,
                                     error=f"{meta_path.name} lacks {missing}")
            text = response_path.read_text(encoding="utf-8").strip()
            responses.append(text)
            models.append(str(meta["model_id_resolved"]))
            turns.append(TurnRecord(
                index=index, user_text=turn["text"], sent_text=prompt, response_text=text,
                started_at=meta.get("asked_at", ""), ended_at=meta["answered_at"],
                duration_s=meta.get("duration_s"), meta={"manual": meta},
            ))
        distinct = sorted(set(models))
        return AdapterResult(
            status=STATUS_OK, turns=turns, final_text=responses[-1],
            model={"provider": spec.platform, "model_id_resolved": distinct[0] if len(distinct) == 1 else distinct,
                   "channel": sorted({t.meta["manual"]["channel"] for t in turns})},
            artifacts={"manual_dir": str(directory.relative_to(store.root))},
        )
