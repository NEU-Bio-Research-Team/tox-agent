"""Re-reading a withheld run's reply from the deployment, without rerunning it."""
from __future__ import annotations

import json

import httpx
import pytest

from evals.investigation.recapture import recapture

RUN = "run_" + "1" * 32
SESSION = "ses_" + "1" * 32


def _study(root):
    source = root / "study-a"
    (source / "raw").mkdir(parents=True)
    (source / "raw" / "trace.json").write_text(json.dumps({
        "session_id": SESSION, "evidence": [],
        "runs": [{"turn": 0, "answer": None}, {"turn": 1, "answer": None}],
    }))
    turns = [
        {"index": 0, "response_text": "a published answer", "meta": {"disposition": "publish", "run_id": "run_p"}},
        {"index": 1, "response_text": "", "meta": {"disposition": "partial_facts", "run_id": RUN}},
    ]
    record = {"record_id": "rec_1", "system_id": "D2_investigator", "case_id": "c1", "trial": 1,
              "status": "ok", "turns": turns, "final_text": "", "artifacts": {"trace": "raw/trace.json"}}
    (source / "records.jsonl").write_text(json.dumps(record) + "\n")
    (source / "study-manifest.json").write_text(json.dumps({"study_id": "study-a"}))
    return source


def _client(messages, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer t"
        return httpx.Response(200, json={"messages": messages, "count": len(messages)})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_withheld_turn_gets_the_text_the_reader_was_shown_and_says_so(tmp_path):
    source = _study(tmp_path)
    shown = "The answer could not be released.\nWhat the server confirmed: review."
    seen: list[str] = []
    out = recapture(source=source, derived=tmp_path / "study-a-r1", urls={"D2_investigator": "http://d"},
                    token="t", client=_client([{"role": "assistant", "parts": [
                        {"type": "text", "content": {"text": shown}},
                        {"type": "result_ref", "content": {"run_id": RUN}}]}], seen))
    assert out["amended"] == 1 and out["unread"] == 0
    assert seen == [f"http://d/v1/sessions/{SESSION}/messages?limit=500"]
    (record,) = [json.loads(l) for l in (tmp_path / "study-a-r1" / "records.jsonl").read_text().splitlines()]
    assert record["turns"][0]["response_text"] == "a published answer"
    assert record["turns"][1]["response_text"] == shown
    assert record["turns"][1]["meta"]["recaptured"]["adapter"] == "toxagent-replay-adapter-v2"
    assert record["final_text"] == shown
    manifest = json.loads((tmp_path / "study-a-r1" / "study-manifest.json").read_text())
    assert manifest["derived_from"]["agents_rerun"] == 0
    # The source study is untouched.
    assert json.loads((source / "records.jsonl").read_text())["turns"][1]["response_text"] == ""


def test_a_turn_no_message_names_is_left_and_listed(tmp_path):
    source = _study(tmp_path)
    out = recapture(source=source, derived=tmp_path / "d", urls={"D2_investigator": "http://d"},
                    token="t", client=_client([], []))
    assert out == {"derived": str(tmp_path / "d"), "amended": 0, "unread": 1}


def test_a_derived_study_is_never_overwritten(tmp_path):
    source = _study(tmp_path)
    (tmp_path / "d").mkdir()
    with pytest.raises(SystemExit):
        recapture(source=source, derived=tmp_path / "d", urls={}, token="t", client=_client([], []))
