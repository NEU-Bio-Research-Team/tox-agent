from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from agents import ADK_AVAILABLE, root_agent


async def _run(args: argparse.Namespace) -> int:
    if not ADK_AVAILABLE:
        print("ADK_AVAILABLE: False")
        print("google-adk is not installed/importable in current environment.")
        return 1

    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part
    except Exception as exc:
        print(f"Failed to import ADK runtime: {type(exc).__name__}: {exc}")
        return 1

    print("ADK_AVAILABLE: True")
    print(f"Runner agent root: {getattr(root_agent, 'name', type(root_agent).__name__)}")

    session_service = InMemorySessionService()
    app_name = args.app_name
    user_id = args.user_id
    session_id = args.session_id or f"agent_live_{uuid.uuid4().hex[:10]}"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"smiles_input": args.smiles},
    )

    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name=app_name,
    )

    user_message = Content(
        role="user",
        parts=[Part(text=f"Analyze toxicity for SMILES: {args.smiles}")],
    )

    print("\n=== Agent Events ===")
    event_count = 0
    tool_call_count = 0
    tool_calls: List[Dict[str, Any]] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        event_count += 1
        author = getattr(event, "author", "unknown")
        event_type = getattr(event, "type", "unknown")

        getter = getattr(event, "get_function_calls", None)
        if callable(getter):
            try:
                calls = getter() or []
            except Exception:
                calls = []
        else:
            calls = []

        for call in calls:
            tool_call_count += 1
            name = getattr(call, "name", None)
            args_payload = getattr(call, "args", None)
            call_record = {
                "event": event_count,
                "author": author,
                "name": name,
                "args": args_payload,
            }
            tool_calls.append(call_record)
            print(f"[TOOL CALL #{tool_call_count}] {name}({args_payload})")

        is_final = False
        final_checker = getattr(event, "is_final_response", None)
        if callable(final_checker):
            try:
                is_final = bool(final_checker())
            except Exception:
                is_final = False

        content = getattr(event, "content", None)
        text_preview = None
        parts = getattr(content, "parts", None) if content is not None else None
        if isinstance(parts, list):
            text_chunks = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    text_chunks.append(str(text))
            if text_chunks:
                text_preview = " ".join(text_chunks)[:220]

        if text_preview:
            print(f"[EVENT #{event_count}] type={event_type} author={author} text={text_preview}")
        if is_final:
            print(f"[FINAL EVENT #{event_count}] type={event_type} author={author}")

    final_session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    state = getattr(final_session, "state", {}) if final_session is not None else {}
    if not isinstance(state, dict):
        state = {}

    print("\n=== Summary ===")
    print(json.dumps(
        {
            "event_count": event_count,
            "tool_call_count": tool_call_count,
            "session_id": session_id,
            "state_keys": sorted(state.keys()),
            "validation_status": state.get("validation_status")
            or (state.get("validation_result") or {}).get("validation_status")
            if isinstance(state.get("validation_result"), dict)
            else state.get("validation_status"),
            "has_final_report": isinstance(state.get("final_report"), dict),
        },
        indent=2,
        ensure_ascii=False,
    ))

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "tool_calls": tool_calls,
            "state": state,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved live ADK trace to: {output_path}")

    return 0 if tool_call_count > 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live ADK function-calling test for ToxAgent.")
    parser.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O", help="SMILES to analyze")
    parser.add_argument("--app-name", default="tox-agent-test", help="ADK app name")
    parser.add_argument("--user-id", default="test_user", help="ADK user id")
    parser.add_argument("--session-id", default="", help="Optional fixed session id")
    parser.add_argument("--save-json", default="", help="Optional file path to save trace JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = asyncio.run(_run(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
