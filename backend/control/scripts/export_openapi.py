#!/usr/bin/env python3
"""Write the control plane's OpenAPI document for the frontend to type against.

    python backend/control/scripts/export_openapi.py          # write
    python backend/control/scripts/export_openapi.py --check  # fail if stale

The document is built from the app factory, not from a running server, so it
needs no database, predictor or runtime. The frontend generates its API types
from the written file (``npm run openapi:types``); CI runs both steps with
``--check`` / ``git diff --exit-code`` so a response model changed here and not
regenerated there fails the build rather than the browser.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
#: backend/control/scripts/this.py -> the repository root.
REPO_ROOT = HERE.parents[3]
OUT = REPO_ROOT / "frontend" / "src" / "shared" / "api" / "openapi.json"


def document() -> str:
    from toxagent.api.app import create_app

    return json.dumps(create_app().openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = parser.parse_args(argv)
    text = document()
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(REPO_ROOT)} is stale; run backend/control/scripts/export_openapi.py")
            return 1
        print("openapi.json is current")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
