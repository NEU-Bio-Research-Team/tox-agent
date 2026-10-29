"""Capture full text once from the live provider, for frozen replay (#43).

    python -m evals.investigation.capture_full_text \\
        --snapshot evals/fixtures/<fixture>.json --out <store>.json [--records MED:1,PMC:2]

Every record of a frozen evidence snapshot (or the ones named) is fetched from
Europe PMC exactly as the product fetches it, split by the same parser, and
written as a ``frozen-full-text-v1`` store with what the capture was: the
provider, the time, the sha256 of the bytes received and the parser version.
A deployment then replays it offline with
``TOXAGENT_RESEARCH_FULL_TEXT_PATH``/``_SHA256``; the sha256 printed here is
the pin.

A transient failure (an outage, a rate limit) is reported and not frozen: it
says nothing about the document, and a store must not record it as "no text".
Run the capture again for those records.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from toxagent.domain.evidence import SourceIdentifier
from toxagent.research.fulltext_store import freeze_document, write_store


async def capture(records: Mapping[str, Mapping[str, Any]], provider, *,
                  captured_at: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Freeze each record's text. Returns the documents and the record ids
    whose fetch was transient (left out, to be captured again)."""
    documents: dict[str, dict[str, Any]] = {}
    transient: list[str] = []
    for record_id, record in sorted(records.items()):
        identifier = record.get("identifier") or {}
        fetch = await provider.fetch_full_text(
            provider_record_id=record_id,
            identifier=SourceIdentifier(**{k: v for k, v in identifier.items() if v}),
        )
        if fetch.transient:
            transient.append(record_id)
            continue
        documents[record_id] = freeze_document(
            fetch, provider=provider.name, captured_at=captured_at,
        )
    return documents, transient


def main(argv: list[str] | None = None) -> int:
    from toxagent.platform.config import ResearchSettings
    from toxagent.research.providers.europepmc import EuropePmcProvider

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", default="", help="comma-separated provider record ids")
    args = parser.parse_args(argv)
    fixture = json.loads(args.snapshot.read_text())
    records = dict((fixture.get("evidence") or {}).get("records") or {})
    wanted = [r for r in args.records.split(",") if r]
    if wanted:
        unknown = sorted(set(wanted) - set(records))
        if unknown:
            raise SystemExit(f"not in the snapshot: {unknown}")
        records = {k: v for k, v in records.items() if k in wanted}

    async def run() -> tuple[dict, list[str]]:
        provider = EuropePmcProvider(ResearchSettings.from_env())
        try:
            return await capture(records, provider,
                                 captured_at=datetime.now(timezone.utc).isoformat())
        finally:
            await provider.aclose()

    documents, transient = asyncio.run(run())
    digest = write_store(documents, args.out)
    for record_id, document in sorted(documents.items()):
        print(f"{record_id}: {document['outcome']} ({len(document['passages'])} passages)")
    if transient:
        print(f"not frozen (transient, capture again): {', '.join(transient)}")
    print("sha256", digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
