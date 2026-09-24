"""``python -m toxagent.worker``: execute runs without serving HTTP (WS08 / PR-14).

P1-7 of the 2026-09-13 audit: run execution lived inside the FastAPI process.
A web replica restart cancelled whatever agent turns it was holding, the web
tier could not be scaled without scaling execution with it, and a slow report
occupied the same event loop that answered health checks.

A worker is the same composition as the API — every handler, gateway, tool
registry and runtime adapter a run might reach — started through the app's own
lifespan and then simply not bound to a port. Reusing the lifespan rather than
writing a second composition root is deliberate: two object graphs that must
stay in step are two places for a handler to be registered in one and missing
from the other, which is I01's failure in a new shape.

It refuses to start unless ``external_worker_mode`` is on. With it off the API
processes execute their own runs, and a worker process beside them would not be
wrong — the lease still gives each run one owner — but it would be a topology
nobody chose.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import replace

from . import metrics
from .api.app import create_app
from .config import Settings
from .flags import is_enabled

log = logging.getLogger("toxagent.worker")


async def run_worker(
    settings: Settings | None = None,
    *,
    stop: asyncio.Event | None = None,
    **app_overrides,
) -> None:
    settings = settings or Settings.from_env()
    if not is_enabled("external_worker_mode"):
        raise RuntimeError(
            "toxagent.worker needs TOXAGENT_FLAG_EXTERNAL_WORKER_MODE=1; with it off the "
            "API processes execute their own runs"
        )
    if settings.worker.role == "api":
        raise RuntimeError("TOXAGENT_PROCESS_ROLE=api cannot run a worker")
    settings = replace(settings, worker=replace(settings.worker, role="worker"))
    app = create_app(settings, **app_overrides)
    stop = stop or asyncio.Event()
    listener = (
        await serve_metrics(settings.worker.metrics_port)
        if settings.worker.metrics_port else None
    )
    try:
        await _run(app, settings, stop)
    finally:
        if listener is not None:
            listener.close()
            await listener.wait_closed()


async def serve_metrics(port: int, host: str = "0.0.0.0") -> asyncio.base_events.Server:
    """A minimal `GET /metrics` listener for a process with no web framework.

    One request, one response, connection closed. Anything but `GET /metrics`
    is a 404. It carries only what the API's own `/metrics` route carries, and
    every value there has passed the registry's label guard.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            while (await asyncio.wait_for(reader.readline(), timeout=5)) not in (b"\r\n", b"\n", b""):
                pass
            if request_line.split(b" ")[:2] == [b"GET", b"/metrics"]:
                body = metrics.REGISTRY.render().encode("utf-8")
                head = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
            else:
                body = b"not found\n"
                head = b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n"
            writer.write(head + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port)


async def _run(app, settings: Settings, stop: asyncio.Event) -> None:
    async with app.router.lifespan_context(app):
        scheduler = app.state.scheduler
        log.info(
            "worker %s claiming from %s (max in flight %d)",
            scheduler.worker_id,
            ",".join(scheduler.queues or ()),
            settings.worker.max_in_flight,
        )
        await stop.wait()
        log.info("worker %s stopping; %d run(s) in flight", scheduler.worker_id, scheduler.in_flight)


def main() -> None:
    loop = asyncio.new_event_loop()
    stop = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    try:
        loop.run_until_complete(run_worker(stop=stop))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
