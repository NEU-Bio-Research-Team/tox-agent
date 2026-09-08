#!/usr/bin/env python3
"""Expose a loopback-only OpenCode server to Docker's private bridge only."""
from __future__ import annotations

import argparse
import socket
import threading


def relay(source: socket.socket, target: socket.socket) -> None:
    try:
        while data := source.recv(65_536):
            target.sendall(data)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def serve(client: socket.socket, upstream_host: str, upstream_port: int) -> None:
    try:
        upstream = socket.create_connection((upstream_host, upstream_port), timeout=10)
        threading.Thread(target=relay, args=(client, upstream), daemon=True).start()
        relay(upstream, client)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=4096)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=4096)
    args = parser.parse_args()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.port))
    listener.listen(32)
    while True:
        client, _ = listener.accept()
        threading.Thread(
            target=serve,
            args=(client, args.upstream_host, args.upstream_port),
            daemon=True,
        ).start()


if __name__ == "__main__":
    main()
