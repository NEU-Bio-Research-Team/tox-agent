"""Private OpenCode hosts, one per AI connection, so a run uses the user's own AI.

I12/K05: OpenCode V1 takes ``{providerID, modelID}`` per prompt and reads the
credential from its *process-wide* auth store (checked live on 1.17.11: a
``PUT /auth/openai`` is visible from every project directory of that process).
A shared runtime host can therefore only ever bill the account it was set up
with, which is why the V1 adapter refused every connection that carried a
credential.

G1_Web_Design solved the same problem by giving each user a private OpenCode
process whose credential lives only in that process's own data directory.
This module does that for ToxAgent:

- A host is started for one connection (one owner, one credential) with
  ``HOME`` and every XDG directory inside its own 0700 root, the checked-in
  deny-all agent profile (``OPENCODE_CONFIG``), ``serve --pure`` on loopback,
  and a random server password — the same isolation ``devops/agent-runtime``
  uses for the shared host, plus authentication on the management API.
- The credential is installed the way OpenCode itself installs one:
  ``PUT /auth/{provider}`` for a provider in its catalog (API key or ChatGPT
  OAuth token pair), or a custom OpenAI-compatible provider whose key arrives
  through an environment variable (``{env:...}``) — both verified live
  against the pinned binary before this was written.
- After every turn the host's auth file is re-read. A ChatGPT token pair the
  runtime refreshed is written back through ``credential_sink``; otherwise the
  next run would start from a refresh token the provider already rotated away.
- Hosts are bounded (``max_hosts``) and reaped after ``idle_ttl_s`` without a
  lease. A busy pool refuses a run rather than queueing behind it or, worse,
  falling back to the shared host's account.

What this is not: a sandbox against a hostile OpenCode binary. The process
runs as the control plane's user. For a multi-tenant deployment, run hosts in
containers (as G1 does) and point ``spawner`` at that.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..connections.runtime_install import (
    OAUTH_FIELDS,
    HostCredential,
    RuntimeInstall,
    install_for,
    parse_oauth,
)
from ..domain.errors import RuntimeUnavailable

log = logging.getLogger("toxagent.harness.owner_hosts")


@dataclass
class OwnerHost:
    key: str
    root: Path
    port: int
    password: str = field(repr=False)
    process: Any
    client: httpx.AsyncClient
    runtime_provider_id: str
    credential: HostCredential | None
    fingerprint: str
    leases: int = 0
    last_used: float = field(default_factory=time.monotonic)

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def auth_file(self) -> Path:
        return self.root / "data" / "opencode" / "auth.json"

    def read_auth(self, provider_id: str) -> dict[str, Any] | None:
        try:
            value = json.loads(self.auth_file.read_text(encoding="utf-8")).get(provider_id)
        except (OSError, ValueError, AttributeError):
            return None
        return value if isinstance(value, dict) else None

    def alive(self) -> bool:
        return self.process.returncode is None


#: ``(argv, env, cwd, log_path) -> process`` — injectable so tests need no binary.
Spawner = Callable[[list[str], dict[str, str], Path, Path], Any]


async def _default_spawner(argv: list[str], env: dict[str, str], cwd: Path, log_path: Path):
    handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 - owned by the process
    try:
        return await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            cwd=str(cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
        )
    finally:
        handle.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OpenCodeHostPool:
    def __init__(
        self,
        *,
        binary: str,
        profile_path: Path,
        root: Path,
        required_agents: tuple[str, ...] = ("toxagent",),
        max_hosts: int = 2,
        idle_ttl_s: float = 600.0,
        startup_timeout_s: float = 45.0,
        request_timeout_s: float = 10.0,
        spawner: Spawner | None = None,
        client_factory: Callable[[str, str], httpx.AsyncClient] | None = None,
        port_picker: Callable[[], int] = _free_port,
    ) -> None:
        self._binary = binary
        self._profile = Path(profile_path)
        self._root = Path(root)
        self._required_agents = required_agents
        self._max_hosts = max(1, max_hosts)
        self._idle_ttl_s = idle_ttl_s
        self._startup_timeout_s = startup_timeout_s
        self._request_timeout_s = request_timeout_s
        self._spawner = spawner or _default_spawner
        self._client_factory = client_factory or self._make_client
        self._port_picker = port_picker
        self._hosts: dict[str, OwnerHost] = {}
        self._lock = asyncio.Lock()
        #: ``(secret_ref, owner_id, secret) -> None``; set by the app so a
        #: refreshed OAuth pair is persisted under the same reference.
        self.credential_sink: Callable[[str, str, str], None] | None = None

    def _make_client(self, base_url: str, password: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            auth=("opencode", password),
            timeout=httpx.Timeout(self._request_timeout_s),
            trust_env=False,
        )

    @property
    def size(self) -> int:
        return len(self._hosts)

    def _host_root(self, key: str) -> Path:
        return self._root / hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    async def acquire(self, credential: HostCredential) -> OwnerHost:
        """A running host for this connection, with one lease taken on it."""
        install = install_for(credential)
        key = "conn:" + credential.connection_id
        async with self._lock:
            host = self._hosts.get(key)
            if host is not None and (
                not host.alive() or host.fingerprint != credential.fingerprint()
            ):
                if host.leases:
                    raise RuntimeUnavailable(
                        "this AI connection changed while one of its runs is still active; "
                        "retry when that run ends"
                    )
                await self._stop(host)
                host = None
            if host is None:
                host = await self._start(key, install, credential)
            host.leases += 1
            host.last_used = time.monotonic()
            return host

    async def enrollment_host(self, owner_id: str) -> OwnerHost:
        """A credential-less host for one owner's ChatGPT sign-in, leased once."""
        key = "enroll:" + owner_id
        async with self._lock:
            existing = self._hosts.get(key)
            if existing is not None:
                await self._stop(existing)
            install = RuntimeInstall(runtime_provider_id="openai", config={})
            host = await self._start(key, install, None)
            host.leases += 1
            return host

    async def release(self, host: OwnerHost) -> None:
        """Return a lease; persist a credential the runtime refreshed."""
        self.persist_rotation(host)
        async with self._lock:
            host.leases = max(0, host.leases - 1)
            host.last_used = time.monotonic()
        await self.reap()

    async def discard(self, host: OwnerHost) -> None:
        async with self._lock:
            await self._stop(host)

    def persist_rotation(self, host: OwnerHost) -> bool:
        credential = host.credential
        if credential is None or credential.auth_mode != "chatgpt_subscription":
            return False
        current = host.read_auth(host.runtime_provider_id)
        if not current or current.get("type") != "oauth":
            return False
        try:
            stored = parse_oauth(credential.secret)
        except RuntimeUnavailable:
            stored = {}
        if current == stored:
            return False
        refreshed = json.dumps(
            {k: v for k, v in current.items() if k in OAUTH_FIELDS}, separators=(",", ":")
        )
        if credential.secret_ref and self.credential_sink is not None:
            try:
                self.credential_sink(credential.secret_ref, credential.owner_id, refreshed)
            except Exception:  # noqa: BLE001 - never fail a finished turn over it
                log.exception("could not persist a refreshed ChatGPT credential")
                return False
        # The host keeps serving under the refreshed pair; record it as current
        # so the next acquire does not restart a host that is already right.
        host.credential = HostCredential(
            connection_id=credential.connection_id,
            owner_id=credential.owner_id,
            provider_id=credential.provider_id,
            model_id=credential.model_id,
            auth_mode=credential.auth_mode,
            base_url=credential.base_url,
            secret=refreshed,
            secret_ref=credential.secret_ref,
        )
        host.fingerprint = host.credential.fingerprint()
        return True

    async def reap(self) -> int:
        """Stop hosts idle past the TTL, and any whose process has exited."""
        now = time.monotonic()
        async with self._lock:
            stale = [
                host
                for host in self._hosts.values()
                if host.leases == 0
                and (not host.alive() or now - host.last_used >= self._idle_ttl_s)
            ]
            for host in stale:
                await self._stop(host)
        return len(stale)

    async def forget_connection(self, connection_id: str) -> None:
        """Stop a deleted connection's host so its credential leaves memory and disk."""
        async with self._lock:
            host = self._hosts.get("conn:" + connection_id)
            if host is not None and host.leases == 0:
                await self._stop(host)

    async def aclose(self) -> None:
        async with self._lock:
            for host in list(self._hosts.values()):
                await self._stop(host)

    async def _make_room(self) -> None:
        if len(self._hosts) < self._max_hosts:
            return
        idle = sorted(
            (host for host in self._hosts.values() if host.leases == 0),
            key=lambda host: host.last_used,
        )
        while idle and len(self._hosts) >= self._max_hosts:
            await self._stop(idle.pop(0))
        if len(self._hosts) >= self._max_hosts:
            raise RuntimeUnavailable(
                "every private AI runtime host is busy with another run; retry shortly",
                max_hosts=self._max_hosts,
            )

    async def _start(
        self, key: str, install: RuntimeInstall, credential: HostCredential | None
    ) -> OwnerHost:
        await self._make_room()
        if not self._profile.is_file():
            raise RuntimeUnavailable("the OpenCode agent profile is missing on this deployment")
        root = self._host_root(key)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        for name in ("home", "data", "config", "state", "cache", "workspace"):
            (root / name).mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        password = secrets.token_urlsafe(32)
        port = self._port_picker()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root / "home"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_STATE_HOME": str(root / "state"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "OPENCODE_CONFIG": str(self._profile),
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                {"autoupdate": False, "share": "disabled", **install.config}
            ),
            "OPENCODE_SERVER_PASSWORD": password,
            "TERM": "xterm",
            **install.env,
        }
        argv = [self._binary, "serve", "--pure", "--hostname", "127.0.0.1", "--port", str(port)]
        try:
            process = await self._spawner(argv, env, root / "workspace", root / "opencode.log")
        except OSError as exc:
            shutil.rmtree(root, ignore_errors=True)
            raise RuntimeUnavailable("could not start a private OpenCode runtime host") from exc
        client = self._client_factory(f"http://127.0.0.1:{port}", password)
        host = OwnerHost(
            key=key,
            root=root,
            port=port,
            password=password,
            process=process,
            client=client,
            runtime_provider_id=install.runtime_provider_id,
            credential=credential,
            fingerprint=credential.fingerprint() if credential else "",
        )
        try:
            await self._wait_ready(host)
            if install.auth is not None:
                response = await client.put(
                    f"/auth/{install.runtime_provider_id}", json=install.auth
                )
                if response.status_code != 200 or response.json() is not True:
                    raise RuntimeUnavailable("the private runtime host refused the credential")
        except BaseException:
            await self._terminate(host)
            raise
        self._hosts[key] = host
        return host

    async def _wait_ready(self, host: OwnerHost) -> None:
        deadline = time.monotonic() + self._startup_timeout_s
        directory = str(host.workspace)
        while True:
            if not host.alive():
                raise RuntimeUnavailable("the private OpenCode runtime host exited during start-up")
            try:
                response = await host.client.get("/agent", params={"directory": directory})
                if response.status_code == 200:
                    names = {item.get("name") for item in response.json() if isinstance(item, dict)}
                    missing = [name for name in self._required_agents if name not in names]
                    if missing:
                        raise RuntimeUnavailable(
                            "the private runtime host does not expose the product agent",
                            missing_agents=missing,
                        )
                    return
            except (httpx.HTTPError, ValueError, TypeError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeUnavailable("the private OpenCode runtime host did not become ready")
            await asyncio.sleep(0.25)

    async def _stop(self, host: OwnerHost) -> None:
        self._hosts.pop(host.key, None)
        self.persist_rotation(host)
        await self._terminate(host)

    async def _terminate(self, host: OwnerHost) -> None:
        try:
            await host.client.aclose()
        except Exception:  # noqa: BLE001
            pass
        process = host.process
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, AttributeError, TypeError):
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, AttributeError, TypeError):
                    process.kill()
                await process.wait()
        # The credential file goes with the host.
        shutil.rmtree(host.root, ignore_errors=True)
