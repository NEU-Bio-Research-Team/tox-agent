# Run the back end on this machine and serve it to the hosted frontend

The frontend is static and lives on Firebase Hosting (project `tox-agent`).
Everything else runs here: the control plane, the predictor, OCR, PostgreSQL
and the agent runtime. A tunnel gives the control plane's API a public HTTPS
address, and the browser calls that address directly.

```
browser ── https://tox-agent.web.app ──────────► Firebase Hosting (static files)
   │
   └───── https://<api-host>/v1/... ──► tunnel ──► 127.0.0.1:18080  API gateway
                                                        └► control plane ─► predictor, OCR, PostgreSQL
```

Day-to-day operation of the stack itself (backup, restore, logs) is in
[operations.md](operations.md). This page covers only what is different when
the stack is public.

## 1. Configure

Run `./bin/toxagent setup` once if this machine has never run the stack. Then
add to `.env`:

```
# Who may sign in: accounts of the Firebase project.
TOXAGENT_OIDC_ISSUER=https://securetoken.google.com/tox-agent
TOXAGENT_OIDC_AUDIENCE=tox-agent
TOXAGENT_OIDC_JWKS_URL=https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com

# Which sites may call the API from a browser.
TOXAGENT_CORS_ALLOW_ORIGINS=https://tox-agent.web.app,https://tox-agent.firebaseapp.com
```

The static token `setup` generated stays in `.env` for local development. The
public stack ignores it: it runs as production, which refuses static tokens.

## 2. Start

```
./bin/toxagent up --public
```

This starts the user-AI stack as production and an API gateway on
`127.0.0.1:18080`. The command fails unless `/health/ready` reports agent mode
and `/metrics`, `/internal/mcp`, `/docs` and `/openapi.json` answer 404 through
the gateway. `down`, `status` and `logs` take `--public` too.

The gateway forwards `/v1/` and the two health probes, nothing else. Point the
tunnel at the gateway port, never at the control plane.

## 3. Give the gateway a public address

The address must stay the same across restarts, because it is compiled into
the frontend and listed nowhere else. It must also carry a long-lived
streaming response: session events are server-sent events. Cloudflare's quick
tunnels (`trycloudflare.com`) meet neither condition and are not an option.

| | Tailscale Funnel | Cloudflare Tunnel (named) |
|---|---|---|
| Address | `https://<machine>.<tailnet>.ts.net`, fixed | a hostname on a domain you own |
| Cost | free | the domain |
| Needs | a Tailscale account | a domain whose DNS is on Cloudflare |
| Limits | bandwidth cap, not published; ports 443, 8443, 10000 | none that matter here |

**Tailscale Funnel**, when there is no domain:

```
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                 # opens a browser link to sign in, once
sudo tailscale funnel --bg 18080  # prints the public https address
```

`--bg` keeps the funnel across restarts of `tailscaled`. Under WSL, `tailscaled`
needs systemd enabled (`/etc/wsl.conf`, `[boot] systemd=true`) to start on its
own.

**Cloudflare Tunnel**, when there is a domain on Cloudflare:

```
cloudflared tunnel login
cloudflared tunnel create toxagent-api
cloudflared tunnel route dns toxagent-api api.<your-domain>
cloudflared tunnel run --url http://127.0.0.1:18080 toxagent-api
```

Run it as a service (`cloudflared service install`) so it survives a reboot.
The tunnel's credential file stays outside the repository.

Check from another network before going further:

```
curl -s https://<api-host>/health/ready        # JSON, mode agent_enabled
curl -s -o /dev/null -w '%{http_code}\n' https://<api-host>/metrics   # 404
```

## 4. Point the frontend at it

Set two repository variables (Settings → Secrets and variables → Actions →
Variables):

- `TOXAGENT_API_ORIGIN`: `https://<api-host>`, no trailing slash.
- `FIREBASE_WEB_API_KEY`: the project's web API key.

The `Frontend hosting` workflow builds with them and deploys to Firebase
Hosting on every push to `main` that touches the frontend. Until
`TOXAGENT_API_ORIGIN` is set it builds and tests but does not deploy, so the
live site is never replaced by a frontend with no back end to call.

Changing the API address means changing the variable and re-running the
workflow: the address is fixed at build time.

## 5. Keep it running

- The machine must not sleep. On Windows with WSL, disable sleep on AC power
  and keep WSL alive (a WSL instance stops when its last process exits; the
  Docker daemon counts).
- The public stack is its own Compose project (`toxagent-public`), with its
  own volumes. Back up with `./bin/toxagent backup --public` (plain `backup`
  dumps the development stack's database) and copy the
  `toxagent-public_toxagent-state` volume with it. The database refers to files in that volume; one without the other
  restores dangling references.
- Losing `TOXAGENT_SECRET_ENCRYPTION_KEY` makes every stored AI credential
  unreadable. Keep a copy of `.env` somewhere other than this disk.

## Signing in with Google

The sign-in page offers "Tiếp tục với Google" next to email and password. It
opens the Firebase project's own handler in a popup, so it needs no client id
in this repository and works on the project's `web.app` and `firebaseapp.com`
addresses and on preview channels, which Firebase lists as authorized domains.

One step is not in the repository: switch the Google provider on in the
Firebase console (Authentication → Sign-in method → Google → Enable, with a
support email). The hosting workflow asks Firebase whether the provider is on
and ships the button only when it is, so run the workflow again after enabling
it. A build made by hand shows the button whenever `VITE_FIREBASE_PROJECT_ID`
is set, and the button then says so if the provider is still off. A Google account and a password account with the same email are one
person to Firebase by default, so signing in either way reaches the same data.

## What is not covered

- Roles. A Firebase account has no `roles` claim unless one is set with the
  Admin SDK, so everyone signs in as an ordinary user and expert threshold
  overrides are refused.
- ChatGPT Plus/Pro sign-in is off unless the operator turns it on. Set
  `TOXAGENT_PUBLIC_CHATGPT_OAUTH=hosted` in `.env` and run
  `./bin/toxagent up --public` again; each user then signs in with their own
  account (a device code entered on OpenAI's page) and their credential is
  kept encrypted for them alone. OpenAI asks hosted services that offer this
  to register with it first
  ([token sharing](https://developers.openai.com/siwc/token-sharing-open-source)),
  and whether to do so is the operator's call. Each signed-in user holds a
  private OpenCode process of about 300 MB while they chat, capped by
  `TOXAGENT_OWNER_HOSTS_MAX` (2 by default); a third user's run is refused
  until one goes idle, so raise it only as far as the machine's memory allows.
  Without the setting, users connect an API key or an OpenAI-compatible server.
