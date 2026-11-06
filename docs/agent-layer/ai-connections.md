# Users' own AI connections

Written for engineers and operators of this repository.

A user can run the agent on their own AI instead of the deployment's: an API
key (OpenAI, Claude, OpenRouter, DeepSeek, or any OpenAI-compatible server) or
a ChatGPT Plus/Pro sign-in. The run then bills that user's account, reaches the
endpoint they chose, and is recorded as such in the run's runtime binding.

The design follows G1_Web_Design's AI connections (encrypted per-user
credentials, one model picker fed by each provider's own model list, ChatGPT
sign-in through OpenCode), adapted to how this control plane runs OpenCode.

## Why a run needs its own runtime host

OpenCode 1.17.11 takes `{providerID, modelID}` per prompt and reads the
credential from a store that is global to its process: a `PUT /auth/openai` is
visible from every project directory of that process (checked live). A shared
runtime host can therefore only bill the account it was set up with. Until
this change the adapter refused every connection that carried a credential
(I12) rather than spend the wrong account.

`harness/owner_hosts.py` starts a private OpenCode process per connection:

| Property | How |
|---|---|
| Isolation | `HOME` and every XDG directory inside the host's own 0700 root; the checked-in deny-all agent profile (`OPENCODE_CONFIG`); `serve --pure` on 127.0.0.1 |
| Management API | A random `OPENCODE_SERVER_PASSWORD`; a request without it gets 401 |
| Credential, catalog provider | `PUT /auth/{provider}`: `{"type":"api","key":…}`, or the ChatGPT OAuth token pair |
| Credential, custom endpoint | `options.apiKey = {env:TOXAGENT_PROVIDER_KEY}` on a custom OpenAI-compatible provider. OpenCode does **not** read `/auth` for a custom provider: the endpoint received no Authorization header when that was tried |
| Model the binary predates | Declared in `OPENCODE_CONFIG_CONTENT`, which merges over the agent profile without losing the product agents |
| Refreshed ChatGPT tokens | Re-read from the host's `auth.json` after every turn and written back under the same secret reference |
| Bounds | At most `TOXAGENT_OWNER_HOSTS_MAX` hosts; idle ones stop after `TOXAGENT_OWNER_HOSTS_IDLE_S`. A full pool refuses the run. It never falls back to the shared host |
| Clean-up | A host's root, credential file included, is deleted when it stops |

A run whose session pins a connection happens entirely on that connection's
host: session, MCP install, prompt, events and close. The shared host serves
only runs on the deployment's own runtime.

This is process isolation under the control plane's own user, not a sandbox
against a hostile OpenCode binary. For a multi-tenant deployment, run hosts in
containers (as G1 does) by giving `OpenCodeHostPool` a container `spawner`.

## ChatGPT Plus/Pro sign-in

`connections/chatgpt.py`, through OpenCode's own OAuth:

1. `POST /v1/model-connections:chatgpt-start` starts a credential-less host for
   the user and picks OpenCode's **headless** (device-code) OpenAI method by
   its label, not by index. The browser method is not used, because its
   redirect goes to a loopback port on the server.
2. The user opens `https://auth.openai.com/codex/device` and enters the code.
   Any other link is refused before it is shown.
3. A background task waits on OpenCode's callback. On approval the token pair
   is read from that host's `auth.json`, validated against OpenCode's OAuth
   schema, and the models the sign-in may use are read. The host stops right
   away, so the pair is held only in memory.
4. `POST /v1/model-connections:chatgpt-complete {model_id}` stores the pair,
   encrypted and bound to the user, as an `openai` connection with
   `auth_mode=chatgpt_subscription`.

A sign-in cannot be pasted: `POST /v1/model-connections` refuses
`chatgpt_subscription`.

Whether a deployment may offer this is the operator's decision. OpenAI
documents subscription sign-in for open-source and local tools, and asks
hosted services to register first
([token sharing](https://developers.openai.com/siwc/token-sharing-open-source)).

| `TOXAGENT_CHATGPT_OAUTH` | Offered when |
|---|---|
| `local` (default) | Owner hosts are on **and** `TOXAGENT_EGRESS_POLICY=local` (single-user self-hosting) |
| `hosted` | Owner hosts are on, and the operator has opted in after registering |
| `off` | Never |

When it is not offered, the provider list says why. A Claude.ai Pro/Max
sign-in is never offered: Anthropic does not allow third-party apps to use a
consumer subscription for inference. Use an Anthropic API key instead.

## Storage

`TOXAGENT_SECRET_ENCRYPTION_KEY` (a Fernet key) switches the secret store to
`EncryptedSecretStore`. Each file holds a token whose authenticated plaintext
names the owner, so a file copied under another user's connection does not
decrypt into a usable credential. `./bin/toxagent setup` generates the key.
Production refuses to start without one.

A plaintext secret written before the key was set is read once more, then
re-encrypted for the owner whose connection points at it. Keep the key: if it
is lost, users re-enter their keys.

## API

All routes act for the authenticated user only. No route returns a credential.

| Route | Does |
|---|---|
| `GET /v1/model-connections:providers` | Providers, their auth modes, whether each lists models, and ChatGPT availability with its reason |
| `POST /v1/model-connections:models` | Models a key can use, before it is saved. The key is not stored |
| `POST /v1/model-connections` | Create an API-key or keyless connection |
| `GET /v1/model-connections/{id}/models` | Models the saved connection's own credential can use |
| `PATCH /v1/model-connections/{id}` | Switch to another model of the same account. Status returns to `untested` |
| `POST /v1/model-connections/{id}:test` | Probe it: Messages API for Anthropic, chat/completions otherwise, the private host for a ChatGPT sign-in |
| `DELETE /v1/model-connections/{id}` | Delete it, its secret and its host |
| `POST /v1/model-connections:chatgpt-start` · `GET` · `:chatgpt-complete` · `DELETE /v1/model-connections:chatgpt` | The sign-in above |

Model lists come from each provider's free listing endpoint. No inference is
spent to fill a picker. They are cached for 10 minutes under a digest of
provider, endpoint and key, and pass the same egress check as the probe (I15).

## Running it locally

```bash
export TOXAGENT_RUNTIME_KIND=opencode TOXAGENT_OPENCODE_URL=http://127.0.0.1:4096
export TOXAGENT_OWNER_RUNTIME_HOSTS=1            # private hosts for users' AI
export TOXAGENT_OPENCODE_BINARY=$HOME/.opencode/bin/opencode   # pinned 1.17.11
export TOXAGENT_SECRET_ENCRYPTION_KEY=...        # ./bin/toxagent setup writes one
# optional: TOXAGENT_OWNER_HOSTS_DIR, TOXAGENT_OWNER_HOSTS_MAX=2,
#           TOXAGENT_OWNER_HOSTS_IDLE_S=600, TOXAGENT_CHATGPT_OAUTH=local
```

The private hosts must be able to reach `TOXAGENT_MCP_RUNTIME_URL`, as the
shared host does. In **Settings → AI của bạn**, add a key and press *Tải danh
sách model*, or choose *ChatGPT Plus/Pro*. Then *Test* the connection and pick
it as the connection for sessions.

Each host is a full OpenCode process (~300 MB). On an 8 GB machine keep
`TOXAGENT_OWNER_HOSTS_MAX` at 2.

## Known limits

- A custom base URL on the `openai` provider must serve OpenAI's **Responses
  API**, because that is what OpenCode's OpenAI provider calls (seen live). For
  a gateway that only speaks chat/completions, use *OpenAI-compatible*.
- The Compose stack runs the control plane in a container without the OpenCode
  binary, so owner hosts are off there. They need the control plane and the
  binary on the same machine, or a container spawner.
- A ChatGPT connection's capabilities are recorded as not proven. The test
  proves that the sign-in is accepted, not what the model can do.
- Hosts live in one control-plane process. A second replica starts its own.

## Tests

| Suite | Covers |
|---|---|
| `tests/unit/test_user_ai_credentials.py` | Encryption, owner binding, plaintext migration, how each connection is presented to OpenCode |
| `tests/unit/test_owner_runtime_hosts.py` | Isolation, reuse and replacement, bounds, reaping, write-back of refreshed tokens |
| `tests/unit/test_chatgpt_sign_in.py` | Method choice, link validation, approval, expiry, cancel, availability |
| `tests/unit/test_model_catalog_and_anthropic_probe.py` | Model listing per provider, cache, egress, the Anthropic probe |
| `tests/contract/test_opencode_v1_owner_hosts.py` | A pinned run happens entirely on its own host |
| `tests/integration/test_user_ai_connections_api.py` | The routes end to end |
| `tests/contract/test_owner_hosts_live.py` | `TOXAGENT_LIVE_OPENCODE=1`: the real binary forwards each connection's key to its endpoint, refuses unauthenticated calls, deletes its files, and returns an OpenAI device link |
