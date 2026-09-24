# ToxAgent

ToxAgent is a containerized toxicity decision-support workspace: a browser
frontend, control plane, ToxPred molecular predictor, MolScribe structure OCR
and PostgreSQL. It reports separate model measurements and provenance; it does
not produce a medical or safety decision.

## Setup, in order

Follow these seven steps in order on a fresh clone. Each one either succeeds or
stops with a message naming what is missing — nothing here fails silently, so if
a step prints an error, fix that error before running the next step.

### Prerequisites

| Needed | Why | Check it |
|---|---|---|
| Docker Engine + Compose v2 | Every service runs in a container | `docker compose version` |
| `curl` | Downloads the pinned OCR checkpoint | `curl --version` |
| `python3` with PyYAML | `setup` and `doctor` read the predictor manifest on the host, before any container exists | `python3 -c "import yaml"` |
| `sha256sum`, `gzip`, `sed`, `grep` | Checksum verification, backup/restore | Present on any standard Linux install |
| ~12 GB free disk | Images, plus a one-time 1.1 GB MolScribe checkpoint | `df -h .` |

If `python3 -c "import yaml"` fails, install it (`sudo apt install python3-yaml`,
or `pip install pyyaml`) before continuing — `setup` cannot verify model
artifacts without it.

### 1. Clone

```bash
git clone <customer-repo-url> toxagent
cd toxagent
```

Run every command from this directory. `bin/toxagent` finds the repository root
itself, so it also works from a subdirectory, but the examples below assume the
root.

### 2. Provide the predictor model weights

**This is the step a clean clone cannot skip.** The hERG/Tox21 weights are not
in the repository — they are about 14 MB of checkpoint plus tokenizer files,
distributed separately because their redistribution rights are decided per
deployment. Without them the stack builds every image and then never becomes
ready, so `setup` refuses early instead.

Choose one:

```bash
# (a) You have an artifact bucket. Create .env from the template and point at it.
cp .env.example .env
#   then set, in .env:
#   MODEL_ARTIFACTS_URI=gs://your-bucket/toxagent/artifacts
#   or s3://your-bucket/toxagent/artifacts   (gs:// and s3:// are the supported
#   schemes; the host needs google-cloud-storage or boto3 and working credentials)

# (b) You were given the files directly. Put them here:
mkdir -p .data/models
#   copy the bundle so that this path exists:
#   .data/models/pretrained_2head_herg_chemberta_model/best_model.pt
```

The complete file list, with a SHA-256 for every file, is
[`backend/predictor/registry/models/herg-tox21-chemberta-v1.yaml`](backend/predictor/registry/models/herg-tox21-chemberta-v1.yaml).
`setup` verifies every one of those checksums; a partial or mismatched copy is
reported by name rather than accepted.

To keep the weights somewhere other than `.data/models`, set
`TOXPRED_MODELS_HOST_PATH` in `.env` instead of moving them.

### 3. Run setup

```bash
./bin/toxagent setup
```

It creates `.env` from `.env.example` if you have not already, generates the
PostgreSQL password, the capability secret and a local access token, downloads
the pinned MolScribe OCR checkpoint (about 1.1 GB, once) into `.artifacts/` and
verifies its SHA-256, then verifies the predictor artifacts from step 2.

It never overwrites an existing secret or an artifact that is already present,
so re-running it is safe and is the right response to a failed download.

**It prints the access token. Keep that terminal, or read it back later with
`grep TOXAGENT_STATIC_TOKENS .env`** — the part before the first `:`.

### 4. Check the machine

```bash
./bin/toxagent doctor
```

Docker, Compose, the Compose configuration, both artifact sets and the browser
port. Run it before a long build whenever the machine has changed. It is safe
to run against a stack that is already up.

### 5. Start the stack

```bash
./bin/toxagent up
```

The first run builds every image and can take a long time on a cold machine;
later restarts are much faster. `up` waits up to ten minutes for health checks.
If it times out on the first build, the built layers are cached — run `up`
again and it will continue rather than start over.

### 6. Prove it works

```bash
./bin/toxagent smoke
```

This checks readiness and sends one real prediction (`CCO`) through the browser
route, so a pass means the frontend, control plane and predictor are genuinely
connected. It is deliberately small; it is not a benchmark.

### 7. Open it

Open the URL `up` printed — normally `http://localhost:8088` — and paste the
access token from step 3.

That token is a development-only local credential. Do not share it and do not
put this stack on a public address with it.

### If a step fails

| What you see | What it means | What to do |
|---|---|---|
| `predictor model artifacts are missing or do not match the manifest` | Step 2 is incomplete, or a file is truncated | Re-check the paths and checksums in the manifest named above, then re-run `setup` |
| `OCR checkpoint checksum mismatch` | The 1.1 GB download was interrupted | Delete `.artifacts/toxocr/swin_base_char_aux_1m.pth` and re-run `setup` |
| `port 8088 is already in use by another process` | Something else owns the browser port | Change `FRONTEND_PORT` in `.env`, then `up` again |
| `ModuleNotFoundError: No module named 'yaml'` | Host prerequisite missing | Install PyYAML for the `python3` on your PATH |
| `up` fails, or `smoke` reports a service | A container is unhealthy | `./bin/toxagent logs <service>` using the name in the message, e.g. `./bin/toxagent logs toxagent-control toxpred` |

### What this gives you

A CPU evaluation stack: browser frontend, control plane, ToxPred predictor,
MolScribe structure OCR and PostgreSQL. It serves the **hERG** and **Tox21**
endpoints. ClinTox is deliberately shown as unavailable — see [model
admission](#model-admission) below.

There is no external LLM dependency and nothing calls a paid API. The agent
runtime, which does, is opt-in and covered under [Agent runtime](#agent-runtime-local-byoc)
below; you do not need it for prediction, batch prediction or OCR.

## Daily operation

```bash
./bin/toxagent status
./bin/toxagent logs
./bin/toxagent down
```

Run `./bin/toxagent doctor` before a long build if a machine has changed. The
default stack deliberately has no external LLM dependency. An approved
OpenCode runtime is optional and documented in [configuration](docs/CONFIGURATION.md).

### Local ports

Use the launcher rather than killing processes by port: it preserves the
PostgreSQL volume and also stops the private OpenCode bridge when applicable.

```bash
# Start / stop the standard local stack (frontend at http://localhost:8088).
./bin/toxagent up
./bin/toxagent down

# Start / stop the agent-enabled stack.
./bin/toxagent up --agent
./bin/toxagent down --agent

# Inspect ports and service state.
./bin/toxagent status
./bin/toxagent status --agent
ss -ltnp | rg ':(8088|8000|4096)\b'
```

`FRONTEND_PORT` in `.env` controls the browser port (default `8088`). The
control-plane port `8000` and OpenCode port `4096` are loopback-only; OpenCode
is additionally reachable from the Docker bridge only through the local
agent launcher, never from the LAN.

## Agent runtime (local BYOC)

Each local user connects their own provider. Credentials are stored only in
the ignored, mode-0700 `.data/opencode-auth/` directory; the ToxAgent database,
events, logs, and `.env` receive only the non-secret provider/model IDs.

```bash
# Opens the provider login in an isolated ToxAgent auth store.
./bin/toxagent setup --agent --auth login --provider openai --model gpt-5.6-luna

# Or import an auth file that you own, once.
./bin/toxagent setup --agent --auth existing --auth-file /path/to/auth.json \
  --provider openai --model gpt-5.6-luna

# Starts OpenCode, its Docker-private bridge, and the agent-enabled stack.
./bin/toxagent up --agent
./bin/toxagent status --agent
./bin/toxagent logs --agent
./bin/toxagent down --agent
```

Use `opencode models` to choose a provider/model pair available to your own
account. Provider requests consume that account's quota.

## Documentation

- [Getting started](docs/GETTING_STARTED.md)
- [Configuration](docs/CONFIGURATION.md)
- [Operations](docs/OPERATIONS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Model card](docs/MODEL_CARD.md)
- [Development](docs/DEVELOPMENT.md)

ClinTox is intentionally shown as unavailable: its release artifact lacks the
tokenizer needed for reproducible inference. hERG and Tox21 are the served
predictor capabilities in this release. See the [model card](docs/MODEL_CARD.md)
for intended use, calibration and limitations.

## Model admission

Checkpoints on disk are not automatically offered in the product. Review them
through the admission workflow; only a hash-verified, provider-compatible
manifest entry can be served.

```bash
./bin/toxagent models scan
./bin/toxagent models inspect .data/models/example/best_model.pt
./bin/toxagent models validate herg-tox21-chemberta-v1
./bin/toxagent models admit herg-tox21-chemberta-v1
```
