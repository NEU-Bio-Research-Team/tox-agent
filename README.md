# ToxAgent

ToxAgent is a containerized toxicity decision-support workspace: a browser
frontend, control plane, ToxPred molecular predictor, MolScribe structure OCR
and PostgreSQL. It reports separate model measurements and provenance; it does
not produce a medical or safety decision.

## Quick start

Prerequisites: Docker Engine with Docker Compose v2, `curl`, and roughly 12 GB
of free disk space for images and the one-time MolScribe checkpoint download.

```bash
git clone <customer-repo-url> toxagent
cd toxagent
./bin/toxagent setup
./bin/toxagent up
./bin/toxagent smoke
```

Open the URL printed by `up` (normally `http://localhost:8088`) and paste the
one-time local access token printed by `setup`. `setup` never overwrites
existing secrets or artifacts. The default is a CPU evaluation environment;
first startup builds images and can take time, while ordinary restart is much
faster.

## Daily operation

```bash
./bin/toxagent status
./bin/toxagent logs
./bin/toxagent down
```

Run `./bin/toxagent doctor` before a long build if a machine has changed. The
default stack deliberately has no external LLM dependency. An approved
OpenCode runtime is optional and documented in [configuration](docs/CONFIGURATION.md).

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
