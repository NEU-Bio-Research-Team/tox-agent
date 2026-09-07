# Getting started

From a clean clone, run the five commands in the root README. `setup` creates
`.env`, generates local credentials, downloads the pinned OCR checkpoint into
the ignored `.artifacts/` directory and verifies its SHA-256. It is the only
network download initiated by the wrapper outside Docker image builds.

`up` waits for Compose health checks. If it fails, run `./bin/toxagent logs`
and use the named service in the error. A successful `smoke` proves the browser
route, control plane and predictor route are connected. It intentionally uses
a small SMILES request and does not invoke a model-heavy benchmark.

The token printed by setup is a development-only local token. Paste it into the
frontend gate; do not send it to other users or deploy it to a public host.
