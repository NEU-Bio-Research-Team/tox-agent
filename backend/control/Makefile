# One documented way to test this package from a fresh clone.
#
# P2-3 of the 2026-09-13 audit: there was no such target. Running the suite
# meant knowing which workstation virtualenv happened to have `aiosqlite` in
# it, which is not a thing a new contributor or a CI job can know. Every target
# here builds its own environment under `$(VENV)` and never touches an ambient
# one.
#
#   make test          backend suite + 3 scripted eval trials, from nothing
#   make unit          the fast half, for an edit loop
#   make eval          the scripted eval only
#   make docker-test   the same suite inside the image's `test` stage
#   make clean         remove the throwaway virtualenv

PYTHON ?= python3
VENV ?= .venv-test
BIN := $(VENV)/bin
STAMP := $(VENV)/.installed
DOCKER ?= docker
IMAGE ?= toxagent-control

.PHONY: test unit contract integration eval docker-test clean

$(STAMP): pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	# `dev` carries pytest, aiosqlite and trio: the suite runs on SQLite, and
	# the production image deliberately does not ship any of it.
	$(BIN)/pip install -e '.[dev]'
	@touch $(STAMP)

## Everything a pull request must pass before it is reviewed.
test: $(STAMP)
	$(BIN)/python -m pytest tests -q
	$(BIN)/python -m evals.runner --runtime scripted --trials 3

unit: $(STAMP)
	$(BIN)/python -m pytest tests/unit -q

contract: $(STAMP)
	$(BIN)/python -m pytest tests/contract -q

integration: $(STAMP)
	$(BIN)/python -m pytest tests/integration -q

eval: $(STAMP)
	$(BIN)/python -m evals.runner --runtime scripted --trials 3

## The same thing with no Python on the host at all.
docker-test:
	$(DOCKER) build --target test -f deploy/Dockerfile -t $(IMAGE):test .
	$(DOCKER) run --rm $(IMAGE):test

clean:
	rm -rf $(VENV)
