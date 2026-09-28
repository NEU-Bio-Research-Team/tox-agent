# Developer entry point for the whole repository. Operators use bin/toxagent;
# this file is for changing the code. Each target delegates to the service's
# own tooling, so CI and a workstation run the same commands.
#
#   make lint            ruff (blocking rules) + frontend ESLint
#   make lint-report     wider ruff rule set, report only
#   make typecheck       mypy (report only) + frontend tsc
#   make test            every suite; narrow with SERVICE=control|predictor|ocr|frontend|devops
#   make check           docs links, stray workspace roots, handoff surface
#
# Python tools come from the active environment (`pip install -e
# 'backend/control[dev]'` provides ruff and mypy). Override with RUFF=… MYPY=…

PYTHON ?= python3
RUFF ?= ruff
MYPY ?= mypy
NPM ?= npm
SERVICE ?= all

PY_DIRS := backend devops

.PHONY: lint lint-report fmt typecheck test check \
        test-control test-predictor test-ocr test-frontend test-devops

lint:
	$(RUFF) check $(PY_DIRS)
	cd frontend && $(NPM) run lint

## Import order, pyupgrade and formatting drift — what `lint` will block on next.
lint-report:
	-$(RUFF) check $(PY_DIRS) --extend-select I,UP --statistics
	-$(RUFF) format --check $(PY_DIRS) --quiet

## Format only the files you name: FILES="path/a.py path/b.py". Formatting the
## whole tree at once would bury structural moves in the diff.
fmt:
	@test -n "$(FILES)" || { echo 'usage: make fmt FILES="path/a.py ..."'; exit 2; }
	$(RUFF) format $(FILES)
	$(RUFF) check --fix $(FILES)

typecheck:
	-cd backend/control && $(MYPY) src/toxagent
	-cd backend/predictor && $(MYPY) src/toxpred
	cd frontend && $(NPM) run typecheck

ifeq ($(SERVICE),all)
test: test-control test-predictor test-ocr test-frontend test-devops
else
test: test-$(SERVICE)
endif

test-control:
	$(MAKE) -C backend/control test

test-predictor:
	cd backend/predictor && $(PYTHON) -m pytest -q

test-ocr:
	cd backend/ocr && PYTHONPATH=src $(PYTHON) -m pytest -q tests

test-frontend:
	cd frontend && $(NPM) run typecheck && $(NPM) test

test-devops:
	$(PYTHON) -m pytest -q devops/tests

check:
	$(PYTHON) devops/scripts/check_docs.py
	$(PYTHON) devops/scripts/check_workspace.py
	$(PYTHON) devops/scripts/handoff.py --check
