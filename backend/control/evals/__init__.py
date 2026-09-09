"""ToxAgent agent-evaluation suite (plan section 16).

Not part of the shipped ``toxagent`` package. Four things live here:

* ``schema/task.schema.json`` — the ``eval-task-v1`` contract (Phase 0).
* ``tasks/`` — the initial 50-task set (plan section 16.2).
* ``fixtures/`` — frozen, content-hashed predictor/evidence bundles (plan 16.3).
* ``graders/`` and ``runner.py`` — deterministic grading and ``pass@k``.

The rubric and SME graders (plan 16.4) are out of scope for this deterministic
layer; a task that needs them lists them in ``graders`` and the runner marks
those dimensions ``deferred``.
"""
