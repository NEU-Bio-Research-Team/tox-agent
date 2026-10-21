"""Frozen activity screenings for replay tests (#44): the eval builders,
so a test and a study replay the same records."""
from evals.investigation.activity_fixtures import (  # noqa: F401
    CONTEXT,
    LIGAND,
    NOW,
    frozen_screening,
    replay_file,
    screening_id,
    synthetic_set,
)
