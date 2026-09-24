"""Prediction endpoints and the frozen Tox21 task order.

The three endpoints are scientifically distinct and are never merged into a
single verdict:

* ``clintox``  — clinical-trial toxicity signal (ClinTox-trained model)
* ``herg``     — hERG channel blockade / cardiotoxicity liability
* ``tox21``    — 12 Tox21 assay activities

The task order below is the order baked into the serving checkpoint
(``task_names`` inside ``models/pretrained_2head_herg_chemberta_model/best_model.pt``).
It is a versioned constant, never derived from dict iteration or from a dataset
loaded at runtime: the model's output vector is positional, so a reordering here
would silently relabel every assay.
"""
from __future__ import annotations

from enum import Enum
from typing import Final


class Endpoint(str, Enum):
    CLINTOX = "clintox"
    HERG = "herg"
    TOX21 = "tox21"


TOX21_TASK_ORDER_VERSION: Final[str] = "tox21-12task-v1"

TOX21_TASKS: Final[tuple[str, ...]] = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)

TOX21_TASK_INDEX: Final[dict[str, int]] = {t: i for i, t in enumerate(TOX21_TASKS)}


def validate_task_order(task_names: list[str] | tuple[str, ...]) -> None:
    """Raise if a checkpoint's task order disagrees with the frozen constant.

    Called at model load. A mismatch means the artifact and this code disagree
    about which column is which assay, which cannot be resolved at runtime.
    """
    actual = tuple(task_names)
    if actual != TOX21_TASKS:
        raise ValueError(
            "Tox21 task order mismatch between artifact and "
            f"{TOX21_TASK_ORDER_VERSION}.\n"
            f"  artifact: {actual}\n"
            f"  expected: {TOX21_TASKS}"
        )
