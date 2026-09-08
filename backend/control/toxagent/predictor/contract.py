"""Constants pinned from the ToxPred contract.

These mirror values the predictor owns. They are duplicated here deliberately
rather than imported: importing would couple the two deployables, and a shared
constant that silently follows the predictor is exactly what would let a task
reordering pass unnoticed. Duplicated, a drift is a loud
``predictor_protocol_error`` at the boundary, which is what SCI-01 and SCI-05
require.
"""
from __future__ import annotations

from typing import Final

ENDPOINTS: Final[tuple[str, ...]] = ("clintox", "herg", "tox21")

TOX21_TASK_ORDER_VERSION: Final = "tox21-12task-v1"

TOX21_TASKS: Final[tuple[str, ...]] = (
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
)

#: The probability field each endpoint publishes. There is exactly one per
#: endpoint and they are never interchangeable: a hERG blockade probability
#: serialised under a clinical key is the specific bug the predictor rewrite
#: existed to remove (SCI-01).
PROBABILITY_FIELD: Final[dict[str, str]] = {
    "clintox": "probability_clinical_toxicity",
    "herg": "probability_blocker",
    "tox21": "probability_activity",
}

#: Labels each endpoint may emit. A validator compares against these exactly;
#: no synonym, no "safe", no "non-toxic".
LABELS: Final[dict[str, tuple[str, ...]]] = {
    "clintox": ("positive", "negative"),
    "herg": ("blocker", "non_blocker"),
}

APPLICABILITY_STATUSES: Final[tuple[str, ...]] = ("ok", "limited", "out_of_domain")

#: SCI-07: the applicability check is an element whitelist, not a learned OOD
#: detector, and the payload says so in this field.
APPLICABILITY_METHOD_PREFIX: Final = "element_rules"
