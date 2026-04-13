"""Task configuration helpers for toxicity datasets."""

from dataclasses import dataclass, replace
from typing import List


TOX21_TASKS: List[str] = [
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
]

CLINTOX_TASKS: List[str] = ["CT_TOX"]


@dataclass
class TaskConfig:
    """Metadata describing a dataset's task layout and training defaults."""

    name: str
    task_names: List[str]
    primary_metric: str
    loss_type: str

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)

    @property
    def is_multitask(self) -> bool:
        return self.num_tasks > 1


CLINTOX_CONFIG = TaskConfig(
    name="clintox",
    task_names=CLINTOX_TASKS,
    primary_metric="f1",
    loss_type="focal",
)

TOX21_CONFIG = TaskConfig(
    name="tox21",
    task_names=TOX21_TASKS,
    primary_metric="mean_auc_roc",
    loss_type="masked_focal",
)


def get_task_config(dataset_name: str, loss_type: str = None) -> TaskConfig:
    """Return a TaskConfig for a supported dataset."""

    dataset_key = dataset_name.strip().lower()
    registry = {"clintox": CLINTOX_CONFIG, "tox21": TOX21_CONFIG}
    if dataset_key not in registry:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Available: {sorted(registry.keys())}"
        )

    config = registry[dataset_key]
    if loss_type is not None and loss_type != config.loss_type:
        config = replace(config, loss_type=loss_type)
    return config
