"""SQLAlchemy Core repositories.

Each takes the connection of the enclosing unit of work, so everything a
workflow touches — including the events it emits — lands in one transaction.
None of these classes opens a transaction of its own.

SQLAlchemy stores, one module per aggregate.

Each store implements its protocol in ``persistence/interfaces.py`` over one
connection; ``database.UnitOfWork`` constructs them. Every store is re-exported
here, so ``from .repositories import SqlRunStore`` keeps working.
"""
from __future__ import annotations

from .conversation import (  # noqa: F401
    SqlSessionSettingsStore,
    SqlSessionStore,
    SqlMessageStore,
    SqlAnswerStore,
    SqlAttachmentStore,
)
from .runs import (  # noqa: F401
    SqlRunConfigurationSnapshotStore,
    SqlRunStore,
    SqlRunJobStore,
    SqlConcurrencySlotStore,
    SqlRuntimeBindingStore,
    SqlRuntimeUsageStore,
    SqlToolCallStore,
    SqlCapabilityTokenStore,
)
from .investigation import (  # noqa: F401
    SqlInvestigationStore,
    SqlDecisionStateStore,
    SqlScientificCaseStore,
    SqlDevelopmentPostureStore,
    SqlSkillDraftStore,
)
from .connections import (  # noqa: F401
    SqlModelConnectionStore,
)
from .analyses import (  # noqa: F401
    SqlAnalysisStore,
    SqlObservationStore,
    SqlExplanationCheckpointStore,
)
from .evidence import (  # noqa: F401
    SqlEvidenceStore,
    SqlEvidenceRelationStore,
)
from .reports import (  # noqa: F401
    SqlReportStore,
)
from ._common import (  # noqa: F401
    _parse_ts,
)
