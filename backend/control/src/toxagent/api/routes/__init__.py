"""Product HTTP routes (plan sections 6.1-6.5).

Handlers stay thin: authenticate, parse, delegate, serialise. Ownership is
enforced in the application and the store, not here, so a new endpoint cannot
forget it. Reads are all reconstructions of committed state, which is what makes
"the stream died" a non-event.

The product HTTP API, one module per resource.

Every module registers on the shared ``router`` (``/v1``) or ``health`` from
``_common``; importing them here, in this order, is what registers the routes.
The order is the one the single ``routes.py`` used, except that ``cancel_run``
now registers with the other run routes — no earlier route can match its path.
"""
# ruff: noqa: I001  -- import order is route registration order
from __future__ import annotations

from ._common import (  # noqa: F401
    router,
    health,
    actor,
    _services,
    _isoformat,
    _decode_image,
)
from .liveness import (  # noqa: F401
    root,
    live,
    ready,
)
from .predict import (  # noqa: F401
    quick_predict,
    _quick_attributions,
    quick_predict_batch,
    _refuse_unadmitted,
    quick_predict_compare,
    effective_product,
    predict_capabilities,
    quick_recognize,
    quick_explain,
)
from .connections import (  # noqa: F401
    list_supported_providers,
    create_model_connection,
    list_model_connections,
    get_model_connection,
    test_model_connection,
    delete_model_connection,
)
from .sessions import (  # noqa: F401
    list_sessions,
    create_session,
    update_session,
    get_session,
    get_session_settings,
    update_session_settings,
    list_messages,
    send_message,
)
from .reports import (  # noqa: F401
    create_report,
    list_reports,
    get_report,
    download_report_rendering,
    get_report_figure,
    get_report_build,
    cancel_report_build,
)
from .runs import (  # noqa: F401
    get_run,
    get_decision_state,
    list_evidence_relations,
    cancel_run,
)
from .cases import (  # noqa: F401
    _case_summary,
    _owned_case,
    list_scientific_cases,
    get_scientific_case,
    get_scientific_case_events,
    get_latest_decision_dossier,
    get_run_decision_dossier,
    _user_case_update,
    add_scientific_case_context,
    set_scientific_case_question,
    set_scientific_case_scope,
    close_scientific_case,
)
from .skill_drafts import (  # noqa: F401
    _drafts_enabled,
    _visible_draft,
    propose_skill_draft,
    list_skill_drafts,
    get_skill_draft,
    review_skill_draft,
    withdraw_skill_draft,
    export_skill_draft,
)
from .artifacts import (  # noqa: F401
    get_analysis,
    list_attributions,
    get_answer,
    get_observation,
    list_evidence,
    get_evidence,
)
from .events import (  # noqa: F401
    stream_events,
    list_events,
)
