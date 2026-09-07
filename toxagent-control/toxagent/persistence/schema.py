"""The relational schema (plan section 13.2).

SQLAlchemy Core rather than the ORM: the mapping is the contract between the
domain and the store, and an explicit table definition is the version of it a
reviewer can read against the plan. The same metadata runs on PostgreSQL in
production and SQLite in tests, so what CI exercises is what ships (ADR 0003).

Immutability of ``analysis_snapshots``, ``observations``, ``answers`` and
``claims`` is an application contract here — the repositories expose no update
path — and should additionally be a database grant or trigger in production.
"""
from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

#: JSONB where it exists, JSON where it does not. Payload semantics are
#: identical; only the index and containment operators differ.
Json = JSON().with_variant(JSONB, "postgresql")

_ID = String(40)
_TS = DateTime(timezone=True)


sessions = Table(
    "sessions", metadata,
    Column("id", _ID, primary_key=True),
    Column("owner_id", String(255), nullable=False),
    Column("status", String(32), nullable=False),
    Column("preferred_language", String(8), nullable=False),
    Column("title", Text),
    Column("active_analysis_id", _ID),
    Column("context_epoch", Integer, nullable=False, server_default="0"),
    # The ordering authority for the change feed. Bumped in the same
    # transaction as the events it numbers.
    Column("event_sequence", Integer, nullable=False, server_default="0"),
    Column("created_at", _TS, nullable=False),
    Column("updated_at", _TS, nullable=False),
    Column("version", Integer, nullable=False),
    Column("client_session_id", String(255)),
    UniqueConstraint("owner_id", "client_session_id", name="uq_session_idempotency"),
    Index("ix_sessions_owner", "owner_id", "created_at"),
)

messages = Table(
    "messages", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("client_message_id", String(255)),
    Column("role", String(16), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("created_at", _TS, nullable=False),
    UniqueConstraint("session_id", "sequence", name="uq_message_sequence"),
    UniqueConstraint("session_id", "client_message_id", name="uq_message_idempotency"),
)

message_parts = Table(
    "message_parts", metadata,
    Column("id", _ID, primary_key=True),
    Column("message_id", _ID, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
    Column("index", Integer, nullable=False),
    Column("type", String(32), nullable=False),
    Column("content", Json, nullable=False),
    Column("version", Integer, nullable=False, server_default="1"),
    UniqueConstraint("message_id", "index", name="uq_part_index"),
)

runs = Table(
    "runs", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("trigger_message_id", _ID, ForeignKey("messages.id"), nullable=False),
    Column("lane", String(16), nullable=False),
    Column("intent", String(32), nullable=False),
    Column("status", String(16), nullable=False),
    Column("runtime_binding_id", _ID, ForeignKey("runtime_bindings.id")),
    Column("recovery_of_run_id", _ID, ForeignKey("runs.id")),
    Column("deadline_at", _TS, nullable=False),
    Column("failure_code", String(64)),
    Column("potentially_billed", Boolean, nullable=False, server_default="0"),
    Column("cancel_requested", Boolean, nullable=False, server_default="0"),
    Column("created_at", _TS, nullable=False),
    Column("started_at", _TS),
    Column("ended_at", _TS),
    Column("version", Integer, nullable=False, server_default="1"),
    CheckConstraint(
        "lane <> 'deterministic' OR runtime_binding_id IS NULL",
        name="ck_deterministic_lane_has_no_runtime",
    ),
    Index("ix_runs_session", "session_id", "created_at"),
)

runtime_bindings = Table(
    "runtime_bindings", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("runtime_kind", String(16), nullable=False),
    Column("runtime_version", String(64), nullable=False),
    Column("runtime_session_id", String(255), nullable=False),
    Column("provider_id", String(128), nullable=False),
    Column("model_id", String(128), nullable=False),
    Column("profile_hash", String(64), nullable=False),
    Column("tool_schema_hash", String(64), nullable=False),
    Column("system_prompt_hash", String(64), nullable=False),
    Column("capabilities", Json, nullable=False),
    Column("status", String(16), nullable=False),
    Column("selection_reason", Text, nullable=False, server_default=""),
    Column("created_at", _TS, nullable=False),
    Column("closed_at", _TS),
)

# W2-13/14: provider reports are immutable events. Nullable numeric fields
# mean "the provider did not report this"; zero remains a real reported zero.
runtime_usage_events = Table(
    "runtime_usage_events", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
    Column("runtime_binding_id", _ID, ForeignKey("runtime_bindings.id", ondelete="CASCADE"), nullable=False),
    Column("provider_id", String(128), nullable=False),
    Column("model_id", String(128), nullable=False),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("reasoning_tokens", Integer),
    Column("cache_read_tokens", Integer),
    Column("cache_write_tokens", Integer),
    Column("total_tokens", Integer),
    Column("cost_amount", Numeric(18, 8)),
    Column("cost_currency", String(8)),
    Column("reported_at", _TS, nullable=False),
    Index("ix_runtime_usage_events_run", "run_id", "reported_at"),
)

analysis_snapshots = Table(
    "analysis_snapshots", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("input_smiles", Text, nullable=False),
    Column("canonical_smiles", Text, nullable=False),
    Column("requested_endpoints", Json, nullable=False),
    # Lossless. Projections are computed on read; this column is never rewritten.
    Column("predictor_response", Json, nullable=False),
    Column("predictor_base_url_id", String(128), nullable=False),
    Column("predictor_service_version", String(64)),
    Column("predictor_git_commit", String(64)),
    Column("artifact_hashes", Json, nullable=False),
    Column("policy_snapshot", Json, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("idempotency_key", String(64), nullable=False),
    Column("created_at", _TS, nullable=False),
    UniqueConstraint("session_id", "idempotency_key", name="uq_analysis_idempotency"),
)

observations = Table(
    "observations", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("analysis_id", _ID, ForeignKey("analysis_snapshots.id")),
    Column("producer", String(32), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("schema_version", String(64), nullable=False),
    Column("canonical_payload", Json, nullable=False),
    Column("model_projection", Json, nullable=False),
    Column("projection_version", String(32), nullable=False),
    Column("required_limitations", Json, nullable=False),
    Column("provenance", Json, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("created_at", _TS, nullable=False),
    Index("ix_observations_session", "session_id", "created_at"),
)

evidence_records = Table(
    "evidence_records", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("provider", String(64), nullable=False),
    Column("provider_record_id", String(255), nullable=False),
    Column("source_type", String(32), nullable=False),
    Column("title", Text, nullable=False),
    Column("authors", Json, nullable=False),
    Column("published_at", String(10)),
    Column("retrieved_at", _TS, nullable=False),
    Column("canonical_url", Text),
    Column("identifier", Json, nullable=False),
    Column("dedupe_key", String(255), nullable=False),
    Column("abstract_or_excerpt", Text),
    Column("normalized_facts", Json, nullable=False),
    Column("source_quality_tier", String(32), nullable=False),
    Column("raw_payload_ref", Text),
    Column("status", String(16), nullable=False),
    Column("rejection_reason", Text),
    Column("content_sha256", String(64), nullable=False),
    # The same source retrieved twice in one session is one record.
    UniqueConstraint("session_id", "dedupe_key", name="uq_evidence_dedupe"),
)

answers = Table(
    "answers", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("schema_version", String(32), nullable=False),
    Column("answer_markdown", Text, nullable=False),
    Column("limitations", Json, nullable=False),
    Column("recommended_next_steps", Json, nullable=False),
    Column("candidate_generation", Integer, nullable=False),
    Column("is_fallback", Boolean, nullable=False, server_default="0"),
    Column("content_sha256", String(64), nullable=False),
    Column("created_at", _TS, nullable=False),
    # At most one accepted answer per candidate generation, and the application
    # refuses a second generation once one is accepted (plan section 8.4).
    UniqueConstraint("run_id", "candidate_generation", name="uq_answer_generation"),
)

claims = Table(
    "claims", metadata,
    Column("id", _ID, primary_key=True),
    Column("answer_id", _ID, ForeignKey("answers.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("text", Text, nullable=False),
    Column("observation_id", _ID, ForeignKey("observations.id")),
    Column("field_path", Text),
    Column("source_value", Json),
    Column("rendered_value", Text),
    Column("transform", String(32), nullable=False),
    Column("input_claim_ids", Json, nullable=False),
    Column("position", Integer, nullable=False),
    CheckConstraint(
        "kind NOT IN ('numeric', 'classification') "
        "OR (observation_id IS NOT NULL AND field_path IS NOT NULL)",
        name="ck_field_backed_claim_has_source",
    ),
    Index("ix_claims_answer", "answer_id", "position"),
)

claim_sources = Table(
    "claim_sources", metadata,
    Column("claim_id", _ID, ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True),
    Column("evidence_id", _ID, ForeignKey("evidence_records.id"), primary_key=True),
)

attachments = Table(
    "attachments", metadata,
    Column("id", _ID, primary_key=True),
    Column("owner_id", String(255), nullable=False),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("media_type", String(128), nullable=False),
    Column("object_uri", Text, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("retention_class", String(16), nullable=False),
    Column("created_at", _TS, nullable=False),
    Column("expires_at", _TS),
)

# --- beyond the plan's table list, and why ---------------------------------

# Plan section 8.5 requires capability tokens to be auditable by jti. Recording
# them also makes revocation possible without waiting for expiry.
capability_tokens = Table(
    "capability_tokens", metadata,
    Column("jti", String(64), primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("runtime_binding_id", _ID, ForeignKey("runtime_bindings.id")),
    Column("allowed_tools", Json, nullable=False),
    Column("issued_at", _TS, nullable=False),
    Column("expires_at", _TS, nullable=False),
    Column("revoked_at", _TS),
)

# Plan sections 14.5 and 15.2: the duplicate/cyclic call detector and the tool
# metrics both need the per-call record, and the transcript grader reads it.
tool_calls = Table(
    "tool_calls", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("tool_name", String(64), nullable=False),
    Column("arguments_sha256", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("error_code", String(64)),
    Column("observation_ids", Json, nullable=False),
    Column("duration_ms", Integer),
    Column("started_at", _TS, nullable=False),
    Column("ended_at", _TS),
    Index("ix_tool_calls_run", "run_id", "started_at"),
)

event_outbox = Table(
    "event_outbox", metadata,
    Column("event_id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("type", String(64), nullable=False),
    Column("entity_type", String(32), nullable=False),
    Column("entity_id", String(64), nullable=False),
    Column("entity_version", Integer, nullable=False, server_default="1"),
    Column("run_id", _ID),
    Column("payload", Json, nullable=False),
    Column("occurred_at", _TS, nullable=False),
    Column("dispatched_at", _TS),
    UniqueConstraint("session_id", "sequence", name="uq_outbox_sequence"),
    Index("ix_outbox_undispatched", "dispatched_at", "sequence"),
)

#: Written once, never updated. Repositories expose no update path for these.
IMMUTABLE_TABLES = frozenset(
    {"analysis_snapshots", "observations", "answers", "claims", "claim_sources", "event_outbox"}
)
