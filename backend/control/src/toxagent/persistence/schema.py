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
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
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
    Column("title_source", String(24)),
    Column("title_status", String(24), nullable=False, server_default="pending"),
    Column("title_updated_at", _TS),
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

# Product configuration is deliberately separate from a runtime binding: users
# choose a provider/profile and scientific predictors for a session; a runtime
# may then create many ephemeral bindings while the selection remains stable.
session_settings = Table(
    "session_settings", metadata,
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("ai_profile_id", _ID, nullable=True),
    Column("predictor_bindings", Json, nullable=False),
    Column("updated_at", _TS, nullable=False),
)

run_configuration_snapshots = Table(
    "run_configuration_snapshots", metadata,
    Column("run_id", _ID, ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
    Column("ai_profile_id", _ID, nullable=True),
    Column("predictor_bindings", Json, nullable=False),
    # Which router decided this run's intent, and why. Nullable: runs routed
    # before WS07 recorded neither, and a routing complaint is unanswerable
    # without knowing which rules were in force (P1-9).
    Column("intent_decision", Json),
    Column("created_at", _TS, nullable=False),
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
    Column("auth_mode", String(32), nullable=False, server_default="none"),
    Column("connection_id", _ID),
    Column("profile_hash", String(64), nullable=False),
    Column("tool_schema_hash", String(64), nullable=False),
    Column("system_prompt_hash", String(64), nullable=False),
    Column("capabilities", Json, nullable=False),
    Column("status", String(16), nullable=False),
    Column("selection_reason", Text, nullable=False, server_default=""),
    # Nullable on purpose: bindings written before WS01 have no manifest, and
    # inventing one for them would be a fabricated audit row.
    Column("runtime_manifest", Json),
    Column("created_at", _TS, nullable=False),
    Column("closed_at", _TS),
)

# Product-owned investigation state.  The current row is the fast read model;
# every accepted update is also appended to case_revisions for audit/recovery.
cases = Table(
    "cases", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("goal", String(64), nullable=False),
    Column("subject", Json, nullable=False),
    Column("active_analysis_id", _ID, ForeignKey("analysis_snapshots.id")),
    Column("active_plan_id", _ID),
    Column("state", Json, nullable=False),
    Column("revision", Integer, nullable=False),
    Column("revision_reason", Text, nullable=False),
    Column("created_at", _TS, nullable=False),
    Column("updated_at", _TS, nullable=False),
    UniqueConstraint("session_id", "id", name="uq_cases_session_id"),
    Index("ix_cases_session_updated", "session_id", "updated_at"),
)

case_revisions = Table(
    "case_revisions", metadata,
    Column("case_id", _ID, ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("reason", Text, nullable=False),
    Column("state", Json, nullable=False),
    Column("created_at", _TS, nullable=False),
)

investigation_plans = Table(
    "investigation_plans", metadata,
    Column("id", _ID, primary_key=True),
    Column("case_id", _ID, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("reason", Text, nullable=False),
    Column("created_at", _TS, nullable=False),
    UniqueConstraint("case_id", "revision", name="uq_plan_case_revision"),
)

investigation_steps = Table(
    "investigation_steps", metadata,
    Column("id", _ID, primary_key=True),
    Column("plan_id", _ID, ForeignKey("investigation_plans.id", ondelete="CASCADE"), nullable=False),
    Column("position", Integer, nullable=False),
    Column("question", Text, nullable=False),
    Column("capability", String(64), nullable=False),
    Column("input_refs", Json, nullable=False),
    Column("expected_output", Text, nullable=False),
    Column("success_condition", Text, nullable=False),
    Column("case_revision", Integer, nullable=False),
    Column("status", String(24), nullable=False),
    Column("output_refs", Json, nullable=False),
    Column("failure_reason", Text),
    UniqueConstraint("plan_id", "position", name="uq_plan_step_position"),
)

kernel_transitions = Table(
    "kernel_transitions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", _ID, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
    Column("state", String(32), nullable=False),
    Column("detail", Text, nullable=False, server_default=""),
    Column("occurred_at", _TS, nullable=False),
    Index("ix_kernel_transition_case", "case_id", "id"),
)

model_connections = Table(
    "model_connections", metadata,
    Column("id", _ID, primary_key=True),
    Column("owner_id", String(255), nullable=False),
    Column("provider_id", String(128), nullable=False),
    Column("model_id", String(128), nullable=False),
    Column("display_name", String(160), nullable=False, server_default=""),
    Column("base_url", Text),
    Column("auth_mode", String(32), nullable=False),
    Column("credential_ref", String(255)),
    Column("capabilities", Json, nullable=False),
    Column("status", String(24), nullable=False),
    Column("created_at", _TS, nullable=False),
    Column("updated_at", _TS, nullable=False),
    Index("ix_model_connections_owner", "owner_id", "created_at"),
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
    # WS02 source identity. Nullable: rows written before normalization have
    # none, and a backfill would be a guess about what a provider reported.
    Column("source_event_id", String(64)),
    Column("source_event_type", String(64)),
    Column("provider_message_id", String(128)),
    Column("provider_step_id", String(128)),
    Column("revision", Integer),
    Column("semantics", String(16), nullable=False, server_default="unknown"),
    Column("is_normalized", Boolean, nullable=False, server_default="0"),
    Column("raw_payload_hash", String(64)),
    Index("ix_runtime_usage_events_run", "run_id", "reported_at"),
    # The last line of defence against a duplicate. In-memory deduplication
    # cannot survive a worker restart mid-run; this can. Partial, so the rows
    # that predate source identity are not all collapsed onto one NULL key.
    Index(
        "uq_runtime_usage_source",
        "runtime_binding_id",
        "source_event_id",
        unique=True,
        sqlite_where=text("source_event_id IS NOT NULL"),
        postgresql_where=text("source_event_id IS NOT NULL"),
    ),
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
    # Why this record is, or is not, about the compound and endpoint the run
    # asked about. Nullable: records stored before WS04 were never assessed,
    # and an empty judgement is the honest value for them.
    Column("relevance_assessment", Json),
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

run_jobs = Table(
    "run_jobs", metadata,
    Column("run_id", _ID, ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
    # Everything needed to execute this run again in a process that has never
    # seen it. Without it a restart can only close the run out (I18).
    Column("envelope", Json, nullable=False),
    # Who currently owns execution, until when. NULL/expired means unowned:
    # the previous owner died, or nobody has claimed it yet.
    Column("worker_id", String(64)),
    Column("lease_expires_at", _TS),
    # The fencing token. Every claim increments it, so a worker that was
    # slow rather than dead finds its own writes refused by the epoch it
    # still holds, instead of racing the new owner (I17).
    Column("lease_epoch", Integer, nullable=False, server_default="0"),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("created_at", _TS, nullable=False),
    Column("updated_at", _TS, nullable=False),
    # WS08: which worker class may take this job, and in what order. NULL on
    # a job written before 0014; readers derive the queue from its intent.
    Column("queue_name", String(32)),
    Column("priority", Integer, nullable=False, server_default="0"),
    # Not before this moment — a job deferred because a concurrency slot was
    # full is not claimable again until then.
    Column("available_at", _TS),
    Column("last_error_code", String(64)),
    Index("ix_run_jobs_claimable", "lease_expires_at"),
    Index("ix_run_jobs_queue", "queue_name", "priority", "created_at"),
)

concurrency_slots = Table(
    "concurrency_slots", metadata,
    # WS08 / PR-15: global, tenant, provider and queue caps that hold across
    # every worker. One row per occupied slot index; the primary key is the
    # mutual exclusion, the lease is what frees a dead worker's slot.
    Column("scope", String(16), nullable=False),
    Column("scope_key", String(128), nullable=False),
    Column("slot_index", Integer, nullable=False),
    # No foreign key on purpose: a slot outliving its run by one lease period
    # is harmless, and a cascade would make deleting a session contend with
    # the claim path for these rows.
    Column("run_id", _ID, nullable=False),
    Column("worker_id", String(64), nullable=False),
    Column("expires_at", _TS, nullable=False),
    Column("acquired_at", _TS, nullable=False),
    PrimaryKeyConstraint("scope", "scope_key", "slot_index"),
    Index("ix_concurrency_slots_run", "run_id"),
)

explanation_checkpoints = Table(
    "explanation_checkpoints", metadata,
    # sha256 over canonical SMILES, endpoint, task and the model that produced
    # the probability. Two runs asking for the same explanation of the same
    # molecule from the same model are asking for the same artifact.
    Column("key", String(72), primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("endpoint", String(32), nullable=False),
    Column("task", String(64)),
    Column("model_id", String(128)),
    Column("canonical_smiles", Text, nullable=False),
    Column("payload", Json, nullable=False),
    Column("created_at", _TS, nullable=False),
    Index("ix_explanation_checkpoints_session", "session_id", "created_at"),
)

# --- report builder (spec section 12.1) ------------------------------------
#
# Bytes never live here. A figure's SVG and a rendering's Markdown/HTML/PDF go
# to the object store; these rows carry ownership, hashes, provenance and the
# object refs. Evidence and observation payloads are referenced, never copied:
# a report that duplicated them could disagree with the observation it cites,
# which is the one thing the whole trust chain exists to prevent.

report_builds = Table(
    "report_builds", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("analysis_id", _ID, ForeignKey("analysis_snapshots.id"), nullable=False),
    # The frozen request. Which endpoints were *asked* for is not recoverable
    # from the finished report: two served endpoints could be a two-endpoint
    # request that succeeded or a three-endpoint one that lost a section.
    Column("request", Json, nullable=False),
    Column("stage", String(32), nullable=False),
    Column("report_id", _ID),
    # The one permitted correction attempt, counted durably so a restart
    # cannot buy a second one.
    Column("correction_attempts", Integer, nullable=False, server_default="0"),
    Column("failure_code", String(64)),
    Column("failure_detail", Text),
    # Stage outputs already paid for, so a resumed build does not re-run a
    # billable predictor or provider call (spec section 10).
    Column("stage_state", Json, nullable=False),
    Column("deadline_at", _TS),
    Column("created_at", _TS, nullable=False),
    Column("updated_at", _TS, nullable=False),
    Index("ix_report_builds_session", "session_id", "created_at"),
)

report_artifacts = Table(
    "report_artifacts", metadata,
    Column("id", _ID, primary_key=True),
    Column("report_build_id", _ID, ForeignKey("report_builds.id"), nullable=False),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("analysis_id", _ID, ForeignKey("analysis_snapshots.id"), nullable=False),
    Column("schema_version", String(32), nullable=False),
    Column("title", Text, nullable=False),
    Column("status", String(32), nullable=False),
    Column("report_language", String(8), nullable=False),
    # The whole canonical artifact. Written once; the renderers read it rather
    # than re-deriving anything, so a rendering can never say something the
    # validated artifact does not.
    Column("document", Json, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    # A rebuild links to what it supersedes rather than replacing it: the old
    # report stays readable and its provenance stays true (eval scenario 15).
    Column("supersedes_report_id", _ID, ForeignKey("report_artifacts.id")),
    Column("version", Integer, nullable=False, server_default="1"),
    Column("created_at", _TS, nullable=False),
    Index("ix_report_artifacts_session", "session_id", "created_at"),
    Index("ix_report_artifacts_build", "report_build_id"),
)

report_figures = Table(
    "report_figures", metadata,
    Column("figure_id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("attachment_id", _ID, ForeignKey("attachments.id"), nullable=False),
    # What the image depicts. Stored as columns rather than only inside the
    # artifact JSON so the validator's figure-to-observation check is a lookup,
    # not a scan of a document (eval scenario 12).
    Column("observation_id", _ID, ForeignKey("observations.id")),
    Column("endpoint", String(32)),
    Column("task", String(64)),
    Column("media_type", String(64), nullable=False),
    Column("caption", Text, nullable=False),
    Column("alt_text", Text, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("renderer_version", String(64), nullable=False),
    Column("created_at", _TS, nullable=False),
    Index("ix_report_figures_session", "session_id", "created_at"),
)

report_renderings = Table(
    "report_renderings", metadata,
    Column("id", _ID, primary_key=True),
    Column("report_id", _ID, ForeignKey("report_artifacts.id", ondelete="CASCADE"), nullable=False),
    Column("format", String(16), nullable=False),
    Column("media_type", String(64), nullable=False),
    Column("object_uri", Text, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("renderer_version", String(64), nullable=False),
    Column("created_at", _TS, nullable=False),
    # One rendering per format per report. A second Markdown for the same
    # immutable artifact would be a second answer to a settled question.
    UniqueConstraint("report_id", "format", name="uq_report_rendering_format"),
)

report_claim_links = Table(
    "report_claim_links", metadata,
    Column("report_id", _ID, ForeignKey("report_artifacts.id", ondelete="CASCADE"), primary_key=True),
    Column("claim_id", _ID, primary_key=True),
    Column("section_id", String(64), nullable=False),
    Column("kind", String(24), nullable=False),
    Column("source_class", String(24), nullable=False),
    Column("observation_id", _ID),
    Column("field_path", Text),
    Index("ix_report_claim_links_observation", "observation_id"),
)

report_evidence_links = Table(
    "report_evidence_links", metadata,
    Column("report_id", _ID, ForeignKey("report_artifacts.id", ondelete="CASCADE"), primary_key=True),
    Column("evidence_id", _ID, primary_key=True),
    Column("relation", String(24), nullable=False),
    Column("section_id", String(64), nullable=False),
    Index("ix_report_evidence_links_evidence", "evidence_id"),
)


#: Adaptive decision support (docs/spec/TOXAGENT_ADAPTIVE_DECISION_SUPPORT_PLAN_VI.md
#: section 9.3, ADR 0010). One row per source-vs-proposition assessment for a
#: decision_support run; distinct from report_evidence_links above, which is
#: the report-build capability's own, narrower relation link.
evidence_relation_assessments = Table(
    "evidence_relation_assessments", metadata,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("proposition_id", _ID, nullable=False),
    Column("source_class", String(32), nullable=False),
    Column("source_id", _ID, nullable=False),
    Column("relation", String(24), nullable=False),
    Column("directness", String(16), nullable=False),
    Column("applicability", String(16), nullable=False),
    Column("strength", String(16), nullable=False),
    Column("reason_codes", Json, nullable=False),
    Column("scope", Json, nullable=False),
    Column("created_at", _TS, nullable=False),
    Index("ix_evidence_relation_run", "run_id"),
    Index("ix_evidence_relation_proposition", "session_id", "proposition_id"),
)


#: ADS plan section 10.1/10.2, W6. One row per answer that carried a
#: development posture — ``answer_id`` is the primary key rather than a
#: separately minted id, since the relationship is 1:1 (an answer either has
#: one posture or none) and domain/development_posture.py's
#: ``DevelopmentPosture`` is deliberately a value object with no identity of
#: its own.
development_postures = Table(
    "development_postures", metadata,
    Column("answer_id", _ID, ForeignKey("answers.id", ondelete="CASCADE"), primary_key=True),
    Column("session_id", _ID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("run_id", _ID, ForeignKey("runs.id"), nullable=False),
    Column("value", String(24), nullable=False),
    Column("scope", String(24), nullable=False),
    Column("confidence_band", String(16), nullable=False),
    Column("basis_claim_ids", Json, nullable=False),
    Column("contrary_claim_ids", Json, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("conditions", Json, nullable=False),
    Column("recommended_next_steps", Json, nullable=False),
    Column("created_at", _TS, nullable=False),
    Index("ix_development_postures_run", "run_id"),
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
    {"analysis_snapshots", "observations", "answers", "claims", "claim_sources", "event_outbox",
     "case_revisions", "investigation_plans", "kernel_transitions",
     # A report is immutable for the same reason an answer is: it is cited,
     # downloaded and audited. A changed report needs a new version row, not an
     # UPDATE that quietly rewrites what someone already read.
     "report_artifacts", "report_claim_links", "report_evidence_links",
     # A posture belongs to the answer it was submitted with; a changed
     # posture is a new answer, not an UPDATE on this row (same reasoning as
     # claims above).
     "development_postures"}
)
