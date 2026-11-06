import type { ActivityLive, ToxAgentEvent, Violation } from '../../../shared/api/types';
import type { ConnectionStatus } from '../../../shared/events/eventBus';
import type { ArtifactSelection } from '../hooks/useArtifactSelection';
import { reduceReportProgress, type ReportProgress } from '../report/reportProgress';

/** The live-only projection is intentionally small. Durable session, message,
 * run, answer, and artifact data remains in REST/React Query; these fields
 * only make an already-open workspace responsive between REST refetches. */
export interface ToolCallLive {
  call_id: string;
  tool_name: string;
  state: 'running' | 'completed' | 'failed';
  started_at: string;
  ended_at?: string;
  duration_ms?: number;
  error_code?: string;
  observation_ids?: string[];
}

/** One line of a run's live log, in the order the server recorded it. */
export type TraceEntry =
  | { kind: 'turn'; id: string; at: string; phase: string | null; providerId: string; modelId: string; toolCount: number; promptTokens: number | null }
  | { kind: 'skills'; id: string; at: string; mode: string; skills: { skill_id: string; version?: string; description?: string }[] }
  | { kind: 'reasoning'; id: string; at: string; updatedAt: string; text: string; done: boolean; truncated: boolean }
  | { kind: 'tool'; id: string; at: string; toolName: string; args: string | null; state: ToolCallLive['state']; endedAt?: string; durationMs?: number; errorCode?: string }
  | { kind: 'rejected'; id: string; at: string; violations: number }
  | { kind: 'validating'; id: string; at: string };

export interface RecoveryBanner {
  originalRunId: string;
  recoveryRunId: string;
}

export interface SessionEventsState {
  status: ConnectionStatus;
  cursor: number;
  liveToolCalls: Record<string, ToolCallLive[]>;
  liveActivities: Record<string, ActivityLive[]>;
  liveRejections: Record<string, Violation[][]>;
  recoveryBanners: RecoveryBanner[];
  analysisIdByRun: Record<string, string>;
  latestArtifact: (ArtifactSelection & { sequence: number }) | null;
  /** run_id -> report build progress from `report.stage_changed` (WS09). */
  liveReportProgress: Record<string, ReportProgress>;
  /** run_id -> the run's live log: turns, skills, reasoning, tool calls. */
  liveTrace: Record<string, TraceEntry[]>;
}

/** A short replay window gives event_id an explicit role in deduplication,
 * while the monotonic sequence remains the durable, unbounded cursor. */
export interface SessionEventsReducerState extends SessionEventsState {
  recentEventIds: readonly string[];
  /** Highest immutable session cursor whose outbox history has been projected
   * after a reload. It is separate from `cursor`, which may already be ahead
   * because the SSE stream receives a new event while history is paging. */
  historyHydratedThroughSequence: number;
}

export type SessionEventsAction =
  | { type: 'reset'; initialCursor: number }
  | { type: 'connection.status'; status: ConnectionStatus }
  | { type: 'history.hydrated'; events: readonly ToxAgentEvent[]; throughSequence: number }
  | { type: 'event.received'; event: ToxAgentEvent };

const EVENT_ID_REPLAY_WINDOW = 1_024;

export function createSessionEventsState(initialCursor: number): SessionEventsReducerState {
  return {
    status: 'connecting',
    cursor: initialCursor,
    recentEventIds: [],
    historyHydratedThroughSequence: -1,
    liveToolCalls: {},
    liveActivities: {},
    liveRejections: {},
    recoveryBanners: [],
    analysisIdByRun: {},
    latestArtifact: null,
    liveReportProgress: {},
    liveTrace: {},
  };
}

/**
 * Pure projection of one durable outbox event. Keeping this reducer free of
 * fetches and React Query writes lets tests exercise replay/duplicate paths
 * without pretending browser state is authoritative.
 */
export function sessionEventsReducer(
  state: SessionEventsReducerState,
  action: SessionEventsAction,
): SessionEventsReducerState {
  switch (action.type) {
    case 'reset':
      return createSessionEventsState(action.initialCursor);

    case 'connection.status':
      return state.status === action.status ? state : { ...state, status: action.status };

    case 'history.hydrated':
      return hydrateHistory(state, action.events, action.throughSequence);

    case 'event.received':
      return reduceEvent(state, action.event);
  }
}

/** Rebuild only the small live projections that need historical outbox rows.
 * The REST entities stay in React Query. Crucially, this does not rewind the
 * cursor or replace an SSE event which arrived while the bootstrap request
 * was in flight. Historical artifacts are not marked "new", preventing a
 * page reload from auto-opening an old analysis or answer. */
function hydrateHistory(
  state: SessionEventsReducerState,
  events: readonly ToxAgentEvent[],
  throughSequence: number,
): SessionEventsReducerState {
  if (throughSequence <= state.historyHydratedThroughSequence) return state;

  let history = createSessionEventsState(0);
  for (const event of [...events]
    .filter((event) => event.sequence <= throughSequence)
    .sort((a, b) => a.sequence - b.sequence)) {
    history = reduceEvent(history, event);
  }

  return {
    ...state,
    historyHydratedThroughSequence: throughSequence,
    // New SSE state wins on a key collision. It follows the immutable cursor
    // and is necessarily newer than this requested history snapshot.
    liveToolCalls: mergeToolCalls(history.liveToolCalls, state.liveToolCalls),
    liveActivities: { ...history.liveActivities, ...state.liveActivities },
    liveRejections: mergeRejections(history.liveRejections, state.liveRejections),
    analysisIdByRun: { ...history.analysisIdByRun, ...state.analysisIdByRun },
    liveReportProgress: { ...history.liveReportProgress, ...state.liveReportProgress },
    liveTrace: mergeTrace(history.liveTrace, state.liveTrace),
    recoveryBanners: uniqueRecoveryBanners([...history.recoveryBanners, ...state.recoveryBanners]),
  };
}

function mergeToolCalls(
  history: Record<string, ToolCallLive[]>,
  live: Record<string, ToolCallLive[]>,
): Record<string, ToolCallLive[]> {
  const merged: Record<string, ToolCallLive[]> = { ...history };
  for (const [runId, calls] of Object.entries(live)) {
    const byId = new Map((merged[runId] ?? []).map((call) => [call.call_id, call]));
    for (const call of calls) byId.set(call.call_id, call);
    merged[runId] = [...byId.values()];
  }
  return merged;
}

function mergeTrace(
  history: Record<string, TraceEntry[]>,
  live: Record<string, TraceEntry[]>,
): Record<string, TraceEntry[]> {
  const merged: Record<string, TraceEntry[]> = { ...history };
  for (const [runId, entries] of Object.entries(live)) {
    let current = merged[runId] ?? [];
    for (const entry of entries) current = upsertTrace(current, entry);
    merged[runId] = current;
  }
  return merged;
}

/** Replace the entry with the same kind and id (a reasoning part that grew,
 * a tool call that finished), or append a new one. */
function upsertTrace(entries: TraceEntry[], entry: TraceEntry): TraceEntry[] {
  const index = entries.findIndex((item) => item.kind === entry.kind && item.id === entry.id);
  if (index < 0) return [...entries, entry];
  const next = entries.slice();
  next[index] = { ...entries[index], ...entry, at: entries[index].at } as TraceEntry;
  return next;
}

function withTrace(state: SessionEventsReducerState, runId: string, entry: TraceEntry): SessionEventsReducerState {
  return { ...state, liveTrace: { ...state.liveTrace, [runId]: upsertTrace(state.liveTrace[runId] ?? [], entry) } };
}

function mergeRejections(
  history: Record<string, Violation[][]>,
  live: Record<string, Violation[][]>,
): Record<string, Violation[][]> {
  const merged: Record<string, Violation[][]> = { ...history };
  for (const [runId, rejections] of Object.entries(live)) {
    merged[runId] = [...(merged[runId] ?? []), ...rejections];
  }
  return merged;
}

function uniqueRecoveryBanners(banners: RecoveryBanner[]): RecoveryBanner[] {
  const byRecoveryRun = new Map<string, RecoveryBanner>();
  for (const banner of banners) byRecoveryRun.set(banner.recoveryRunId, banner);
  return [...byRecoveryRun.values()];
}

function reduceEvent(state: SessionEventsReducerState, event: ToxAgentEvent): SessionEventsReducerState {
  // Sequence protects every event since the initial cursor, while event_id
  // makes duplicate delivery explicit even if a broken/replayed frame carries
  // an unexpected sequence. Never advance the cursor for either duplicate.
  if (event.sequence <= state.cursor || state.recentEventIds.includes(event.event_id)) return state;

  const next = {
    ...state,
    cursor: event.sequence,
    recentEventIds: [...state.recentEventIds, event.event_id].slice(-EVENT_ID_REPLAY_WINDOW),
  };
  const runId = event.run_id ?? undefined;

  switch (event.type) {
    case 'activity.started':
    case 'activity.progress':
    case 'activity.completed':
    case 'activity.failed': {
      if (!runId) return next;
      const status = event.type.split('.')[1] as ActivityLive['status'];
      const activity: ActivityLive = {
        activity_id: event.entity_id,
        phase: (event.payload.phase as ActivityLive['phase']) ?? 'analysis',
        kind: String(event.payload.kind ?? 'cross_check'),
        label_key: String(event.payload.label_key ?? 'activity.processing'),
        status,
        progress: event.payload.progress as ActivityLive['progress'],
      };
      const prior = next.liveActivities[runId] ?? [];
      const index = prior.findIndex((item) => item.activity_id === activity.activity_id);
      const activities = index < 0 ? [...prior, activity] : prior.map((item, i) => i === index ? { ...item, ...activity } : item);
      return { ...next, liveActivities: { ...next.liveActivities, [runId]: activities } };
    }
    case 'tool.started': {
      if (!runId) return next;
      const traced = withTrace(next, runId, {
        kind: 'tool', id: event.entity_id, at: event.occurred_at,
        toolName: String(event.payload.tool_name ?? 'tool'),
        args: typeof event.payload.arguments_preview === 'string' ? event.payload.arguments_preview : null,
        state: 'running',
      });
      return {
        ...traced,
        liveToolCalls: {
          ...next.liveToolCalls,
          [runId]: [
            ...(next.liveToolCalls[runId] ?? []),
            {
              call_id: event.entity_id,
              tool_name: String(event.payload.tool_name ?? 'tool'),
              state: 'running',
              started_at: event.occurred_at,
            },
          ],
        },
      };
    }

    case 'tool.completed':
    case 'tool.failed': {
      if (!runId) return next;
      const prior = (next.liveTrace[runId] ?? []).find((item) => item.kind === 'tool' && item.id === event.entity_id);
      const traced = withTrace(next, runId, {
        kind: 'tool', id: event.entity_id, at: prior?.at ?? event.occurred_at,
        toolName: prior?.kind === 'tool' ? prior.toolName : String(event.payload.tool_name ?? 'tool'),
        args: prior?.kind === 'tool' ? prior.args : null,
        state: event.type === 'tool.completed' ? 'completed' : 'failed',
        endedAt: event.occurred_at,
        durationMs: event.payload.duration_ms as number | undefined,
        errorCode: event.payload.error_code as string | undefined,
      });
      return {
        ...traced,
        liveToolCalls: {
          ...next.liveToolCalls,
          [runId]: (next.liveToolCalls[runId] ?? []).map((call) =>
            call.call_id === event.entity_id
              ? {
                  ...call,
                  state: event.type === 'tool.completed' ? 'completed' : 'failed',
                  ended_at: event.occurred_at,
                  duration_ms: event.payload.duration_ms as number | undefined,
                  error_code: event.payload.error_code as string | undefined,
                  observation_ids: event.payload.observation_ids as string[] | undefined,
                }
              : call,
          ),
        },
      };
    }

    case 'answer.rejected':
      if (!runId || !Array.isArray(event.payload.violations)) return next;
      return {
        ...withTrace(next, runId, {
          kind: 'rejected', id: event.event_id, at: event.occurred_at,
          violations: (event.payload.violations as unknown[]).length,
        }),
        liveRejections: {
          ...next.liveRejections,
          [runId]: [...(next.liveRejections[runId] ?? []), event.payload.violations as Violation[]],
        },
      };

    case 'answer.accepted':
      return {
        ...next,
        latestArtifact: { kind: 'answer', entityId: event.entity_id, sequence: event.sequence },
      };

    case 'analysis.created':
      return {
        ...next,
        analysisIdByRun: runId ? { ...next.analysisIdByRun, [runId]: event.entity_id } : next.analysisIdByRun,
        latestArtifact: { kind: 'analysis', entityId: event.entity_id, sequence: event.sequence },
      };

    case 'report.stage_changed':
    case 'report.completed':
    case 'report.completed_with_gaps':
    case 'report.failed':
    case 'report.cancelled': {
      if (!runId) return next;
      const prior = next.liveReportProgress[runId];
      const progress = reduceReportProgress(prior, event);
      if (!progress || progress === prior) return next;
      return { ...next, liveReportProgress: { ...next.liveReportProgress, [runId]: progress } };
    }

    case 'agent.trace': {
      if (!runId) return next;
      const p = event.payload;
      if (p.kind === 'turn') {
        return withTrace(next, runId, {
          kind: 'turn', id: event.event_id, at: event.occurred_at,
          phase: typeof p.phase === 'string' ? p.phase : null,
          providerId: String(p.provider_id ?? ''), modelId: String(p.model_id ?? ''),
          toolCount: Number(p.tool_count ?? 0),
          promptTokens: typeof p.prompt_tokens === 'number' ? p.prompt_tokens : null,
        });
      }
      if (p.kind === 'skills' && Array.isArray(p.skills)) {
        return withTrace(next, runId, {
          kind: 'skills', id: event.event_id, at: event.occurred_at, mode: String(p.mode ?? ''),
          skills: p.skills as { skill_id: string; version?: string; description?: string }[],
        });
      }
      return next;
    }

    case 'agent.reasoning': {
      if (!runId) return next;
      return withTrace(next, runId, {
        kind: 'reasoning', id: event.entity_id, at: event.occurred_at, updatedAt: event.occurred_at,
        text: String(event.payload.text ?? ''), done: Boolean(event.payload.done),
        truncated: Boolean(event.payload.truncated),
      });
    }

    case 'run.validating':
      if (!runId) return next;
      return withTrace(next, runId, { kind: 'validating', id: runId, at: event.occurred_at });

    case 'runtime.recovery_started': {
      const originalRunId = String(event.payload.recovery_of_run_id ?? '');
      if (!originalRunId) return next;
      return {
        ...next,
        recoveryBanners: [...next.recoveryBanners, { originalRunId, recoveryRunId: event.entity_id }],
      };
    }

    default:
      return next;
  }
}
