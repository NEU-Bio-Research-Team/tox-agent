import type { ToxAgentEvent, Violation } from '../api/types';
import type { ConnectionStatus } from './eventBus';
import type { ArtifactSelection } from '../../hooks/useArtifactSelection';

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

export interface RecoveryBanner {
  originalRunId: string;
  recoveryRunId: string;
}

export interface SessionEventsState {
  status: ConnectionStatus;
  cursor: number;
  liveToolCalls: Record<string, ToolCallLive[]>;
  liveRejections: Record<string, Violation[][]>;
  recoveryBanners: RecoveryBanner[];
  analysisIdByRun: Record<string, string>;
  latestArtifact: (ArtifactSelection & { sequence: number }) | null;
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
    liveRejections: {},
    recoveryBanners: [],
    analysisIdByRun: {},
    latestArtifact: null,
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
    liveRejections: mergeRejections(history.liveRejections, state.liveRejections),
    analysisIdByRun: { ...history.analysisIdByRun, ...state.analysisIdByRun },
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
    case 'tool.started':
      if (!runId) return next;
      return {
        ...next,
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

    case 'tool.completed':
    case 'tool.failed':
      if (!runId) return next;
      return {
        ...next,
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

    case 'answer.rejected':
      if (!runId || !Array.isArray(event.payload.violations)) return next;
      return {
        ...next,
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
