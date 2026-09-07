import { useEffect, useReducer } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { SessionEventBus } from '../lib/store/eventBus';
import type { ToxAgentEvent } from '../lib/api/types';
import {
  createSessionEventsState,
  sessionEventsReducer,
  type SessionEventsState,
} from '../lib/store/sessionEventsReducer';

export type { RecoveryBanner, SessionEventsState, ToolCallLive } from '../lib/store/sessionEventsReducer';

const sessionKey = (sessionId: string) => ['session', sessionId];
const messagesKey = (sessionId: string) => ['messages', sessionId];
const runKey = (sessionId: string, runId: string) => ['run', sessionId, runId];
const runEventsKey = (sessionId: string, runId: string) => ['run-events', sessionId, runId];
const sessionsListKey = ['sessions'];

export function useSessionEvents(
  sessionId: string,
  initialCursor: number,
  historyEvents?: readonly ToxAgentEvent[],
): SessionEventsState {
  const queryClient = useQueryClient();
  const [state, dispatch] = useReducer(sessionEventsReducer, initialCursor, createSessionEventsState);

  // The SSE cursor starts at `initialCursor`, so it intentionally never
  // replays the old outbox. Hydrate the small historical projections from the
  // separately paged snapshot without treating its old artifacts as newly
  // arrived UI work.
  useEffect(() => {
    if (!historyEvents) return;
    dispatch({ type: 'history.hydrated', events: historyEvents, throughSequence: initialCursor });
  }, [historyEvents, initialCursor]);

  useEffect(() => {
    const bus = new SessionEventBus(sessionId, initialCursor);
    dispatch({ type: 'reset', initialCursor });

    const offStatus = bus.onStatus((status) => dispatch({ type: 'connection.status', status }));
    const offEvent = bus.onEvent((event: ToxAgentEvent) => {
      dispatch({ type: 'event.received', event });
      invalidateQueriesForEvent(queryClient, sessionId, event);
    });

    bus.start();
    return () => {
      offStatus();
      offEvent();
      bus.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- initialCursor is a mount-time seed only
  }, [sessionId, queryClient]);

  return state;
}

/** Side effects deliberately live outside the reducer: React Query retains
 * the durable REST projections, while the reducer only projects live hints. */
function invalidateQueriesForEvent(
  queryClient: ReturnType<typeof useQueryClient>,
  sessionId: string,
  event: ToxAgentEvent,
): void {
  const runId = event.run_id ?? undefined;

  switch (event.type) {
    case 'message.created':
      void queryClient.invalidateQueries({ queryKey: messagesKey(sessionId) });
      void queryClient.invalidateQueries({ queryKey: sessionsListKey });
      return;

    case 'run.queued':
    case 'run.started':
    case 'run.validating':
    case 'run.cancelled':
      void queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      void queryClient.invalidateQueries({ queryKey: sessionsListKey });
      if (runId) void queryClient.invalidateQueries({ queryKey: runKey(sessionId, runId) });
      return;

    case 'run.completed':
    case 'run.failed':
      void queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      void queryClient.invalidateQueries({ queryKey: messagesKey(sessionId) });
      void queryClient.invalidateQueries({ queryKey: sessionsListKey });
      if (runId) void queryClient.invalidateQueries({ queryKey: runKey(sessionId, runId) });
      return;

    case 'tool.started':
    case 'tool.completed':
    case 'tool.failed':
      if (runId) void queryClient.invalidateQueries({ queryKey: runKey(sessionId, runId) });
      return;

    case 'answer.rejected':
      // ValidationTab reads this exact durable-outbox query, not the live
      // reducer hint; an open inspector must not show stale history.
      if (runId) void queryClient.invalidateQueries({ queryKey: runEventsKey(sessionId, runId) });
      return;

    case 'answer.accepted':
      void queryClient.invalidateQueries({ queryKey: messagesKey(sessionId) });
      if (runId) void queryClient.invalidateQueries({ queryKey: runEventsKey(sessionId, runId) });
      return;

    case 'analysis.created':
    case 'runtime.recovery_started':
      void queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      return;

    default:
      return;
  }
}
