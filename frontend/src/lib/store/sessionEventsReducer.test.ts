// @vitest-environment node
import { describe, expect, it } from 'vitest';
import type { EventType, ToxAgentEvent } from '../api/types';
import {
  createSessionEventsState,
  sessionEventsReducer,
  type SessionEventsReducerState,
} from './sessionEventsReducer';

function event(
  sequence: number,
  type: EventType,
  overrides: Partial<ToxAgentEvent> = {},
): ToxAgentEvent {
  return {
    event_id: `evt-${sequence}`,
    session_id: 'ses_test',
    sequence,
    type,
    entity_type: 'run',
    entity_id: `entity-${sequence}`,
    entity_version: 1,
    run_id: 'run_test',
    occurred_at: '2026-09-06T00:00:00Z',
    payload: {},
    ...overrides,
  };
}

function receive(state: SessionEventsReducerState, item: ToxAgentEvent): SessionEventsReducerState {
  return sessionEventsReducer(state, { type: 'event.received', event: item });
}

describe('sessionEventsReducer', () => {
  it('moves the cursor only forward and deduplicates both sequence and event_id', () => {
    let state = createSessionEventsState(4);
    state = receive(state, event(5, 'tool.started', { entity_id: 'call-1', payload: { tool_name: 'get_analysis_slice' } }));

    const afterFirst = state;
    expect(receive(state, event(5, 'tool.started', { event_id: 'evt-replayed' }))).toBe(afterFirst);
    expect(receive(state, event(6, 'tool.started', { event_id: 'evt-5' }))).toBe(afterFirst);

    state = receive(state, event(7, 'answer.accepted', { entity_id: 'ans-7' }));
    expect(state.cursor).toBe(7);
    expect(state.liveToolCalls.run_test).toHaveLength(1);
    expect(state.latestArtifact).toEqual({ kind: 'answer', entityId: 'ans-7', sequence: 7 });
  });

  it('projects live tool, validation, analysis, and recovery hints without mutating durable state', () => {
    let state = createSessionEventsState(0);
    state = receive(state, event(1, 'tool.started', { entity_id: 'call-1', payload: { tool_name: 'search_toxicology_evidence' } }));
    state = receive(state, event(2, 'tool.completed', {
      entity_id: 'call-1',
      payload: { duration_ms: 42, observation_ids: ['obs-1'] },
    }));
    state = receive(state, event(3, 'answer.rejected', {
      payload: { violations: [{ code: 'missing_source', message: 'Claim needs a source.' }] },
    }));
    state = receive(state, event(4, 'analysis.created', { entity_id: 'analysis-4' }));
    state = receive(state, event(5, 'runtime.recovery_started', {
      entity_id: 'recovery-5',
      payload: { recovery_of_run_id: 'run-original' },
    }));

    expect(state.cursor).toBe(5);
    expect(state.liveToolCalls.run_test).toEqual([
      {
        call_id: 'call-1',
        tool_name: 'search_toxicology_evidence',
        state: 'completed',
        started_at: '2026-09-06T00:00:00Z',
        ended_at: '2026-09-06T00:00:00Z',
        duration_ms: 42,
        error_code: undefined,
        observation_ids: ['obs-1'],
      },
    ]);
    expect(state.liveRejections.run_test).toEqual([[{ code: 'missing_source', message: 'Claim needs a source.' }]]);
    expect(state.analysisIdByRun).toEqual({ run_test: 'analysis-4' });
    expect(state.latestArtifact).toEqual({ kind: 'analysis', entityId: 'analysis-4', sequence: 4 });
    expect(state.recoveryBanners).toEqual([{ originalRunId: 'run-original', recoveryRunId: 'recovery-5' }]);
  });

  it('resets live-only hints when the session event controller is replaced', () => {
    let state = createSessionEventsState(0);
    state = receive(state, event(1, 'tool.started'));
    state = sessionEventsReducer(state, { type: 'connection.status', status: 'live' });
    state = sessionEventsReducer(state, { type: 'reset', initialCursor: 12 });

    expect(state).toEqual(createSessionEventsState(12));
  });

  it('hydrates reload-time history without rewinding or replacing newer SSE state', () => {
    let state = createSessionEventsState(8);
    state = receive(state, event(9, 'answer.accepted', { entity_id: 'answer-live' }));

    state = sessionEventsReducer(state, {
      type: 'history.hydrated',
      throughSequence: 8,
      events: [
        event(1, 'analysis.created', { entity_id: 'analysis-old', run_id: 'run-old' }),
        event(2, 'answer.rejected', {
          run_id: 'run-old',
          payload: { violations: [{ code: 'missing_source', message: 'source required' }] },
        }),
        event(3, 'runtime.recovery_started', {
          entity_id: 'recovery-old',
          run_id: 'recovery-old',
          payload: { recovery_of_run_id: 'run-original' },
        }),
      ],
    });

    expect(state.cursor).toBe(9);
    expect(state.latestArtifact).toEqual({ kind: 'answer', entityId: 'answer-live', sequence: 9 });
    expect(state.analysisIdByRun).toEqual({ 'run-old': 'analysis-old' });
    expect(state.liveRejections['run-old']).toHaveLength(1);
    expect(state.recoveryBanners).toEqual([{ originalRunId: 'run-original', recoveryRunId: 'recovery-old' }]);

    const afterFirstHydration = state;
    expect(sessionEventsReducer(state, {
      type: 'history.hydrated',
      throughSequence: 8,
      events: [],
    })).toBe(afterFirstHydration);
  });
});
