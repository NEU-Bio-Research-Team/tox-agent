// @vitest-environment node
import { describe, expect, it } from 'vitest';
import type { EventType, ToxAgentEvent } from '../../../shared/api/types';
import { createSessionEventsState, sessionEventsReducer, type SessionEventsReducerState } from './sessionEventsReducer';

function event(sequence: number, type: EventType, overrides: Partial<ToxAgentEvent> = {}): ToxAgentEvent {
  return {
    event_id: `evt-${sequence}`, session_id: 'ses_test', sequence, type,
    entity_type: 'run', entity_id: `entity-${sequence}`, entity_version: 1,
    run_id: 'run_test', occurred_at: `2026-11-07T00:00:${String(sequence).padStart(2, '0')}Z`, payload: {},
    ...overrides,
  };
}

const receive = (state: SessionEventsReducerState, item: ToxAgentEvent) =>
  sessionEventsReducer(state, { type: 'event.received', event: item });

describe('the live trace', () => {
  it('records a turn, its skills, streaming reasoning and tool calls in order', () => {
    let state = createSessionEventsState(0);
    state = receive(state, event(1, 'agent.trace', { payload: { kind: 'turn', phase: 'investigator', provider_id: 'openai', model_id: 'gpt-5.1', tool_count: 14, prompt_tokens: 8200 } }));
    state = receive(state, event(2, 'agent.trace', { payload: { kind: 'skills', mode: 'dynamic', skills: [{ skill_id: 'assess-conflicting-evidence', version: '1' }] } }));
    state = receive(state, event(3, 'agent.reasoning', { entity_id: 'prt_1', payload: { part_id: 'prt_1', text: 'First', done: false } }));
    state = receive(state, event(4, 'agent.reasoning', { entity_id: 'prt_1', payload: { part_id: 'prt_1', text: 'First, the hERG', done: true } }));
    state = receive(state, event(5, 'tool.started', { entity_id: 'call_1', payload: { tool_name: 'read_scientific_skill', arguments_preview: '{"skill_id": "assess-conflicting-evidence"}' } }));
    state = receive(state, event(6, 'tool.completed', { entity_id: 'call_1', payload: { duration_ms: 120 } }));

    const trace = state.liveTrace.run_test;
    expect(trace.map((item) => item.kind)).toEqual(['turn', 'skills', 'reasoning', 'tool']);
    const reasoning = trace[2];
    // One row per reasoning part: it grows in place and keeps its start time.
    expect(reasoning).toMatchObject({ kind: 'reasoning', text: 'First, the hERG', done: true, at: '2026-11-07T00:00:03Z', updatedAt: '2026-11-07T00:00:04Z' });
    expect(trace[3]).toMatchObject({ kind: 'tool', toolName: 'read_scientific_skill', state: 'completed', durationMs: 120, args: '{"skill_id": "assess-conflicting-evidence"}' });
    // The older projection the rest of the page reads is unchanged.
    expect(state.liveToolCalls.run_test[0]).toMatchObject({ call_id: 'call_1', state: 'completed' });
  });

  it('notes a rejected answer and the validation step', () => {
    let state = createSessionEventsState(0);
    state = receive(state, event(1, 'answer.rejected', { payload: { violations: [{}, {}] } }));
    state = receive(state, event(2, 'run.validating'));
    expect(state.liveTrace.run_test).toMatchObject([{ kind: 'rejected', violations: 2 }, { kind: 'validating' }]);
  });

  it('rebuilds the trace from history after a reload without duplicating live rows', () => {
    let state = createSessionEventsState(3);
    state = receive(state, event(4, 'agent.reasoning', { entity_id: 'prt_1', payload: { part_id: 'prt_1', text: 'more', done: true } }));
    state = sessionEventsReducer(state, {
      type: 'history.hydrated',
      throughSequence: 3,
      events: [
        event(1, 'agent.trace', { payload: { kind: 'turn', provider_id: 'openai', model_id: 'gpt-5.1', tool_count: 3 } }),
        event(2, 'agent.reasoning', { entity_id: 'prt_1', payload: { part_id: 'prt_1', text: 'so', done: false } }),
      ],
    });
    const trace = state.liveTrace.run_test;
    expect(trace.map((item) => item.kind)).toEqual(['turn', 'reasoning']);
    expect(trace[1]).toMatchObject({ text: 'more', done: true });
  });
});
