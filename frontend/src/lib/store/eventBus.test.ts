// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { listEventsOnce, openEventStream } = vi.hoisted(() => ({
  listEventsOnce: vi.fn(),
  openEventStream: vi.fn(),
}));

vi.mock('../api/endpoints', () => ({ listEventsOnce }));
vi.mock('../events/sse', () => ({ openEventStream }));

import { SessionEventBus } from './eventBus';
import type { ToxAgentEvent } from '../api/types';

function event(sequence: number, eventId = `evt-${sequence}`): ToxAgentEvent {
  return {
    event_id: eventId,
    session_id: 'ses_test',
    sequence,
    type: 'run.queued',
    entity_type: 'run',
    entity_id: `run-${sequence}`,
    entity_version: 1,
    run_id: `run-${sequence}`,
    occurred_at: '2026-09-06T00:00:00Z',
    payload: {},
  };
}

describe('SessionEventBus', () => {
  let callbacks: {
    onEvent: (item: ToxAgentEvent) => void | Promise<void>;
    onStatus: (status: 'open' | 'connecting' | 'closed' | 'error') => void;
  }[];

  beforeEach(() => {
    callbacks = [];
    listEventsOnce.mockReset();
    openEventStream.mockReset().mockImplementation((_sessionId, _cursor, handlers) => {
      callbacks.push(handlers);
      return vi.fn();
    });
    vi.useFakeTimers();
  });

  afterEach(() => vi.useRealTimers());

  it('keeps a monotonic cursor and deduplicates at-least-once events', async () => {
    const applied = vi.fn();
    const bus = new SessionEventBus('ses_test', 0);
    bus.onEvent(applied);
    bus.start();

    callbacks[0].onEvent(event(1));
    callbacks[0].onEvent(event(1, 'evt-replayed'));
    callbacks[0].onEvent(event(0));
    await vi.runAllTicks();

    expect(applied).toHaveBeenCalledTimes(1);
    expect(bus.getCursor()).toBe(1);
  });

  it('fills every REST page in a sequence gap before applying the triggering event', async () => {
    listEventsOnce
      .mockResolvedValueOnce({ events: [event(1)], count: 1, latest_sequence: 2 })
      .mockResolvedValueOnce({ events: [event(2)], count: 1, latest_sequence: 2 });
    const applied: number[] = [];
    const bus = new SessionEventBus('ses_test', 0);
    bus.onEvent((item) => applied.push(item.sequence));
    bus.start();

    await callbacks[0].onEvent(event(2));

    expect(listEventsOnce).toHaveBeenNthCalledWith(1, 'ses_test', { after_sequence: 0, limit: 500 });
    expect(listEventsOnce).toHaveBeenNthCalledWith(2, 'ses_test', { after_sequence: 1, limit: 500 });
    expect(applied).toEqual([1, 2]);
    expect(bus.getCursor()).toBe(2);
  });

  it('reconciles the REST outbox before opening a replacement SSE connection', async () => {
    const order: string[] = [];
    listEventsOnce.mockImplementation(async () => {
      order.push('reconcile');
      return { events: [], count: 0, latest_sequence: 3 };
    });
    openEventStream.mockImplementation((_sessionId, _cursor, handlers) => {
      order.push('connect');
      callbacks.push(handlers);
      return vi.fn();
    });

    const bus = new SessionEventBus('ses_test', 3);
    bus.start();
    callbacks[0].onStatus('closed');
    await vi.advanceTimersByTimeAsync(1_000);

    expect(order).toEqual(['connect', 'reconcile', 'connect']);
  });
});
