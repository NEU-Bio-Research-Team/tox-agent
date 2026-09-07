// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { listEventsOnce, openEventStream } = vi.hoisted(() => ({
  listEventsOnce: vi.fn(),
  openEventStream: vi.fn(),
}));

vi.mock('../api/endpoints', () => ({ listEventsOnce }));
vi.mock('../events/sse', () => ({ openEventStream }));

import { SessionEventBus } from './eventBus';
import type { ToxAgentEvent } from '../api/types';

type SseHandlers = {
  onEvent: (item: ToxAgentEvent) => void | Promise<void>;
  onStatus: (status: 'open' | 'connecting' | 'closed' | 'error') => void;
};

function event(sequence: number): ToxAgentEvent {
  return {
    event_id: `evt-${sequence}`,
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

function setOnline(value: boolean) {
  Object.defineProperty(navigator, 'onLine', { value, configurable: true });
}

describe('SessionEventBus environment awareness (W5-03)', () => {
  let callbacks: SseHandlers[];

  beforeEach(() => {
    callbacks = [];
    listEventsOnce.mockReset().mockResolvedValue({ events: [], count: 0, latest_sequence: 0 });
    openEventStream.mockReset().mockImplementation((_sessionId, _cursor, handlers: SseHandlers) => {
      callbacks.push(handlers);
      return vi.fn();
    });
    setOnline(true);
  });

  afterEach(() => {
    setOnline(true);
    vi.useRealTimers();
  });

  it('drops to offline and stops retrying when the network goes away', () => {
    const bus = new SessionEventBus('ses_test', 0);
    const seen: string[] = [];
    bus.onStatus((s) => seen.push(s));
    bus.start();
    callbacks[0].onStatus('open');

    setOnline(false);
    window.dispatchEvent(new Event('offline'));

    expect(bus.getStatus()).toBe('offline');
    expect(seen).toContain('offline');
    bus.stop();
  });

  it('reconciles then reopens SSE immediately when the network returns', async () => {
    listEventsOnce.mockResolvedValue({ events: [event(1)], count: 1, latest_sequence: 1 });
    const bus = new SessionEventBus('ses_test', 0);
    bus.start();
    callbacks[0].onStatus('closed'); // now in backoff/reconnecting

    setOnline(true);
    window.dispatchEvent(new Event('online'));
    await vi.waitFor(() => expect(bus.getCursor()).toBe(1));

    expect(listEventsOnce).toHaveBeenCalledWith('ses_test', { after_sequence: 0, limit: 500 });
    // A fresh SSE connection was opened after the reconcile.
    expect(openEventStream.mock.calls.length).toBeGreaterThanOrEqual(2);
    bus.stop();
  });

  it('reconciles the durable outbox on tab wake without tearing down a live socket', async () => {
    listEventsOnce.mockResolvedValue({ events: [event(2)], count: 1, latest_sequence: 2 });
    const bus = new SessionEventBus('ses_test', 0);
    bus.start();
    callbacks[0].onStatus('open');
    const connectionsBefore = openEventStream.mock.calls.length;

    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(bus.getCursor()).toBe(2));

    expect(bus.getStatus()).toBe('live');
    expect(openEventStream.mock.calls.length).toBe(connectionsBefore);
    bus.stop();
  });

  it('detaches every environment listener on stop', () => {
    const remove = vi.spyOn(window, 'removeEventListener');
    const bus = new SessionEventBus('ses_test', 0);
    bus.start();
    bus.stop();

    expect(remove).toHaveBeenCalledWith('online', expect.any(Function));
    expect(remove).toHaveBeenCalledWith('offline', expect.any(Function));
    remove.mockRestore();
  });
});
