import { listEventsOnce } from '../api/endpoints';
import { openEventStream, type SseStatus } from '../events/sse';
import type { ToxAgentEvent } from '../api/types';

export type ConnectionStatus = 'connecting' | 'live' | 'reconnecting' | 'offline';

type Listener = (event: ToxAgentEvent) => void;
type StatusListener = (status: ConnectionStatus) => void;

const BACKOFF_STEPS_MS = [1000, 2000, 4000, 8000, 16000, 30000];

/**
 * Owns exactly one number: the cursor. Everything else — session, messages,
 * runs, answers — is a REST read the caller re-fetches when this bus says
 * something changed. That split is section 7 of the redesign plan: the bus
 * is the "server is truth, client is a cache with a cursor" primitive, not a
 * second copy of product state.
 */
export class SessionEventBus {
  private cursor: number;
  private readonly sessionId: string;
  private closeStream: (() => void) | null = null;
  private stopped = false;
  private backoffIndex = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly listeners = new Set<Listener>();
  private readonly statusListeners = new Set<StatusListener>();
  private status: ConnectionStatus = 'connecting';
  private lastAppliedSequence: number;
  private readonly detachEnvironment: () => void;

  constructor(sessionId: string, initialCursor: number) {
    this.sessionId = sessionId;
    this.cursor = initialCursor;
    this.lastAppliedSequence = initialCursor;
    this.detachEnvironment = this.attachEnvironmentListeners();
  }

  /**
   * A slept laptop or a phone with its screen off keeps a half-open SSE
   * socket that fires no error until the next `reader.read()` finally times
   * out — minutes later. And exponential backoff means a device that just
   * regained connectivity can sit in `reconnecting` for up to 30s. Both are
   * cases where the environment already knows more than the socket does, so
   * wake/online should short-circuit the backoff and reconcile immediately.
   */
  private attachEnvironmentListeners(): () => void {
    if (typeof window === 'undefined') return () => {};
    const wake = () => this.handleEnvironmentWake();
    const offline = () => this.handleEnvironmentOffline();
    window.addEventListener('online', wake);
    window.addEventListener('offline', offline);
    const doc = typeof document !== 'undefined' ? document : null;
    const onVisible = () => {
      if (doc && doc.visibilityState === 'visible') wake();
    };
    doc?.addEventListener('visibilitychange', onVisible);
    return () => {
      window.removeEventListener('online', wake);
      window.removeEventListener('offline', offline);
      doc?.removeEventListener('visibilitychange', onVisible);
    };
  }

  private handleEnvironmentOffline(): void {
    if (this.stopped) return;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.closeStream?.();
    this.setStatus('offline');
  }

  private handleEnvironmentWake(): void {
    if (this.stopped) return;
    if (typeof navigator !== 'undefined' && navigator.onLine === false) {
      this.setStatus('offline');
      return;
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.backoffIndex = 0;
    if (this.status === 'live') {
      // The socket still reports live, but a slept device can hold a
      // half-open connection that never errored. Reconcile the durable
      // outbox without tearing down a socket that may in fact be fine; the
      // cursor makes this a no-op when nothing was missed.
      void this.reconcileFromRest().catch(() => {});
      return;
    }
    this.setStatus('reconnecting');
    void this.reconcileThenReconnect();
  }

  getStatus(): ConnectionStatus {
    return this.status;
  }

  getCursor(): number {
    return this.cursor;
  }

  onEvent(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  start(): void {
    if (this.stopped) return;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.closeStream?.();
    this.detachEnvironment();
  }

  /** Jump the cursor forward without replaying — used after a caller has
   * already fetched fresh state up to a known sequence (e.g. session
   * bootstrap) so the stream doesn't re-apply events already reflected. */
  fastForward(sequence: number): void {
    if (sequence > this.cursor) {
      this.cursor = sequence;
      this.lastAppliedSequence = sequence;
    }
  }

  private setStatus(next: ConnectionStatus) {
    this.status = next;
    for (const l of this.statusListeners) l(next);
  }

  private connect(): void {
    this.closeStream?.();
    this.closeStream = openEventStream(this.sessionId, this.cursor, {
      onEvent: (event) => this.handleIncoming(event),
      onStatus: (sseStatus) => this.handleSseStatus(sseStatus),
    });
  }

  private handleSseStatus(sseStatus: SseStatus): void {
    if (this.stopped) return;
    if (sseStatus === 'open') {
      this.backoffIndex = 0;
      this.setStatus('live');
      return;
    }
    if (sseStatus === 'connecting') {
      this.setStatus(this.backoffIndex === 0 ? 'connecting' : 'reconnecting');
      return;
    }
    if (sseStatus === 'closed' || sseStatus === 'error') {
      this.setStatus('reconnecting');
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer) return;
    const delay = BACKOFF_STEPS_MS[Math.min(this.backoffIndex, BACKOFF_STEPS_MS.length - 1)];
    this.backoffIndex += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.stopped) void this.reconcileThenReconnect();
    }, delay);
  }

  private async handleIncoming(event: ToxAgentEvent): Promise<void> {
    if (event.sequence <= this.lastAppliedSequence) return; // duplicate, at-least-once delivery

    if (event.sequence > this.lastAppliedSequence + 1) {
      // A gap: something was missed between reconnects. Fill it from the
      // same outbox the stream reads, via the non-streaming replay endpoint,
      // rather than trusting that this one event is safe to apply in
      // isolation.
      await this.fillGapThenApply(event);
      return;
    }

    this.apply(event);
  }

  private async fillGapThenApply(triggering: ToxAgentEvent): Promise<void> {
    try {
      await this.reconcileFromRest();
      // The triggering event may already be included in the page; `apply`
      // is a no-op for anything at or before the cursor it just advanced to.
      this.apply(triggering);
    } catch {
      // Couldn't fill the gap over REST either; the next reconnect (or the
      // caller's own reconcile-on-focus) will retry from the same cursor.
    }
  }

  /** A reconnect must reconcile durable outbox state *before* opening SSE.
   * Otherwise a fast reconnect can make a UI look live while it is still
   * missing a state transition written by another control-plane instance. */
  private async reconcileThenReconnect(): Promise<void> {
    try {
      await this.reconcileFromRest();
    } catch {
      // SSE remains useful even when this one REST replay fails. It starts
      // from the unchanged cursor and a later reconnect tries again.
    }
    if (!this.stopped) this.connect();
  }

  private async reconcileFromRest(): Promise<void> {
    // `events:list` is deliberately capped at 500 rows. Reconcile to the
    // first page's durable snapshot, rather than opening a replacement SSE
    // after only one page and silently skipping a long offline interval.
    // Events committed after that snapshot belong to the new SSE connection;
    // this also prevents a busy session from keeping this loop open forever.
    let snapshotSequence: number | null = null;
    while (snapshotSequence === null || this.lastAppliedSequence < snapshotSequence) {
      const cursorBeforePage = this.lastAppliedSequence;
      const page = await listEventsOnce(this.sessionId, {
        after_sequence: cursorBeforePage,
        limit: 500,
      });
      snapshotSequence ??= page.latest_sequence;
      for (const event of page.events) this.apply(event);

      // A malformed/empty page must not spin the reconnect loop. Keep the
      // unchanged cursor so the replacement SSE (and a future reconcile) can
      // retry the same durable rows.
      if (this.lastAppliedSequence === cursorBeforePage) return;
    }
  }

  private apply(event: ToxAgentEvent): void {
    if (event.sequence <= this.lastAppliedSequence) return;
    this.lastAppliedSequence = event.sequence;
    this.cursor = event.sequence;
    for (const l of this.listeners) l(event);
  }
}
