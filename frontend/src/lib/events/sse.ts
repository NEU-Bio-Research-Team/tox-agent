import { API_BASE_URL, getToken } from '../api/client';
import type { ToxAgentEvent } from '../api/types';

interface RawFrame {
  event?: string;
  id?: string;
  data?: string;
}

function parseFrame(block: string): RawFrame | null {
  const lines = block.split('\n');
  const frame: RawFrame = {};
  const dataLines: string[] = [];
  for (const rawLine of lines) {
    const line = rawLine.replace(/\r$/, '');
    if (!line || line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
    if (field === 'event') frame.event = value;
    else if (field === 'id') frame.id = value;
    else if (field === 'data') dataLines.push(value);
  }
  if (dataLines.length) frame.data = dataLines.join('\n');
  return frame.event || frame.data ? frame : null;
}

export type SseStatus = 'connecting' | 'open' | 'closed' | 'error';

export interface SseHandlers {
  onEvent: (event: ToxAgentEvent) => void;
  onStatus?: (status: SseStatus) => void;
}

/**
 * `EventSource` cannot attach an `Authorization` header, and every `/v1`
 * route requires a Bearer token, so the change feed is read by hand over
 * `fetch` + a streamed body instead. Reconnection and `Last-Event-ID` are
 * therefore this module's job, not the browser's — see the redesign plan,
 * section 7.4.
 */
export function openEventStream(
  sessionId: string,
  afterSequence: number,
  handlers: SseHandlers,
): () => void {
  const controller = new AbortController();
  let stopped = false;

  async function run() {
    handlers.onStatus?.('connecting');
    try {
      const token = getToken();
      const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/events?after_sequence=${afterSequence}`, {
        headers: {
          accept: 'text/event-stream',
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        handlers.onStatus?.('error');
        return;
      }
      handlers.onStatus?.('open');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (!stopped) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // sse-starlette (the backend's actual SSE library) frames with
        // `\r\n\r\n`, not the bare `\n\n` the SSE spec's simplest form uses —
        // searching for `\n\n` alone never matched a single real frame.
        // Normalizing the whole accumulated buffer is safe against a `\r`/`\n`
        // pair split across two chunks: the replace only fires once both
        // characters have actually arrived.
        buffer = buffer.replace(/\r\n/g, '\n');

        let separator = buffer.indexOf('\n\n');
        while (separator >= 0) {
          const block = buffer.slice(0, separator);
          buffer = buffer.slice(separator + 2);
          const frame = parseFrame(block);
          if (frame?.data && frame.event && frame.event !== 'heartbeat') {
            try {
              handlers.onEvent(JSON.parse(frame.data) as ToxAgentEvent);
            } catch {
              // A frame that doesn't parse is dropped, not fatal — the
              // reducer's gap detection (via `sequence`) recovers state
              // through reconciliation regardless of what caused the gap.
            }
          }
          separator = buffer.indexOf('\n\n');
        }
      }
      if (!stopped) handlers.onStatus?.('closed');
    } catch (error) {
      if (!stopped && (error as Error).name !== 'AbortError') {
        handlers.onStatus?.('error');
      }
    }
  }

  void run();

  return () => {
    stopped = true;
    controller.abort();
  };
}
