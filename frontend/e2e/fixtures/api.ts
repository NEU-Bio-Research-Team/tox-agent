import type { Page, Request } from '@playwright/test';

/** A route's answer: a body (200), or a function of the request that returns
 * a body or `[status, body]`. Keys are `METHOD /path`, matched on the pathname. */
export type Handler = unknown | ((request: Request, url: URL) => unknown);
export type Routes = Record<string, Handler>;

export interface StoryApi {
  /** Every POST body, in order, by path. */
  posts: Array<{ path: string; body: Record<string, unknown> }>;
  /** Requests no route answered: a test that leaves one of these is
   * exercising an API this fixture did not declare. */
  unmocked: string[];
}

const QUIET = {
  'GET /v1/sessions': { sessions: [], next_offset: null },
  'GET /health/ready': { ready: true, capabilities: { structure_recognition: false } },
  'GET /v1/health/ready': { ready: true, capabilities: { structure_recognition: false } },
  'GET /v1/predict/capabilities': {
    served_endpoints: ['herg'], default_endpoints: ['herg'], ocr_available: false,
    endpoints: [{ id: 'herg', display_name: 'hERG', enabled: true, models: [{ model_id: 'm', display_name: 'Model m' }] }],
    tox21_tasks: [],
  },
};

/** Serves a story's API to the page: REST snapshots from the fixtures, an
 * empty event stream (nothing live happens in a fixture), and a 404 the app
 * already handles for anything undeclared. */
export async function installStoryApi(page: Page, sessionId: string, routes: Routes): Promise<StoryApi> {
  const api: StoryApi = { posts: [], unmocked: [] };
  const table: Routes = {
    ...QUIET,
    [`GET /v1/sessions/${sessionId}/events:list`]: { events: [], count: 0, latest_sequence: 0 },
    [`GET /v1/sessions/${sessionId}/cases`]: { cases: [] },
    // Ambient reads every workbench makes, unrelated to a story.
    [`GET /v1/sessions/${sessionId}/settings`]: { ai_profile_id: null, predictor_bindings: {} },
    'GET /v1/model-connections': { connections: [] },
    [`GET /v1/sessions/${sessionId}/evidence`]: { evidence: [], count: 0 },
    ...routes,
  };
  await page.addInitScript(() => localStorage.setItem('toxagent.bearer_token', 'e2e-token'));
  await page.route(/\/(v1|health)\//, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (!url.pathname.startsWith('/v1/') && !url.pathname.startsWith('/health/')) return route.continue();
    const key = `${request.method()} ${url.pathname}`;
    if (url.pathname === `/v1/sessions/${sessionId}/events`) {
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body: '' });
    }
    if (request.method() === 'POST') {
      api.posts.push({ path: url.pathname, body: (request.postDataJSON() ?? {}) as Record<string, unknown> });
    }
    const handler = table[key];
    if (handler === undefined) {
      api.unmocked.push(key);
      return route.fulfill({
        status: 404, contentType: 'application/json',
        body: JSON.stringify({ error: { code: 'not_found', message: 'unmocked endpoint', retryable: false, details: {} } }),
      });
    }
    const answer = typeof handler === 'function' ? (handler as (r: Request, u: URL) => unknown)(request, url) : handler;
    const [status, body] = Array.isArray(answer) && answer.length === 2 && typeof answer[0] === 'number'
      ? (answer as [number, unknown]) : [200, answer];
    return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  });
  return api;
}

/** The transcript endpoint pages by `after_sequence`; serve it that way so the
 * client's "short page ends history" rule terminates. */
export function messages(list: Array<{ sequence: number }>) {
  return (_request: Request, url: URL) => {
    const after = Number(url.searchParams.get('after_sequence') ?? 0);
    const page = list.filter((message) => message.sequence > after);
    return { messages: page, count: page.length };
  };
}
