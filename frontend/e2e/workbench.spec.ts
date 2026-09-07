import { expect, test, type Page } from '@playwright/test';

const SESSION_ID = 'ses_e2e';
const NOW = '2026-09-06T00:00:00Z';
const ONE_PIXEL_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL7VwAAAABJRU5ErkJggg==',
  'base64',
);

type AcceptedMessage = {
  body: Record<string, unknown>;
};

async function installApi(page: Page): Promise<AcceptedMessage[]> {
  const accepted: AcceptedMessage[] = [];
  await page.addInitScript(() => localStorage.setItem('toxagent.bearer_token', 'e2e-token'));
  await page.route('**/v1/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const respond = (body: unknown, status = 200, contentType = 'application/json') =>
      route.fulfill({ status, contentType, body: contentType === 'application/json' ? JSON.stringify(body) : String(body) });

    if (url.pathname === '/v1/sessions' && request.method() === 'GET') {
      return respond({ sessions: [], next_offset: null });
    }
    if (url.pathname === `/v1/sessions/${SESSION_ID}` && request.method() === 'GET') {
      return respond({
        session_id: SESSION_ID,
        status: 'active',
        preferred_language: 'vi',
        title: 'E2E workspace',
        version: 1,
        created_at: NOW,
        updated_at: NOW,
        latest_event_sequence: 0,
        active_run: null,
        recent_runs: [],
        active_analysis: null,
      });
    }
    if (url.pathname === `/v1/sessions/${SESSION_ID}/messages` && request.method() === 'GET') {
      return respond({ messages: [], count: 0 });
    }
    if (url.pathname === `/v1/sessions/${SESSION_ID}/events:list`) {
      return respond({ events: [], count: 0, latest_sequence: 0 });
    }
    if (url.pathname === `/v1/sessions/${SESSION_ID}/events`) {
      return respond('', 200, 'text/event-stream');
    }
    if (url.pathname === '/v1/health/ready') {
      return respond({ ready: true, capabilities: { structure_recognition: true } });
    }
    if (url.pathname === `/v1/sessions/${SESSION_ID}/messages` && request.method() === 'POST') {
      accepted.push({ body: request.postDataJSON() as Record<string, unknown> });
      return respond({
        message_id: `msg_${accepted.length}`,
        run_id: `run_${accepted.length}`,
        run_status: 'queued',
        selected_intent: 'analysis',
        lane: 'deterministic',
        events_url: `/v1/sessions/${SESSION_ID}/events`,
      }, 202);
    }
    return respond({ error: { code: 'not_found', message: 'unmocked endpoint', retryable: false, details: {} } }, 404);
  });
  return accepted;
}

async function openWorkbench(page: Page): Promise<AcceptedMessage[]> {
  const accepted = await installApi(page);
  await page.goto(`/s/${SESSION_ID}`);
  await expect(page.getByRole('heading', { name: 'E2E workspace' })).toBeVisible();
  return accepted;
}

test('submits a pasted SMILES as a deterministic molecule request', async ({ page }) => {
  const accepted = await openWorkbench(page);
  await page.getByPlaceholder(/Nhập SMILES hoặc mô tả yêu cầu/i).fill('CCO');
  await page.getByRole('button', { name: 'Gửi' }).click();
  await expect.poll(() => accepted.length).toBe(1);
  expect(accepted[0].body.molecule).toEqual({ smiles: 'CCO' });
  expect(accepted[0].body.content).toBeUndefined();
});

test('stages a safe PNG preview and sends an image envelope', async ({ page }) => {
  const accepted = await openWorkbench(page);
  await page.getByRole('button', { name: 'Ảnh', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Tải ảnh cấu trúc' });
  await dialog.locator('input[type=file]').setInputFiles({
    name: 'structure.png',
    mimeType: 'image/png',
    buffer: ONE_PIXEL_PNG,
  });
  await expect(page.getByText('structure.png')).toBeVisible();
  await page.getByRole('button', { name: 'Gửi' }).click();
  await expect.poll(() => accepted.length).toBe(1);
  expect(accepted[0].body.image).toMatchObject({ mime_type: 'image/png' });
  expect(typeof (accepted[0].body.image as { data_base64: unknown }).data_base64).toBe('string');
});

test('opens the keyboard-accessible structure drawing dialog and survives reload', async ({ page }) => {
  await openWorkbench(page);
  await page.getByRole('button', { name: 'Vẽ cấu trúc', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Vẽ cấu trúc phân tử' })).toBeVisible();
  await page.getByRole('button', { name: 'Đóng hộp thoại' }).click();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'E2E workspace' })).toBeVisible();
});
