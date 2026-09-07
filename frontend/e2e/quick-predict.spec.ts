import { expect, test, type Page } from '@playwright/test';

const ONE_PIXEL_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL7VwAAAABJRU5ErkJggg==',
  'base64',
);

const CAPABILITIES = {
  served_endpoints: ['herg', 'tox21'],
  models: [{ model_id: 'm', capabilities: ['herg', 'tox21'], loaded: true, required: true, detail: '', blocked_reason: null }],
  predictor_id: 'toxpred-local',
  ocr_available: true,
};

const PROJECTION = {
  analysis_id: null,
  persisted: false,
  input_smiles: 'CCO',
  canonical_smiles: 'CCO',
  requested_endpoints: ['herg', 'tox21'],
  served_endpoints: ['herg', 'tox21'],
  unavailable_endpoints: [],
  sections: {
    herg: {
      measurement: 'hERG channel blockade liability',
      probability_blocker: 0.73,
      label: 'blocker',
      threshold: 0.5,
      threshold_source: 'model_default',
      model_id: 'm',
    },
    tox21: {
      measurement: 'Twelve independent Tox21 assay activities',
      task_order_version: 'tox21-12task-v1',
      model_id: 'm',
      assays: {
        'SR-MMP': { probability_activity: 0.8, active: true, threshold: 0.5, threshold_source: 'model_default' },
      },
    },
  },
  applicability: { status: 'ok', method: 'element_rules_v1', reasons: [] },
  provenance: { content_sha256: 'abc', predictor_service_version: '0.1.0' },
  policy_snapshot: {},
  required_limitations: ['uncalibrated_probability'],
  created_at: '2026-09-06T00:00:00Z',
};

const EXPLANATION = {
  status: 'completed',
  endpoint: 'herg',
  task: null,
  input_smiles: 'CCO',
  canonical_smiles: 'CCO',
  atom_order_version: 'rdkit-output-order-v1',
  probability: 0.73,
  atoms: [
    { atom_index: 0, symbol: 'C', importance: 0.5, relative_importance: 0.5 },
    { atom_index: 2, symbol: 'O', importance: 0.3, relative_importance: 0.3 },
  ],
  unmapped_importance: 0.2,
  tokens: [{ token: 'C', importance: 0.5, offsets: [0, 1] }],
  method: 'grad_x_embedding_l2_v1+token_atom_align_v1',
  metadata: { model_id: 'm', deterministic: true, duration_ms: 5, note: null },
  limitations: ['attribution_not_causality'],
};

type Calls = { paths: string[] };

async function installApi(page: Page): Promise<Calls> {
  const calls: Calls = { paths: [] };
  await page.addInitScript(() => localStorage.setItem('toxagent.bearer_token', 'e2e-token'));
  await page.route('**/v1/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    calls.paths.push(`${request.method()} ${url.pathname}`);
    const respond = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

    if (url.pathname === '/v1/predict/capabilities') return respond(CAPABILITIES);
    if (url.pathname === '/v1/predict:batch' && request.method() === 'POST') {
      return respond({
        count: 2,
        results: [PROJECTION, { ...PROJECTION, canonical_smiles: 'CCN' }],
        errors: [{ index: 1, input_smiles: 'nope', error: 'invalid_smiles', detail: '' }],
      });
    }
    if (url.pathname === '/v1/predict' && request.method() === 'POST') return respond(PROJECTION);
    if (url.pathname === '/v1/predict/recognize') {
      return respond({ smiles: 'CCO', canonical_smiles: 'CCO', confidence: 0.88 });
    }
    if (url.pathname === '/v1/predict/explain') return respond(EXPLANATION);
    return respond({ error: { code: 'not_found', message: 'unmocked', retryable: false, details: {} } }, 404);
  });
  return calls;
}

test('predicts a typed SMILES with no session request', async ({ page }) => {
  const calls = await installApi(page);
  await page.goto('/predict');

  await page.getByLabel('SMILES', { exact: true }).fill('CCO');
  await page.getByRole('button', { name: 'Phân tích' }).click();

  await expect(page.getByRole('heading', { name: 'hERG' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Tox21' })).toBeVisible();
  expect(calls.paths.some((p) => p.includes('/v1/sessions'))).toBe(false);
});

test('recognises an uploaded image into an editable SMILES then predicts', async ({ page }) => {
  await installApi(page);
  await page.goto('/predict');

  await page.getByRole('button', { name: 'Tải ảnh' }).click();
  const dialog = page.getByRole('dialog', { name: 'Tải ảnh cấu trúc' });
  await dialog.locator('input[type=file]').setInputFiles({
    name: 'structure.png',
    mimeType: 'image/png',
    buffer: ONE_PIXEL_PNG,
  });

  await expect(page.getByLabel(/SMILES nhận diện được/)).toHaveValue('CCO');
  await page.getByRole('button', { name: 'Phân tích' }).click();
  await expect(page.getByRole('heading', { name: 'hERG' })).toBeVisible();
});

test('runs a multi-SMILES batch with a per-item error', async ({ page }) => {
  await installApi(page);
  await page.goto('/predict');

  await page.getByRole('checkbox', { name: /Nhiều phân tử/ }).click();
  await page.getByLabel('Danh sách SMILES').fill('CCO\nnope\nCCN');
  await page.getByRole('button', { name: 'Phân tích' }).click();

  await expect(page.getByText(/1 phân tử lỗi/)).toBeVisible();
  await expect(page.getByRole('heading', { name: 'hERG' })).toHaveCount(2);
});

test('opens the XAI section and renders atom attribution', async ({ page }) => {
  await installApi(page);
  await page.goto('/predict');

  await page.getByLabel('SMILES', { exact: true }).fill('CCO');
  await page.getByRole('button', { name: 'Phân tích' }).click();
  await expect(page.getByRole('heading', { name: 'hERG' })).toBeVisible();

  await page.getByRole('button', { name: /Giải thích \(XAI\) · herg/ }).click();
  await page.getByRole('button', { name: 'Giải thích', exact: true }).click();

  await expect(page.getByText(/20% trọng số nằm ở liên kết\/tô-pô/)).toBeVisible();
  await expect(page.getByText(/không phải quan hệ nhân quả/)).toBeVisible();
});
