import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { explainPrediction, quickPredict, recognizeStructure } from './endpoints';

function mockJson(body: unknown, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response);
}

describe('quick predict endpoints', () => {
  beforeEach(() => {
    localStorage.setItem('toxagent.bearer_token', 'test-token');
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it('POSTs /v1/predict with the bearer header and body', async () => {
    const fetchMock = mockJson({ persisted: false, analysis_id: null });
    vi.stubGlobal('fetch', fetchMock);

    await quickPredict({ smiles: 'CCO', endpoints: ['herg'] });

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/v1/predict');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>).authorization).toBe('Bearer test-token');
    expect(JSON.parse(init.body as string)).toEqual({ smiles: 'CCO', endpoints: ['herg'] });
  });

  it('POSTs /v1/predict/recognize', async () => {
    const fetchMock = mockJson({ smiles: 'CCO', canonical_smiles: 'CCO', confidence: 0.9 });
    vi.stubGlobal('fetch', fetchMock);

    const result = await recognizeStructure({ mime_type: 'image/png', data_base64: 'AAAA' });

    expect(String(fetchMock.mock.calls[0][0])).toContain('/v1/predict/recognize');
    expect(result.canonical_smiles).toBe('CCO');
  });

  it('POSTs /v1/predict/explain with the task', async () => {
    const fetchMock = mockJson({ status: 'completed', atoms: [], limitations: [] });
    vi.stubGlobal('fetch', fetchMock);

    await explainPrediction({ smiles: 'CCO', endpoint: 'tox21', task: 'NR-ER' });

    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(String(fetchMock.mock.calls[0][0])).toContain('/v1/predict/explain');
    expect(body).toEqual({ smiles: 'CCO', endpoint: 'tox21', task: 'NR-ER' });
  });
});
