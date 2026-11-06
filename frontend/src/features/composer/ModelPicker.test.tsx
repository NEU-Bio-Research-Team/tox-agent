import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ listConnectionModels: vi.fn() }));
vi.mock('../../shared/api/endpoints', () => api);

import { ModelPicker } from './ModelPicker';
import type { ModelConnection } from '../../shared/api/types';

afterEach(cleanup);

const conn = (id: string, provider: string, model: string, auth = 'api_key') => ({
  connection_id: id, provider_id: provider, model_id: model, display_name: `${provider} · ${model}`,
  auth_mode: auth, base_url: null, has_credential: true, status: 'ready',
  capabilities: { streaming: true, tool_calls: true, structured_output: true, context_size: null },
}) as unknown as ModelConnection;

const CONNECTIONS = [conn('con_openai', 'openai', 'gpt-a'), conn('con_claude', 'anthropic', 'claude-x')];
const NAMES = { openai: 'OpenAI', anthropic: 'Claude (Anthropic)' };

function renderPicker(onChoose = vi.fn(), connections = CONNECTIONS) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ModelPicker choice={{ connectionId: 'con_openai', modelId: 'gpt-a' }} connections={connections} providerNames={NAMES} onChoose={onChoose} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return onChoose;
}

beforeEach(() => {
  api.listConnectionModels.mockReset();
  api.listConnectionModels.mockImplementation(async (id: string) => ({
    models: id === 'con_openai'
      ? [{ id: 'gpt-a', name: 'gpt-a' }, { id: 'gpt-b', name: 'gpt-b' }]
      : [{ id: 'claude-x', name: 'Claude X' }, { id: 'claude-y', name: 'Claude Y' }],
  }));
});

describe('ModelPicker', () => {
  it('lists every model of every connected account, grouped by provider', async () => {
    renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Model AI: gpt-a/ }));
    expect(await screen.findByText('claude-y')).toBeTruthy();
    expect(screen.getByText('gpt-b')).toBeTruthy();
    expect(screen.getByText('OpenAI')).toBeTruthy();
    expect(screen.getByText('Claude (Anthropic)')).toBeTruthy();
    expect(screen.getByRole('option', { name: /gpt-a/ }).getAttribute('aria-selected')).toBe('true');
  });

  it('picks a model with one click, on its own connection', async () => {
    const onChoose = renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Model AI/ }));
    fireEvent.click(await screen.findByRole('option', { name: /claude-y/ }));
    expect(onChoose).toHaveBeenCalledWith({ connectionId: 'con_claude', modelId: 'claude-y' });
  });

  it('filters as the user types and picks with Enter', async () => {
    const onChoose = renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Model AI/ }));
    await screen.findByText('claude-y');
    const search = screen.getByLabelText('Tìm model');
    fireEvent.change(search, { target: { value: 'gpt-b' } });
    await waitFor(() => expect(screen.queryByText('claude-y')).toBeNull());
    fireEvent.keyDown(search, { key: 'Enter' });
    expect(onChoose).toHaveBeenCalledWith({ connectionId: 'con_openai', modelId: 'gpt-b' });
  });

  it('keeps the tested model when the provider cannot list its models', async () => {
    api.listConnectionModels.mockRejectedValue(new Error('no listing'));
    renderPicker(vi.fn(), [conn('con_openai', 'openai', 'gpt-a')]);
    fireEvent.click(screen.getByRole('button', { name: /Model AI/ }));
    expect(await screen.findByText(/chỉ có model đã kiểm tra/)).toBeTruthy();
    expect(screen.getByRole('option', { name: /gpt-a/ })).toBeTruthy();
  });

  it('points to Settings when nothing is connected', () => {
    renderPicker(vi.fn(), []);
    fireEvent.click(screen.getByRole('button', { name: /Model AI: Kết nối AI/ }));
    expect(screen.getByRole('link', { name: /Mở Cài đặt/ }).getAttribute('href')).toBe('/settings');
  });
});
