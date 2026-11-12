import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ listConnectionModels: vi.fn(), checkConnectionModel: vi.fn() }));
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
  api.checkConnectionModel.mockReset();
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
    expect(screen.getByRole('link', { name: /Dùng API key/ }).getAttribute('href')).toBe('/settings');
    // The ChatGPT sign-in is one click away, on its own method rather than on the key form.
    expect(screen.getByRole('link', { name: /Đăng nhập ChatGPT Plus\/Pro/ }).getAttribute('href')).toBe('/settings?connect=chatgpt');
  });
});


describe('ModelPicker on a ChatGPT sign-in', () => {
  const CHATGPT = [conn('con_chatgpt', 'openai', 'gpt-5.3-codex-spark', 'chatgpt_subscription')];
  const verdict = (status: string, model = 'gpt-5.6') => ({ model_id: model, status, message: 'from the provider' });

  function renderChatGPT(onChoose = vi.fn()) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <ModelPicker
            choice={{ connectionId: 'con_chatgpt', modelId: 'gpt-5.3-codex-spark' }}
            connections={CHATGPT}
            providerNames={NAMES}
            onChoose={onChoose}
          />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    return onChoose;
  }

  beforeEach(() => {
    api.listConnectionModels.mockImplementation(async () => ({
      models: [
        { id: 'gpt-5.3-codex-spark', name: 'GPT-5.3 Codex Spark', availability: 'runtime' },
        { id: 'gpt-5.6', name: 'GPT-5.6', availability: 'catalogue' },
        { id: 'gpt-5.4-mini', name: 'GPT-5.4 mini', availability: 'runtime' },
      ],
    }));
  });

  const open = async () => {
    fireEvent.click(screen.getByRole('button', { name: /Model AI/ }));
    await screen.findByRole('option', { name: /gpt-5\.6/ });
  };

  it('keeps a model only after it answers a real turn', async () => {
    api.checkConnectionModel.mockResolvedValue(verdict('usable'));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.6/ }));
    await waitFor(() => expect(onChoose).toHaveBeenCalledWith({ connectionId: 'con_chatgpt', modelId: 'gpt-5.6' }));
    expect(api.checkConnectionModel).toHaveBeenCalledWith('con_chatgpt', 'gpt-5.6');
  });

  it('does not keep a model the account cannot use, and says so next to it', async () => {
    api.checkConnectionModel.mockResolvedValue(verdict('unsupported', 'gpt-5.4-mini'));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.4-mini/ }));
    expect(await screen.findByText(/không dùng được model này/)).toBeTruthy();
    expect(onChoose).not.toHaveBeenCalled();
    // The picker stays open on the same list, so the user picks another.
    expect(screen.getByRole('option', { name: /gpt-5\.6/ })).toBeTruthy();
  });

  it('tells an expired sign-in apart from a refused model', async () => {
    api.checkConnectionModel.mockResolvedValue(verdict('signin_rejected'));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.6/ }));
    expect(await screen.findByText(/kết nối lại trong Cài đặt/)).toBeTruthy();
    expect(onChoose).not.toHaveBeenCalled();
  });

  it('does not keep a model it could not check, and lets the user try again', async () => {
    api.checkConnectionModel.mockRejectedValueOnce(new Error('network')).mockResolvedValueOnce(verdict('usable'));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.6/ }));
    expect(await screen.findByText(/Chưa kiểm tra được model này/)).toBeTruthy();
    expect(onChoose).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.6/ }));
    await waitFor(() => expect(onChoose).toHaveBeenCalledWith({ connectionId: 'con_chatgpt', modelId: 'gpt-5.6' }));
  });

  it('marks models only the catalogue knows as not yet checked', async () => {
    renderChatGPT();
    await open();
    expect(screen.getByText('chưa kiểm tra')).toBeTruthy();
  });

  it('asks once while a check is running', async () => {
    let finish: (value: unknown) => void = () => {};
    api.checkConnectionModel.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.6/ }));
    expect(await screen.findByText(/Đang kiểm tra/)).toBeTruthy();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.4-mini/ }));
    expect(api.checkConnectionModel).toHaveBeenCalledTimes(1);
    finish(verdict('usable'));
    await waitFor(() => expect(onChoose).toHaveBeenCalledWith({ connectionId: 'con_chatgpt', modelId: 'gpt-5.6' }));
  });

  it('checks a previously saved model too', async () => {
    api.checkConnectionModel.mockResolvedValue(verdict('unsupported', 'gpt-5.3-codex-spark'));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.3-codex-spark/ }));
    expect(await screen.findByText(/không dùng được model này/)).toBeTruthy();
    expect(api.checkConnectionModel).toHaveBeenCalledWith('con_chatgpt', 'gpt-5.3-codex-spark');
    expect(onChoose).not.toHaveBeenCalled();
  });

  it('does not change the selection if the picker was closed during a check', async () => {
    let finish: (value: unknown) => void = () => {};
    api.checkConnectionModel.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.click(screen.getByRole('option', { name: /gpt-5\.6/ }));
    expect(await screen.findByText(/Đang kiểm tra/)).toBeTruthy();
    fireEvent.keyDown(screen.getByLabelText('Tìm model'), { key: 'Escape' });
    finish(verdict('usable'));
    await waitFor(() => expect(screen.queryByText(/Đang kiểm tra/)).toBeNull());
    expect(onChoose).not.toHaveBeenCalled();
  });

  it('lets the user name a model no list has yet, and proves it first', async () => {
    api.checkConnectionModel.mockResolvedValue(verdict('usable', 'gpt-6.1-terra'));
    const onChoose = renderChatGPT();
    await open();
    fireEvent.change(screen.getByLabelText('Mã model khác'), { target: { value: ' gpt-6.1-terra ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Kiểm tra' }));
    await waitFor(() => expect(onChoose).toHaveBeenCalledWith({ connectionId: 'con_chatgpt', modelId: 'gpt-6.1-terra' }));
    expect(api.checkConnectionModel).toHaveBeenCalledWith('con_chatgpt', 'gpt-6.1-terra');
  });

  it('refuses a typed id the server could never select, without asking the server', async () => {
    renderChatGPT();
    await open();
    fireEvent.change(screen.getByLabelText('Mã model khác'), { target: { value: 'not a model' } });
    fireEvent.click(screen.getByRole('button', { name: 'Kiểm tra' }));
    expect(await screen.findByText('Mã model không hợp lệ.')).toBeTruthy();
    expect(api.checkConnectionModel).not.toHaveBeenCalled();
  });

  it('shows a typed model the session already holds even though no list names it', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <ModelPicker choice={{ connectionId: 'con_chatgpt', modelId: 'gpt-7-future' }} connections={CHATGPT} providerNames={NAMES} onChoose={vi.fn()} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: /Model AI: gpt-7-future/ }));
    expect((await screen.findByRole('option', { name: /gpt-7-future/ })).getAttribute('aria-selected')).toBe('true');
  });

  it('never asks for a key-based provider, whose models its provider lists', async () => {
    api.listConnectionModels.mockImplementation(async () => ({
      models: [{ id: 'gpt-a', name: 'gpt-a' }, { id: 'gpt-b', name: 'gpt-b' }],
    }));
    const onChoose = renderPicker(vi.fn(), [conn('con_openai', 'openai', 'gpt-a')]);
    fireEvent.click(screen.getByRole('button', { name: /Model AI/ }));
    fireEvent.click(await screen.findByRole('option', { name: /gpt-b/ }));
    expect(onChoose).toHaveBeenCalledWith({ connectionId: 'con_openai', modelId: 'gpt-b' });
    expect(api.checkConnectionModel).not.toHaveBeenCalled();
  });
});
