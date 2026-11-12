import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

afterEach(cleanup);

const api = vi.hoisted(() => ({
  cancelChatGPTSignIn: vi.fn(),
  completeChatGPTSignIn: vi.fn(),
  createModelConnection: vi.fn(),
  deleteModelConnection: vi.fn(),
  getChatGPTSignIn: vi.fn(),
  listModelConnections: vi.fn(),
  listModelsForKey: vi.fn(),
  listSupportedProviders: vi.fn(),
  startChatGPTSignIn: vi.fn(),
  testModelConnection: vi.fn(),
}));
vi.mock('../../shared/api/endpoints', () => api);

import { AiConnectionsCard } from './AiConnectionsCard';

/** The card reads `?connect=` from the address, so it needs a router. */
function renderCard(url = '/settings') {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <AiConnectionsCard agentEnabled />
    </MemoryRouter>,
  );
}

const OPENAI = {
  provider_id: 'openai', display_name: 'OpenAI', protocol: 'openai_chat_completions',
  default_base_url: 'https://api.openai.com/v1', base_url_required: false,
  auth_modes: ['api_key', 'chatgpt_subscription'], note: '', lists_models: true,
  chatgpt_oauth: { available: true, reason: 'local', documentation_url: 'https://developers.openai.com/x' },
};
const ANTHROPIC = {
  ...OPENAI, provider_id: 'anthropic', display_name: 'Claude (Anthropic)', protocol: 'anthropic_messages',
  auth_modes: ['api_key'], chatgpt_oauth: null,
};

function connection(overrides = {}) {
  return {
    connection_id: 'con_' + 'a'.repeat(32), provider_id: 'openai', model_id: 'gpt-a',
    display_name: 'openai · gpt-a', auth_mode: 'api_key', base_url: null, has_credential: true,
    capabilities: { streaming: true, tool_calls: true, structured_output: true, context_size: null },
    status: 'ready', ...overrides,
  };
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  api.listSupportedProviders.mockResolvedValue({ providers: [OPENAI, ANTHROPIC] });
  api.listModelConnections.mockResolvedValue({ connections: [] });
  api.getChatGPTSignIn.mockResolvedValue({ state: 'none' });
});

describe('AiConnectionsCard', () => {
  it('connects with just a key: tests it on a listed model and never asks for one', async () => {
    api.listModelsForKey.mockResolvedValue({ models: [{ id: 'gpt-b', name: 'gpt-b' }, { id: 'gpt-c', name: 'gpt-c' }] });
    api.createModelConnection.mockResolvedValue(connection({ status: 'untested' }));
    api.testModelConnection.mockResolvedValue(connection());
    renderCard();
    await screen.findByRole('option', { name: 'OpenAI' });
    // Models are picked in the chat, not here.
    expect(screen.queryByLabelText('AI model')).toBeNull();
    expect(screen.queryByLabelText('Model ID để kiểm tra')).toBeNull();

    fireEvent.change(screen.getByLabelText('AI API key'), { target: { value: 'sk-user' } });
    fireEvent.click(screen.getByRole('button', { name: 'Kết nối' }));
    await waitFor(() => expect(api.listModelsForKey).toHaveBeenCalledWith({ provider_id: 'openai', base_url: undefined, credential: 'sk-user' }));
    await waitFor(() => expect(api.createModelConnection).toHaveBeenCalledWith({
      provider_id: 'openai', model_id: 'gpt-b', auth_mode: 'api_key', base_url: undefined, credential: 'sk-user',
    }));
    await waitFor(() => expect(api.testModelConnection).toHaveBeenCalledWith('con_' + 'a'.repeat(32)));
    expect(await screen.findByText(/Chọn model ngay trong khung chat/)).toBeTruthy();
    // The key leaves the form once saved.
    expect((screen.getByLabelText('AI API key') as HTMLInputElement).value).toBe('');
  });

  it('asks for a model id only when the provider lists none', async () => {
    api.listModelsForKey.mockResolvedValue({ models: [] });
    renderCard();
    await screen.findByRole('option', { name: 'OpenAI' });
    fireEvent.change(screen.getByLabelText('AI API key'), { target: { value: 'sk-user' } });
    fireEvent.click(screen.getByRole('button', { name: 'Kết nối' }));
    expect(await screen.findByLabelText('Model ID để kiểm tra')).toBeTruthy();
    expect(api.createModelConnection).not.toHaveBeenCalled();
  });

  it('has no model controls on a saved connection', async () => {
    api.listModelConnections.mockResolvedValue({ connections: [connection()] });
    renderCard();
    await screen.findByRole('button', { name: 'Kiểm tra lại' });
    expect(screen.queryByRole('button', { name: 'Đổi model' })).toBeNull();
    expect(screen.queryByLabelText('Kết nối dùng cho các phiên')).toBeNull();
  });

  it('offers ChatGPT sign-in only for a provider that has it', async () => {
    renderCard();
    await screen.findByRole('button', { name: 'ChatGPT Plus/Pro' });
    fireEvent.change(screen.getByLabelText('AI provider'), { target: { value: 'anthropic' } });
    await waitFor(() => expect(screen.queryByRole('button', { name: 'ChatGPT Plus/Pro' })).toBeNull());
  });

  it('runs the device-code sign-in and saves it as soon as it is approved', async () => {
    api.startChatGPTSignIn.mockResolvedValue({
      state: 'pending', verification_url: 'https://auth.openai.com/codex/device',
      instructions: 'Enter code: ABCD-12345', expires_in_s: 900,
    });
    api.completeChatGPTSignIn.mockResolvedValue(connection({ auth_mode: 'chatgpt_subscription' }));
    renderCard();
    fireEvent.click(await screen.findByRole('button', { name: 'ChatGPT Plus/Pro' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Kết nối ChatGPT Plus/Pro' }));

    const link = await screen.findByRole('link', { name: 'https://auth.openai.com/codex/device' });
    expect(link.getAttribute('target')).toBe('_blank');
    expect(screen.getByText(/Enter code: ABCD-12345/)).toBeTruthy();

    // The server reports the approval on the next poll.
    api.getChatGPTSignIn.mockResolvedValue({ state: 'connected', models: [{ id: 'gpt-5.5', name: 'GPT-5.5' }], expires_in_s: 800 });
    await waitFor(() => expect(api.completeChatGPTSignIn).toHaveBeenCalledWith({ model_id: 'gpt-5.5' }), { timeout: 5000 });
  }, 10_000);

  it('opens on the ChatGPT sign-in when sent there from the chat', async () => {
    api.listSupportedProviders.mockResolvedValue({ providers: [ANTHROPIC, OPENAI] });
    renderCard('/settings?connect=chatgpt');
    expect(await screen.findByRole('button', { name: 'Kết nối ChatGPT Plus/Pro' })).toBeTruthy();
    expect((screen.getByLabelText('AI provider') as HTMLSelectElement).value).toBe('openai');
    expect(screen.getByRole('button', { name: 'ChatGPT Plus/Pro', pressed: true })).toBeTruthy();
  });

  it('opens on the key form, as before, when ChatGPT is asked for but not offered', async () => {
    api.listSupportedProviders.mockResolvedValue({
      providers: [{ ...OPENAI, chatgpt_oauth: { available: false, reason: 'turned off', documentation_url: null } }, ANTHROPIC],
    });
    renderCard('/settings?connect=chatgpt');
    await screen.findByRole('option', { name: 'OpenAI' });
    expect((screen.getByLabelText('AI provider') as HTMLSelectElement).value).toBe('openai');
    expect(screen.queryByRole('button', { name: 'Kết nối ChatGPT Plus/Pro' })).toBeNull();
    expect(screen.getByLabelText('AI API key')).toBeTruthy();
  });

  it('says why ChatGPT sign-in is unavailable instead of offering a button that fails', async () => {
    api.listSupportedProviders.mockResolvedValue({ providers: [{
      ...OPENAI, auth_modes: ['api_key'],
      chatgpt_oauth: { available: false, reason: 'this deployment runs no private AI runtime hosts', documentation_url: null },
    }] });
    renderCard();
    fireEvent.click(await screen.findByRole('button', { name: 'ChatGPT Plus/Pro' }));
    expect(await screen.findByText(/no private AI runtime hosts/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Kết nối ChatGPT Plus/Pro' })).toBeNull();
    expect(api.startChatGPTSignIn).not.toHaveBeenCalled();
  });

  it('shows the server reason when a key is refused', async () => {
    const { ApiError } = await import('../../shared/api/types');
    api.listModelsForKey.mockRejectedValue(new ApiError(400, { error: { code: 'invalid_request', message: 'the provider rejected this credential', retryable: false, details: {} } }));
    renderCard();
    await screen.findByRole('option', { name: 'OpenAI' });
    fireEvent.change(screen.getByLabelText('AI API key'), { target: { value: 'sk-bad' } });
    fireEvent.click(screen.getByRole('button', { name: 'Kết nối' }));
    expect(await screen.findByText('the provider rejected this credential')).toBeTruthy();
  });
});
