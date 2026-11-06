import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

afterEach(cleanup);

const api = vi.hoisted(() => ({
  cancelChatGPTSignIn: vi.fn(),
  changeConnectionModel: vi.fn(),
  completeChatGPTSignIn: vi.fn(),
  createModelConnection: vi.fn(),
  deleteModelConnection: vi.fn(),
  getChatGPTSignIn: vi.fn(),
  listConnectionModels: vi.fn(),
  listModelConnections: vi.fn(),
  listModelsForKey: vi.fn(),
  listSupportedProviders: vi.fn(),
  startChatGPTSignIn: vi.fn(),
  testModelConnection: vi.fn(),
}));
vi.mock('../../shared/api/endpoints', () => api);

const prefs = vi.hoisted(() => ({ getDefaultAiProfileId: vi.fn(), setDefaultAiProfileId: vi.fn() }));
vi.mock('../../shared/lib/preferences', () => prefs);

import { AiConnectionsCard } from './AiConnectionsCard';

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
  prefs.getDefaultAiProfileId.mockReturnValue(undefined);
  api.listSupportedProviders.mockResolvedValue({ providers: [OPENAI, ANTHROPIC] });
  api.listModelConnections.mockResolvedValue({ connections: [] });
  api.getChatGPTSignIn.mockResolvedValue({ state: 'none' });
});

describe('AiConnectionsCard', () => {
  it('lists the models a key can use and saves the chosen one', async () => {
    api.listModelsForKey.mockResolvedValue({ models: [{ id: 'gpt-b', name: 'gpt-b' }, { id: 'gpt-c', name: 'gpt-c' }] });
    api.createModelConnection.mockResolvedValue(connection());
    render(<AiConnectionsCard agentEnabled />);
    await screen.findByRole('option', { name: 'OpenAI' });

    fireEvent.change(screen.getByLabelText('AI API key'), { target: { value: 'sk-user' } });
    fireEvent.click(screen.getByRole('button', { name: 'Tải danh sách model' }));
    await waitFor(() => expect(api.listModelsForKey).toHaveBeenCalledWith({ provider_id: 'openai', base_url: undefined, credential: 'sk-user' }));
    await waitFor(() => expect((screen.getByLabelText('AI model') as HTMLInputElement).value).toBe('gpt-b'));

    fireEvent.click(screen.getByRole('button', { name: 'Thêm kết nối' }));
    await waitFor(() => expect(api.createModelConnection).toHaveBeenCalledWith({
      provider_id: 'openai', model_id: 'gpt-b', auth_mode: 'api_key', base_url: undefined, credential: 'sk-user',
    }));
    // The key leaves the form once saved.
    await waitFor(() => expect((screen.getByLabelText('AI API key') as HTMLInputElement).value).toBe(''));
  });

  it('offers ChatGPT sign-in only for a provider that has it', async () => {
    render(<AiConnectionsCard agentEnabled />);
    await screen.findByRole('button', { name: 'ChatGPT Plus/Pro' });
    fireEvent.change(screen.getByLabelText('AI provider'), { target: { value: 'anthropic' } });
    await waitFor(() => expect(screen.queryByRole('button', { name: 'ChatGPT Plus/Pro' })).toBeNull());
  });

  it('runs the device-code sign-in and saves the chosen model', async () => {
    api.startChatGPTSignIn.mockResolvedValue({
      state: 'pending', verification_url: 'https://auth.openai.com/codex/device',
      instructions: 'Enter code: ABCD-12345', expires_in_s: 900,
    });
    api.completeChatGPTSignIn.mockResolvedValue(connection({ auth_mode: 'chatgpt_subscription' }));
    render(<AiConnectionsCard agentEnabled />);
    fireEvent.click(await screen.findByRole('button', { name: 'ChatGPT Plus/Pro' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Kết nối ChatGPT Plus/Pro' }));

    const link = await screen.findByRole('link', { name: 'https://auth.openai.com/codex/device' });
    expect(link.getAttribute('target')).toBe('_blank');
    expect(screen.getByText(/Enter code: ABCD-12345/)).toBeTruthy();

    // The server reports the approval on the next poll.
    api.getChatGPTSignIn.mockResolvedValue({ state: 'connected', models: [{ id: 'gpt-5.5', name: 'GPT-5.5' }], expires_in_s: 800 });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Lưu kết nối' })).toBeTruthy(), { timeout: 5000 });
    await waitFor(() => expect((screen.getByLabelText('AI model') as HTMLInputElement).value).toBe('gpt-5.5'));
    fireEvent.click(screen.getByRole('button', { name: 'Lưu kết nối' }));
    await waitFor(() => expect(api.completeChatGPTSignIn).toHaveBeenCalledWith({ model_id: 'gpt-5.5' }));
  }, 10_000);

  it('says why ChatGPT sign-in is unavailable instead of offering a button that fails', async () => {
    api.listSupportedProviders.mockResolvedValue({ providers: [{
      ...OPENAI, auth_modes: ['api_key'],
      chatgpt_oauth: { available: false, reason: 'this deployment runs no private AI runtime hosts', documentation_url: null },
    }] });
    render(<AiConnectionsCard agentEnabled />);
    fireEvent.click(await screen.findByRole('button', { name: 'ChatGPT Plus/Pro' }));
    expect(await screen.findByText(/no private AI runtime hosts/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Kết nối ChatGPT Plus/Pro' })).toBeNull();
    expect(api.startChatGPTSignIn).not.toHaveBeenCalled();
  });

  it('switches a saved connection to another of its models', async () => {
    api.listModelConnections.mockResolvedValue({ connections: [connection()] });
    api.listConnectionModels.mockResolvedValue({ models: [{ id: 'gpt-a', name: 'gpt-a' }, { id: 'gpt-z', name: 'gpt-z' }] });
    api.changeConnectionModel.mockResolvedValue(connection({ model_id: 'gpt-z', status: 'untested' }));
    render(<AiConnectionsCard agentEnabled />);
    fireEvent.click(await screen.findByRole('button', { name: 'Đổi model' }));
    await waitFor(() => expect(api.listConnectionModels).toHaveBeenCalledWith('con_' + 'a'.repeat(32)));
    const inputs = screen.getAllByLabelText('AI model') as HTMLInputElement[];
    fireEvent.change(inputs[0], { target: { value: 'gpt-z' } });
    fireEvent.click(screen.getByRole('button', { name: 'Lưu model' }));
    await waitFor(() => expect(api.changeConnectionModel).toHaveBeenCalledWith('con_' + 'a'.repeat(32), { model_id: 'gpt-z' }));
  });

  it('shows the server reason when a key is refused', async () => {
    const { ApiError } = await import('../../shared/api/types');
    api.listModelsForKey.mockRejectedValue(new ApiError(400, { error: { code: 'invalid_request', message: 'the provider rejected this credential', retryable: false, details: {} } }));
    render(<AiConnectionsCard agentEnabled />);
    await screen.findByRole('option', { name: 'OpenAI' });
    fireEvent.change(screen.getByLabelText('AI API key'), { target: { value: 'sk-bad' } });
    fireEvent.click(screen.getByRole('button', { name: 'Tải danh sách model' }));
    expect(await screen.findByText('the provider rejected this credential')).toBeTruthy();
  });
});
