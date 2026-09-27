/**
 * What the composer sends, and when it refuses to.
 *
 * Three defects, all reachable with default settings:
 *
 * - I03: `explanation_mode` was hardcoded to `required` whenever a SMILES was
 *   present, so the Tox21 assay list became a precondition for *any*
 *   prediction. Typing `CCO` on the default endpoints could not be sent, and
 *   the reason lived in an advanced popover. The composer now exposes neither
 *   the mode nor the assay list: it always sends `on_demand`.
 * - I04: a bare word matched the SMILES heuristic and the composer then
 *   replaced the user's text with it; a sentence containing a molecule
 *   matched nothing and reached the backend with no subject.
 * - I21: Enter submitted during input-method composition.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { MessageComposer } from './MessageComposer';
import type { SendMessageInput } from '../../lib/api/endpoints';

type SendSpy = ReturnType<typeof makeSendSpy>;

/** Typed so `mock.calls[0][0]` is a SendMessageInput rather than `undefined`. */
function makeSendSpy() {
  return vi.fn(async (_input: SendMessageInput) => true);
}

function renderComposer(onSend: SendSpy) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MessageComposer
        sessionId="ses_test"
        hasActiveAnalysis={false}
        disabled={false}
        onSend={onSend}
      />
    </QueryClientProvider>,
  );
}

const sendButton = () => screen.getByRole('button', { name: 'Gửi' });
const textbox = () => screen.getByPlaceholderText('Nhập SMILES hoặc mô tả yêu cầu…');

beforeEach(() => {
  window.localStorage.clear();
});

describe('I03 — prediction does not require an explanation target', () => {
  it('sends a bare SMILES on default settings with no assay chosen', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'CCO');
    await waitFor(() => expect(sendButton()).not.toBeDisabled());
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    const input = onSend.mock.calls[0][0];
    expect(input.molecule).toEqual({ smiles: 'CCO' });
    // The default is on_demand. `required` was what blocked the send.
    expect(input.analysis_options?.explanation_mode).toBe('on_demand');
  });

  it('leaves endpoints to the deployment default and exposes no router controls', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    // The intent select, explanation radios and endpoint/assay checkboxes are
    // gone: the router decides the intent, the server applies its default
    // endpoints, and an explanation is asked for in words.
    expect(screen.queryByRole('radio')).toBeNull();
    expect(screen.queryByRole('combobox')).toBeNull();
    expect(screen.queryByRole('checkbox')).toBeNull();

    await userEvent.type(textbox(), 'CCO');
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.intent_hint).toBe('auto');
    expect(input.analysis_options?.endpoints).toBeUndefined();
    expect(input.analysis_options?.explanation_targets).toBeUndefined();
  });
});

describe('I04 — the question survives the molecule', () => {
  it('keeps a Vietnamese question and extracts the molecule from it', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'Phân tích CCO giúp tôi');
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.molecule).toEqual({ smiles: 'CCO' });
    expect(input.content).toEqual([{ type: 'text', text: 'Phân tích CCO giúp tôi' }]);
  });

  it('does not turn an ordinary word into a molecule', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'hello');
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.molecule).toBeUndefined();
    expect(input.content).toEqual([{ type: 'text', text: 'hello' }]);
  });

  it('drops the text only when the message is nothing but a molecule', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'CC(=O)Oc1ccccc1C(=O)O');
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.molecule).toEqual({ smiles: 'CC(=O)Oc1ccccc1C(=O)O' });
    expect(input.content).toBeUndefined();
  });

  it('asks rather than guessing when a sentence names two molecules', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'Compare CCO and c1ccccc1');
    await waitFor(() => expect(sendButton()).toBeDisabled());
    expect(screen.getByRole('status').textContent).toMatch(/Chọn chuỗi/);
  });

  it('unblocks the ambiguous case once the user picks one candidate', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'Compare CCO and c1ccccc1');
    await userEvent.click(screen.getByRole('button', { name: 'CCO' }));

    await waitFor(() => expect(sendButton()).not.toBeDisabled());
    await userEvent.click(sendButton());
    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.molecule).toEqual({ smiles: 'CCO' });
    expect(input.content).toEqual([{ type: 'text', text: 'Compare CCO and c1ccccc1' }]);
  });
});

describe('the molecule chip', () => {
  it('shows the detected molecule, and removing it sends the text alone', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'Phân tích CCO giúp tôi');
    await userEvent.click(screen.getByRole('button', { name: 'Không phân tích chuỗi này như phân tử' }));
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.molecule).toBeUndefined();
    expect(input.analysis_options).toBeUndefined();
    expect(input.content).toEqual([{ type: 'text', text: 'Phân tích CCO giúp tôi' }]);
  });

  it('keeps a dismissed bare molecule as text rather than dropping it', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'CCO');
    await userEvent.click(screen.getByRole('button', { name: 'Không phân tích chuỗi này như phân tử' }));
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    expect(onSend.mock.calls[0][0].content).toEqual([{ type: 'text', text: 'CCO' }]);
  });

  it('opens an explicit SMILES field from the attach menu', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    expect(screen.queryByRole('textbox', { name: 'SMILES' })).toBeNull();
    await userEvent.click(screen.getByRole('button', { name: 'Thêm SMILES, ảnh hoặc bản vẽ' }));
    await userEvent.click(screen.getByRole('button', { name: 'SMILES' }));
    const field = screen.getByRole('textbox', { name: 'SMILES' });
    await waitFor(() => expect(field).toHaveFocus());

    await userEvent.type(field, 'c1ccccc1');
    await userEvent.click(sendButton());
    await waitFor(() => expect(onSend).toHaveBeenCalled());
    expect(onSend.mock.calls[0][0].molecule).toEqual({ smiles: 'c1ccccc1' });
  });
});

describe('threshold override', () => {
  it('is hidden unless expert mode is on', () => {
    renderComposer(makeSendSpy());
    expect(screen.queryByRole('button', { name: 'Ngưỡng hERG (chuyên gia)' })).toBeNull();
  });

  it('is offered, and sent, in expert mode', async () => {
    window.localStorage.setItem('toxagent.expert_mode', '1');
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.click(screen.getByRole('button', { name: 'Ngưỡng hERG (chuyên gia)' }));
    await userEvent.type(screen.getByLabelText(/hERG threshold override/), '0.3');
    await userEvent.type(textbox(), 'CCO');
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    expect(onSend.mock.calls[0][0].analysis_options?.threshold_overrides).toEqual({ herg: 0.3 });
  });
});

describe('I21 — Enter belongs to the input method while composing', () => {
  it('does not send on Enter during composition', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    const box = textbox();
    fireEvent.change(box, { target: { value: 'xin chao' } });
    fireEvent.compositionStart(box);
    fireEvent.keyDown(box, { key: 'Enter' });

    expect(onSend).not.toHaveBeenCalled();
  });

  it('sends once on Enter after composition ends', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    const box = textbox();
    fireEvent.change(box, { target: { value: 'xin chào' } });
    fireEvent.compositionStart(box);
    fireEvent.compositionEnd(box);
    fireEvent.keyDown(box, { key: 'Enter' });

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
  });

  it('respects isComposing even without composition events', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    const box = textbox();
    fireEvent.change(box, { target: { value: 'xin chao' } });
    fireEvent.keyDown(box, { key: 'Enter', isComposing: true });

    expect(onSend).not.toHaveBeenCalled();
  });
});
