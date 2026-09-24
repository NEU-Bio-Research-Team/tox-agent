/**
 * What the composer sends, and when it refuses to.
 *
 * Three defects, all reachable with default settings:
 *
 * - I03: `explanation_mode` was hardcoded to `required` whenever a SMILES was
 *   present, so the Tox21 assay list became a precondition for *any*
 *   prediction. Typing `CCO` on the default endpoints could not be sent, and
 *   the reason lived in an advanced popover.
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

vi.mock('../../lib/api/endpoints', async () => {
  const actual = await vi.importActual<Record<string, unknown>>('../../lib/api/endpoints');
  return {
    ...actual,
    quickPredictCapabilities: vi.fn(async () => ({
      capability_version: 'predict-capabilities-v2',
      endpoints: [
        { id: 'herg', display_name: 'hERG', enabled: true, tasks: [], blocked_reason: null },
        {
          id: 'tox21',
          display_name: 'Tox21',
          enabled: true,
          tasks: ['NR-AR', 'SR-MMP'],
          blocked_reason: null,
        },
      ],
    })),
  };
});

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

async function chooseExplanationMode(label: string) {
  await userEvent.click(screen.getByRole('radio', { name: label }));
}
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

  it('blocks an explicitly required Tox21 explanation with no assay, and says why next to the button', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'CCO');
    await chooseExplanationMode('Bắt buộc kèm giải thích');

    await waitFor(() => expect(sendButton()).toBeDisabled());
    expect(screen.getByRole('status').textContent).toMatch(/assay Tox21/i);
  });

  it('sends nothing for explanation targets when the user asked for prediction only', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'CCO');
    await chooseExplanationMode('Chỉ dự đoán');
    await userEvent.click(sendButton());

    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.analysis_options?.explanation_mode).toBe('none');
    expect(input.analysis_options?.explanation_targets).toEqual([]);
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
    expect(screen.getByRole('status').textContent).toMatch(/ô SMILES/);
  });

  it('unblocks the ambiguous case as soon as the SMILES field is filled in', async () => {
    const onSend = makeSendSpy();
    renderComposer(onSend);

    await userEvent.type(textbox(), 'Compare CCO and c1ccccc1');
    await userEvent.type(screen.getByPlaceholderText('SMILES (tuỳ chọn)'), 'CCO');

    await waitFor(() => expect(sendButton()).not.toBeDisabled());
    await userEvent.click(sendButton());
    await waitFor(() => expect(onSend).toHaveBeenCalled());
    const input = onSend.mock.calls[0][0];
    expect(input.molecule).toEqual({ smiles: 'CCO' });
    expect(input.content).toEqual([{ type: 'text', text: 'Compare CCO and c1ccccc1' }]);
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
