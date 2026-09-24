// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { addPendingSend, confirmPendingSends, pendingSendFromInput } from './pendingSends';
import type { Message } from '../api/types';

const durableMessage: Message = {
  message_id: 'msg_1',
  role: 'user',
  sequence: 1,
  created_at: '2026-09-06T00:00:00Z',
  client_message_id: 'web-1',
  parts: [],
};

describe('pending user sends', () => {
  it('is idempotent by client_message_id and only confirms on the durable user message', () => {
    const pending = pendingSendFromInput({
      client_message_id: 'web-1',
      content: [{ type: 'text', text: 'Please analyze this.' }],
      molecule: { smiles: 'CCO' },
    }, 123);
    const initial = addPendingSend([], pending);

    expect(addPendingSend(initial, pending)).toHaveLength(1);
    expect(confirmPendingSends(initial, [])).toEqual(initial);
    expect(confirmPendingSends(initial, [durableMessage])).toEqual([]);
  });

  it('does not create a pending row when the caller omitted an idempotency key', () => {
    expect(pendingSendFromInput({ content: [{ type: 'text', text: 'hello' }] })).toBeNull();
  });
});
