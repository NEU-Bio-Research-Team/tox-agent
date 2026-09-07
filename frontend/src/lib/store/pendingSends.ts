import type { SendMessageInput } from '../api/endpoints';
import type { Message } from '../api/types';

/** The only optimistic server-shaped UI state. It has no fabricated run,
 * assistant answer, analysis, or event sequence; the durable user message
 * confirms it by its idempotency key. */
export interface PendingUserSend {
  clientMessageId: string;
  text?: string;
  smiles?: string;
  hasImage: boolean;
  submittedAt: number;
}

export function pendingSendFromInput(input: SendMessageInput, submittedAt = Date.now()): PendingUserSend | null {
  if (!input.client_message_id) return null;
  return {
    clientMessageId: input.client_message_id,
    text: input.content?.find((part) => part.type === 'text')?.text,
    smiles: input.molecule?.smiles,
    hasImage: input.image !== undefined,
    submittedAt,
  };
}

export function addPendingSend(
  current: readonly PendingUserSend[],
  pending: PendingUserSend | null,
): PendingUserSend[] {
  if (!pending || current.some((item) => item.clientMessageId === pending.clientMessageId)) return [...current];
  return [...current, pending];
}

export function confirmPendingSends(
  current: readonly PendingUserSend[],
  messages: readonly Message[],
): PendingUserSend[] {
  const durableIds = new Set(
    messages
      .filter((message) => message.role === 'user' && message.client_message_id !== null)
      .map((message) => message.client_message_id),
  );
  return current.filter((pending) => !durableIds.has(pending.clientMessageId));
}
