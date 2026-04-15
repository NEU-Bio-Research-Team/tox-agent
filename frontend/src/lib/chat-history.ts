import { doc, getDoc, runTransaction, serverTimestamp } from 'firebase/firestore';
import { db } from '../firebase-config';

export type ChatRole = 'user' | 'assistant';

export interface PersistedChatMessage {
  role: ChatRole;
  content: string;
  timestamp: number;
}

export interface ChatSessionRecord {
  sessionId: string;
  analysisSessionId?: string | null;
  smiles?: string | null;
  title?: string;
  messages: PersistedChatMessage[];
  messageCount: number;
  lastMessagePreview?: string;
  createdAtMs?: number;
  updatedAtMs?: number;
}

export interface AppendChatTurnInput {
  sessionId: string;
  analysisSessionId?: string | null;
  smiles?: string | null;
  messages: PersistedChatMessage[];
}

const MAX_MESSAGES_PER_SESSION = 60;
const TITLE_MAX_LENGTH = 80;
const PREVIEW_MAX_LENGTH = 220;

function chatSessionRef(uid: string, sessionId: string) {
  return doc(db, 'users', uid, 'chatSessions', sessionId);
}

function toMillis(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (value && typeof value === 'object') {
    const maybeTimestamp = value as { toMillis?: () => number };
    if (typeof maybeTimestamp.toMillis === 'function') {
      try {
        const result = maybeTimestamp.toMillis();
        return Number.isFinite(result) ? result : undefined;
      } catch {
        return undefined;
      }
    }
  }

  return undefined;
}

function normalizeMessage(value: unknown): PersistedChatMessage | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const data = value as {
    role?: unknown;
    content?: unknown;
    timestamp?: unknown;
  };

  const role = data.role;
  const content = typeof data.content === 'string' ? data.content.trim() : '';
  const timestamp = typeof data.timestamp === 'number' && Number.isFinite(data.timestamp) ? data.timestamp : Date.now();

  if ((role !== 'user' && role !== 'assistant') || !content) {
    return null;
  }

  return {
    role,
    content,
    timestamp,
  };
}

function clampText(text: string, maxLength: number): string {
  const normalized = text.trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3)}...`;
}

export async function loadChatSessionFromFirestore(uid: string, sessionId: string): Promise<ChatSessionRecord | null> {
  if (!uid || !sessionId) {
    return null;
  }

  const snap = await getDoc(chatSessionRef(uid, sessionId));
  if (!snap.exists()) {
    return null;
  }

  const data = snap.data();
  const rawMessages = Array.isArray(data.messages) ? data.messages : [];
  const messages = rawMessages
    .map((item) => normalizeMessage(item))
    .filter((item): item is PersistedChatMessage => item !== null);

  const createdAtMs = toMillis(data.createdAt);
  const updatedAtMs = toMillis(data.updatedAt);

  return {
    sessionId,
    analysisSessionId: typeof data.analysisSessionId === 'string' ? data.analysisSessionId : null,
    smiles: typeof data.smiles === 'string' ? data.smiles : null,
    title: typeof data.title === 'string' ? data.title : undefined,
    messages,
    messageCount: typeof data.messageCount === 'number' ? data.messageCount : messages.length,
    lastMessagePreview: typeof data.lastMessagePreview === 'string' ? data.lastMessagePreview : undefined,
    createdAtMs,
    updatedAtMs,
  };
}

export async function appendChatTurnToFirestore(uid: string, input: AppendChatTurnInput): Promise<void> {
  if (!uid || !input.sessionId) {
    return;
  }

  const incomingMessages = input.messages
    .map((item) => normalizeMessage(item))
    .filter((item): item is PersistedChatMessage => item !== null);

  if (incomingMessages.length === 0) {
    return;
  }

  const ref = chatSessionRef(uid, input.sessionId);

  await runTransaction(db, async (transaction) => {
    const snapshot = await transaction.get(ref);
    const existingData = snapshot.exists() ? snapshot.data() : null;

    const existingMessages = Array.isArray(existingData?.messages)
      ? existingData.messages
          .map((item: unknown) => normalizeMessage(item))
          .filter((item: PersistedChatMessage | null): item is PersistedChatMessage => item !== null)
      : [];

    const mergedMessages = [...existingMessages, ...incomingMessages].slice(-MAX_MESSAGES_PER_SESSION);
    const lastMessage = mergedMessages.length > 0 ? mergedMessages[mergedMessages.length - 1] : incomingMessages[incomingMessages.length - 1];
    const firstUserMessage = mergedMessages.find((item) => item.role === 'user');

    const existingTitle = typeof existingData?.title === 'string' ? existingData.title : '';
    const derivedTitle = existingTitle || firstUserMessage?.content || incomingMessages[0].content;

    const payload: Record<string, unknown> = {
      sessionId: input.sessionId,
      analysisSessionId:
        input.analysisSessionId ??
        (typeof existingData?.analysisSessionId === 'string' ? existingData.analysisSessionId : null),
      smiles: input.smiles ?? (typeof existingData?.smiles === 'string' ? existingData.smiles : null),
      title: clampText(derivedTitle, TITLE_MAX_LENGTH),
      messages: mergedMessages,
      messageCount: mergedMessages.length,
      lastMessagePreview: clampText(lastMessage?.content ?? '', PREVIEW_MAX_LENGTH),
      updatedAt: serverTimestamp(),
    };

    if (!snapshot.exists()) {
      payload.createdAt = serverTimestamp();
    }

    transaction.set(ref, payload, { merge: true });
  });
}
