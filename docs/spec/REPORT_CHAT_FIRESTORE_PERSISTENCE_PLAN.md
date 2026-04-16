# Report Chat Firestore Persistence Plan

## Context

Current report chat works for a single runtime instance but does not persist chat turns in Firestore. The backend already returns `chat_session_id` from `/agent/analyze` and `/agent/chat`, while frontend currently persists only analysis history.

## Goals

- Persist report-chat sessions by user at `users/{uid}/chatSessions/{sessionId}`.
- Keep chat ownership strict: a user can only read/write their own sessions.
- Support chat resume in `ChatbotPage` when a known `chatSessionId` exists.
- Keep backend API contract unchanged for this phase.

## Proposed Data Model

Document path:
- `users/{uid}/chatSessions/{sessionId}`

Document fields:
- `sessionId: string`
- `analysisSessionId: string | null`
- `smiles: string | null`
- `title: string` (derived from first user message)
- `messages: Array<{ role: 'user' | 'assistant'; content: string; timestamp: number }>`
- `messageCount: number`
- `lastMessagePreview: string`
- `createdAt: serverTimestamp`
- `updatedAt: serverTimestamp`

Guard rails:
- cap messages to last 60 per session to avoid document growth.
- append successful turns only (user + assistant pair).

## Implementation Plan

### Layer 1: Firestore Rules

- Extend `firestore.rules` with:
  - `match /users/{userId}/chatSessions/{sessionId}`
  - owner check: `request.auth.uid == userId`
  - session id consistency on create/update:
    - `request.resource.data.sessionId == sessionId`

### Layer 2: Frontend Chat Persistence Service

Create `frontend/src/lib/chat-history.ts` with:
- `loadChatSessionFromFirestore(uid, sessionId)`
- `appendChatTurnToFirestore(uid, payload)`
- robust parsing for timestamps and message arrays
- transactional append + trim behavior

### Layer 3: Chat Flow Wiring

Update `frontend/src/app/pages/chatbot-page.tsx`:
- load existing messages when user is authenticated and `activeChatSessionId` exists.
- after each successful `/agent/chat`, append the user+assistant turn to Firestore.
- update active session id from backend response and persist under that id.
- keep guest flow unchanged (no Firestore writes).

### Layer 4: Backend

No schema or endpoint changes required in this pass.
Reason: `/agent/chat` already returns `chat_session_id`, and request already supports `chat_session_id`, `analysis_session_id`, and `report_state`.

## Validation

- `npm --prefix frontend run build`
- open report -> ask question -> navigate to chat -> refresh -> confirm session messages reload for logged-in user.
- verify guest mode still functions without persistence.

## Deployment Scope

- Backend deploy to Cloud Run (requested).
- Firestore rules should be deployed in a separate Firebase deploy step when ready.

## Risks and Follow-ups

- In-memory backend chat session remains a known runtime limitation across scaled instances.
- Long-term recommendation: move server-side chat state to Redis/Firestore-backed session cache.
- Future enhancement: list/reopen previous chat sessions from UI sidebar.
