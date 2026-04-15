# Chat Persistence E2E Checklist

## Scope

Validate Firestore-backed chat persistence for authenticated users and no-write behavior for guests.

## Preconditions

- Frontend is deployed or running locally against project `tox-agent`.
- Backend `/agent/chat` is reachable and healthy.
- Firestore rules are deployed from `firestore.rules`.
- Test account available (email/password) and a guest browser session available.

## Test Data

- Use a known report flow and ask at least 2 short questions.
- Example prompts:
  - "Summarize key toxicity risk"
  - "What evidence supports this risk?"

## Authenticated User Flow

- [ ] Login with test user.
  - Expected: authenticated UI state is visible.

- [ ] Run analysis and open chat from report page.
  - Expected: chat page opens with report context.

- [ ] Send first question, wait for assistant reply.
  - Expected: reply is returned and rendered.

- [ ] Send second question, wait for assistant reply.
  - Expected: both user+assistant turns are visible.

- [ ] Hard refresh browser on chat page.
  - Expected: previous conversation reloads from Firestore.

- [ ] Close tab, reopen app, navigate back to same chat session.
  - Expected: conversation is restored for same user.

## Firestore Data Validation

- [ ] In Firestore console, open path `users/{uid}/chatSessions/{sessionId}`.
  - Expected fields present:
    - `sessionId`
    - `analysisSessionId`
    - `messages` (array)
    - `messageCount`
    - `lastMessagePreview`
    - `createdAt`, `updatedAt`

- [ ] Verify `messages` contains both roles (`user`, `assistant`) in saved turns.
  - Expected: each successful chat round appends a user+assistant pair.

- [ ] Verify message cap behavior after many turns.
  - Expected: only the most recent 60 messages are retained.

## Guest Flow (No Firestore Writes)

- [ ] Open app in guest mode (not logged in).
- [ ] Open chat and send at least one message.
  - Expected: chat still works, but no `users/{uid}/chatSessions/*` writes occur for guest.

## Security Rule Spot Checks

- [ ] User A cannot read/write User B chat sessions.
  - Expected: denied by rules.

- [ ] Create/update payload must keep `sessionId` matching document id.
  - Expected: mismatch writes are denied.

## Regression Checks

- [ ] Existing analysis history still works (`users/{uid}/analyses`).
- [ ] Chat initial prompt and normal UX behavior unchanged.
- [ ] No frontend build/type errors.

## Exit Criteria

- All checklist items pass.
- No console errors related to Firestore permission denied in valid authenticated flow.
- Persisted sessions survive refresh/reopen for authenticated user.
