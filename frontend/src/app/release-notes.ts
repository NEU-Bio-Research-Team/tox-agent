import { APP_VERSION } from './build-info';

export const RELEASE_NOTES_EVENT = 'toxagent:open-release-notes';
export const RELEASE_NOTES_STORAGE_KEY = `toxagent.release-notes.seen.${APP_VERSION}`;

export const RELEASE_NOTES_ITEMS = [
  'Added production wiring for the report-level chatbot agent with stable chat-session handoff from /agent/analyze to /agent/chat.',
  'Enabled MolRAG request options in frontend API payloads and backend agent runtime flow (top-k retrieval + similarity threshold).',
  'Added a dedicated MolRAG Evidence & Reasoning section on the report page, including analog retrieval table and fusion snapshot.',
  'Updated report sidebar navigation and section ordering to include the new MolRAG block.',
  'Released frontend bundle version 0.0.8 with updated report experience and compatibility fields for MolRAG payloads.',
];
