import { APP_VERSION } from './build-info';

export const RELEASE_NOTES_EVENT = 'toxagent:open-release-notes';
export const RELEASE_NOTES_STORAGE_KEY = `toxagent.release-notes.seen.${APP_VERSION}`;

export const RELEASE_NOTES_ITEMS = [
  'Support diverse methods to insert SMILES inputs for users: Text, Draw, Upload photo.',
  'A new page automatically appeared when users prompted and interacted with chatbot agent.',
  'Released frontend bundle version 0.0.9 with updated input experiences and chatbot interaction experiences.',
];
