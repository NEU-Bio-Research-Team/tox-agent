import { APP_VERSION } from './build-info';

export const RELEASE_NOTES_EVENT = 'toxagent:open-release-notes';
export const RELEASE_NOTES_STORAGE_KEY = `toxagent.release-notes.seen.${APP_VERSION}`;

export const RELEASE_NOTES_ITEMS = [
  'Added a new Documents page with a report-style vertical sidebar and a full User Guide section.',
  'Integrated all User Guide screenshots directly into the Documents page for quick walkthrough access.',
  'Added a highlighted cyan Rate us button in the top task bar that routes directly to the Google Form.',
  'Added a Documents entry in navigation and routing so users can open docs from both desktop and mobile menus.',
  'Switched the frontend to English-only mode and removed the Vietnamese language option from Settings.',
];
