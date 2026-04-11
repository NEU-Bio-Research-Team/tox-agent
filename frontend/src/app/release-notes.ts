import { APP_VERSION } from './build-info';

export const RELEASE_NOTES_EVENT = 'toxagent:open-release-notes';
export const RELEASE_NOTES_STORAGE_KEY = `toxagent.release-notes.seen.${APP_VERSION}`;

export const RELEASE_NOTES_ITEMS = [
  'Added support for new model-predict payload fields so the UI stays compatible with both binary_tox_model and tox_type_model routing updates.',
  'Fixed report rendering when risk_level is returned as an object (level + description), preventing production crashes.',
  'Kept release notes popup and version badge behavior so build updates are visible immediately after deployment.',
  'Preserved screening + reporting pipeline stability after warm-start and deterministic fallback hardening.',
];
