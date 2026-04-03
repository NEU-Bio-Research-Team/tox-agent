export const RELEASE_NOTES_EVENT = 'toxagent:open-release-notes';
export const RELEASE_NOTES_STORAGE_KEY = `toxagent.release-notes.seen.${__APP_VERSION__}`;

export const RELEASE_NOTES_ITEMS = [
  'Fixed structural rendering so molecule image and attribution heatmap no longer collapse into the same fallback output.',
  'Added a clickable build/version badge in the navbar to reopen release notes anytime.',
  'Introduced an automatic first-visit popup for this version to highlight important production changes.',
  'Kept screening + reporting pipeline stable after warm-start and deterministic rebuild hardening.',
];
