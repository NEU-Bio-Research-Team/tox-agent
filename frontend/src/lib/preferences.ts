const EXPERT_MODE_KEY = 'toxagent.expert_mode';
const DRAFT_KEY_PREFIX = 'toxagent.draft.';
const LAYOUT_KEY = 'toxagent.layout.v1';

/** Section 7.6: only presentation preferences persist, versioned so a future
 * shape change can migrate or drop the key instead of crashing on old JSON.
 * Selection, unread markers and the "manually closed" flag stay in memory —
 * they are session-lifetime UI state, not durable preference. */
export interface LayoutPreferences {
  sidebarCollapsed: boolean;
  artifactsWidthPct: number;
}

const DEFAULT_LAYOUT: LayoutPreferences = {
  sidebarCollapsed: false,
  artifactsWidthPct: 28,
};

export function getLayoutPreferences(): LayoutPreferences {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY);
    if (!raw) return DEFAULT_LAYOUT;
    const parsed = JSON.parse(raw) as Partial<LayoutPreferences>;
    return {
      sidebarCollapsed: Boolean(parsed.sidebarCollapsed),
      artifactsWidthPct: clampArtifactsWidth(parsed.artifactsWidthPct ?? DEFAULT_LAYOUT.artifactsWidthPct),
    };
  } catch {
    return DEFAULT_LAYOUT;
  }
}

/** Distinguishes "never toggled" from "explicitly expanded" so a smart
 * per-breakpoint default (section 8.2.2: tablet starts collapsed to rail)
 * only applies until the user actually makes a choice — after that, their
 * choice wins regardless of viewport. */
export function getStoredSidebarCollapsed(): boolean | null {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<LayoutPreferences>;
    return typeof parsed.sidebarCollapsed === 'boolean' ? parsed.sidebarCollapsed : null;
  } catch {
    return null;
  }
}

export function setLayoutPreferences(next: Partial<LayoutPreferences>): void {
  try {
    const current = getLayoutPreferences();
    const merged = { ...current, ...next };
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(merged));
  } catch {
    // best effort — a blocked storage just means the default applies next load
  }
}

/** 320–560px is the plan's proposed clamp (section 8.2); expressed as a
 * percentage of viewport width here since the panel is laid out with
 * react-resizable-panels, which sizes in percentages, not pixels. */
export function clampArtifactsWidth(pct: number): number {
  if (typeof window === 'undefined') return pct;
  const minPct = (320 / window.innerWidth) * 100;
  const maxPct = (560 / window.innerWidth) * 100;
  return Math.min(Math.max(pct, minPct), maxPct);
}

/** The one piece of client state this app is allowed to own outright — see
 * redesign plan section 7.1, rule 2: everything else is a server read. */
export function getExpertModeEnabled(): boolean {
  try {
    return localStorage.getItem(EXPERT_MODE_KEY) === '1';
  } catch {
    return false;
  }
}

export function setExpertModeEnabled(enabled: boolean): void {
  try {
    if (enabled) localStorage.setItem(EXPERT_MODE_KEY, '1');
    else localStorage.removeItem(EXPERT_MODE_KEY);
  } catch {
    // best effort
  }
}

export function getDraft(sessionId: string): string {
  try {
    return localStorage.getItem(`${DRAFT_KEY_PREFIX}${sessionId}`) ?? '';
  } catch {
    return '';
  }
}

export function setDraft(sessionId: string, text: string): void {
  try {
    if (text) localStorage.setItem(`${DRAFT_KEY_PREFIX}${sessionId}`, text);
    else localStorage.removeItem(`${DRAFT_KEY_PREFIX}${sessionId}`);
  } catch {
    // best effort
  }
}
