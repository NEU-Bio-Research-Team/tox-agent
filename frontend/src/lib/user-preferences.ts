export type AppLanguage = 'vi' | 'en';

export interface UserPreferences {
  language: AppLanguage;
  clinicalThreshold: number;
  mechanismThreshold: number;
}

const STORAGE_KEY = 'toxagent:user-preferences:v1';

const DEFAULTS: UserPreferences = {
  language: 'vi',
  clinicalThreshold: 0.35,
  mechanismThreshold: 0.5,
};

function clampThreshold(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(1, Math.max(0, value));
}

export function getDefaultPreferences(): UserPreferences {
  return { ...DEFAULTS };
}

export function loadUserPreferences(): UserPreferences {
  if (typeof window === 'undefined') {
    return getDefaultPreferences();
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return getDefaultPreferences();

    const parsed = JSON.parse(raw) as Partial<UserPreferences>;
    const language = parsed.language === 'en' ? 'en' : 'vi';
    return {
      language,
      clinicalThreshold: clampThreshold(Number(parsed.clinicalThreshold), DEFAULTS.clinicalThreshold),
      mechanismThreshold: clampThreshold(Number(parsed.mechanismThreshold), DEFAULTS.mechanismThreshold),
    };
  } catch {
    return getDefaultPreferences();
  }
}

export function saveUserPreferences(prefs: UserPreferences): UserPreferences {
  const normalized: UserPreferences = {
    language: prefs.language === 'en' ? 'en' : 'vi',
    clinicalThreshold: clampThreshold(prefs.clinicalThreshold, DEFAULTS.clinicalThreshold),
    mechanismThreshold: clampThreshold(prefs.mechanismThreshold, DEFAULTS.mechanismThreshold),
  };

  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
  }

  return normalized;
}
