export type AppLanguage = 'vi' | 'en';
export type InferenceBackend = 'xsmiles' | 'chemberta' | 'pubchem' | 'molformer';

export interface UserPreferences {
  language: AppLanguage;
  clinicalThreshold: number;
  mechanismThreshold: number;
  inferenceBackend: InferenceBackend;
}

const STORAGE_KEY = 'toxagent:user-preferences:v1';

const DEFAULTS: UserPreferences = {
  language: 'en',
  clinicalThreshold: 0.35,
  mechanismThreshold: 0.5,
  inferenceBackend: 'xsmiles',
};

function normalizeInferenceBackend(value: unknown): InferenceBackend {
  const candidate = String(value ?? '').trim().toLowerCase();
  if (candidate === 'chembert') return 'chemberta';
  if (candidate === 'chemberta' || candidate === 'pubchem' || candidate === 'molformer') {
    return candidate;
  }
  return 'xsmiles';
}

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
    const language = parsed.language === 'vi' ? 'vi' : 'en';
    return {
      language,
      clinicalThreshold: clampThreshold(Number(parsed.clinicalThreshold), DEFAULTS.clinicalThreshold),
      mechanismThreshold: clampThreshold(Number(parsed.mechanismThreshold), DEFAULTS.mechanismThreshold),
      inferenceBackend: normalizeInferenceBackend(parsed.inferenceBackend),
    };
  } catch {
    return getDefaultPreferences();
  }
}

export function saveUserPreferences(prefs: UserPreferences): UserPreferences {
  const normalized: UserPreferences = {
    language: prefs.language === 'vi' ? 'vi' : 'en',
    clinicalThreshold: clampThreshold(prefs.clinicalThreshold, DEFAULTS.clinicalThreshold),
    mechanismThreshold: clampThreshold(prefs.mechanismThreshold, DEFAULTS.mechanismThreshold),
    inferenceBackend: normalizeInferenceBackend(prefs.inferenceBackend),
  };

  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
  }

  return normalized;
}
