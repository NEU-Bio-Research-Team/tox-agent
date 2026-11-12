/** Light/dark preference. The key and the `dark` class on <html> are the ones
 * the production site already used, so a returning visitor keeps their choice. */
export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'theme';

function storedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value === 'dark' || value === 'light' ? value : null;
  } catch {
    return null;
  }
}

export function getTheme(): Theme {
  if (typeof window === 'undefined') return 'light';
  return storedTheme() ?? (document.documentElement.classList.contains('dark') ? 'dark' : 'light');
}

export function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle('dark', theme === 'dark');
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Storage unavailable: the choice holds for this page only.
  }
}

/** Called once before the first render so a stored dark preference does not
 * flash light. With nothing stored the page stays as the markup left it. */
export function initTheme(): void {
  const theme = storedTheme();
  if (theme) document.documentElement.classList.toggle('dark', theme === 'dark');
}
