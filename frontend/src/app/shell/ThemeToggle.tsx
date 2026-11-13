import { useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { Button } from '../../shared/ui/button';
import { applyTheme, getTheme, type Theme } from '../../shared/lib/theme';

/** Light/dark switch. The choice is stored by `applyTheme` under the key the
 * production site already used. */
export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>(() => getTheme());
  const next: Theme = theme === 'dark' ? 'light' : 'dark';
  const label = next === 'dark' ? 'Chuyển sang giao diện tối' : 'Chuyển sang giao diện sáng';
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className={className}
      title={label}
      aria-label={label}
      onClick={() => {
        applyTheme(next);
        setTheme(next);
      }}
    >
      {theme === 'dark' ? <Sun /> : <Moon />}
    </Button>
  );
}
