import { useEffect, useState } from 'react';

/**
 * Which of the given section ids is currently under the top of the viewport.
 * Used by the Settings section list and the docs table of contents. Works with a
 * scrolling ancestor, because an IntersectionObserver clips through overflow.
 * Where IntersectionObserver is missing (jsdom), the first id stays active.
 */
export function useActiveSection(ids: ReadonlyArray<string>): readonly [string, (id: string) => void] {
  const [active, setActive] = useState(ids[0] ?? '');
  const key = ids.join('|');

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return undefined;
    const elements = ids.map((id) => document.getElementById(id)).filter((el): el is HTMLElement => el !== null);
    if (elements.length === 0) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) if (entry.isIntersecting) setActive(entry.target.id);
      },
      { rootMargin: '-12% 0px -78% 0px', threshold: 0 },
    );
    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
    // `ids` is represented by `key`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return [active, setActive] as const;
}

/** Scroll a section into view and mark it active at once. */
export function scrollToSection(id: string, setActive: (id: string) => void): void {
  setActive(id);
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
