import { useEffect, useState } from 'react';

/**
 * Cycles through `phrases` on a timer — the ChatGPT/Claude-style rotating
 * one-liner under a big empty-state title. `index` is exposed alongside the
 * text so callers can key a fade-in animation off it (a plain text swap has
 * no way to signal "this is a new phrase" to CSS).
 */
export function useRotatingText(phrases: readonly string[], intervalMs = 4000): { text: string; index: number } {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (phrases.length <= 1) return;
    const id = setInterval(() => {
      setIndex((prev) => (prev + 1) % phrases.length);
    }, intervalMs);
    return () => clearInterval(id);
  }, [phrases, intervalMs]);

  return { text: phrases[index % phrases.length] ?? '', index };
}
