import { useEffect, useState } from 'react';

export type Breakpoint = 'mobile' | 'tablet' | 'desktop';

/** Plan section 8.2.2: <768px is a single mobile column, 768–1279px is the
 * rail/Sheet tablet layout, >=1280px is the full three-region desktop
 * layout. Distinct from `use-mobile.ts`'s 768px-only check, which the
 * shadcn sidebar primitive owns for its own Sheet-vs-inline decision. */
function computeBreakpoint(width: number): Breakpoint {
  if (width < 768) return 'mobile';
  if (width < 1280) return 'tablet';
  return 'desktop';
}

export function useBreakpoint(): Breakpoint {
  const [breakpoint, setBreakpoint] = useState<Breakpoint>(() =>
    typeof window === 'undefined' ? 'desktop' : computeBreakpoint(window.innerWidth),
  );

  useEffect(() => {
    const update = () => setBreakpoint(computeBreakpoint(window.innerWidth));
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return breakpoint;
}
