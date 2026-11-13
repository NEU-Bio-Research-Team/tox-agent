import logoImage from '../../assets/logo-tox.png';
import { cn } from './utils';

/** The ToxAgent mark. On a dark surface it is drawn as a white silhouette (see
 * `.logo-mark` in theme.css) until a dual-tone asset replaces the navy PNG. */
export function Logo({ className, withWordmark = false, decorative = false }: { className?: string; withWordmark?: boolean; decorative?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <img src={logoImage} alt={withWordmark || decorative ? '' : 'ToxAgent'} className={cn('logo-mark h-6 w-auto shrink-0', className)} />
      {withWordmark ? <span className="text-15 font-semibold text-ink">ToxAgent</span> : null}
    </span>
  );
}
