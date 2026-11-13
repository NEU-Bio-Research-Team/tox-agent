import type { ReactNode } from 'react';
import { Logo } from '../../../shared/ui/logo';

/** A person's message is a purple bubble on the right; ToxAgent's is a card on the
 * left behind the product mark, so the two voices are told apart at a glance. */
export function MessageBubble({ role, children }: { role: 'user' | 'assistant'; children: ReactNode }) {
  if (role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl bg-brand px-4 py-3 text-sm leading-relaxed text-brand-foreground">{children}</div>
      </div>
    );
  }
  return (
    <div className="flex items-start gap-3">
      <span className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-full border border-line bg-brand-soft" aria-hidden="true">
        <Logo decorative className="h-4" />
      </span>
      <div className="min-w-0 max-w-[88%] flex-1 rounded-2xl border border-line bg-card px-4 py-3 text-sm leading-relaxed text-ink">{children}</div>
    </div>
  );
}
