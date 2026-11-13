import * as React from 'react';

import { cn } from './utils';

/** The title block every workspace page opens with: title, one line of purpose,
 * and the page's primary action on the right. */
export function PageHeader({
  title,
  description,
  actions,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-wrap items-end justify-between gap-4', className)}>
      <div className="min-w-0 space-y-1.5">
        <h1 className="text-3xl font-bold leading-tight tracking-tight text-ink">{title}</h1>
        {description ? <p className="max-w-[62ch] text-sm leading-relaxed text-ink-secondary">{description}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
