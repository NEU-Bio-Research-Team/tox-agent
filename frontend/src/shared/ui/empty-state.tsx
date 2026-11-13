import * as React from 'react';

import { cn } from './utils';

/** What a list or pane shows when it has nothing yet: say what will appear and
 * what to do, instead of leaving a blank area. */
export function EmptyState({
  icon,
  title,
  description,
  action,
  dashed = false,
  className,
}: {
  icon?: React.ReactNode;
  title: React.ReactNode;
  description?: React.ReactNode;
  action?: React.ReactNode;
  dashed?: boolean;
  className?: string;
}) {
  return (
    <div
      data-slot="empty-state"
      className={cn(
        'flex flex-col items-center gap-3 rounded-xl px-6 py-10 text-center',
        dashed ? 'border-[1.5px] border-dashed border-line-strong bg-canvas-subtle' : 'border border-line bg-card',
        className,
      )}
    >
      {icon ? <span className="text-ink-tertiary [&_svg]:size-6" aria-hidden="true">{icon}</span> : null}
      <div className="space-y-1">
        <p className="text-sm font-semibold text-ink">{title}</p>
        {description ? <p className="mx-auto max-w-[46ch] text-13 leading-relaxed text-ink-secondary">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}
