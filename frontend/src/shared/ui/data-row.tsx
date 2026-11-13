import * as React from 'react';

import { cn } from './utils';

/** A bordered group of `DataRow`s: the settings pattern of a label and its
 * description on the left and the control on the right. */
export function DataList({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="data-list"
      className={cn('divide-y divide-line overflow-hidden rounded-xl border border-line bg-card', className)}
      {...props}
    />
  );
}

export function DataListHeading({ className, ...props }: React.ComponentProps<'p'>) {
  return (
    <p
      className={cn('border-b border-line bg-canvas-subtle px-5 py-2.5 text-xs font-semibold uppercase tracking-wider text-ink-tertiary', className)}
      {...props}
    />
  );
}

export function DataRow({
  title,
  description,
  titleId,
  children,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  /** Lets a control point at its label with `aria-labelledby`. */
  titleId?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div data-slot="data-row" className={cn('flex flex-wrap items-center justify-between gap-x-6 gap-y-3 px-5 py-4', className)}>
      <div className="min-w-0 flex-[1_1_260px]">
        <p id={titleId} className="text-sm font-semibold text-ink">{title}</p>
        {description ? <p className="mt-0.5 text-13 leading-snug text-ink-secondary">{description}</p> : null}
      </div>
      {children ? <div className="flex shrink-0 items-center gap-3">{children}</div> : null}
    </div>
  );
}
