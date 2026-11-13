import * as React from 'react';
import { CircleAlert, CircleCheck, Info, ShieldCheck, TriangleAlert } from 'lucide-react';

import { cn } from './utils';

export type Tone = 'neutral' | 'brand' | 'info' | 'success' | 'warning' | 'danger';

const SURFACE: Record<Tone, string> = {
  neutral: 'bg-surface-muted border-line',
  brand: 'bg-brand-soft border-brand-soft-strong',
  info: 'bg-info-soft border-info-border',
  success: 'bg-success-soft border-success-border',
  warning: 'bg-warning-soft border-warning-border',
  danger: 'bg-danger-soft border-danger-border',
};

const ICON_INK: Record<Tone, string> = {
  neutral: 'text-ink-secondary',
  brand: 'text-brand-ink',
  info: 'text-info-ink',
  success: 'text-success-ink',
  warning: 'text-warning-ink',
  danger: 'text-danger-ink',
};

const DEFAULT_ICON: Record<Tone, React.ComponentType<{ className?: string }>> = {
  neutral: Info,
  brand: ShieldCheck,
  info: Info,
  success: CircleCheck,
  warning: TriangleAlert,
  danger: CircleAlert,
};

/**
 * One message about one thing: a measurement, an answer, a connection. Tone is
 * the message's own status; per ADR-0002 a Callout never summarises several
 * measurements.
 */
export function Callout({
  tone = 'neutral',
  title,
  icon,
  className,
  children,
  ...props
}: Omit<React.ComponentProps<'div'>, 'title'> & {
  tone?: Tone;
  title?: React.ReactNode;
  /** `false` removes the icon. */
  icon?: React.ReactNode | false;
}) {
  const Icon = DEFAULT_ICON[tone];
  return (
    <div
      data-slot="callout"
      data-tone={tone}
      className={cn('flex gap-3 rounded-[var(--radius-control)] border px-4 py-3 text-sm leading-relaxed text-ink', SURFACE[tone], className)}
      {...props}
    >
      {icon === false ? null : (
        <span className={cn('mt-0.5 shrink-0 [&_svg]:size-[18px]', ICON_INK[tone])} aria-hidden="true">
          {icon ?? <Icon className="size-[18px]" />}
        </span>
      )}
      <div className="min-w-0 flex-1 space-y-1">
        {title ? <p className="font-semibold">{title}</p> : null}
        <div className="[&_a]:font-medium [&_a]:text-link [&_a]:underline">{children}</div>
      </div>
    </div>
  );
}
