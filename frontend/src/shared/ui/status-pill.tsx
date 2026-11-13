import * as React from 'react';

import { cn } from './utils';
import type { Tone } from './callout';

const TONE: Record<Tone, string> = {
  neutral: 'bg-surface-muted text-ink-secondary border-line',
  brand: 'bg-brand-soft-strong text-brand-ink border-transparent',
  info: 'bg-info-soft text-info-ink border-info-border',
  success: 'bg-success-soft text-success-ink border-transparent',
  warning: 'bg-warning-soft text-warning-ink border-warning-border',
  danger: 'bg-danger-soft text-danger-ink border-danger-border',
};

const DOT: Record<Tone, string> = {
  neutral: 'bg-ink-tertiary',
  brand: 'bg-brand',
  info: 'bg-info',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-danger',
};

/**
 * A short status for one thing. Model labels follow ADR-0002: a positive class
 * ("blocker", "active") is `warning`, a negative class is `neutral`, and "not
 * run" or "unavailable" is `neutral` or `info`. `danger` is for failures of the
 * system (a connection, a request), never for a prediction.
 */
export function StatusPill({
  tone = 'neutral',
  dot = false,
  pulse = false,
  className,
  children,
  ...props
}: React.ComponentProps<'span'> & { tone?: Tone; dot?: boolean; pulse?: boolean }) {
  return (
    <span
      data-slot="status-pill"
      data-tone={tone}
      className={cn('inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-0.5 text-xs font-semibold', TONE[tone], className)}
      {...props}
    >
      {dot ? <span className={cn('size-1.5 rounded-full', DOT[tone], pulse && 'animate-pulse')} aria-hidden="true" /> : null}
      {children}
    </span>
  );
}
