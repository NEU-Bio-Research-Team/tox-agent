import * as React from 'react';

import { cn } from './utils';

export interface SegmentedOption<T extends string> {
  value: T;
  label: React.ReactNode;
  icon?: React.ReactNode;
  disabled?: boolean;
}

/** A small group of mutually exclusive choices that act immediately (a mode, a
 * theme). Navigation between panels of content is `Tabs`. */
export function Segmented<T extends string>({
  value,
  onValueChange,
  options,
  label,
  size = 'md',
  fill = false,
  className,
}: {
  value: T;
  onValueChange: (value: T) => void;
  options: ReadonlyArray<SegmentedOption<T>>;
  /** Accessible name of the group. */
  label: string;
  size?: 'sm' | 'md';
  /** Stretch the options to share the full width. */
  fill?: boolean;
  className?: string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      data-slot="segmented"
      className={cn('inline-flex gap-0.5 rounded-[11px] bg-surface-muted p-[3px]', fill && 'flex w-full', className)}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            disabled={option.disabled}
            onClick={() => onValueChange(option.value)}
            className={cn(
              'inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-[9px] border text-sm transition-colors disabled:pointer-events-none disabled:opacity-50',
              size === 'sm' ? 'h-7 px-2.5 text-xs' : 'h-8 px-3.5',
              fill && 'flex-1',
              active
                ? 'border-line bg-surface-solid font-semibold text-ink shadow-xs'
                : 'border-transparent font-medium text-ink-secondary hover:text-ink',
            )}
          >
            {option.icon ? <span className="[&_svg]:size-3.5" aria-hidden="true">{option.icon}</span> : null}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
