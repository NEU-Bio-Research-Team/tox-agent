import * as React from 'react';
import { ChevronDown } from 'lucide-react';

import { cn } from './utils';

/** A styled native `<select>`. Used where the browser's own picker is the right
 * control (long lists, phones, and tests that drive `selectOption`); everything
 * else uses the Radix `Select`. */
const NativeSelect = React.forwardRef<HTMLSelectElement, React.ComponentProps<'select'>>(
  ({ className, children, ...props }, ref) => (
    <div className="relative inline-flex w-full min-w-0">
      <select
        ref={ref}
        data-slot="native-select"
        className={cn(
          'h-9 w-full min-w-0 appearance-none rounded-lg border border-line-strong bg-surface-solid pl-3 pr-9 text-sm text-ink outline-none transition-colors disabled:opacity-50 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40',
          className,
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-ink-secondary" aria-hidden="true" />
    </div>
  ),
);
NativeSelect.displayName = 'NativeSelect';

export { NativeSelect };
