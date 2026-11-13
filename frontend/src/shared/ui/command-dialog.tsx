import * as React from 'react';
import { Search } from 'lucide-react';

import { Dialog, DialogContent, DialogDescription, DialogTitle } from './dialog';
import { cn } from './utils';

export interface CommandItem {
  id: string;
  label: string;
  /** Right-aligned secondary text (a section, a date). */
  hint?: string;
  group?: string;
  icon?: React.ReactNode;
  onSelect: () => void;
}

/** Lower-case and strip diacritics so "phan tich" finds "Phân tích". */
export function fold(text: string): string {
  return text.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/g, 'd').toLowerCase();
}

/**
 * A keyboard-first finder: type to filter, arrows to move, Enter to choose.
 * The workspace frame opens it with Ctrl/Cmd+K over pages and sessions; the docs
 * frame opens it over articles and headings.
 */
export function CommandDialog({
  open,
  onOpenChange,
  items,
  placeholder = 'Tìm…',
  emptyText = 'Không có kết quả.',
  label,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  items: ReadonlyArray<CommandItem>;
  placeholder?: string;
  emptyText?: string;
  label: string;
}) {
  const [query, setQuery] = React.useState('');
  const [active, setActive] = React.useState(0);
  const listId = React.useId();

  React.useEffect(() => {
    if (!open) {
      setQuery('');
      setActive(0);
    }
  }, [open]);

  const matches = React.useMemo(() => {
    const needle = fold(query.trim());
    if (!needle) return items;
    return items.filter((item) => fold(`${item.label} ${item.hint ?? ''} ${item.group ?? ''}`).includes(needle));
  }, [items, query]);

  React.useEffect(() => setActive(0), [query]);

  const choose = (item: CommandItem | undefined) => {
    if (!item) return;
    onOpenChange(false);
    item.onSelect();
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActive((index) => (matches.length ? (index + 1) % matches.length : 0));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActive((index) => (matches.length ? (index - 1 + matches.length) % matches.length : 0));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      choose(matches[active]);
    }
  };

  let lastGroup: string | undefined;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="top-[20%] translate-y-0 gap-0 overflow-hidden p-0 sm:max-w-xl">
        <DialogTitle className="sr-only">{label}</DialogTitle>
        <DialogDescription className="sr-only">Gõ để lọc, dùng phím mũi tên để chọn và Enter để mở.</DialogDescription>
        <div className="flex items-center gap-3 border-b border-line px-4">
          <Search className="size-4 shrink-0 text-ink-tertiary" aria-hidden="true" />
          <input
            autoFocus
            role="combobox"
            aria-expanded="true"
            aria-controls={listId}
            aria-activedescendant={matches[active] ? `${listId}-${matches[active].id}` : undefined}
            aria-label={label}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            className="h-12 w-full bg-transparent pr-8 text-sm text-ink outline-none placeholder:text-ink-tertiary"
          />
        </div>
        <ul id={listId} role="listbox" aria-label={label} className="max-h-[min(24rem,60vh)] overflow-y-auto p-2">
          {matches.length === 0 ? <li className="px-3 py-6 text-center text-sm text-ink-secondary">{emptyText}</li> : null}
          {matches.map((item, index) => {
            const showGroup = item.group && item.group !== lastGroup;
            lastGroup = item.group;
            return (
              <React.Fragment key={item.id}>
                {showGroup ? (
                  <li role="presentation" className="px-3 pb-1 pt-3 text-2xs font-semibold uppercase tracking-wider text-ink-tertiary first:pt-1">
                    {item.group}
                  </li>
                ) : null}
                <li
                  id={`${listId}-${item.id}`}
                  role="option"
                  aria-selected={index === active}
                  onMouseMove={() => setActive(index)}
                  onClick={() => choose(item)}
                  className={cn(
                    'flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-sm text-ink',
                    index === active && 'bg-brand-soft-strong text-brand-ink',
                  )}
                >
                  {item.icon ? <span className="shrink-0 [&_svg]:size-4" aria-hidden="true">{item.icon}</span> : null}
                  <span className="min-w-0 flex-1 truncate">{item.label}</span>
                  {item.hint ? <span className="shrink-0 text-xs text-ink-tertiary">{item.hint}</span> : null}
                </li>
              </React.Fragment>
            );
          })}
        </ul>
      </DialogContent>
    </Dialog>
  );
}
