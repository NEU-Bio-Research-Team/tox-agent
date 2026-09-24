import { useState, type ReactNode } from 'react';
import { Check, ChevronDown, Copy, Pencil } from 'lucide-react';
import { SidebarTrigger } from '../ui/sidebar';
import { Separator } from '../ui/separator';
import { ConnectionIndicator } from './ConnectionIndicator';
import type { ConnectionStatus } from '../../lib/store/eventBus';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover';
import { Button } from '../ui/button';
import { Input } from '../ui/input';

/** Top bar inside the chat column (plan section 8.2 wireframe row 1). Not a
 * page-wide Navbar — the sidebar toggle lives here so mobile/tablet users
 * always have one, regardless of which region (rail vs Sheet) the shadcn
 * sidebar primitive is currently rendering. */
export function WorkspaceHeader({
  title,
  sessionId,
  status,
  actions,
  onRename,
}: {
  title: string;
  sessionId?: string;
  status?: ConnectionStatus;
  actions?: ReactNode;
  onRename?: (title: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [nextTitle, setNextTitle] = useState(title);
  const [copied, setCopied] = useState(false);
  const save = async () => {
    const normalized = nextTitle.trim();
    if (!normalized || !onRename) return;
    await onRename(normalized);
    setEditing(false);
  };
  return (
    <header
      className="ta-glass flex h-14 shrink-0 items-center gap-2 border-b px-3 md:px-4"
      style={{ borderColor: 'var(--line)', backgroundColor: 'var(--surface)' }}
    >
      <SidebarTrigger />
      <Separator orientation="vertical" className="h-5" />
      <div className="min-w-0 flex-1">
        {sessionId ? <Popover onOpenChange={(open) => { if (open) setNextTitle(title); if (!open) setEditing(false); }}>
          <PopoverTrigger asChild>
            <button type="button" className="flex max-w-full items-center gap-1 rounded-md text-left focus-visible:outline-none">
              <h1 className="truncate text-sm font-semibold" style={{ color: 'var(--text)' }}>{title}</h1>
              <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
            </button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-80 space-y-3">
            {editing ? (
              <form className="flex gap-2" onSubmit={(event) => { event.preventDefault(); void save(); }}>
                <Input value={nextTitle} maxLength={120} onChange={(event) => setNextTitle(event.target.value)} aria-label="Tên session" autoFocus />
                <Button type="submit" size="icon" aria-label="Lưu tên session"><Check className="h-4 w-4" /></Button>
              </form>
            ) : (
              <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => setEditing(true)}><Pencil className="h-3.5 w-3.5" /> Đổi tên session</Button>
            )}
            <div className="flex items-center justify-between gap-2 rounded-lg bg-muted p-2">
              <code className="truncate text-[11px] text-muted-foreground">{sessionId}</code>
              <Button variant="ghost" size="icon" className="h-7 w-7" aria-label="Sao chép ID session" onClick={() => { void navigator.clipboard?.writeText(sessionId); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }}>
                {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              </Button>
            </div>
          </PopoverContent>
        </Popover> : <h1 className="truncate text-sm font-semibold" style={{ color: 'var(--text)' }}>{title}</h1>}
      </div>
      {status && <ConnectionIndicator status={status} />}
      {actions}
    </header>
  );
}
