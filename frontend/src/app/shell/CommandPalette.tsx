import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import { BookOpen, FlaskConical, MessageSquare, Plus, Settings } from 'lucide-react';
import { CommandDialog, type CommandItem } from '../../shared/ui/command-dialog';
import { relativeTimeVi } from '../../features/sessions/dateGroups';
import { useSessionsList } from '../../features/sessions/useSessionsList';
import { useCreateSession } from '../../features/sessions/useCreateSession';

export const OPEN_PALETTE_EVENT = 'toxagent:open-command-palette';

/** Ask the mounted palette to open (the rail's search button uses this). */
export function openCommandPalette(): void {
  window.dispatchEvent(new Event(OPEN_PALETTE_EVENT));
}

/** Ctrl/Cmd+K over the workspace's pages, a "new session" action and the
 * sessions already loaded in the rail. */
export function CommandPalette() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const { rows } = useSessionsList();
  const create = useCreateSession();

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setOpen((current) => !current);
      }
    };
    const onOpen = () => setOpen(true);
    window.addEventListener('keydown', onKey);
    window.addEventListener(OPEN_PALETTE_EVENT, onOpen);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener(OPEN_PALETTE_EVENT, onOpen);
    };
  }, []);

  const items = useMemo<CommandItem[]>(
    () => [
      { id: 'new', group: 'Hành động', label: 'Session mới', icon: <Plus />, onSelect: () => create.mutate() },
      { id: 'analyze', group: 'Đi tới', label: 'Phân tích nhanh', icon: <FlaskConical />, onSelect: () => navigate('/analyze') },
      { id: 'sessions', group: 'Đi tới', label: 'Sessions', icon: <MessageSquare />, onSelect: () => navigate('/sessions') },
      { id: 'docs', group: 'Đi tới', label: 'Tài liệu', icon: <BookOpen />, onSelect: () => navigate('/docs') },
      { id: 'settings', group: 'Đi tới', label: 'Cài đặt', icon: <Settings />, onSelect: () => navigate('/settings') },
      ...rows.map((row) => ({
        id: `s-${row.session_id}`,
        group: 'Session gần đây',
        label: row.title ?? 'Session mới',
        hint: relativeTimeVi(row.updated_at),
        onSelect: () => navigate(`/s/${row.session_id}`),
      })),
    ],
    [create, navigate, rows],
  );

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      items={items}
      label="Tìm trang hoặc session"
      placeholder="Tìm trang hoặc session…"
      emptyText="Không có trang hay session nào khớp."
    />
  );
}
