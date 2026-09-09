import { useMemo, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';
import { Plus, Search } from 'lucide-react';
import { WorkspaceLayout } from '../components/shell/WorkspaceLayout';
import { WorkspaceHeader } from '../components/shell/WorkspaceHeader';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Card } from '../components/ui/card';
import { createSession } from '../lib/api/endpoints';
import { useSessionsList } from '../hooks/useSessionsList';
import { SessionCardRow } from '../components/shell/SessionRow';

/**
 * Plan section 5.2: the expanded history view, sharing the sidebar's own
 * `['sessions']` query cache rather than a second copy — the sidebar stays
 * usable alongside it because both live inside the same `WorkspaceLayout`.
 */
export function SessionsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const { rows, isLoading, isError, hasNextPage, isFetchingNextPage, fetchNextPage } = useSessionsList();

  const createMutation = useMutation({
    mutationFn: () => createSession({ preferred_language: 'vi' }),
    onSuccess: (session) => {
      void queryClient.invalidateQueries({ queryKey: ['sessions'] });
      navigate(`/s/${session.session_id}`);
    },
  });

  const filtered = useMemo(() => {
    if (!search.trim()) return rows;
    const needle = search.toLowerCase();
    return rows.filter((row) => `${row.title ?? ''} ${row.last_message_preview ?? ''} ${row.session_id}`.toLowerCase().includes(needle));
  }, [rows, search]);

  return (
    <WorkspaceLayout>
      <div className="flex h-full flex-col">
        <WorkspaceHeader title="Tất cả sessions" />
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
          <div className="mx-auto max-w-3xl">
            <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {search.trim() ? `${filtered.length} kết quả trong danh sách đã tải` : 'Lịch sử session, mới nhất trước'}
              </p>
              <Button onClick={() => createMutation.mutate()} disabled={createMutation.isPending} className="gap-2">
                <Plus className="h-4 w-4" />
                {createMutation.isPending ? 'Đang tạo…' : 'Session mới'}
              </Button>
            </div>

            <div className="relative mb-6">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" style={{ color: 'var(--text-faint)' }} />
              <Input
                placeholder="Tìm theo tiêu đề hoặc nội dung trong danh sách đã tải…"
                className="pl-9"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </div>

            {isLoading && (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Đang tải…
              </p>
            )}
            {isError && (
              <p className="text-sm" style={{ color: 'var(--accent-red)' }}>
                Không tải được danh sách session.
              </p>
            )}
            {!isLoading && filtered.length === 0 && (
              <Card className="p-8 text-center" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  {search.trim()
                    ? 'Không có kết quả tìm kiếm trong danh sách đã tải.'
                    : 'Chưa có session nào. Tạo một session mới để bắt đầu phân tích.'}
                </p>
              </Card>
            )}

            <div className="space-y-3">
              {filtered.map((row) => (
                <SessionCardRow key={row.session_id} row={row} onOpen={() => navigate(`/s/${row.session_id}`)} />
              ))}
            </div>

            {hasNextPage && !search.trim() && (
              <div className="mt-4 flex justify-center">
                <Button variant="outline" size="sm" disabled={isFetchingNextPage} onClick={() => void fetchNextPage()}>
                  {isFetchingNextPage ? 'Đang tải thêm…' : 'Tải thêm'}
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </WorkspaceLayout>
  );
}
