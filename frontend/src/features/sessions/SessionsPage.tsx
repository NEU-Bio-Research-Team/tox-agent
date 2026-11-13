import { useMemo, useState } from 'react';
import { Link } from 'react-router';
import { FlaskConical, MessageSquare, Plus, Search } from 'lucide-react';
import { WorkspacePage } from '../../app/shell/WorkspacePage';
import { Button } from '../../shared/ui/button';
import { Callout } from '../../shared/ui/callout';
import { EmptyState } from '../../shared/ui/empty-state';
import { NativeSelect } from '../../shared/ui/native-select';
import { PageHeader } from '../../shared/ui/page-header';
import { Segmented } from '../../shared/ui/segmented';
import { Skeleton } from '../../shared/ui/skeleton';
import { StatusPill } from '../../shared/ui/status-pill';
import { RUN_STATUS_LABEL_VI } from '../../shared/lib/labels';
import type { SessionListRow } from '../../shared/api/types';
import { fold } from '../../shared/ui/command-dialog';
import { groupByRecency, relativeTimeVi } from './dateGroups';
import { useCreateSession } from './useCreateSession';
import { useSessionsList } from './useSessionsList';

type Filter = 'all' | 'running';
type Sort = 'updated' | 'created' | 'runs';

function SessionListItem({ row }: { row: SessionListRow }) {
  const running = row.active_run !== null;
  return (
    <Link
      to={`/s/${row.session_id}`}
      className="flex flex-wrap items-center gap-x-4 gap-y-1 px-5 py-3.5 outline-none transition-colors hover:bg-surface-muted focus-visible:bg-surface-muted focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
    >
      <div className="min-w-0 flex-[1_1_280px]">
        <p className="flex items-center gap-2 text-sm font-semibold text-ink">
          {running ? <span className="size-[7px] shrink-0 animate-pulse rounded-full bg-brand" aria-hidden="true" /> : null}
          <span className="truncate">{row.title ?? 'Session mới'}</span>
        </p>
        {row.last_message_preview ? (
          <p className="mt-0.5 truncate text-13 text-ink-secondary">{row.last_message_preview}</p>
        ) : (
          <p className="mt-0.5 text-13 italic text-ink-tertiary">Chưa có tin nhắn</p>
        )}
      </div>
      {running ? <StatusPill tone="brand">{RUN_STATUS_LABEL_VI[row.active_run!.status]}</StatusPill> : null}
      <span className="w-14 text-right font-mono text-xs text-ink-tertiary">{row.run_count} run</span>
      <span className="w-24 text-right text-xs text-ink-tertiary">{relativeTimeVi(row.updated_at)}</span>
    </Link>
  );
}

/**
 * The expanded history (plan section 6.4): what is running, a search that says
 * what it covers, one dense list grouped by day. It shares the rail's
 * `['sessions']` cache, so there is one list in the app.
 */
export function SessionsPage() {
  const { rows, isLoading, isError, hasNextPage, isFetchingNextPage, fetchNextPage, refetch } = useSessionsList();
  const create = useCreateSession();
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<Filter>('all');
  const [sort, setSort] = useState<Sort>('updated');

  const running = useMemo(() => rows.filter((row) => row.active_run !== null), [rows]);

  const visible = useMemo(() => {
    const needle = fold(search.trim());
    let list = rows.filter((row) => {
      if (filter === 'running' && row.active_run === null) return false;
      if (!needle) return true;
      return fold(`${row.title ?? ''} ${row.last_message_preview ?? ''} ${row.session_id}`).includes(needle);
    });
    if (sort === 'created') list = [...list].sort((a, b) => b.created_at.localeCompare(a.created_at));
    if (sort === 'runs') list = [...list].sort((a, b) => b.run_count - a.run_count);
    return list;
  }, [rows, search, filter, sort]);

  const groups = useMemo(
    () => (sort === 'updated' ? groupByRecency(visible, (row) => row.updated_at) : [{ label: 'Tất cả', rows: visible }]),
    [visible, sort],
  );

  const newSession = (
    <Button onClick={() => create.mutate()} disabled={create.isPending} className="h-10 gap-2 rounded-lg px-4 font-semibold">
      <Plus />
      {create.isPending ? 'Đang tạo…' : 'Session mới'}
    </Button>
  );

  return (
    <WorkspacePage title="Sessions" width="page">
      <PageHeader
        title="Sessions"
        description="Mỗi session giữ các phân tích, cuộc trò chuyện với agent và dấu vết kiểm chứng của chúng."
        actions={newSession}
      />

      {running.length > 0 ? (
        <section aria-label="Đang chạy" className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-tertiary">Đang chạy · {running.length}</h2>
          {running.map((row) => (
            <div key={row.session_id} className="flex flex-wrap items-center gap-3 rounded-[14px] border border-brand-soft-strong bg-brand-soft px-4 py-3.5">
              <span className="size-2 shrink-0 animate-pulse rounded-full bg-brand" aria-hidden="true" />
              <div className="min-w-0 flex-[1_1_240px]">
                <p className="truncate text-sm font-semibold text-ink">{row.title ?? 'Session mới'}</p>
                <p className="text-13 text-ink-secondary">
                  Run {RUN_STATUS_LABEL_VI[row.active_run!.status]} · cập nhật {relativeTimeVi(row.updated_at)}
                </p>
              </div>
              <Button asChild variant="outline" size="sm" className="bg-surface-solid">
                <Link to={`/s/${row.session_id}`}>Mở session</Link>
              </Button>
            </div>
          ))}
        </section>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex h-10 min-w-[260px] flex-1 items-center gap-2.5 rounded-lg border border-line-strong bg-surface-solid px-3.5 focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/40">
          <Search className="size-4 shrink-0 text-ink-tertiary" aria-hidden="true" />
          <input
            type="search"
            aria-label="Tìm session"
            placeholder="Tìm theo tiêu đề hoặc nội dung"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-ink-tertiary"
          />
          <span className="hidden whitespace-nowrap text-xs text-ink-tertiary sm:inline">trong {rows.length} session đã tải</span>
        </div>
        <Segmented<Filter>
          label="Lọc session"
          value={filter}
          onValueChange={setFilter}
          options={[
            { value: 'all', label: 'Tất cả' },
            { value: 'running', label: `Đang chạy · ${running.length}` },
          ]}
        />
        <div className="w-44">
          <NativeSelect aria-label="Sắp xếp" className="h-10 rounded-lg" value={sort} onChange={(event) => setSort(event.target.value as Sort)}>
            <option value="updated">Mới cập nhật</option>
            <option value="created">Mới tạo</option>
            <option value="runs">Nhiều run nhất</option>
          </NativeSelect>
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-px overflow-hidden rounded-xl border border-line" aria-busy="true" aria-label="Đang tải danh sách session">
          {Array.from({ length: 5 }, (_, index) => (
            <Skeleton key={index} className="h-[66px] rounded-none" />
          ))}
        </div>
      ) : null}

      {isError ? (
        <Callout
          tone="danger"
          title="Không tải được danh sách session"
        >
          <Button variant="outline" size="sm" className="mt-1 bg-surface-solid" onClick={() => void refetch()}>
            Thử lại
          </Button>
        </Callout>
      ) : null}

      {!isLoading && !isError && rows.length === 0 ? (
        <EmptyState
          dashed
          icon={<MessageSquare />}
          title="Chưa có session nào"
          description="Một session giữ các phân tích, cho phép bạn hỏi agent về từng con số và lưu dấu vết kiểm chứng."
          action={
            <div className="flex flex-wrap justify-center gap-2">
              <Button onClick={() => create.mutate()} disabled={create.isPending} className="gap-2 rounded-lg">
                <Plus />
                Tạo session đầu tiên
              </Button>
              <Button asChild variant="outline" className="gap-2 rounded-lg">
                <Link to="/analyze">
                  <FlaskConical />
                  Thử Phân tích nhanh
                </Link>
              </Button>
            </div>
          }
        />
      ) : null}

      {!isLoading && !isError && rows.length > 0 && visible.length === 0 ? (
        <EmptyState icon={<Search />} title="Không có session nào khớp" description="Tìm kiếm chỉ áp dụng cho các session đã tải. Bỏ bộ lọc hoặc tải thêm." />
      ) : null}

      {visible.length > 0 ? (
        <section aria-label="Danh sách session" className="overflow-hidden rounded-xl border border-line bg-card">
          {groups.map((group) => (
            <div key={group.label}>
              <h2 className="border-b border-line bg-canvas-subtle px-5 py-2 text-xs font-semibold uppercase tracking-wider text-ink-tertiary [&:not(:first-child)]:border-t">
                {group.label}
              </h2>
              <div className="divide-y divide-line">
                {group.rows.map((row) => (
                  <SessionListItem key={row.session_id} row={row} />
                ))}
              </div>
            </div>
          ))}
        </section>
      ) : null}

      {rows.length > 0 ? (
        <div className="flex items-center justify-center gap-4">
          <span className="text-13 text-ink-tertiary">
            Đang hiển thị {visible.length} trong {rows.length} session đã tải
          </span>
          {hasNextPage && !search.trim() ? (
            <Button variant="outline" size="sm" disabled={isFetchingNextPage} onClick={() => void fetchNextPage()}>
              {isFetchingNextPage ? 'Đang tải thêm…' : 'Tải thêm'}
            </Button>
          ) : null}
        </div>
      ) : null}
    </WorkspacePage>
  );
}
