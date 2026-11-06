import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useQueries } from '@tanstack/react-query';
import { Link } from 'react-router';
import { Check, ChevronDown, Cpu, Loader2, Search, Settings2 } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '../../shared/ui/popover';
import { listConnectionModels, type ModelOption } from '../../shared/api/endpoints';
import type { ModelConnection } from '../../shared/api/types';
import type { AiChoice } from '../../shared/lib/aiProfile';

// The chat's model picker. A connection in Settings is only a credential; here
// the user sees every model each connected account can use — listed by the
// provider itself, so nothing is guessed — scrolls, and clicks one.

function authLabel(mode: string): string {
  if (mode === 'chatgpt_subscription') return 'ChatGPT Plus/Pro';
  if (mode === 'api_key') return 'API key';
  return 'Không cần key';
}

/** A short, stable two-letter badge for a provider, tinted from its id. */
function ProviderBadge({ providerId, size = 18 }: { providerId: string; size?: number }) {
  const hue = [...providerId].reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) % 360, 7);
  const letters = providerId.replace(/[^a-z0-9]/gi, '').slice(0, 2).toUpperCase() || 'AI';
  return (
    <span
      aria-hidden
      className="inline-flex shrink-0 items-center justify-center rounded-md font-semibold"
      style={{
        width: size, height: size, fontSize: size * 0.45,
        background: `hsl(${hue} 70% 55% / .16)`, color: `hsl(${hue} 55% 42%)`,
      }}
    >
      {letters}
    </span>
  );
}

interface Row {
  connection: ModelConnection;
  model: ModelOption;
}

function useModelsByConnection(connections: ModelConnection[], enabled: boolean) {
  const results = useQueries({
    queries: connections.map((connection) => ({
      queryKey: ['connection-models', connection.connection_id],
      queryFn: () => listConnectionModels(connection.connection_id),
      staleTime: 10 * 60_000,
      retry: false,
      enabled,
    })),
  });
  return connections.map((connection, index) => {
    const result = results[index];
    const listed = result?.data?.models ?? [];
    // The model the connection was tested on is always usable, even when the
    // provider cannot list models or the listing failed.
    const models = listed.some((item) => item.id === connection.model_id)
      ? listed
      : [{ id: connection.model_id, name: connection.model_id }, ...listed];
    return { connection, models, loading: Boolean(result?.isLoading), failed: Boolean(result?.isError) };
  });
}

export function ModelPicker({
  choice,
  connections,
  providerNames,
  loading,
  onChoose,
  disabled,
}: {
  choice: AiChoice | null;
  connections: ModelConnection[];
  providerNames: Record<string, string>;
  loading?: boolean;
  onChoose: (choice: AiChoice) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const [warm, setWarm] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  // Listing asks each provider; do it when the user reaches for the picker
  // (hover or focus), not on every page load.
  const groups = useModelsByConnection(connections, open || warm);

  const current = connections.find((item) => item.connection_id === choice?.connectionId) ?? null;
  const currentModel = current ? (choice?.modelId ?? current.model_id) : null;

  const needle = query.trim().toLowerCase();
  const filtered = useMemo(
    () =>
      groups
        .map((group) => ({
          ...group,
          models: needle
            ? group.models.filter((model) =>
                `${model.id} ${model.name} ${providerNames[group.connection.provider_id] ?? group.connection.provider_id}`
                  .toLowerCase()
                  .includes(needle),
              )
            : group.models,
        }))
        .filter((group) => group.models.length > 0 || (!needle && group.loading)),
    [groups, needle, providerNames],
  );
  const rows: Row[] = useMemo(
    () => filtered.flatMap((group) => group.models.map((model) => ({ connection: group.connection, model }))),
    [filtered],
  );

  useEffect(() => { setActive(0); }, [needle]);
  useEffect(() => {
    if (!open) return;
    // Start on the selected model, so the list opens where the user is.
    const index = rows.findIndex((row) => row.connection.connection_id === current?.connection_id && row.model.id === currentModel);
    if (index >= 0) setActive(index);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only when opening
  }, [open]);
  useEffect(() => {
    listRef.current?.querySelector(`[data-row="${active}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [active]);

  const pick = (row: Row) => {
    onChoose({ connectionId: row.connection.connection_id, modelId: row.model.id });
    setOpen(false);
    setQuery('');
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'ArrowDown') { event.preventDefault(); setActive((i) => Math.min(rows.length - 1, i + 1)); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setActive((i) => Math.max(0, i - 1)); }
    else if (event.key === 'Enter' && rows[active]) { event.preventDefault(); pick(rows[active]); }
  };

  const label = current && currentModel
    ? currentModel
    : loading
      ? 'Đang tải…'
      : connections.length === 0
        ? 'Kết nối AI'
        : 'Chọn model';

  let rowIndex = -1;
  return (
    <Popover open={open} onOpenChange={(next) => { setOpen(next); if (!next) setQuery(''); }}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          aria-label={`Model AI: ${label}`}
          onPointerEnter={() => setWarm(true)}
          onFocus={() => setWarm(true)}
          className="group flex h-8 max-w-[240px] items-center gap-1.5 rounded-full border px-2.5 text-xs transition-colors hover:bg-[var(--surface-hover)] disabled:opacity-50"
          style={{ borderColor: 'var(--line)', color: 'var(--text-muted)' }}
        >
          {current ? <ProviderBadge providerId={current.provider_id} size={16} /> : <Cpu className="h-3.5 w-3.5" />}
          <span className="truncate font-medium" style={{ color: 'var(--text)' }}>{label}</span>
          <ChevronDown className="h-3 w-3 shrink-0 transition-transform group-data-[state=open]:rotate-180" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" side="top" sideOffset={8} className="w-[340px] overflow-hidden p-0">
        {connections.length === 0 ? (
          <div className="space-y-3 p-4 text-sm">
            <p className="font-medium">Chưa có AI nào được kết nối</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Thêm API key hoặc đăng nhập ChatGPT Plus/Pro trong Cài đặt. Sau đó mọi model của tài khoản đó sẽ hiện ở đây.
            </p>
            <Link to="/settings" className="inline-flex items-center gap-1.5 text-xs font-medium underline" style={{ color: 'var(--purple-600)' }}>
              <Settings2 className="h-3.5 w-3.5" />Mở Cài đặt
            </Link>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2 border-b px-3 py-2" style={{ borderColor: 'var(--line)' }}>
              <Search className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--text-faint)' }} />
              <input
                autoFocus
                aria-label="Tìm model"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Tìm model…"
                className="h-7 w-full bg-transparent text-sm outline-none placeholder:text-[var(--text-faint)]"
              />
            </div>
            <div ref={listRef} role="listbox" aria-label="Các model khả dụng" className="max-h-[320px] overflow-y-auto overscroll-contain py-1">
              {filtered.map((group) => (
                <div key={group.connection.connection_id} className="py-1">
                  <div
                    className="sticky top-0 z-10 flex items-center gap-2 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide backdrop-blur"
                    style={{ color: 'var(--text-faint)', background: 'var(--surface)' }}
                  >
                    <ProviderBadge providerId={group.connection.provider_id} size={14} />
                    <span className="truncate">{providerNames[group.connection.provider_id] ?? group.connection.provider_id}</span>
                    <span className="font-normal normal-case tracking-normal">· {authLabel(group.connection.auth_mode)}</span>
                    {group.loading && <Loader2 className="ml-auto h-3 w-3 animate-spin" />}
                  </div>
                  {group.models.map((model) => {
                    rowIndex += 1;
                    const index = rowIndex;
                    const selected = group.connection.connection_id === current?.connection_id && model.id === currentModel;
                    return (
                      <button
                        key={model.id}
                        type="button"
                        role="option"
                        aria-selected={selected}
                        data-row={index}
                        onMouseEnter={() => setActive(index)}
                        onClick={() => pick({ connection: group.connection, model })}
                        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition-colors"
                        style={{ background: index === active ? 'var(--surface-hover)' : undefined }}
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-mono text-[13px]">{model.id}</span>
                          {model.name && model.name !== model.id && (
                            <span className="block truncate text-[11px]" style={{ color: 'var(--text-faint)' }}>{model.name}</span>
                          )}
                        </span>
                        {selected && <Check className="h-4 w-4 shrink-0" style={{ color: 'var(--purple-600)' }} />}
                      </button>
                    );
                  })}
                  {group.failed && !needle && (
                    <p className="px-3 pb-1 text-[11px]" style={{ color: 'var(--text-faint)' }}>
                      Provider không trả về danh sách model; chỉ có model đã kiểm tra.
                    </p>
                  )}
                </div>
              ))}
              {rows.length === 0 && !filtered.some((group) => group.loading) && (
                <p className="px-3 py-6 text-center text-xs" style={{ color: 'var(--text-faint)' }}>Không có model nào khớp “{query}”.</p>
              )}
            </div>
            <div className="flex items-center justify-between border-t px-3 py-2 text-[11px]" style={{ borderColor: 'var(--line)', color: 'var(--text-faint)' }}>
              <span>↑↓ để chọn · Enter để dùng</span>
              <Link to="/settings" className="inline-flex items-center gap-1 hover:underline">
                <Settings2 className="h-3 w-3" />Quản lý kết nối
              </Link>
            </div>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}
