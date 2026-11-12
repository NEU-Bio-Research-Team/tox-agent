import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useQueries } from '@tanstack/react-query';
import { Link } from 'react-router';
import { AlertTriangle, Check, ChevronDown, Cpu, Loader2, LogIn, Search, Settings2 } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '../../shared/ui/popover';
import {
  checkConnectionModel,
  listConnectionModels,
  type ModelCheck,
  type ModelOption,
} from '../../shared/api/endpoints';
import type { ModelConnection } from '../../shared/api/types';
import type { AiChoice } from '../../shared/lib/aiProfile';

// The chat's model picker. A connection in Settings is only a credential; here
// the user sees the models each connected account can use, scrolls, and clicks
// one.
//
// A key's models are listed by its provider, so a click keeps the model. A
// ChatGPT sign-in is different: the list is what the runtime and its catalogue
// *name*, which is more than the account can use (gpt-5.4-mini was listed and
// refused by the Codex backend, and every run on it failed). So a click on a
// ChatGPT model first asks it a real question, and the pick is kept only when
// it answers. Any model id can be typed too: a model released after this
// build is one the user can name and prove, not one that waits for a release.

function authLabel(mode: string): string {
  if (mode === 'chatgpt_subscription') return 'ChatGPT Plus/Pro';
  if (mode === 'api_key') return 'API key';
  return 'Không cần key';
}

const CHATGPT = 'chatgpt_subscription';
/** The shape the server accepts as a model id (connections/models.py MODEL_ID). */
const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$/;

const rowKey = (connectionId: string, modelId: string) => `${connectionId}\u0000${modelId}`;

type Verdict = { state: 'checking' } | { state: 'done'; result: ModelCheck };

/** What a check that did not pass means, for the person choosing. */
function verdictText(result: ModelCheck): string {
  switch (result.status) {
    case 'unsupported':
      return 'Tài khoản ChatGPT của bạn không dùng được model này.';
    case 'signin_rejected':
      return 'Đăng nhập ChatGPT đã hết hạn hoặc bị từ chối. Hãy kết nối lại trong Cài đặt.';
    case 'busy':
      return 'Kết nối này đang chạy một câu hỏi khác. Thử lại khi nó xong.';
    default:
      return 'Chưa kiểm tra được model này. Thử lại sau ít phút.';
  }
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

function useModelsByConnection(
  connections: ModelConnection[],
  enabled: boolean,
  choice: AiChoice | null,
) {
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
    // Keep the connection's saved model visible even if listing fails; a
    // ChatGPT sign-in still needs a real check before selecting it.
    const withTested = listed.some((item) => item.id === connection.model_id)
      ? listed
      : [{ id: connection.model_id, name: connection.model_id }, ...listed];
    // A model the user typed and proved is theirs even though no list names it.
    const picked = choice?.connectionId === connection.connection_id ? choice.modelId : null;
    const models = picked && !withTested.some((item) => item.id === picked)
      ? [{ id: picked, name: picked }, ...withTested]
      : withTested;
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
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [typed, setTyped] = useState<Record<string, string>>({});
  const [typedError, setTypedError] = useState<Record<string, string>>({});
  const listRef = useRef<HTMLDivElement>(null);
  const selectionVersion = useRef(0);
  useEffect(() => () => { selectionVersion.current += 1; }, []);
  // Listing asks each provider; do it when the user reaches for the picker
  // (hover or focus), not on every page load.
  const groups = useModelsByConnection(connections, open || warm, choice);

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

  const checking = Object.values(verdicts).some((item) => item.state === 'checking');

  const keep = (row: Row) => {
    selectionVersion.current += 1;
    onChoose({ connectionId: row.connection.connection_id, modelId: row.model.id });
    setOpen(false);
    setQuery('');
  };

  const pick = async (row: Row) => {
    const { connection, model } = row;
    // A key's model is the provider's own listing; only a ChatGPT sign-in is
    // proven before it is kept, including a previously saved choice.
    if (connection.auth_mode !== CHATGPT) return keep(row);
    const key = rowKey(connection.connection_id, model.id);
    if (checking) return;
    const version = selectionVersion.current;
    setVerdicts((all) => ({ ...all, [key]: { state: 'checking' } }));
    let result: ModelCheck;
    try {
      result = await checkConnectionModel(connection.connection_id, model.id);
    } catch {
      result = { model_id: model.id, status: 'unavailable', message: '' };
    }
    setVerdicts((all) => ({ ...all, [key]: { state: 'done', result } }));
    if (result.status === 'usable' && version === selectionVersion.current) keep(row);
  };

  const submitTyped = (connection: ModelConnection) => {
    const id = (typed[connection.connection_id] ?? '').trim();
    if (!id) return;
    if (id.length > 128 || !MODEL_ID.test(id)) {
      setTypedError((all) => ({ ...all, [connection.connection_id]: 'Mã model không hợp lệ.' }));
      return;
    }
    setTypedError((all) => ({ ...all, [connection.connection_id]: '' }));
    void pick({ connection, model: { id, name: id } });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'ArrowDown') { event.preventDefault(); setActive((i) => Math.min(rows.length - 1, i + 1)); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setActive((i) => Math.max(0, i - 1)); }
    else if (event.key === 'Enter' && rows[active]) { event.preventDefault(); void pick(rows[active]); }
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
    <Popover open={open} onOpenChange={(next) => {
      setOpen(next);
      if (!next) { selectionVersion.current += 1; setQuery(''); }
    }}>
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
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <Link to="/settings?connect=chatgpt" className="inline-flex items-center gap-1.5 text-xs font-medium underline" style={{ color: 'var(--purple-600)' }}>
                <LogIn className="h-3.5 w-3.5" />Đăng nhập ChatGPT Plus/Pro
              </Link>
              <Link to="/settings" className="inline-flex items-center gap-1.5 text-xs font-medium underline" style={{ color: 'var(--purple-600)' }}>
                <Settings2 className="h-3.5 w-3.5" />Dùng API key
              </Link>
            </div>
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
                    const verdict = verdicts[rowKey(group.connection.connection_id, model.id)];
                    const refused = verdict?.state === 'done' && verdict.result.status !== 'usable' ? verdict.result : null;
                    return (
                      <button
                        key={model.id}
                        type="button"
                        role="option"
                        aria-selected={selected}
                        aria-disabled={checking && verdict?.state !== 'checking'}
                        data-row={index}
                        onMouseEnter={() => setActive(index)}
                        onClick={() => void pick({ connection: group.connection, model })}
                        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition-colors"
                        style={{ background: index === active ? 'var(--surface-hover)' : undefined }}
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-mono text-[13px]">{model.id}</span>
                          {model.name && model.name !== model.id && (
                            <span className="block truncate text-[11px]" style={{ color: 'var(--text-faint)' }}>{model.name}</span>
                          )}
                          {refused && (
                            <span
                              role="alert"
                              title={refused.message || undefined}
                              className="mt-0.5 flex items-start gap-1 text-[11px]"
                              style={{ color: 'var(--accent-red)' }}
                            >
                              <AlertTriangle className="mt-px h-3 w-3 shrink-0" />
                              <span>{verdictText(refused)}</span>
                            </span>
                          )}
                        </span>
                        {verdict?.state === 'checking' ? (
                          <span className="flex shrink-0 items-center gap-1 text-[11px]" style={{ color: 'var(--text-faint)' }}>
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />Đang kiểm tra…
                          </span>
                        ) : selected ? (
                          <Check className="h-4 w-4 shrink-0" style={{ color: 'var(--purple-600)' }} />
                        ) : !refused && group.connection.auth_mode === CHATGPT && model.availability === 'catalogue' ? (
                          <span className="shrink-0 text-[11px]" style={{ color: 'var(--text-faint)' }}>chưa kiểm tra</span>
                        ) : null}
                      </button>
                    );
                  })}
                  {group.connection.auth_mode === CHATGPT && !needle && (
                    <div className="px-3 pt-1">
                      <form
                        className="flex items-center gap-1.5"
                        onSubmit={(event) => { event.preventDefault(); submitTyped(group.connection); }}
                      >
                        <input
                          aria-label="Mã model khác"
                          value={typed[group.connection.connection_id] ?? ''}
                          onChange={(event) => setTyped((all) => ({ ...all, [group.connection.connection_id]: event.target.value }))}
                          placeholder="Model khác: nhập mã, ví dụ gpt-6.1-terra"
                          className="h-7 min-w-0 flex-1 rounded-md border bg-transparent px-2 font-mono text-xs outline-none placeholder:font-sans placeholder:text-[var(--text-faint)]"
                          style={{ borderColor: 'var(--line)' }}
                        />
                        <button
                          type="submit"
                          disabled={checking || !(typed[group.connection.connection_id] ?? '').trim()}
                          className="h-7 shrink-0 rounded-md border px-2 text-xs disabled:opacity-50"
                          style={{ borderColor: 'var(--line)' }}
                        >
                          Kiểm tra
                        </button>
                      </form>
                      {(() => {
                        const id = (typed[group.connection.connection_id] ?? '').trim();
                        const verdict = id ? verdicts[rowKey(group.connection.connection_id, id)] : undefined;
                        const message =
                          typedError[group.connection.connection_id] ||
                          (verdict?.state === 'done' && verdict.result.status !== 'usable' ? verdictText(verdict.result) : '');
                        return message ? (
                          <p role="alert" className="mt-1 text-[11px]" style={{ color: 'var(--accent-red)' }}>{message}</p>
                        ) : null;
                      })()}
                    </div>
                  )}
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
