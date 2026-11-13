import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router';
import { CheckCircle2, CircleAlert, CircleDashed, KeyRound, Loader2, Trash2 } from 'lucide-react';
import { Button } from '../../shared/ui/button';
import { Callout } from '../../shared/ui/callout';
import { DataList, DataListHeading } from '../../shared/ui/data-row';
import { Input } from '../../shared/ui/input';
import { Label } from '../../shared/ui/label';
import { Segmented } from '../../shared/ui/segmented';
import { StatusPill } from '../../shared/ui/status-pill';
import { cn } from '../../shared/ui/utils';
import {
  cancelChatGPTSignIn,
  completeChatGPTSignIn,
  createModelConnection,
  deleteModelConnection,
  getChatGPTSignIn,
  listModelConnections,
  listModelsForKey,
  listSupportedProviders,
  startChatGPTSignIn,
  testModelConnection,
  type ChatGPTEnrollment,
  type SupportedProvider,
} from '../../shared/api/endpoints';
import { describeAgentProfileChoice } from '../../shared/lib/aiProfile';
import { ApiError, type ModelConnection } from '../../shared/api/types';

// Settings only connects an account: an API key (OpenAI, Claude, OpenRouter,
// DeepSeek, any OpenAI-compatible server…) or a ChatGPT Plus/Pro sign-in.
// Which model a session runs is picked in the chat, where the composer's model
// picker lists every model each connected account can use.
//
// The server still tests a connection against one model, so one is chosen
// here without asking: the first chat model the provider lists for the key.
// Only a provider that cannot list its models asks for a model id to test.
//
// The provider list and what each one accepts come from the server (I13), so
// the form never offers something the server would refuse — including the
// ChatGPT sign-in, which a deployment may not run.

const POLL_MS = 3000;

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

function authLabel(mode: string): string {
  if (mode === 'chatgpt_subscription') return 'ChatGPT Plus/Pro';
  if (mode === 'api_key') return 'API key';
  return 'Không cần key';
}

function ConnectionStatus({ status }: { status: ModelConnection['status'] }) {
  if (status === 'ready') return <StatusPill tone="success"><CheckCircle2 className="size-3.5" />Sẵn sàng</StatusPill>;
  if (status === 'failed') return <StatusPill tone="danger"><CircleAlert className="size-3.5" />Kết nối lỗi</StatusPill>;
  return <StatusPill tone="neutral"><CircleDashed className="size-3.5" />Chưa kiểm tra</StatusPill>;
}

function ConnectionRow({ connection, providerName, onChanged, onReconnect }: {
  connection: ModelConnection; providerName: string; onChanged: () => Promise<void>; onReconnect: (connection: ModelConnection) => void;
}) {
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState<'test' | 'delete' | null>(null);

  const run = async (kind: 'test' | 'delete', action: () => Promise<unknown>, fallback: string) => {
    setBusy(kind);
    setMessage(null);
    try {
      await action();
    } catch (error) {
      setMessage(errorText(error, fallback));
    } finally {
      setBusy(null);
      await onChanged();
    }
  };

  const subscription = connection.auth_mode === 'chatgpt_subscription';
  return (
    <div className="space-y-3 px-5 py-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-brand-soft-strong text-sm font-bold text-brand-ink" aria-hidden="true">
          {providerName.slice(0, 1).toUpperCase()}
        </span>
        <div className="min-w-0 flex-[1_1_180px]">
          <p className="truncate text-sm font-semibold text-ink">{providerName}</p>
          <p className="truncate text-13 text-ink-secondary">{authLabel(connection.auth_mode)}</p>
        </div>
        <ConnectionStatus status={connection.status} />
        <div className="flex items-center gap-1.5">
          <Button size="sm" variant="outline" disabled={busy !== null} onClick={() => void run('test', () => testModelConnection(connection.connection_id), 'Kiểm tra thất bại.')}>
            {busy === 'test' ? <Loader2 className="animate-spin" /> : null}Kiểm tra lại
          </Button>
          <Button size="icon" variant="ghost" className="size-8" aria-label={`Xoá kết nối ${providerName}`} disabled={busy !== null} onClick={() => void run('delete', () => deleteModelConnection(connection.connection_id), 'Không xoá được.')}>
            <Trash2 />
          </Button>
        </div>
      </div>
      {connection.status === 'failed' ? (
        <Callout tone="danger" className="items-center">
          <span className="flex flex-wrap items-center justify-between gap-3">
            <span>
              {subscription
                ? 'Phiên đăng nhập ChatGPT không còn hợp lệ. Đăng nhập lại để tiếp tục dùng gói của bạn.'
                : 'Provider từ chối kết nối này. Kiểm tra lại key, hoặc xoá và kết nối lại.'}
            </span>
            {subscription ? (
              <Button size="sm" onClick={() => onReconnect(connection)}>Đăng nhập lại</Button>
            ) : null}
          </span>
        </Callout>
      ) : null}
      {message ? <p className="text-xs text-danger" role="alert">{message}</p> : null}
    </div>
  );
}

function ChatGPTSignIn({ provider, onConnected }: { provider: SupportedProvider; onConnected: () => Promise<void> }) {
  const oauth = provider.chatgpt_oauth;
  const [enrollment, setEnrollment] = useState<ChatGPTEnrollment>({ state: 'none' });
  const [model, setModel] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoSaved = useRef(false);

  const stopPolling = () => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
  };

  useEffect(() => {
    if (!oauth?.available) return undefined;
    // A sign-in started before a reload is still on the server; pick it up.
    void getChatGPTSignIn().then(setEnrollment).catch(() => undefined);
    return stopPolling;
  }, [oauth?.available]);

  useEffect(() => {
    if (enrollment.state !== 'pending') { stopPolling(); return; }
    if (timer.current) return;
    timer.current = setInterval(() => {
      void getChatGPTSignIn().then(setEnrollment).catch(() => undefined);
    }, POLL_MS);
  }, [enrollment.state]);

  const save = async (modelId: string) => {
    setError(null);
    setSaving(true);
    try {
      await completeChatGPTSignIn({ model_id: modelId });
      setEnrollment({ state: 'none' });
      setModel('');
      await onConnected();
    } catch (e) {
      setError(errorText(e, 'Không lưu được kết nối ChatGPT.'));
    } finally {
      setSaving(false);
    }
  };

  // Approved: save it straight away on the first listed model. The model used
  // for each chat is picked later, in the chat.
  useEffect(() => {
    if (enrollment.state !== 'connected' || autoSaved.current) return;
    const first = enrollment.models?.[0]?.id;
    if (!first) return;
    autoSaved.current = true;
    void save(first);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- once per approval
  }, [enrollment]);

  if (!oauth?.available) {
    return (
      <p className="text-xs text-ink-tertiary">
        ChatGPT Plus/Pro chưa dùng được trên bản triển khai này: {oauth?.reason ?? 'chưa cấu hình'}.
        {oauth?.documentation_url && <> Xem <a className="underline" href={oauth.documentation_url} target="_blank" rel="noreferrer">hướng dẫn của OpenAI</a>.</>}
      </p>
    );
  }

  const start = async () => {
    setError(null);
    autoSaved.current = false;
    try { setEnrollment(await startChatGPTSignIn()); } catch (e) { setError(errorText(e, 'Không bắt đầu được đăng nhập ChatGPT.')); }
  };
  const cancel = async () => {
    stopPolling();
    await cancelChatGPTSignIn().catch(() => undefined);
    setEnrollment({ state: 'none' });
  };

  return (
    <div className="space-y-3 rounded-xl border border-line bg-canvas-subtle p-4 text-sm text-ink">
      {(enrollment.state === 'none' || enrollment.state === 'expired' || enrollment.state === 'failed' || enrollment.state === 'cancelled') && (
        <>
          <p className="text-xs text-ink-tertiary">
            Dùng gói ChatGPT Plus/Pro của bạn thay cho API key. Bạn sẽ nhập một mã trên trang của OpenAI; ToxAgent không bao giờ thấy mật khẩu.
          </p>
          {enrollment.error && <p className="text-xs text-danger">{enrollment.error}</p>}
          <Button size="sm" onClick={() => void start()}>Kết nối ChatGPT Plus/Pro</Button>
        </>
      )}
      {enrollment.state === 'pending' && (
        <>
          <p>1. Mở <a className="font-medium underline" href={enrollment.verification_url ?? '#'} target="_blank" rel="noreferrer">{enrollment.verification_url}</a></p>
          <p>2. {enrollment.instructions}</p>
          <p className="flex items-center gap-1.5 text-xs text-ink-tertiary">
            <Loader2 className="h-3 w-3 animate-spin" />
            Đang chờ bạn xác nhận trên trang OpenAI… (còn {Math.ceil((enrollment.expires_in_s ?? 0) / 60)} phút)
          </p>
          <Button size="sm" variant="ghost" onClick={() => void cancel()}>Huỷ</Button>
        </>
      )}
      {enrollment.state === 'connected' && (enrollment.models?.length ? (
        <p className="flex items-center gap-1.5 text-xs text-ink-secondary">
          <Loader2 className="h-3 w-3 animate-spin" />Đã đăng nhập ChatGPT — đang lưu kết nối…
        </p>
      ) : (
        <>
          <p>Đã đăng nhập ChatGPT, nhưng OpenAI không trả về danh sách model. Nhập một model ID để kiểm tra kết nối:</p>
          <div className="flex flex-wrap items-center gap-2">
            <Input aria-label="Model ID để kiểm tra" value={model} onChange={(e) => setModel(e.target.value)} placeholder="vd. gpt-5" className="h-9 w-48" />
            <Button size="sm" disabled={!model.trim() || saving} onClick={() => void save(model.trim())}>Lưu kết nối</Button>
            <Button size="sm" variant="ghost" onClick={() => void cancel()}>Huỷ</Button>
          </div>
        </>
      ))}
      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  );
}

type Step = 'idle' | 'listing' | 'saving' | 'testing';

const STEP_LABEL: Record<Step, string> = {
  idle: 'Kết nối',
  listing: 'Đang đọc các model của key…',
  saving: 'Đang lưu…',
  testing: 'Đang thử kết nối…',
};

export function AiConnectionsCard({
  agentEnabled,
  userAiRequired = false,
  onConnectionsChanged,
}: {
  agentEnabled: boolean | null;
  userAiRequired?: boolean;
  /** Lets the chat's model picker see a new or removed connection at once. */
  onConnectionsChanged?: () => void;
}) {
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [providers, setProviders] = useState<SupportedProvider[]>([]);
  const [provider, setProvider] = useState('');
  const [method, setMethod] = useState<'api_key' | 'chatgpt'>('api_key');
  const [searchParams] = useSearchParams();
  // Read once: where the person arrived from decides the first view, and a
  // later change to the address should not reset a choice they have made.
  const [connectIntent] = useState(() => searchParams.get('connect'));
  const [baseUrl, setBaseUrl] = useState('');
  const [credential, setCredential] = useState('');
  const [probeModel, setProbeModel] = useState('');
  const [askModel, setAskModel] = useState(false);
  const [step, setStep] = useState<Step>('idle');
  const [formError, setFormError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const formRef = useRef<HTMLDivElement>(null);

  const refreshConnections = async () => {
    try { setConnections((await listModelConnections()).connections); } catch { setFormError('Không tải được danh sách kết nối AI.'); }
    onConnectionsChanged?.();
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- once, on mount
  useEffect(() => { void refreshConnections(); }, []);
  useEffect(() => {
    void listSupportedProviders()
      .then((result) => {
        setProviders(result.providers);
        // `?connect=chatgpt` (the chat's model picker links here when nothing
        // is connected) opens on the ChatGPT sign-in instead of the API-key
        // form, when this deployment offers it. Otherwise the page opens as
        // it always did, and the sign-in card says why it is unavailable.
        const wantsChatGPT = connectIntent === 'chatgpt';
        const openai = result.providers.find((item) => item.provider_id === 'openai' && item.chatgpt_oauth?.available);
        if (wantsChatGPT && openai) {
          setProvider(openai.provider_id);
          setMethod('chatgpt');
        } else if (result.providers[0]) {
          setProvider(result.providers[0].provider_id);
        }
      })
      .catch(() => setFormError('Không tải được danh sách provider được hỗ trợ.'));
  }, [connectIntent]);

  const selected = useMemo(() => providers.find((item) => item.provider_id === provider) ?? null, [providers, provider]);
  const providerName = (id: string) => providers.find((item) => item.provider_id === id)?.display_name ?? id;
  const offersChatGPT = Boolean(selected?.chatgpt_oauth);
  const keyOptional = selected?.auth_modes.includes('none') ?? false;
  const needsModelId = askModel || selected?.lists_models === false;
  const readyCount = connections.filter((item) => item.status === 'ready').length;
  const notice = describeAgentProfileChoice({ agentEnabled, readyProfileCount: readyCount, userAiRequired }).notice;

  const chooseProvider = (next: string) => {
    setProvider(next);
    setBaseUrl('');
    setProbeModel('');
    setAskModel(false);
    setMethod('api_key');
    setFormError(null);
    setSuccess(null);
  };

  const connect = async () => {
    if (!provider) return;
    if (!keyOptional && !credential.trim()) { setFormError('Nhập API key.'); return; }
    if (selected?.base_url_required && !baseUrl.trim()) { setFormError(`${selected.display_name} cần Base URL.`); return; }
    setFormError(null);
    setSuccess(null);
    const base_url = selected?.base_url_required ? baseUrl.trim() : undefined;
    let created = false;
    try {
      let modelId = probeModel.trim();
      if (!modelId) {
        if (!selected?.lists_models) { setFormError('Nhập một model ID để kiểm tra kết nối.'); return; }
        setStep('listing');
        const listed = await listModelsForKey({ provider_id: provider, base_url, credential: credential.trim() || undefined });
        modelId = listed.models[0]?.id ?? '';
        if (!modelId) {
          setAskModel(true);
          setFormError('Provider không trả về model nào cho key này; nhập một model ID để kiểm tra.');
          return;
        }
      }
      setStep('saving');
      const authMode: ModelConnection['auth_mode'] = credential.trim() ? 'api_key' : 'none';
      const connection = await createModelConnection({
        provider_id: provider, model_id: modelId, auth_mode: authMode, base_url,
        credential: authMode === 'api_key' ? credential : undefined,
      });
      created = true;
      // The key leaves the form as soon as the server has it.
      setCredential(''); setBaseUrl(''); setProbeModel(''); setAskModel(false);
      setStep('testing');
      const tested = await testModelConnection(connection.connection_id).catch((error: unknown) => {
        setFormError(errorText(error, 'Đã lưu, nhưng kiểm tra kết nối thất bại.'));
        return null;
      });
      if (tested?.status === 'ready') setSuccess(`Đã kết nối ${providerName(provider)}. Chọn model ngay trong khung chat.`);
    } catch (error) {
      // The server's message says what is wrong and often what to pick instead.
      setFormError(errorText(error, 'Không thể kết nối.'));
    } finally {
      setStep('idle');
      if (created) await refreshConnections();
    }
  };

  // A failed ChatGPT sign-in is repaired from the form below, opened on it.
  const reconnect = (connection: ModelConnection) => {
    chooseProvider(connection.provider_id);
    setMethod('chatgpt');
    formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const providerNote = (item: SupportedProvider) => {
    if (item.base_url_required) return 'Base URL và API key';
    const parts: string[] = [];
    if (item.auth_modes.includes('api_key')) parts.push('API key');
    if (item.chatgpt_oauth) parts.push('ChatGPT Plus/Pro');
    if (parts.length === 0) parts.push('Không cần key');
    return parts.join(' hoặc ');
  };

  return (
    <div className="space-y-6">
      {notice ? <Callout tone="info" icon={false}>{notice}</Callout> : null}

      {connections.length === 0 ? (
        <p className="rounded-xl border border-dashed border-line-strong bg-canvas-subtle px-5 py-6 text-center text-sm text-ink-secondary">
          Chưa có kết nối nào. Thêm một kết nối bên dưới để dùng agent.
        </p>
      ) : (
        <DataList>
          <DataListHeading>Đã kết nối · {connections.length}</DataListHeading>
          {connections.map((connection) => (
            <ConnectionRow
              key={connection.connection_id}
              connection={connection}
              providerName={providerName(connection.provider_id)}
              onChanged={refreshConnections}
              onReconnect={reconnect}
            />
          ))}
        </DataList>
      )}

      <div ref={formRef} className="space-y-3">
        <h3 className="text-15 font-semibold text-ink">Thêm kết nối</h3>
        {providers.length === 0 && !formError ? (
          <p className="text-13 text-ink-secondary">Đang tải danh sách provider. Nếu mãi không hiện, triển khai này chưa hỗ trợ provider nào.</p>
        ) : null}
        <div role="group" aria-label="AI provider" className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-3">
          {providers.map((item) => {
            const active = item.provider_id === provider;
            return (
              <button
                key={item.provider_id}
                type="button"
                aria-pressed={active}
                onClick={() => chooseProvider(item.provider_id)}
                className={cn(
                  'flex flex-col items-start gap-1 rounded-[14px] border p-4 text-left transition-colors',
                  active ? 'border-2 border-brand bg-brand-soft' : 'border-line-strong bg-card hover:bg-surface-muted',
                )}
              >
                <span className="text-sm font-semibold text-ink">{item.display_name}</span>
                <span className="text-13 text-ink-secondary">{providerNote(item)}</span>
              </button>
            );
          })}
        </div>

        {selected ? (
          <div className="space-y-4 rounded-xl border border-line bg-card p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h4 className="text-15 font-semibold text-ink">Kết nối {selected.display_name}</h4>
              {offersChatGPT ? (
                <Segmented<'api_key' | 'chatgpt'>
                  label="Cách kết nối"
                  value={method}
                  onValueChange={setMethod}
                  options={[
                    { value: 'api_key', label: 'API key' },
                    { value: 'chatgpt', label: 'ChatGPT Plus/Pro' },
                  ]}
                />
              ) : null}
            </div>

            {method === 'chatgpt' ? (
              <ChatGPTSignIn provider={selected} onConnected={refreshConnections} />
            ) : (
              <form className="space-y-4" onSubmit={(event) => { event.preventDefault(); void connect(); }}>
                <div className="space-y-2">
                  <Label htmlFor="ai-credential">API key</Label>
                  <div className="relative">
                    <KeyRound className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-tertiary" aria-hidden="true" />
                    <Input
                      id="ai-credential"
                      aria-label="AI API key"
                      type="password"
                      autoComplete="off"
                      value={credential}
                      onChange={(e) => setCredential(e.target.value)}
                      placeholder={keyOptional ? 'Bỏ trống nếu server không cần key' : 'sk-…'}
                      className="h-10 pl-9 font-mono"
                    />
                  </div>
                  <p className="text-13 leading-snug text-ink-secondary">
                    Key chỉ được gửi lên khi kết nối, được mã hoá khi lưu và không bao giờ trả lại giao diện. Muốn đổi key, hãy xoá và kết nối lại.
                  </p>
                </div>
                {selected.base_url_required ? (
                  <div className="space-y-2">
                    <Label htmlFor="ai-base-url">Base URL</Label>
                    <Input id="ai-base-url" aria-label="AI base URL" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://…/v1" className="h-10 font-mono" />
                  </div>
                ) : null}
                {needsModelId ? (
                  <div className="space-y-2">
                    <Label htmlFor="ai-probe-model">Model để kiểm tra</Label>
                    <Input id="ai-probe-model" aria-label="Model ID để kiểm tra" value={probeModel} onChange={(e) => setProbeModel(e.target.value)} placeholder="Model ID để kiểm tra kết nối" className="h-10 font-mono" />
                  </div>
                ) : null}
                {selected.note ? <p className="text-13 text-ink-secondary">{selected.note}</p> : null}
                <Button type="submit" disabled={step !== 'idle'} className="h-10 gap-2 rounded-lg px-5 font-semibold">
                  {step !== 'idle' ? <Loader2 className="animate-spin" /> : null}
                  {STEP_LABEL[step]}
                </Button>
              </form>
            )}

            {success ? (
              <p className="flex items-center gap-1.5 text-13 text-success" role="status"><CheckCircle2 className="size-4" />{success}</p>
            ) : null}
            {formError ? <p className="text-13 text-danger" role="alert">{formError}</p> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
