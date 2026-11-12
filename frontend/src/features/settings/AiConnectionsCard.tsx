import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router';
import { CheckCircle2, CircleAlert, CircleDashed, KeyRound, Loader2, Trash2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../shared/ui/card';
import { Button } from '../../shared/ui/button';
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

function StatusPill({ status }: { status: ModelConnection['status'] }) {
  const tone = status === 'ready'
    ? { icon: CheckCircle2, text: 'Đã kết nối', color: 'var(--accent-green)' }
    : status === 'failed'
      ? { icon: CircleAlert, text: 'Kết nối lỗi', color: 'var(--accent-red)' }
      : { icon: CircleDashed, text: 'Chưa kiểm tra', color: 'var(--text-faint)' };
  const Icon = tone.icon;
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium" style={{ color: tone.color }}>
      <Icon className="h-3.5 w-3.5" />{tone.text}
    </span>
  );
}

function ConnectionRow({ connection, providerName, onChanged }: {
  connection: ModelConnection; providerName: string; onChanged: () => Promise<void>;
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

  return (
    <div className="space-y-1.5 rounded-xl border px-3 py-2.5 text-sm" style={{ borderColor: 'var(--border)' }}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate font-medium">{providerName}</p>
          <p className="flex flex-wrap items-center gap-x-2 text-xs" style={{ color: 'var(--text-faint)' }}>
            <span>{authLabel(connection.auth_mode)}</span>
            <span aria-hidden>·</span>
            <StatusPill status={connection.status} />
          </p>
        </div>
        <div className="flex gap-1.5">
          <Button size="sm" variant="outline" disabled={busy !== null} onClick={() => void run('test', () => testModelConnection(connection.connection_id), 'Kiểm tra thất bại.')}>
            {busy === 'test' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}Kiểm tra lại
          </Button>
          <Button size="sm" variant="ghost" aria-label="Xoá" disabled={busy !== null} onClick={() => void run('delete', () => deleteModelConnection(connection.connection_id), 'Không xoá được.')}>
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
      {message && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{message}</p>}
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
      <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
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
    <div className="space-y-3 rounded-xl border p-3 text-sm" style={{ borderColor: 'var(--border)' }}>
      {(enrollment.state === 'none' || enrollment.state === 'expired' || enrollment.state === 'failed' || enrollment.state === 'cancelled') && (
        <>
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            Dùng gói ChatGPT Plus/Pro của bạn thay cho API key. Bạn sẽ nhập một mã trên trang của OpenAI; ToxAgent không bao giờ thấy mật khẩu.
          </p>
          {enrollment.error && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{enrollment.error}</p>}
          <Button size="sm" onClick={() => void start()}>Kết nối ChatGPT Plus/Pro</Button>
        </>
      )}
      {enrollment.state === 'pending' && (
        <>
          <p>1. Mở <a className="font-medium underline" href={enrollment.verification_url ?? '#'} target="_blank" rel="noreferrer">{enrollment.verification_url}</a></p>
          <p>2. {enrollment.instructions}</p>
          <p className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-faint)' }}>
            <Loader2 className="h-3 w-3 animate-spin" />
            Đang chờ bạn xác nhận trên trang OpenAI… (còn {Math.ceil((enrollment.expires_in_s ?? 0) / 60)} phút)
          </p>
          <Button size="sm" variant="ghost" onClick={() => void cancel()}>Huỷ</Button>
        </>
      )}
      {enrollment.state === 'connected' && (enrollment.models?.length ? (
        <p className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
          <Loader2 className="h-3 w-3 animate-spin" />Đã đăng nhập ChatGPT — đang lưu kết nối…
        </p>
      ) : (
        <>
          <p>Đã đăng nhập ChatGPT, nhưng OpenAI không trả về danh sách model. Nhập một model ID để kiểm tra kết nối:</p>
          <div className="flex flex-wrap items-center gap-2">
            <input aria-label="Model ID để kiểm tra" value={model} onChange={(e) => setModel(e.target.value)} placeholder="vd. gpt-5" className="h-9 rounded-md border bg-transparent px-3 text-sm" />
            <Button size="sm" disabled={!model.trim() || saving} onClick={() => void save(model.trim())}>Lưu kết nối</Button>
            <Button size="sm" variant="ghost" onClick={() => void cancel()}>Huỷ</Button>
          </div>
        </>
      ))}
      {error && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{error}</p>}
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
  const connectIntent = searchParams.get('connect');
  const [baseUrl, setBaseUrl] = useState('');
  const [credential, setCredential] = useState('');
  const [probeModel, setProbeModel] = useState('');
  const [askModel, setAskModel] = useState(false);
  const [step, setStep] = useState<Step>('idle');
  const [formError, setFormError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

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
  }, []);

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

  return (
    <Card style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
      <CardHeader>
        <CardTitle className="text-base">AI của bạn</CardTitle>
        <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
          Kết nối tài khoản AI ở đây. Model dùng cho từng cuộc trò chuyện được chọn ngay trong khung chat.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {notice && <p className="rounded-lg px-3 py-2 text-xs" style={{ background: 'var(--surface-muted)', color: 'var(--text-muted)' }}>{notice}</p>}

        {connections.length === 0
          ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Chưa có kết nối nào. Key chỉ được gửi lên khi kết nối, được mã hoá khi lưu, và không bao giờ trả lại giao diện.</p>
          : <div className="space-y-2">{connections.map((connection) => (
              <ConnectionRow key={connection.connection_id} connection={connection} providerName={providerName(connection.provider_id)} onChanged={refreshConnections} />
            ))}</div>}

        <div className="space-y-3 border-t pt-4" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm font-medium">Thêm kết nối</p>
          <select aria-label="AI provider" value={provider} onChange={(e) => chooseProvider(e.target.value)} className="h-9 w-full rounded-md border bg-transparent px-3 text-sm">
            {providers.map((item) => <option key={item.provider_id} value={item.provider_id}>{item.display_name}</option>)}
          </select>
          {offersChatGPT && (
            <div className="flex gap-2" role="radiogroup" aria-label="Cách kết nối">
              <Button size="sm" variant={method === 'api_key' ? 'default' : 'outline'} aria-pressed={method === 'api_key'} onClick={() => setMethod('api_key')}>API key</Button>
              <Button size="sm" variant={method === 'chatgpt' ? 'default' : 'outline'} aria-pressed={method === 'chatgpt'} onClick={() => setMethod('chatgpt')}>ChatGPT Plus/Pro</Button>
            </div>
          )}
          {method === 'chatgpt' && selected
            ? <ChatGPTSignIn provider={selected} onConnected={refreshConnections} />
            : (
              <form className="space-y-2" onSubmit={(event) => { event.preventDefault(); void connect(); }}>
                <div className="relative">
                  <KeyRound className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2" style={{ color: 'var(--text-faint)' }} />
                  <input aria-label="AI API key" type="password" value={credential} onChange={(e) => setCredential(e.target.value)} placeholder={keyOptional ? 'API key (bỏ trống nếu server không cần)' : 'API key'} className="h-9 w-full rounded-md border bg-transparent pl-8 pr-3 text-sm" autoComplete="new-password" />
                </div>
                {selected?.base_url_required && <input aria-label="AI base URL" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="Base URL (bắt buộc)" className="h-9 w-full rounded-md border bg-transparent px-3 text-sm" />}
                {needsModelId && <input aria-label="Model ID để kiểm tra" value={probeModel} onChange={(e) => setProbeModel(e.target.value)} placeholder="Model ID để kiểm tra kết nối" className="h-9 w-full rounded-md border bg-transparent px-3 text-sm" />}
                {selected?.note && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>{selected.note}</p>}
                <Button size="sm" type="submit" disabled={step !== 'idle'}>
                  {step !== 'idle' && <Loader2 className="h-3.5 w-3.5 animate-spin" />}{STEP_LABEL[step]}
                </Button>
              </form>
            )}
          {success && <p className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--accent-green)' }}><CheckCircle2 className="h-3.5 w-3.5" />{success}</p>}
          {formError && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{formError}</p>}
        </div>
      </CardContent>
    </Card>
  );
}
