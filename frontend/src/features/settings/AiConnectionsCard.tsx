import { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../shared/ui/card';
import { Label } from '../../shared/ui/label';
import { Button } from '../../shared/ui/button';
import {
  cancelChatGPTSignIn,
  changeConnectionModel,
  completeChatGPTSignIn,
  createModelConnection,
  deleteModelConnection,
  getChatGPTSignIn,
  listConnectionModels,
  listModelConnections,
  listModelsForKey,
  listSupportedProviders,
  startChatGPTSignIn,
  testModelConnection,
  type ChatGPTEnrollment,
  type ModelOption,
  type SupportedProvider,
} from '../../shared/api/endpoints';
import { describeAgentProfileChoice } from '../../shared/lib/aiProfile';
import { getDefaultAiProfileId, setDefaultAiProfileId } from '../../shared/lib/preferences';
import { ApiError, type ModelConnection } from '../../shared/api/types';

// Users connect their own AI here: an API key (OpenAI, Claude, OpenRouter,
// DeepSeek, any OpenAI-compatible server) or a ChatGPT Plus/Pro sign-in. The
// design follows G1_Web_Design's: list the models the key can actually use and
// pick one, rather than guessing an id; a sign-in is a device code the user
// approves on OpenAI's own page, never a token pasted here.
//
// The provider list and what each one accepts come from the server (I13), so
// the form never offers something the server would refuse — including the
// ChatGPT sign-in, which a deployment may not run.

const POLL_MS = 3000;

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

function statusLabel(status: ModelConnection['status']): string {
  if (status === 'ready') return 'Đã kết nối ✓';
  if (status === 'failed') return 'Kết nối lỗi';
  return 'Chưa kiểm tra';
}

function authLabel(mode: string): string {
  if (mode === 'chatgpt_subscription') return 'ChatGPT Plus/Pro';
  if (mode === 'api_key') return 'API key';
  return 'Không cần key';
}

/** A text field with the listed models as suggestions; any id may still be typed. */
function ModelPicker({
  id, value, onChange, models, placeholder,
}: {
  id: string; value: string; onChange: (value: string) => void; models: ModelOption[]; placeholder: string;
}) {
  return (
    <>
      <input
        aria-label="AI model"
        list={`${id}-options`}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className="h-9 rounded-md border bg-transparent px-3 text-sm"
      />
      <datalist id={`${id}-options`}>
        {models.map((model) => (
          <option key={model.id} value={model.id}>{model.name !== model.id ? model.name : undefined}</option>
        ))}
      </datalist>
    </>
  );
}

function ConnectionRow({ connection, onChanged }: { connection: ModelConnection; onChanged: () => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [model, setModel] = useState(connection.model_id);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async (action: () => Promise<unknown>, fallback: string) => {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (error) {
      setMessage(errorText(error, fallback));
    } finally {
      setBusy(false);
      await onChanged();
    }
  };

  const openModelEditor = async () => {
    setEditing(true);
    setMessage(null);
    try {
      setModels((await listConnectionModels(connection.connection_id)).models);
    } catch (error) {
      setModels([]);
      setMessage(errorText(error, 'Không tải được danh sách model; vẫn có thể nhập ID model.'));
    }
  };

  return (
    <div className="space-y-2 rounded-lg border p-3 text-sm" style={{ borderColor: 'var(--border)' }}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="font-medium">{connection.display_name}</p>
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            {connection.provider_id} · {connection.model_id} · {authLabel(connection.auth_mode)} · {statusLabel(connection.status)}
            {connection.has_credential ? ' · đã lưu credential' : ''}
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" disabled={busy} onClick={() => void run(() => testModelConnection(connection.connection_id), 'Kiểm tra thất bại.')}>Test</Button>
          <Button size="sm" variant="outline" disabled={busy} onClick={() => void (editing ? setEditing(false) : openModelEditor())}>Đổi model</Button>
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => void run(() => deleteModelConnection(connection.connection_id), 'Không xoá được.')}>Xoá</Button>
        </div>
      </div>
      {editing && (
        <div className="flex flex-wrap items-center gap-2">
          <ModelPicker id={`edit-${connection.connection_id}`} value={model} onChange={setModel} models={models} placeholder="ID model" />
          <Button
            size="sm"
            disabled={busy || !model.trim() || model.trim() === connection.model_id}
            onClick={() => void run(async () => {
              await changeConnectionModel(connection.connection_id, { model_id: model.trim() });
              setEditing(false);
            }, 'Không đổi được model.')}
          >
            Lưu model
          </Button>
          <span className="text-xs" style={{ color: 'var(--text-faint)' }}>Sau khi đổi, bấm Test lại.</span>
        </div>
      )}
      {message && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{message}</p>}
    </div>
  );
}

function ChatGPTSignIn({ provider, onConnected }: { provider: SupportedProvider; onConnected: () => Promise<void> }) {
  const oauth = provider.chatgpt_oauth;
  const [enrollment, setEnrollment] = useState<ChatGPTEnrollment>({ state: 'none' });
  const [model, setModel] = useState('');
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

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

  useEffect(() => {
    if (enrollment.state === 'connected' && !model && enrollment.models?.length) {
      setModel(enrollment.models[0].id);
    }
  }, [enrollment, model]);

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
    setModel('');
    try { setEnrollment(await startChatGPTSignIn()); } catch (e) { setError(errorText(e, 'Không bắt đầu được đăng nhập ChatGPT.')); }
  };
  const cancel = async () => {
    stopPolling();
    await cancelChatGPTSignIn().catch(() => undefined);
    setEnrollment({ state: 'none' });
  };
  const save = async () => {
    setError(null);
    try {
      await completeChatGPTSignIn({ model_id: model.trim() });
      setEnrollment({ state: 'none' });
      setModel('');
      await onConnected();
    } catch (e) {
      setError(errorText(e, 'Không lưu được kết nối ChatGPT.'));
    }
  };

  return (
    <div className="space-y-3 rounded-lg border p-3 text-sm" style={{ borderColor: 'var(--border)' }}>
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
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Đang chờ bạn xác nhận trên trang OpenAI… (còn {Math.ceil((enrollment.expires_in_s ?? 0) / 60)} phút)</p>
          <Button size="sm" variant="ghost" onClick={() => void cancel()}>Huỷ</Button>
        </>
      )}
      {enrollment.state === 'connected' && (
        <>
          <p>Đã đăng nhập ChatGPT. Chọn model cho kết nối này:</p>
          <div className="flex flex-wrap items-center gap-2">
            <ModelPicker id="chatgpt-model" value={model} onChange={setModel} models={enrollment.models ?? []} placeholder="ID model" />
            <Button size="sm" disabled={!model.trim()} onClick={() => void save()}>Lưu kết nối</Button>
            <Button size="sm" variant="ghost" onClick={() => void cancel()}>Huỷ</Button>
          </div>
        </>
      )}
      {error && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{error}</p>}
    </div>
  );
}

export function AiConnectionsCard({ agentEnabled, userAiRequired = false }: { agentEnabled: boolean | null; userAiRequired?: boolean }) {
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [providers, setProviders] = useState<SupportedProvider[]>([]);
  const [provider, setProvider] = useState('');
  const [method, setMethod] = useState<'api_key' | 'chatgpt'>('api_key');
  const [model, setModel] = useState('');
  const [models, setModels] = useState<ModelOption[]>([]);
  const [baseUrl, setBaseUrl] = useState('');
  const [credential, setCredential] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [listing, setListing] = useState(false);
  const [defaultProfileId, setDefaultProfileIdState] = useState(getDefaultAiProfileId());

  const refreshConnections = async () => {
    try { setConnections((await listModelConnections()).connections); } catch { setFormError('Không tải được danh sách kết nối AI.'); }
  };
  useEffect(() => { void refreshConnections(); }, []);
  useEffect(() => {
    void listSupportedProviders()
      .then((result) => {
        setProviders(result.providers);
        if (result.providers[0]) setProvider(result.providers[0].provider_id);
      })
      .catch(() => setFormError('Không tải được danh sách provider được hỗ trợ.'));
  }, []);

  const selected = useMemo(() => providers.find((item) => item.provider_id === provider) ?? null, [providers, provider]);
  const offersChatGPT = Boolean(selected?.chatgpt_oauth);
  const keyOptional = selected?.auth_modes.includes('none') ?? false;
  const readyConnections = connections.filter((item) => item.status === 'ready');
  const profileChoice = describeAgentProfileChoice({ agentEnabled, readyProfileCount: readyConnections.length, userAiRequired });
  // A default that was deleted or stopped passing its test is not in force;
  // useSessionProfileSync falls back to the server runtime for it.
  const defaultInForce = readyConnections.some((item) => item.connection_id === defaultProfileId) ? defaultProfileId! : '';

  const chooseProvider = (next: string) => {
    setProvider(next);
    setBaseUrl('');
    setModels([]);
    setModel('');
    setMethod('api_key');
    setFormError(null);
  };

  const loadModels = async () => {
    if (!keyOptional && !credential.trim()) { setFormError('Nhập API key để tải danh sách model.'); return; }
    if (selected?.base_url_required && !baseUrl.trim()) { setFormError(`${selected.display_name} cần Base URL.`); return; }
    setListing(true);
    setFormError(null);
    try {
      const result = await listModelsForKey({
        provider_id: provider,
        base_url: selected?.base_url_required ? baseUrl.trim() : undefined,
        credential: credential.trim() || undefined,
      });
      setModels(result.models);
      if (!model && result.models[0]) setModel(result.models[0].id);
      if (result.models.length === 0) setFormError('Provider không trả về model nào; hãy nhập ID model.');
    } catch (error) {
      setModels([]);
      setFormError(errorText(error, 'Không tải được danh sách model.'));
    } finally {
      setListing(false);
    }
  };

  const addConnection = async () => {
    if (!provider || !model.trim()) { setFormError('Chọn provider và model.'); return; }
    if (!keyOptional && !credential.trim()) { setFormError('Nhập API key để tạo kết nối này.'); return; }
    if (selected?.base_url_required && !baseUrl.trim()) { setFormError(`${selected.display_name} cần Base URL.`); return; }
    const authMode: ModelConnection['auth_mode'] = credential.trim() ? 'api_key' : 'none';
    try {
      await createModelConnection({
        provider_id: provider, model_id: model.trim(), auth_mode: authMode,
        base_url: selected?.base_url_required ? baseUrl.trim() : undefined,
        credential: authMode === 'api_key' ? credential : undefined,
      });
      setCredential(''); setModel(''); setModels([]); setBaseUrl(''); setFormError(null);
      await refreshConnections();
    } catch (error) {
      // The server's message says what is wrong and often what to pick instead.
      setFormError(errorText(error, 'Không thể lưu kết nối.'));
    }
  };

  return (
    <Card style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
      <CardHeader><CardTitle className="text-base">AI của bạn</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label htmlFor="default-ai-profile">Kết nối dùng cho các phiên</Label>
          <select
            id="default-ai-profile"
            className="mt-1 h-9 w-full rounded-md border bg-transparent px-3 text-sm"
            disabled={profileChoice.disabled}
            value={defaultInForce}
            onChange={(event) => {
              const next = event.target.value || null;
              setDefaultAiProfileId(next);
              setDefaultProfileIdState(next);
            }}
          >
            <option value="">{profileChoice.defaultOptionLabel}</option>
            {readyConnections.map((connection) => (
              <option key={connection.connection_id} value={connection.connection_id}>
                {connection.display_name || `${connection.provider_id} · ${connection.model_id}`}
              </option>
            ))}
          </select>
          <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
            {profileChoice.notice ?? 'Áp dụng cho run tiếp theo của mọi phiên. Chỉ kết nối đã Test thành công mới chọn được.'}
          </p>
        </div>

        {connections.length === 0
          ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Chưa có kết nối nào. Key chỉ được gửi lên khi tạo kết nối, được mã hoá khi lưu, và không bao giờ trả lại giao diện.</p>
          : connections.map((connection) => <ConnectionRow key={connection.connection_id} connection={connection} onChanged={refreshConnections} />)}

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
              <>
                <div className="grid gap-2 sm:grid-cols-2">
                  <input aria-label="AI API key" type="password" value={credential} onChange={(e) => setCredential(e.target.value)} placeholder={keyOptional ? 'API key (bỏ trống nếu server không cần)' : 'API key'} className="h-9 rounded-md border bg-transparent px-3 text-sm" autoComplete="new-password" />
                  {selected?.base_url_required && <input aria-label="AI base URL" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="Base URL (bắt buộc)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />}
                  <ModelPicker id="new-connection" value={model} onChange={setModel} models={models} placeholder={models.length ? 'Chọn hoặc nhập model' : 'ID model'} />
                  {selected?.lists_models && (
                    <Button size="sm" variant="outline" disabled={listing} onClick={() => void loadModels()}>
                      {listing ? 'Đang tải…' : models.length ? `${models.length} model — tải lại` : 'Tải danh sách model'}
                    </Button>
                  )}
                </div>
                {selected?.note && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>{selected.note}</p>}
                <Button size="sm" onClick={() => void addConnection()}>Thêm kết nối</Button>
              </>
            )}
          {formError && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{formError}</p>}
        </div>
      </CardContent>
    </Card>
  );
}
