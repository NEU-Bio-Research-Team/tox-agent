import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { WorkspaceLayout } from '../components/shell/WorkspaceLayout';
import { WorkspaceHeader } from '../components/shell/WorkspaceHeader';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Button } from '../components/ui/button';
import { getToken, setToken, API_BASE_URL } from '../lib/api/client';
import { getDefaultAiProfileId, getDeveloperModeEnabled, getExpertModeEnabled, setDefaultAiProfileId, setDeveloperModeEnabled, setExpertModeEnabled } from '../lib/preferences';
import { createModelConnection, deleteModelConnection, getHealthReady, listModelConnections, listSupportedProviders, testModelConnection, type HealthReady, type SupportedProvider } from '../lib/api/endpoints';
import { describeAgentProfileChoice } from '../lib/aiProfile';
import { ApiError, type ModelConnection } from '../lib/api/types';

// The provider list comes from the server (I13). The hardcoded one here
// offered Anthropic and Google Gemini, whose wire formats this control plane
// has no adapter for, with a blank default base URL — so the form could be
// filled in correctly, save, and then fail every capability probe with
// "capability probing requires an explicit base_url".
//
// The form asks only for what the provider needs. The authentication mode
// follows from whether a key was given (a provider that accepts none says so
// in `auth_modes`), a display name defaults server-side, and a base URL is
// asked for only when the provider has no fixed endpoint.

export function SettingsPage() {
  const navigate = useNavigate();
  const [expertMode, setExpertMode] = useState(getExpertModeEnabled());
  const [developerMode, setDeveloperMode] = useState(getDeveloperModeEnabled());
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [providers, setProviders] = useState<SupportedProvider[]>([]);
  const [provider, setProvider] = useState('');
  const [model, setModel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [credential, setCredential] = useState('');
  const [providerError, setProviderError] = useState<string | null>(null);
  const [defaultProfileId, setDefaultProfileIdState] = useState(getDefaultAiProfileId());
  const [health, setHealth] = useState<HealthReady | null>(null);

  const refreshConnections = () => listModelConnections().then((result) => setConnections(result.connections)).catch(() => setProviderError('Không tải được danh sách AI provider.'));
  useEffect(() => { void refreshConnections(); }, []);
  useEffect(() => { void getHealthReady().then(setHealth).catch(() => setHealth(null)); }, []);
  useEffect(() => {
    void listSupportedProviders()
      .then((result) => {
        setProviders(result.providers);
        const first = result.providers[0];
        if (first) setProvider(first.provider_id);
      })
      .catch(() => setProviderError('Không tải được danh sách provider được hỗ trợ.'));
  }, []);

  const selected = providers.find((item) => item.provider_id === provider) ?? null;
  const keyOptional = selected?.auth_modes.includes('none') ?? false;
  const readyConnections = connections.filter((item) => item.status === 'ready');
  const profileChoice = describeAgentProfileChoice({
    agentEnabled: health ? health.mode === 'agent_enabled' : null,
    readyProfileCount: readyConnections.length,
  });
  // A default that was deleted or stopped passing its test is not in force;
  // useSessionProfileSync falls back to the server runtime for it.
  const defaultInForce = readyConnections.some((item) => item.connection_id === defaultProfileId) ? defaultProfileId! : '';

  const addProvider = async () => {
    if (!provider.trim() || !model.trim()) { setProviderError('Chọn provider và nhập model.'); return; }
    if (!keyOptional && !credential.trim()) { setProviderError('Nhập API key để tạo kết nối này.'); return; }
    if (selected?.base_url_required && !baseUrl.trim()) { setProviderError(`${selected.display_name} cần Base URL.`); return; }
    const authMode: ModelConnection['auth_mode'] = credential.trim() ? 'api_key' : 'none';
    try {
      await createModelConnection({ provider_id: provider.trim(), model_id: model.trim(), auth_mode: authMode, base_url: selected?.base_url_required ? baseUrl.trim() : undefined, credential: authMode === 'api_key' ? credential : undefined });
      setCredential(''); setModel(''); setBaseUrl(''); setProviderError(null); await refreshConnections();
    } catch (error) {
      // The server's message says what is wrong and often what to pick
      // instead; replacing it with one generic sentence is what left a user
      // with a form that failed and no way to know why.
      setProviderError(error instanceof ApiError ? error.message : 'Không thể lưu provider.');
    }
  };

  return (
    <WorkspaceLayout>
      <div className="flex h-full flex-col">
        <WorkspaceHeader title="Cài đặt" />
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
          <div className="mx-auto max-w-2xl space-y-6">
        <Card style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
          <CardHeader>
            <CardTitle className="text-base">Kết nối</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span style={{ color: 'var(--text-muted)' }}>Control plane</span>
              <code style={{ color: 'var(--text)' }}>{API_BASE_URL || '(same origin)'}</code>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span style={{ color: 'var(--text-muted)' }}>Access token</span>
              <code style={{ color: 'var(--text)' }}>{getToken() ? '••••••••' : 'chưa kết nối'}</code>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setToken(null);
                navigate('/');
              }}
            >
              Ngắt kết nối
            </Button>
          </CardContent>
        </Card>

        <Card style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
          <CardHeader><CardTitle className="text-base">AI</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="default-ai-profile">Profile dùng cho các phiên</Label>
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
                {profileChoice.notice ?? 'Áp dụng cho run tiếp theo của mọi phiên. Chỉ profile đã Test thành công mới chọn được.'}
              </p>
            </div>

            {connections.length === 0 ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Chưa có provider nào. API key chỉ được gửi khi tạo kết nối và không bao giờ trả lại giao diện.</p> : connections.map((connection) => (
              <div key={connection.connection_id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3 text-sm" style={{ borderColor: 'var(--border)' }}>
                <div><p className="font-medium">{connection.display_name}</p><p className="text-xs" style={{ color: 'var(--text-faint)' }}>{connection.provider_id} · {connection.model_id} · {connection.status === 'ready' ? 'Connected ✓' : connection.status === 'failed' ? 'Connection failed' : 'Chưa kiểm tra'} · Credential saved {connection.has_credential ? '✓' : '—'}</p></div>
                <div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => void testModelConnection(connection.connection_id).then(refreshConnections).catch(() => refreshConnections())}>Test</Button><Button size="sm" variant="ghost" onClick={() => void deleteModelConnection(connection.connection_id).then(refreshConnections)}>Xoá</Button></div>
              </div>
            ))}
            <div className="grid gap-2 sm:grid-cols-2">
              <select aria-label="AI provider" value={provider} onChange={(e) => { setProvider(e.target.value); setBaseUrl(''); }} className="h-9 rounded-md border bg-transparent px-3 text-sm">
                {providers.map((item) => <option key={item.provider_id} value={item.provider_id}>{item.display_name}</option>)}
              </select>
              <input aria-label="AI model" value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model (gpt-...)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />
              <input aria-label="AI API key" type="password" value={credential} onChange={(e) => setCredential(e.target.value)} placeholder={keyOptional ? 'API key (bỏ trống nếu server không cần)' : 'API key'} className="h-9 rounded-md border bg-transparent px-3 text-sm" autoComplete="new-password" />
              {selected?.base_url_required && <input aria-label="AI base URL" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="Base URL (bắt buộc)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />}
            </div>
            {selected?.note && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>{selected.note}</p>}
            {providerError && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{providerError}</p>}
            <Button size="sm" onClick={() => void addProvider()}>Add provider</Button>
          </CardContent>
        </Card>

        <details className="rounded-xl border px-6 py-4" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
          <summary className="cursor-pointer text-base font-semibold">Nâng cao</summary>
          <div className="mt-4 space-y-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <Label htmlFor="developer-mode">Hiện Run Details</Label>
                <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
                  Hiện liên kết trace, runtime, usage và raw event trong từng run. Không đưa tool trace vào transcript thông thường.
                </p>
              </div>
              <Switch id="developer-mode" checked={developerMode} onCheckedChange={(checked) => { setDeveloperMode(checked); setDeveloperModeEnabled(checked); }} />
            </div>
            <div className="flex items-center justify-between gap-4">
              <div>
                <Label htmlFor="expert-mode">Chế độ chuyên gia: threshold override</Label>
                <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
                  Chỉ hiển thị control; backend vẫn từ chối (403 forbidden) nếu token của bạn
                  không có role <code>expert</code>. Bật control ở đây không tự cấp quyền.
                </p>
              </div>
              <Switch
                id="expert-mode"
                checked={expertMode}
                onCheckedChange={(checked) => {
                  setExpertMode(checked);
                  setExpertModeEnabled(checked);
                }}
              />
            </div>
          </div>
        </details>
          </div>
        </div>
      </div>
    </WorkspaceLayout>
  );
}
