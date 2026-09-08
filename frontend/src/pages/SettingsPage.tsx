import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { WorkspaceLayout } from '../components/shell/WorkspaceLayout';
import { WorkspaceHeader } from '../components/shell/WorkspaceHeader';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Button } from '../components/ui/button';
import { getToken, setToken, API_BASE_URL } from '../lib/api/client';
import { getDeveloperModeEnabled, getExpertModeEnabled, setDeveloperModeEnabled, setExpertModeEnabled } from '../lib/preferences';
import { createModelConnection, deleteModelConnection, listModelConnections, testModelConnection } from '../lib/api/endpoints';
import type { ModelConnection } from '../lib/api/types';

const PROVIDERS = [
  { id: 'openai', label: 'OpenAI', baseUrl: '' },
  { id: 'anthropic', label: 'Anthropic', baseUrl: '' },
  { id: 'gemini', label: 'Google Gemini', baseUrl: '' },
  { id: 'openrouter', label: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1' },
  { id: 'openai_compatible', label: 'OpenAI-compatible / local', baseUrl: '' },
] as const;

export function SettingsPage() {
  const navigate = useNavigate();
  const [expertMode, setExpertMode] = useState(getExpertModeEnabled());
  const [developerMode, setDeveloperMode] = useState(getDeveloperModeEnabled());
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [provider, setProvider] = useState('openai');
  const [model, setModel] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [authMode, setAuthMode] = useState<ModelConnection['auth_mode']>('api_key');
  const [credential, setCredential] = useState('');
  const [providerError, setProviderError] = useState<string | null>(null);

  const refreshConnections = () => listModelConnections().then((result) => setConnections(result.connections)).catch(() => setProviderError('Không tải được danh sách AI provider.'));
  useEffect(() => { void refreshConnections(); }, []);

  const addProvider = async () => {
    if (!provider.trim() || !model.trim()) { setProviderError('Chọn provider và nhập model.'); return; }
    if (authMode === 'api_key' && !credential.trim()) { setProviderError('Nhập API key để tạo kết nối này.'); return; }
    if (provider === 'openai_compatible' && !baseUrl.trim()) { setProviderError('OpenAI-compatible cần Base URL.'); return; }
    try {
      await createModelConnection({ provider_id: provider.trim(), model_id: model.trim(), display_name: displayName.trim() || undefined, auth_mode: authMode, base_url: baseUrl.trim() || undefined, credential: authMode === 'api_key' ? credential : undefined });
      setCredential(''); setModel(''); setDisplayName(''); setProviderError(null); await refreshConnections();
    } catch { setProviderError('Không thể lưu provider. Kiểm tra endpoint và quyền truy cập.'); }
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
          <CardHeader><CardTitle className="text-base">Developer diagnostics</CardTitle></CardHeader>
          <CardContent>
            <div className="flex items-center justify-between gap-4"><div><Label htmlFor="developer-mode">Hiện Run Details</Label><p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>Hiện liên kết trace, runtime, usage và raw event trong từng run. Không đưa tool trace vào transcript thông thường.</p></div><Switch id="developer-mode" checked={developerMode} onCheckedChange={(checked) => { setDeveloperMode(checked); setDeveloperModeEnabled(checked); }} /></div>
          </CardContent>
        </Card>

        <Card style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
          <CardHeader><CardTitle className="text-base">AI Providers</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {connections.length === 0 ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Chưa có provider nào. API key chỉ được gửi khi tạo kết nối và không bao giờ trả lại giao diện.</p> : connections.map((connection) => (
              <div key={connection.connection_id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3 text-sm" style={{ borderColor: 'var(--border)' }}>
                <div><p className="font-medium">{connection.display_name}</p><p className="text-xs" style={{ color: 'var(--text-faint)' }}>{connection.provider_id} · {connection.model_id} · {connection.status === 'ready' ? 'Connected ✓' : connection.status === 'failed' ? 'Connection failed' : 'Chưa kiểm tra'} · Credential saved {connection.has_credential ? '✓' : '—'}</p></div>
                <div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => void testModelConnection(connection.connection_id).then(refreshConnections).catch(() => refreshConnections())}>Test</Button><Button size="sm" variant="ghost" onClick={() => void deleteModelConnection(connection.connection_id).then(refreshConnections)}>Xoá</Button></div>
              </div>
            ))}
            <div className="grid gap-2 sm:grid-cols-2">
              <select aria-label="AI provider" value={provider} onChange={(e) => { const next = e.target.value; setProvider(next); setBaseUrl(PROVIDERS.find((item) => item.id === next)?.baseUrl ?? ''); }} className="h-9 rounded-md border bg-transparent px-3 text-sm">
                {PROVIDERS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
              </select>
              <input aria-label="AI model" value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model (gpt-...)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />
              <input aria-label="AI profile name" value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder="Tên profile (optional)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />
              <input aria-label="AI base URL" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder={provider === 'openai_compatible' ? 'Base URL (required)' : 'Base URL (optional)'} className="h-9 rounded-md border bg-transparent px-3 text-sm" />
              <select aria-label="AI authentication" value={authMode} onChange={(e) => setAuthMode(e.target.value as ModelConnection['auth_mode'])} className="h-9 rounded-md border bg-transparent px-3 text-sm"><option value="api_key">API key</option><option value="local">Local / no credential</option><option value="none">No authentication</option></select>
              {authMode === 'api_key' && <input aria-label="AI API key" type="password" value={credential} onChange={(e) => setCredential(e.target.value)} placeholder="API key" className="h-9 rounded-md border bg-transparent px-3 text-sm" autoComplete="new-password" />}
            </div>
            {providerError && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>{providerError}</p>}
            <Button size="sm" onClick={() => void addProvider()}>Add provider</Button>
          </CardContent>
        </Card>

        <Card style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
          <CardHeader>
            <CardTitle className="text-base">Chế độ chuyên gia</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="expert-mode">Hiện tuỳ chọn threshold override</Label>
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
          </CardContent>
        </Card>
          </div>
        </div>
      </div>
    </WorkspaceLayout>
  );
}
