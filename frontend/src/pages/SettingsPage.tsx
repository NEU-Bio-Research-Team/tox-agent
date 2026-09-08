import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { WorkspaceLayout } from '../components/shell/WorkspaceLayout';
import { WorkspaceHeader } from '../components/shell/WorkspaceHeader';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Button } from '../components/ui/button';
import { getToken, setToken, API_BASE_URL } from '../lib/api/client';
import { getExpertModeEnabled, setExpertModeEnabled } from '../lib/preferences';
import { createModelConnection, deleteModelConnection, listModelConnections, testModelConnection } from '../lib/api/endpoints';
import type { ModelConnection } from '../lib/api/types';

export function SettingsPage() {
  const navigate = useNavigate();
  const [expertMode, setExpertMode] = useState(getExpertModeEnabled());
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [provider, setProvider] = useState('openai');
  const [model, setModel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [credential, setCredential] = useState('');
  const [providerError, setProviderError] = useState<string | null>(null);

  const refreshConnections = () => listModelConnections().then((result) => setConnections(result.connections)).catch(() => setProviderError('Không tải được danh sách AI provider.'));
  useEffect(() => { void refreshConnections(); }, []);

  const addProvider = async () => {
    if (!provider.trim() || !model.trim() || !credential.trim()) { setProviderError('Nhập provider, model và API key.'); return; }
    try {
      await createModelConnection({ provider_id: provider.trim(), model_id: model.trim(), auth_mode: 'api_key', base_url: baseUrl.trim() || undefined, credential });
      setCredential(''); setModel(''); setProviderError(null); await refreshConnections();
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
          <CardHeader><CardTitle className="text-base">AI Providers</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {connections.length === 0 ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Chưa có provider nào. API key chỉ được gửi khi tạo kết nối và không bao giờ trả lại giao diện.</p> : connections.map((connection) => (
              <div key={connection.connection_id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3 text-sm" style={{ borderColor: 'var(--border)' }}>
                <div><p className="font-medium">{connection.provider_id} · {connection.model_id}</p><p className="text-xs" style={{ color: 'var(--text-faint)' }}>{connection.status === 'ready' ? 'Connected ✓' : connection.status === 'failed' ? 'Connection failed' : 'Chưa kiểm tra'} · Credential saved {connection.has_credential ? '✓' : '—'}</p></div>
                <div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => void testModelConnection(connection.connection_id).then(refreshConnections).catch(() => refreshConnections())}>Test</Button><Button size="sm" variant="ghost" onClick={() => void deleteModelConnection(connection.connection_id).then(refreshConnections)}>Xoá</Button></div>
              </div>
            ))}
            <div className="grid gap-2 sm:grid-cols-2">
              <input aria-label="AI provider" value={provider} onChange={(e) => setProvider(e.target.value)} placeholder="Provider (openai)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />
              <input aria-label="AI model" value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model (gpt-...)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />
              <input aria-label="AI base URL" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="Base URL (optional)" className="h-9 rounded-md border bg-transparent px-3 text-sm" />
              <input aria-label="AI API key" type="password" value={credential} onChange={(e) => setCredential(e.target.value)} placeholder="API key" className="h-9 rounded-md border bg-transparent px-3 text-sm" autoComplete="new-password" />
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
