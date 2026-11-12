import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { WorkspaceLayout } from '../../app/shell/WorkspaceLayout';
import { WorkspaceHeader } from '../../app/shell/WorkspaceHeader';
import { Card, CardContent, CardHeader, CardTitle } from '../../shared/ui/card';
import { Label } from '../../shared/ui/label';
import { Switch } from '../../shared/ui/switch';
import { Button } from '../../shared/ui/button';
import { getToken, API_BASE_URL } from '../../shared/api/client';
import { signOut } from '../auth/firebaseAuth';
import { getDeveloperModeEnabled, getExpertModeEnabled, setDeveloperModeEnabled, setExpertModeEnabled } from '../../shared/lib/preferences';
import { getHealthReady, type HealthReady } from '../../shared/api/endpoints';
import { AiConnectionsCard } from './AiConnectionsCard';

// The AI section (API keys and ChatGPT sign-in only — models are picked in the
// chat) lives in AiConnectionsCard; this page keeps the connection to the control plane and
// the display preferences.

export function SettingsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [expertMode, setExpertMode] = useState(getExpertModeEnabled());
  const [developerMode, setDeveloperMode] = useState(getDeveloperModeEnabled());
  const [health, setHealth] = useState<HealthReady | null>(null);

  useEffect(() => { void getHealthReady().then(setHealth).catch(() => setHealth(null)); }, []);

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
                signOut();
                navigate('/');
              }}
            >
              Ngắt kết nối
            </Button>
          </CardContent>
        </Card>

        <AiConnectionsCard
          agentEnabled={health ? health.mode === 'agent_enabled' : null}
          userAiRequired={health?.runtime?.ai_source === 'user_connections'}
          onConnectionsChanged={() => {
            void queryClient.invalidateQueries({ queryKey: ['model-connections'] });
            void queryClient.invalidateQueries({ queryKey: ['connection-models'] });
          }}
        />

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
