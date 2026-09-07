import { useState } from 'react';
import { useNavigate } from 'react-router';
import { WorkspaceLayout } from '../components/shell/WorkspaceLayout';
import { WorkspaceHeader } from '../components/shell/WorkspaceHeader';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Button } from '../components/ui/button';
import { getToken, setToken, API_BASE_URL } from '../lib/api/client';
import { getExpertModeEnabled, setExpertModeEnabled } from '../lib/preferences';

export function SettingsPage() {
  const navigate = useNavigate();
  const [expertMode, setExpertMode] = useState(getExpertModeEnabled());

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
