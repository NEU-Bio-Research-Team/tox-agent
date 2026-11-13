import { useEffect, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { ExternalLink } from 'lucide-react';
import { WorkspacePage } from '../../app/shell/WorkspacePage';
import { ReleaseNotesModal } from '../../app/shell/ReleaseNotesModal';
import { APP_COMMIT, APP_VERSION, BUILD_TIME_LABEL } from '../../app/build-info';
import { FEEDBACK_URL, REPO_URL } from '../../app/links';
import { Button } from '../../shared/ui/button';
import { DataList, DataRow } from '../../shared/ui/data-row';
import { PageHeader } from '../../shared/ui/page-header';
import { Segmented } from '../../shared/ui/segmented';
import { StatusPill } from '../../shared/ui/status-pill';
import { Switch } from '../../shared/ui/switch';
import { cn } from '../../shared/ui/utils';
import { getToken, API_BASE_URL } from '../../shared/api/client';
import { queryClient as sharedQueryClient } from '../../shared/lib/queryClient';
import { applyTheme, getTheme, type Theme } from '../../shared/lib/theme';
import { scrollToSection, useActiveSection } from '../../shared/hooks/useActiveSection';
import { signOut } from '../auth/firebaseAuth';
import { getDeveloperModeEnabled, getExpertModeEnabled, setDeveloperModeEnabled, setExpertModeEnabled } from '../../shared/lib/preferences';
import { getHealthReady, type HealthReady } from '../../shared/api/endpoints';
import { AiConnectionsCard } from './AiConnectionsCard';

const SECTIONS = [
  { id: 'ai', label: 'Kết nối AI' },
  { id: 'account', label: 'Tài khoản' },
  { id: 'display', label: 'Hiển thị' },
  { id: 'advanced', label: 'Nâng cao' },
  { id: 'about', label: 'Về bản này' },
] as const;

function Section({ id, title, description, children }: { id: string; title: string; description?: ReactNode; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-6 space-y-4">
      <div className="space-y-1.5">
        <h2 id={`${id}-title`} className="text-xl font-semibold tracking-tight text-ink">{title}</h2>
        {description ? <p className="max-w-[64ch] text-sm leading-relaxed text-ink-secondary">{description}</p> : null}
      </div>
      {children}
    </section>
  );
}

function controlPlaneStatus(health: HealthReady | null) {
  if (!health) return <StatusPill tone="neutral">Chưa đọc được trạng thái</StatusPill>;
  if (!health.ready) return <StatusPill tone="warning" dot>Chưa sẵn sàng</StatusPill>;
  const mode = health.mode === 'agent_enabled' ? 'chế độ agent' : health.mode === 'predictor_only' ? 'chỉ predictor' : null;
  return <StatusPill tone="success" dot>{mode ? `Sẵn sàng · ${mode}` : 'Sẵn sàng'}</StatusPill>;
}

/**
 * Settings (plan section 6.4): a section list on the left and label-left,
 * control-right rows. Connecting an AI account is the first section; its logic
 * lives in AiConnectionsCard. Version, build and release notes moved here from
 * the old navigation bar.
 */
export function SettingsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [expertMode, setExpertMode] = useState(getExpertModeEnabled());
  const [developerMode, setDeveloperMode] = useState(getDeveloperModeEnabled());
  const [theme, setTheme] = useState<Theme>(() => getTheme());
  const [health, setHealth] = useState<HealthReady | null>(null);
  const [notesOpen, setNotesOpen] = useState(false);
  const [active, setActive] = useActiveSection(SECTIONS.map((section) => section.id));

  useEffect(() => { void getHealthReady().then(setHealth).catch(() => setHealth(null)); }, []);

  const disconnect = () => {
    signOut();
    sharedQueryClient.clear();
    navigate('/');
  };

  return (
    <WorkspacePage title="Cài đặt" width="page">
      <PageHeader title="Cài đặt" description="Kết nối AI, tài khoản và cách ToxAgent hiển thị." />

      <div className="flex flex-wrap items-start gap-x-10 gap-y-6">
        <nav aria-label="Các mục cài đặt" className="flex w-full shrink-0 gap-1 overflow-x-auto md:sticky md:top-2 md:w-[200px] md:flex-col md:gap-0.5 md:overflow-visible">
          {SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              aria-current={active === section.id ? 'true' : undefined}
              onClick={(event) => {
                event.preventDefault();
                scrollToSection(section.id, setActive);
              }}
              className={cn(
                'flex h-9 shrink-0 items-center rounded-[10px] px-3 text-sm transition-colors',
                active === section.id ? 'bg-brand-soft-strong font-semibold text-brand-ink' : 'text-ink-secondary hover:bg-surface-muted hover:text-ink',
              )}
            >
              {section.label}
            </a>
          ))}
        </nav>

        <div className="flex min-w-0 max-w-3xl flex-[1_1_520px] flex-col gap-12">
          <Section
            id="ai"
            title="Kết nối AI"
            description="Kết nối tài khoản AI của bạn. Model cho từng cuộc trò chuyện được chọn ngay trong khung chat."
          >
            <AiConnectionsCard
              agentEnabled={health ? health.mode === 'agent_enabled' : null}
              userAiRequired={health?.runtime?.ai_source === 'user_connections'}
              onConnectionsChanged={() => {
                void queryClient.invalidateQueries({ queryKey: ['model-connections'] });
                void queryClient.invalidateQueries({ queryKey: ['connection-models'] });
              }}
            />
          </Section>

          <Section id="account" title="Tài khoản" description="Kết nối giữa trình duyệt này và máy chủ ToxAgent.">
            <DataList>
              <DataRow title="Control plane" description="Máy chủ mà giao diện này đang gọi.">
                <code className="font-mono text-13 text-ink">{API_BASE_URL || 'cùng origin'}</code>
                {controlPlaneStatus(health)}
              </DataRow>
              <DataRow title="Access token" description="Token đăng nhập, lưu trong trình duyệt này.">
                <code className="font-mono text-sm tracking-widest text-ink">{getToken() ? '••••••••' : 'chưa kết nối'}</code>
              </DataRow>
              <DataRow title="Ngắt kết nối" description="Xoá token khỏi trình duyệt này. Session của bạn vẫn được lưu trên máy chủ.">
                <Button variant="outline" className="border-danger-border text-danger hover:bg-danger-soft hover:text-danger" onClick={disconnect}>
                  Ngắt kết nối
                </Button>
              </DataRow>
            </DataList>
          </Section>

          <Section id="display" title="Hiển thị">
            <DataList>
              <DataRow title="Giao diện" description="Sáng hoặc tối. Lưu trong trình duyệt này.">
                <Segmented<Theme>
                  label="Giao diện"
                  value={theme}
                  onValueChange={(next) => {
                    applyTheme(next);
                    setTheme(next);
                  }}
                  options={[
                    { value: 'light', label: 'Sáng' },
                    { value: 'dark', label: 'Tối' },
                  ]}
                />
              </DataRow>
            </DataList>
          </Section>

          <Section id="advanced" title="Nâng cao" description="Dành cho người cần xem dấu vết chi tiết của agent.">
            <DataList>
              <DataRow
                titleId="developer-mode-label"
                title="Hiện Run Details"
                description="Hiện liên kết trace, runtime, usage và raw event trong từng run. Không đưa tool trace vào transcript thông thường."
              >
                <Switch
                  id="developer-mode"
                  aria-labelledby="developer-mode-label"
                  checked={developerMode}
                  onCheckedChange={(checked) => { setDeveloperMode(checked); setDeveloperModeEnabled(checked); }}
                />
              </DataRow>
              <DataRow
                titleId="expert-mode-label"
                title="Chế độ chuyên gia: ghi đè ngưỡng"
                description={<>Chỉ hiển thị control. Backend vẫn từ chối (403) nếu token của bạn không có role <code className="font-mono text-xs">expert</code>, nên bật ở đây không tự cấp quyền.</>}
              >
                <Switch
                  id="expert-mode"
                  aria-labelledby="expert-mode-label"
                  checked={expertMode}
                  onCheckedChange={(checked) => { setExpertMode(checked); setExpertModeEnabled(checked); }}
                />
              </DataRow>
            </DataList>
          </Section>

          <Section id="about" title="Về bản này">
            <DataList>
              <DataRow title="Phiên bản" description="Bản đang chạy trong trình duyệt này.">
                <span className="flex items-center gap-2 font-mono text-13 text-ink">
                  v{APP_VERSION} · {APP_COMMIT}
                  <StatusPill tone="brand" className="uppercase tracking-wide">Beta</StatusPill>
                </span>
              </DataRow>
              <DataRow title="Thời điểm build">
                <span className="font-mono text-13 text-ink">{BUILD_TIME_LABEL}</span>
              </DataRow>
              <DataRow title="Ghi chú phát hành" description="Những gì thay đổi trong bản này.">
                <Button variant="outline" onClick={() => setNotesOpen(true)}>Xem ghi chú</Button>
              </DataRow>
              <DataRow title="Góp ý và mã nguồn">
                <Button asChild variant="outline" className="gap-1.5">
                  <a href={FEEDBACK_URL} target="_blank" rel="noopener noreferrer">Góp ý <ExternalLink /></a>
                </Button>
                <Button asChild variant="outline" className="gap-1.5">
                  <a href={REPO_URL} target="_blank" rel="noopener noreferrer">GitHub <ExternalLink /></a>
                </Button>
              </DataRow>
            </DataList>
          </Section>
        </div>
      </div>

      <ReleaseNotesModal open={notesOpen} onOpenChange={setNotesOpen} />
    </WorkspacePage>
  );
}
