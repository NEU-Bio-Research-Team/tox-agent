import { useEffect, useState } from 'react';
import { Link } from 'react-router';
import { SlidersHorizontal } from 'lucide-react';
import { Button } from '../ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover';
import {
  getHealthReady,
  getSessionSettings,
  listModelConnections,
  quickPredictCapabilities,
  updateSessionSettings,
  type HealthReady,
  type SessionSettings,
} from '../../lib/api/endpoints';
import type { Endpoint, PredictCapabilities } from '../../lib/api/types';

interface Connection {
  connection_id: string;
  provider_id: string;
  model_id: string;
  display_name: string;
  status: string;
}

const connectionLabel = (connection: Connection) =>
  connection.display_name || `${connection.provider_id} · ${connection.model_id}`;

export interface AgentProfileChoice {
  /** What the null (no profile saved) option is called. */
  defaultOptionLabel: string;
  /** Shown under the control when something needs configuring. */
  notice: string | null;
  /** True when picking a profile cannot help. */
  disabled: boolean;
}

/**
 * What to call "no profile saved", given what the deployment actually has.
 *
 * I06: this was unconditionally "Runtime mặc định", which reads as "a usable
 * default is in force". It said that in a predictor-only deployment with no
 * runtime at all, and in an agent deployment with no profiles configured —
 * two states where nothing would run. Separated out from the component
 * because the wording is the fix; the popover around it is not.
 *
 * `agentEnabled` is `null` while /health/ready is still in flight: assume
 * nothing, and say nothing, rather than flashing a claim either way.
 */
export function describeAgentProfileChoice({
  agentEnabled,
  readyProfileCount,
}: {
  agentEnabled: boolean | null;
  readyProfileCount: number;
}): AgentProfileChoice {
  if (agentEnabled === false) {
    return {
      defaultOptionLabel: 'Không có agent runtime',
      notice:
        'Bản triển khai này chỉ chạy dự đoán. Câu hỏi về báo cáo, attribution và tra cứu ' +
        'tài liệu không khả dụng ở đây.',
      disabled: true,
    };
  }
  if (readyProfileCount === 0) {
    return {
      defaultOptionLabel: 'Chưa cấu hình AI',
      notice: 'Chưa có profile AI nào sẵn sàng.',
      disabled: false,
    };
  }
  return {
    defaultOptionLabel: 'Runtime mặc định của server',
    notice: null,
    disabled: false,
  };
}

export function SessionConfigPopover({ sessionId }: { sessionId: string }) {
  const [settings, setSettings] = useState<SessionSettings>({
    ai_profile_id: null,
    predictor_bindings: {},
  });
  const [caps, setCaps] = useState<PredictCapabilities | null>(null);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [health, setHealth] = useState<HealthReady | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void Promise.all([
      getSessionSettings(sessionId),
      quickPredictCapabilities(),
      listModelConnections(),
      getHealthReady(),
    ]).then(([saved, catalog, profiles, ready]) => {
      setSettings(saved);
      setCaps(catalog);
      setConnections(profiles.connections);
      setHealth(ready);
    });
  }, [sessionId]);

  const save = async () => {
    setSaving(true);
    try {
      setSettings(await updateSessionSettings(sessionId, settings));
    } finally {
      setSaving(false);
    }
  };

  const readyConnections = connections.filter((item) => item.status === 'ready');
  const activeProfile = connections.find((item) => item.connection_id === settings.ai_profile_id);

  // `mode` comes from /health/ready, which is the only thing that knows
  // whether a runtime is bound rather than merely configured (I01/I05).
  const { defaultOptionLabel, notice, disabled } = describeAgentProfileChoice({
    agentEnabled: health ? health.mode === 'agent_enabled' : null,
    readyProfileCount: readyConnections.length,
  });

  const triggerLabel = activeProfile ? connectionLabel(activeProfile) : 'AI & Predictors';

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="max-w-48 gap-1.5 truncate"
          aria-label="Cấu hình AI và predictor"
        >
          <SlidersHorizontal className="h-4 w-4 shrink-0" />
          <span className="truncate">{triggerLabel}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 space-y-3">
        <div>
          <p className="text-sm font-semibold">Cấu hình phiên</p>
          <p className="text-xs text-muted-foreground">Được pin vào mỗi run khi gửi.</p>
        </div>

        <label className="block text-xs font-medium">
          Agent model
          <select
            className="mt-1 w-full rounded border bg-transparent p-2"
            // Selecting a profile cannot help when nothing can run it.
            disabled={disabled}
            value={settings.ai_profile_id ?? ''}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                ai_profile_id: event.target.value || null,
              }))
            }
          >
            <option value="">{defaultOptionLabel}</option>
            {readyConnections.map((item) => (
              <option key={item.connection_id} value={item.connection_id}>
                {connectionLabel(item)}
              </option>
            ))}
          </select>
        </label>

        {notice && (
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {notice}{' '}
            <Link to="/settings" className="underline">
              Mở Cài đặt
            </Link>
          </p>
        )}

        {(caps?.endpoints ?? [])
          .filter((endpoint) => endpoint.enabled)
          .map((endpoint) => (
            <ModelBinding
              key={endpoint.id}
              endpoint={endpoint.id}
              label={endpoint.display_name}
              models={endpoint.models ?? []}
              value={settings.predictor_bindings[endpoint.id] ?? ''}
              onChange={(modelId) =>
                setSettings((current) => ({
                  ...current,
                  predictor_bindings: { ...current.predictor_bindings, [endpoint.id]: modelId },
                }))
              }
            />
          ))}

        <Button size="sm" className="w-full" disabled={saving} onClick={() => void save()}>
          {saving ? 'Đang lưu…' : 'Lưu cấu hình'}
        </Button>
      </PopoverContent>
    </Popover>
  );
}

function ModelBinding({
  endpoint,
  label,
  models,
  value,
  onChange,
}: {
  endpoint: Endpoint;
  label: string;
  models: Array<{ model_id: string }>;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block text-xs font-medium">
      {label}
      <select
        aria-label={`Mô hình ${label}`}
        className="mt-1 w-full rounded border bg-transparent p-2"
        value={value || models[0]?.model_id || ''}
        onChange={(event) => onChange(event.target.value)}
      >
        {models.map((model) => (
          <option key={model.model_id} value={model.model_id}>
            {model.model_id}
          </option>
        ))}
      </select>
    </label>
  );
}
