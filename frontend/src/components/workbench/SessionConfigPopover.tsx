import { useEffect, useState } from 'react';
import { SlidersHorizontal } from 'lucide-react';
import { Button } from '../ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover';
import { getSessionSettings, listModelConnections, quickPredictCapabilities, updateSessionSettings, type SessionSettings } from '../../lib/api/endpoints';
import type { Endpoint, PredictCapabilities } from '../../lib/api/types';

export function SessionConfigPopover({ sessionId }: { sessionId: string }) {
  const [settings, setSettings] = useState<SessionSettings>({ ai_profile_id: null, predictor_bindings: {} });
  const [caps, setCaps] = useState<PredictCapabilities | null>(null);
  const [connections, setConnections] = useState<Array<{ connection_id: string; provider_id: string; model_id: string; display_name: string; status: string }>>([]);
  const [saving, setSaving] = useState(false);
  useEffect(() => { void Promise.all([getSessionSettings(sessionId), quickPredictCapabilities(), listModelConnections()]).then(([saved, catalog, profiles]) => { setSettings(saved); setCaps(catalog); setConnections(profiles.connections); }); }, [sessionId]);
  const save = async () => { setSaving(true); try { setSettings(await updateSessionSettings(sessionId, settings)); } finally { setSaving(false); } };
  const activeProfile = connections.find((item) => item.connection_id === settings.ai_profile_id);
  const triggerLabel = activeProfile ? activeProfile.display_name || `${activeProfile.provider_id} · ${activeProfile.model_id}` : 'AI & Predictors';
  return <Popover><PopoverTrigger asChild><Button variant="ghost" size="sm" className="max-w-48 gap-1.5 truncate" aria-label="Cấu hình AI và predictor"><SlidersHorizontal className="h-4 w-4 shrink-0" /><span className="truncate">{triggerLabel}</span></Button></PopoverTrigger><PopoverContent align="end" className="w-80 space-y-3"><div><p className="text-sm font-semibold">Cấu hình phiên</p><p className="text-xs text-muted-foreground">Được pin vào mỗi run khi gửi.</p></div><label className="block text-xs font-medium">Agent model<select className="mt-1 w-full rounded border bg-transparent p-2" value={settings.ai_profile_id ?? ''} onChange={(e) => setSettings((current) => ({ ...current, ai_profile_id: e.target.value || null }))}><option value="">Runtime mặc định</option>{connections.filter((item) => item.status === 'ready').map((item) => <option key={item.connection_id} value={item.connection_id}>{item.display_name || `${item.provider_id} · ${item.model_id}`}</option>)}</select></label>{(caps?.endpoints ?? []).filter((endpoint) => endpoint.enabled).map((endpoint) => <ModelBinding key={endpoint.id} endpoint={endpoint.id} label={endpoint.display_name} models={endpoint.models ?? []} value={settings.predictor_bindings[endpoint.id] ?? ''} onChange={(modelId) => setSettings((current) => ({ ...current, predictor_bindings: { ...current.predictor_bindings, [endpoint.id]: modelId } }))} />)}<Button size="sm" className="w-full" disabled={saving} onClick={() => void save()}>{saving ? 'Đang lưu…' : 'Lưu cấu hình'}</Button></PopoverContent></Popover>;
}

function ModelBinding({ endpoint, label, models, value, onChange }: { endpoint: Endpoint; label: string; models: Array<{ model_id: string }>; value: string; onChange: (value: string) => void }) {
  return <label className="block text-xs font-medium">{label}<select aria-label={`Mô hình ${label}`} className="mt-1 w-full rounded border bg-transparent p-2" value={value || models[0]?.model_id || ''} onChange={(e) => onChange(e.target.value)}>{models.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}</select></label>;
}
