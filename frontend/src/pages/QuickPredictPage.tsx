import { Suspense, lazy, useEffect, useState } from 'react';
import { ImageUp, Pencil, Sparkles } from 'lucide-react';
import { Navbar } from '../components/shell/Navbar';
import { Footer } from '../components/shell/Footer';
import { AnalysisPanel } from '../components/workbench/AnalysisPanel';
import { ImageUploadDialog, type StagedImage } from '../components/workbench/ImageUploadDialog';
import { looksLikeSmiles } from '../components/workbench/MessageComposer';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import {
  quickPredict,
  quickPredictBatch,
  quickPredictCapabilities,
  recognizeStructure,
} from '../lib/api/endpoints';
import { ApiError } from '../lib/api/types';
import type {
  Endpoint,
  PredictCapabilities,
  QuickPredictBatchResult,
  QuickPredictResult,
  RecognizedStructure,
} from '../lib/api/types';
import { errorMessageVi } from '../lib/labels';
import { getEndpointSelection, getExpertModeEnabled, setEndpointSelection } from '../lib/preferences';

const StructureEditorDialog = lazy(() =>
  import('../components/workbench/StructureEditorDialog').then((m) => ({ default: m.StructureEditorDialog })),
);

export function QuickPredictPage() {
  const [smiles, setSmiles] = useState('');
  const [batchMode, setBatchMode] = useState(false);
  const [batchText, setBatchText] = useState('');
  const [endpoints, setEndpoints] = useState<Endpoint[]>(() => getEndpointSelection() ?? ['herg', 'tox21']);
  const [modelSelection, setModelSelection] = useState<Partial<Record<Endpoint, string>>>({});
  const [thresholdHerg, setThresholdHerg] = useState('');
  const [drawOpen, setDrawOpen] = useState(false);
  const [imageOpen, setImageOpen] = useState(false);
  const [recognized, setRecognized] = useState<(RecognizedStructure & { previewUrl: string }) | null>(null);
  const [caps, setCaps] = useState<PredictCapabilities | null>(null);
  const [result, setResult] = useState<QuickPredictResult | null>(null);
  const [batchResult, setBatchResult] = useState<QuickPredictBatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ field?: boolean; message: string } | null>(null);

  const expertMode = getExpertModeEnabled();

  useEffect(() => {
    let cancelled = false;
    quickPredictCapabilities()
      .then((c) => !cancelled && setCaps(c))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!caps) return;
    const enabled = new Set((caps.endpoints ?? []).filter((endpoint) => endpoint.enabled).map((endpoint) => endpoint.id));
    setEndpoints((current) => {
      const compatible = current.filter((endpoint) => enabled.has(endpoint));
      const fallback = (caps.default_endpoints ?? caps.served_endpoints).filter((endpoint) => enabled.has(endpoint));
      return compatible.length ? compatible : fallback;
    });
  }, [caps]);

  useEffect(() => {
    if (!caps) return;
    setModelSelection((current) => Object.fromEntries(
      (caps.endpoints ?? []).flatMap((endpoint) => {
        const compatible = endpoint.models ?? [];
        const retained = current[endpoint.id];
        const selected = compatible.some((model) => model.model_id === retained)
          ? retained
          : compatible[0]?.model_id;
        return selected ? [[endpoint.id, selected]] : [];
      }),
    ) as Partial<Record<Endpoint, string>>);
  }, [caps]);

  useEffect(() => { setEndpointSelection(endpoints); }, [endpoints]);

  useEffect(() => {
    return () => {
      if (recognized) URL.revokeObjectURL(recognized.previewUrl);
    };
  }, [recognized]);

  const ocrAvailable = caps?.ocr_available ?? false;
  const endpointCapabilities = caps?.endpoints ?? [];

  const overrides =
    expertMode && thresholdHerg.trim() ? { herg: Number(thresholdHerg) } : null;

  const analyse = async () => {
    if (batchMode) {
      return void analyseBatch();
    }
    const trimmed = smiles.trim();
    if (!trimmed) {
      setError({ field: true, message: 'Nhập SMILES để phân tích.' });
      return;
    }
    setLoading(true);
    setError(null);
    setBatchResult(null);
    try {
      setResult(
        await quickPredict({ smiles: trimmed, endpoints, model_selection: modelSelection, threshold_overrides: overrides }),
      );
    } catch (err) {
      if (err instanceof ApiError) {
        setError({
          field: err.code === 'invalid_smiles',
          message: errorMessageVi(err.code, err.message),
        });
      } else {
        setError({ message: 'Không phân tích được. Thử lại.' });
      }
    } finally {
      setLoading(false);
    }
  };

  const analyseBatch = async () => {
    const lines = batchText
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      setError({ field: true, message: 'Nhập ít nhất một SMILES (mỗi dòng một chuỗi).' });
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setBatchResult(
        await quickPredictBatch({ smiles: lines, endpoints, model_selection: modelSelection, threshold_overrides: overrides }),
      );
    } catch (err) {
      if (err instanceof ApiError) {
        setError({ message: errorMessageVi(err.code, err.message) });
      } else {
        setError({ message: 'Không phân tích được. Thử lại.' });
      }
    } finally {
      setLoading(false);
    }
  };

  const onImageConfirm = async (image: StagedImage) => {
    setError(null);
    try {
      const rec = await recognizeStructure({
        mime_type: image.mimeType,
        data_base64: image.dataBase64,
      });
      setRecognized({ ...rec, previewUrl: image.previewUrl });
      setSmiles(rec.canonical_smiles);
    } catch (err) {
      URL.revokeObjectURL(image.previewUrl);
      const message =
        err instanceof ApiError ? errorMessageVi(err.code, err.message) : 'Không nhận diện được ảnh.';
      setError({ message });
    }
  };

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--canvas-subtle)' }}>
      <Navbar />
      <main className="mx-auto max-w-[1240px] px-4 py-8 md:px-6 md:py-14">
        <header className="mx-auto mb-8 max-w-3xl text-center">
          <p className="mb-3 inline-flex items-center gap-1.5 rounded-full bg-[var(--purple-100)] px-3 py-1 text-xs font-medium text-[var(--purple-700)]"><Sparkles className="h-3.5 w-3.5" /> Phân tích nhanh, không lưu phiên</p>
          <h1 className="text-3xl font-semibold tracking-tight md:text-[40px]" style={{ color: 'var(--ink)' }}>
            Dự đoán độc tính trong vài giây
          </h1>
          <p className="mt-2 text-sm md:text-base" style={{ color: 'var(--ink-secondary)' }}>
            Nhập SMILES, tải ảnh hoặc vẽ cấu trúc. Kết quả chỉ là screening và không được lưu vào lịch sử.
          </p>
        </header>

        <div className="mx-auto max-w-[920px]">
          <div
            className="ta-glass space-y-5 rounded-[var(--radius-floating)] border p-4 shadow-[var(--shadow-float)] md:p-6"
            style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--line-strong)' }}
          >
            <div className="inline-flex rounded-xl bg-[var(--surface-muted)] p-1" role="group" aria-label="Chế độ dự đoán">
              <Button type="button" aria-pressed={!batchMode} variant={batchMode ? 'ghost' : 'secondary'} size="sm" onClick={() => { setBatchMode(false); setError(null); }}>Một phân tử</Button>
              <Button type="button" aria-pressed={batchMode} variant={batchMode ? 'secondary' : 'ghost'} size="sm" onClick={() => { setBatchMode(true); setError(null); }}>Hàng loạt</Button>
            </div>

            {batchMode ? (
              <div>
                <Label htmlFor="qp-batch" className="text-xs">
                  Danh sách SMILES <span className="font-normal text-muted-foreground">· {batchText.split('\n').filter((line) => line.trim()).length} phân tử</span>
                </Label>
                <Textarea
                  id="qp-batch"
                  aria-label="Danh sách SMILES"
                  rows={6}
                  className="mt-1 font-mono text-sm"
                  placeholder={'CCO\nCC(=O)Oc1ccccc1C(=O)O'}
                  value={batchText}
                  onChange={(e) => {
                    setBatchText(e.target.value);
                    if (error?.field) setError(null);
                  }}
                />
              </div>
            ) : (
              <div>
                <Label htmlFor="qp-smiles" className="text-xs">
                  SMILES
                </Label>
                <Input
                  id="qp-smiles"
                  className="mt-1 font-mono text-sm"
                  placeholder="vd. CC(=O)Oc1ccccc1C(=O)O"
                  value={smiles}
                  onChange={(e) => {
                    setSmiles(e.target.value);
                    if (error?.field) setError(null);
                  }}
                  aria-invalid={error?.field ? true : undefined}
                />
                {smiles.trim() && !looksLikeSmiles(smiles.trim()) && (
                  <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
                    Chuỗi này trông không giống SMILES — vẫn gửi được, predictor sẽ xác thực.
                  </p>
                )}
              </div>
            )}

            {!batchMode && (
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" size="sm" className="gap-1.5" onClick={() => setDrawOpen(true)}>
                  <Pencil className="h-3.5 w-3.5" />
                  Vẽ cấu trúc
                </Button>
                {ocrAvailable && (
                  <Button variant="outline" size="sm" className="gap-1.5" onClick={() => setImageOpen(true)}>
                    <ImageUp className="h-3.5 w-3.5" />
                    Tải ảnh
                  </Button>
                )}
              </div>
            )}

            {!batchMode && recognized && (
              <div className="flex gap-3 rounded-lg p-3" style={{ backgroundColor: 'var(--surface-alt)' }}>
                <img
                  src={recognized.previewUrl}
                  alt="Ảnh cấu trúc đã tải"
                  className="h-16 w-16 shrink-0 rounded object-contain"
                  style={{ backgroundColor: '#fff' }}
                />
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Nhận diện được{' '}
                    {recognized.confidence != null
                      ? `(độ tin cậy ${(recognized.confidence * 100).toFixed(0)}%)`
                      : '(không có độ tin cậy)'}
                    . Kiểm tra và sửa SMILES nếu cần.
                  </p>
                  <Input
                    className="font-mono text-xs"
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    aria-label="SMILES nhận diện được (có thể sửa)"
                  />
                </div>
              </div>
            )}

            <fieldset>
              <legend className="mb-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                Endpoints &amp; mô hình
              </legend>
              <div className="flex flex-wrap gap-2">
                {endpointCapabilities.map((endpoint) => {
                  const selected = endpoints.includes(endpoint.id);
                  return (
                    <button key={endpoint.id} type="button" disabled={!endpoint.enabled} title={endpoint.blocked_reason ?? undefined}
                      onClick={() => setEndpoints((current) => selected ? (current.length > 1 ? current.filter((item) => item !== endpoint.id) : current) : [...current, endpoint.id])}
                      className="rounded-xl border px-3 py-2 text-left text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                      style={{ borderColor: selected ? 'var(--purple-500)' : 'var(--line)', backgroundColor: selected ? 'var(--purple-50)' : 'var(--surface-solid)', color: 'var(--ink)' }}>
                      <span className="font-medium">{endpoint.display_name}</span>{!endpoint.enabled && <span className="ml-1 text-[10px] text-[var(--ink-tertiary)]">không khả dụng</span>}
                    </button>
                  );
                })}
              </div>
              <div className="mt-3 space-y-2">
                {endpointCapabilities.filter((endpoint) => endpoints.includes(endpoint.id)).map((endpoint) => {
                  const models = endpoint.models ?? [];
                  if (!endpoint.enabled) return null;
                  return (
                    <label key={`${endpoint.id}-model`} className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-xs" style={{ borderColor: 'var(--line)', backgroundColor: 'var(--surface-solid)' }}>
                      <span className="font-medium" style={{ color: 'var(--ink)' }}>{endpoint.display_name}</span>
                      <select
                        aria-label={`Mô hình ${endpoint.display_name}`}
                        value={modelSelection[endpoint.id] ?? ''}
                        onChange={(event) => setModelSelection((current) => ({ ...current, [endpoint.id]: event.target.value }))}
                        className="max-w-[220px] bg-transparent text-xs outline-none"
                      >
                        {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}
                      </select>
                    </label>
                  );
                })}
              </div>
              {endpoints.length > 1 && new Set(endpoints.map((endpoint) => modelSelection[endpoint])).size === 1 && (
                <p className="mt-2 text-xs" style={{ color: 'var(--purple-700)' }}>⚡ Các endpoint đã chọn dùng chung một model; predictor sẽ deduplicate inference.</p>
              )}
            </fieldset>

            {expertMode && (
              <div>
                <Label htmlFor="qp-threshold" className="text-xs">
                  hERG threshold override (expert — backend từ chối nếu token không có role expert)
                </Label>
                <Input
                  id="qp-threshold"
                  className="mt-1 h-8 text-xs"
                  placeholder="vd. 0.3"
                  value={thresholdHerg}
                  onChange={(e) => setThresholdHerg(e.target.value)}
                />
              </div>
            )}

            {error && !error.field && (
              <p className="text-xs" style={{ color: 'var(--accent-red)' }}>
                {error.message}
              </p>
            )}
            {error?.field && (
              <p className="text-xs" style={{ color: 'var(--accent-red)' }}>
                {error.message}
              </p>
            )}

            <Button
              onClick={() => void analyse()}
              disabled={loading || endpoints.length === 0}
              variant="primary-gloss"
              className="w-full gap-1.5"
            >
              {loading ? 'Đang phân tích…' : 'Phân tích'}
            </Button>
            <p className="text-center text-xs" style={{ color: 'var(--text-faint)' }}>
              Kết quả này không được lưu. Cần audit trail? <a className="text-[var(--purple-700)] underline" href="/sessions">Mở Session</a>.
            </p>
          </div>

          <section aria-live="polite" className="mt-8 space-y-4">
            {batchResult ? (
              <>
                {batchResult.errors.length > 0 && (
                  <div
                    className="rounded-xl border p-3 text-xs"
                    style={{ borderColor: 'var(--border)', color: 'var(--accent-red)' }}
                  >
                    <p className="font-medium">{batchResult.errors.length} phân tử lỗi</p>
                    <ul className="mt-1 space-y-0.5 font-mono">
                      {batchResult.errors.map((e) => (
                        <li key={e.index}>
                          #{e.index} {e.input_smiles || '(rỗng)'} — {errorMessageVi(e.error, e.error)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {batchResult.results.map((r, index) => (
                  <AnalysisPanel key={`${r.canonical_smiles}:${index}`} analysis={r} />
                ))}
              </>
            ) : (
              <AnalysisPanel analysis={result} />
            )}
          </section>
        </div>
      </main>
      <Footer />

      {drawOpen && (
        <Suspense fallback={null}>
          <StructureEditorDialog
            open={drawOpen}
            onOpenChange={setDrawOpen}
            onConfirm={(s) => {
              setSmiles(s);
              setDrawOpen(false);
            }}
          />
        </Suspense>
      )}

      <ImageUploadDialog
        open={imageOpen}
        onOpenChange={setImageOpen}
        available={ocrAvailable}
        onConfirm={(image) => void onImageConfirm(image)}
      />
    </div>
  );
}
