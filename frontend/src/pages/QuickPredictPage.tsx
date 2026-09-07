import { Suspense, lazy, useEffect, useState } from 'react';
import { FlaskConical, ImageUp, Pencil } from 'lucide-react';
import { Navbar } from '../components/shell/Navbar';
import { Footer } from '../components/shell/Footer';
import { AnalysisPanel } from '../components/workbench/AnalysisPanel';
import { ImageUploadDialog, type StagedImage } from '../components/workbench/ImageUploadDialog';
import { looksLikeSmiles } from '../components/workbench/MessageComposer';
import { Button } from '../components/ui/button';
import { Checkbox } from '../components/ui/checkbox';
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
import { getExpertModeEnabled } from '../lib/preferences';

const StructureEditorDialog = lazy(() =>
  import('../components/workbench/StructureEditorDialog').then((m) => ({ default: m.StructureEditorDialog })),
);

const SELECTABLE: Endpoint[] = ['herg', 'tox21', 'clintox'];

export function QuickPredictPage() {
  const [smiles, setSmiles] = useState('');
  const [batchMode, setBatchMode] = useState(false);
  const [batchText, setBatchText] = useState('');
  const [endpoints, setEndpoints] = useState<Endpoint[]>(['herg', 'tox21']);
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
    return () => {
      if (recognized) URL.revokeObjectURL(recognized.previewUrl);
    };
  }, [recognized]);

  const clintoxServed = caps?.served_endpoints.includes('clintox') ?? false;
  const ocrAvailable = caps?.ocr_available ?? false;

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
        await quickPredict({ smiles: trimmed, endpoints, threshold_overrides: overrides }),
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
        await quickPredictBatch({ smiles: lines, endpoints, threshold_overrides: overrides }),
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
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)' }}>
      <Navbar />
      <main className="mx-auto max-w-6xl px-4 py-8 md:px-6">
        <header className="mb-6">
          <h1 className="flex items-center gap-2 text-xl font-bold" style={{ color: 'var(--text)' }}>
            <FlaskConical className="h-5 w-5" style={{ color: 'var(--accent-blue)' }} />
            Phân tích nhanh
          </h1>
          <p className="mt-1 text-sm" style={{ color: 'var(--text-muted)' }}>
            SMILES vào, số ra. Không tạo session, không lưu vào lịch sử — cần bản ghi có
            audit trail thì dùng workbench.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-2">
          <div
            className="space-y-4 rounded-xl border p-4"
            style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
          >
            <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <Checkbox
                checked={batchMode}
                onCheckedChange={(checked) => {
                  setBatchMode(Boolean(checked));
                  setError(null);
                }}
              />
              Nhiều phân tử (mỗi dòng một SMILES)
            </label>

            {batchMode ? (
              <div>
                <Label htmlFor="qp-batch" className="text-xs">
                  Danh sách SMILES
                </Label>
                <Textarea
                  id="qp-batch"
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
              <legend className="mb-1 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                Endpoint
              </legend>
              <div className="flex flex-col gap-1">
                {SELECTABLE.map((endpoint) => {
                  const disabled = endpoint === 'clintox' && !clintoxServed;
                  return (
                    <label
                      key={endpoint}
                      className="flex items-center gap-2 text-xs"
                      style={{ color: disabled ? 'var(--text-faint)' : 'var(--text)' }}
                      title={
                        disabled ? 'Bản predictor này không phục vụ ClinTox (thiếu artifact).' : undefined
                      }
                    >
                      <Checkbox
                        checked={endpoints.includes(endpoint)}
                        disabled={disabled}
                        onCheckedChange={(checked) =>
                          setEndpoints((prev) =>
                            checked ? [...prev, endpoint] : prev.filter((e) => e !== endpoint),
                          )
                        }
                      />
                      {endpoint}
                      {disabled && ' — không khả dụng'}
                    </label>
                  );
                })}
              </div>
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
              className="w-full gap-1.5"
            >
              {loading ? 'Đang phân tích…' : 'Phân tích'}
            </Button>
            <p className="text-center text-xs" style={{ color: 'var(--text-faint)' }}>
              Phân tích nhanh (không lưu vào session)
            </p>
          </div>

          <div className="space-y-4">
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
          </div>
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
