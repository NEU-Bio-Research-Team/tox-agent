import { Suspense, lazy, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { ChevronRight, ImageIcon, Info, Loader2, Pencil, Type, Zap } from 'lucide-react';
import { WorkspacePage } from '../../app/shell/WorkspacePage';
import { PageHeader } from '../../shared/ui/page-header';
import { StatusPill } from '../../shared/ui/status-pill';
import { Button } from '../../shared/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../shared/ui/tabs';
import {
  createSession,
  quickPredict,
  quickPredictBatch,
  quickPredictCompare,
  quickPredictCapabilities,
  sendMessage,
} from '../../shared/api/endpoints';
import { ApiError } from '../../shared/api/types';
import type {
  Endpoint,
  PredictCapabilities,
  QuickPredictBatchResult,
  QuickPredictCompareResult,
  QuickPredictResult,
} from '../../shared/api/types';
import { queryClient } from '../../shared/lib/queryClient';
import { errorMessageVi } from '../../shared/lib/labels';
import { getEndpointSelection, getExpertModeEnabled, setDraft, setEndpointSelection } from '../../shared/lib/preferences';
import { looksLikeSmiles } from '../composer/smiles';
import { AdvancedOptions } from './AdvancedOptions';
import { EndpointPicker } from './EndpointPicker';
import { ResultsPane } from './ResultsPane';
import { SaveBar } from './SaveBar';
import { SmilesField, type Validation } from './SmilesField';
import { StructureImagePanel } from './StructureImagePanel';
import { StructurePreview } from './StructurePreview';

const StructureDrawPanel = lazy(() => import('./StructureDrawPanel').then((m) => ({ default: m.StructureDrawPanel })));

type InputMode = 'text' | 'draw' | 'image';

/** The value a preview is drawn for trails the input, so a structure is not
 * re-laid-out on every keystroke of a long SMILES. */
function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

/**
 * Phân tích nhanh in the workspace frame (plan section 6.4): the input on the
 * left, one card per measurement on the right. Every request is the control
 * plane's stateless prediction API, unchanged. Endpoints and models come from the
 * capability inventory, not from a list in the page; results are one card per
 * measurement with no combined label (ADR-0002); nothing is saved, and "Lưu vào
 * session mới" is the explicit way to keep a result.
 */
export function QuickPredictPage() {
  return (
    <WorkspacePage
      title="Phân tích nhanh"
      width="wide"
      headerActions={
        <Link to="/docs/reading-results" className="flex items-center gap-1 text-13 font-medium text-link hover:underline">
          Cách đọc kết quả
          <ChevronRight className="size-3.5" />
        </Link>
      }
    >
      <QuickPredictContent />
    </WorkspacePage>
  );
}

export function QuickPredictContent() {
  const navigate = useNavigate();
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [smiles, setSmiles] = useState('');
  const [batchMode, setBatchMode] = useState(false);
  const [batchText, setBatchText] = useState('');
  const [endpoints, setEndpoints] = useState<Endpoint[]>(() => getEndpointSelection() ?? ['herg', 'tox21']);
  const [modelSelection, setModelSelection] = useState<Partial<Record<Endpoint, string>>>({});
  const [compareMode, setCompareMode] = useState(false);
  const [compareSelection, setCompareSelection] = useState<Partial<Record<Endpoint, string[]>>>({});
  const [thresholdHerg, setThresholdHerg] = useState('');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [caps, setCaps] = useState<PredictCapabilities | null>(null);
  const [result, setResult] = useState<QuickPredictResult | null>(null);
  const [batchResult, setBatchResult] = useState<QuickPredictBatchResult | null>(null);
  const [compareResult, setCompareResult] = useState<QuickPredictCompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState<'save' | 'ask' | null>(null);
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
        // The deployment's declared default, not the first entry: the list is
        // ordered by identifier, which has nothing to do with which release
        // is the standard one.
        const declared = compatible.find((model) => (model.default_for ?? []).includes(endpoint.id));
        const selected = compatible.some((model) => model.model_id === retained)
          ? retained
          : (declared ?? compatible[0])?.model_id;
        return selected ? [[endpoint.id, selected]] : [];
      }),
    ) as Partial<Record<Endpoint, string>>);
  }, [caps]);

  useEffect(() => { setEndpointSelection(endpoints); }, [endpoints]);

  const ocrAvailable = caps?.ocr_available ?? false;
  const endpointCapabilities = caps?.endpoints ?? [];
  // A list is typed, so batch applies to the text tab only; a drawn or
  // recognised structure is always one molecule.
  const batchActive = batchMode && inputMode === 'text';
  const trimmedSmiles = smiles.trim();
  const batchLines = batchText.split('\n').map((line) => line.trim()).filter(Boolean);
  const previewSmiles = useDebounced(trimmedSmiles, 400);

  const overrides = expertMode && thresholdHerg.trim() ? { herg: Number(thresholdHerg) } : null;

  const changeSmiles = (value: string) => {
    setSmiles(value);
    if (error?.field) setError(null);
  };

  const fail = (err: unknown, field: boolean) => {
    if (err instanceof ApiError) {
      setError({ field: field && err.code === 'invalid_smiles', message: errorMessageVi(err.code, err.message) });
    } else {
      setError({ message: 'Không phân tích được. Thử lại.' });
    }
  };

  const analyse = async () => {
    if (batchActive) {
      return void analyseBatch();
    }
    if (!trimmedSmiles) {
      setError({ field: true, message: 'Nhập SMILES để phân tích.' });
      return;
    }
    setLoading(true);
    setError(null);
    setBatchResult(null);
    setCompareResult(null);
    try {
      if (compareMode) {
        const selected = Object.fromEntries(Object.entries(compareSelection)
          .filter(([endpoint, models]) => endpoints.includes(endpoint as Endpoint) && models && models.length));
        const count = Object.values(selected).reduce((total, models) => total + models.length, 0);
        if (count < 2) {
          setError({ message: 'Chọn ít nhất hai cặp phép đo/mô hình để so sánh.' });
          setAdvancedOpen(true);
          return;
        }
        setResult(null);
        setCompareResult(await quickPredictCompare({ smiles: trimmedSmiles, model_selection: selected, threshold_overrides: overrides }));
      } else {
        setResult(await quickPredict({ smiles: trimmedSmiles, endpoints, model_selection: modelSelection, threshold_overrides: overrides }));
      }
    } catch (err) {
      fail(err, true);
    } finally {
      setLoading(false);
    }
  };

  const analyseBatch = async () => {
    if (batchLines.length === 0) {
      setError({ field: true, message: 'Nhập ít nhất một SMILES (mỗi dòng một chuỗi).' });
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setCompareResult(null);
    try {
      setBatchResult(
        await quickPredictBatch({ smiles: batchLines, endpoints, model_selection: modelSelection, threshold_overrides: overrides }),
      );
    } catch (err) {
      fail(err, false);
    } finally {
      setLoading(false);
    }
  };

  // Keep the result: open a session carrying the same molecule(s) and options,
  // and go to it. The analysis then runs there, persisted, with an audit trail.
  // "Hỏi agent" also leaves a draft question in the composer without sending it.
  const saveToSession = async (askAgent: boolean) => {
    setSaving(askAgent ? 'ask' : 'save');
    setError(null);
    try {
      const session = await createSession({ preferred_language: 'vi' });
      await sendMessage(session.session_id, {
        molecule: batchResult ? { batch_smiles: batchLines } : { smiles: trimmedSmiles },
        analysis_options: { endpoints, model_selection: modelSelection, threshold_overrides: overrides },
      });
      if (askAgent) setDraft(session.session_id, 'Giải thích kết quả phân tích này.');
      void queryClient.invalidateQueries({ queryKey: ['sessions'] });
      navigate(`/s/${session.session_id}`);
    } catch (err) {
      setSaving(null);
      setError({ message: err instanceof ApiError ? errorMessageVi(err.code, err.message) : 'Không lưu được vào session. Thử lại.' });
    }
  };

  const hasInput = batchActive ? batchLines.length > 0 : Boolean(trimmedSmiles);
  const analyzeDisabled = loading || !hasInput || endpoints.length === 0;
  const hasResult = Boolean(result || batchResult || compareResult);
  const canSave = Boolean(result || batchResult) && !compareResult && !loading;

  // The page cannot validate chemistry — the predictor does. So this states
  // only what is known: the input is ready to send, or it does not read like
  // SMILES, or the predictor has already rejected it.
  const validation: Validation = batchActive || !trimmedSmiles
    ? null
    : error?.field
      ? { tone: 'invalid', message: error.message }
      : looksLikeSmiles(trimmedSmiles)
        ? { tone: 'valid', message: 'Cú pháp trông hợp lệ. Backend sẽ kiểm tra đầy đủ khi phân tích.' }
        : { tone: 'note', message: 'Chuỗi này không giống SMILES. Vẫn có thể gửi; backend sẽ kiểm tra.' };

  return (
    <>
      <PageHeader
        title="Phân tích nhanh"
        description="Một phân tử, một câu hỏi. Kết quả chỉ hiện ở trang này và không được lưu."
        actions={
          <StatusPill tone="neutral" className="gap-2 px-3 py-1.5 text-xs font-normal text-ink">
            <Info className="size-3.5" aria-hidden="true" />
            Không lưu kết quả
          </StatusPill>
        }
      />

      <div className="flex flex-wrap items-start gap-6">
        <section aria-label="Cấu trúc đầu vào" className="flex min-w-0 max-w-full flex-[0_0_440px] flex-col gap-5 rounded-xl border border-line bg-card p-5">
          <Tabs value={inputMode} onValueChange={(next) => setInputMode(next as InputMode)} className="gap-4">
            <TabsList className="grid h-10 w-full grid-cols-3 overflow-visible">
              <TabsTrigger value="text" className="gap-1.5"><Type />SMILES</TabsTrigger>
              <TabsTrigger value="draw" className="gap-1.5"><Pencil />Vẽ</TabsTrigger>
              <TabsTrigger value="image" className="gap-1.5"><ImageIcon />Ảnh</TabsTrigger>
            </TabsList>

            <TabsContent value="text">
              <SmilesField
                smiles={smiles}
                onSmiles={changeSmiles}
                batchMode={batchMode}
                onBatchMode={(batch) => { setBatchMode(batch); setError(null); }}
                batchText={batchText}
                onBatchText={(value) => { setBatchText(value); if (error?.field) setError(null); }}
                batchCount={batchLines.length}
                validation={validation}
                invalid={Boolean(error?.field)}
                loading={loading}
                onSubmit={() => { if (!analyzeDisabled) void analyse(); }}
              />
            </TabsContent>

            <TabsContent value="draw">
              <Suspense fallback={<p className="rounded-xl border border-line p-4 text-sm text-ink-secondary">Đang tải công cụ vẽ…</p>}>
                <StructureDrawPanel onSmilesExtracted={changeSmiles} disabled={loading} />
              </Suspense>
            </TabsContent>

            <TabsContent value="image">
              <StructureImagePanel available={ocrAvailable} smiles={smiles} onSmilesChange={changeSmiles} disabled={loading} />
            </TabsContent>
          </Tabs>

          {!batchActive ? <StructurePreview smiles={trimmedSmiles} rendered={previewSmiles} /> : null}

          <EndpointPicker
            capabilities={endpointCapabilities}
            selected={endpoints}
            onChange={setEndpoints}
            modelSelection={modelSelection}
            disabled={loading}
          />

          <AdvancedOptions
            capabilities={endpointCapabilities}
            selected={endpoints}
            modelSelection={modelSelection}
            onModelSelection={setModelSelection}
            compareMode={compareMode}
            onCompareMode={setCompareMode}
            compareSelection={compareSelection}
            onCompareSelection={setCompareSelection}
            batchActive={batchActive}
            expertMode={expertMode}
            thresholdHerg={thresholdHerg}
            onThresholdHerg={setThresholdHerg}
            disabled={loading}
            open={advancedOpen}
            onOpenChange={setAdvancedOpen}
          />

          <div className="space-y-2 max-md:sticky max-md:bottom-0 max-md:-mx-5 max-md:-mb-5 max-md:rounded-b-xl max-md:border-t max-md:border-line max-md:bg-card max-md:p-4">
            <Button
              onClick={() => void analyse()}
              disabled={analyzeDisabled}
              className="h-[46px] w-full gap-2 rounded-lg text-sm font-semibold"
            >
              {loading ? <Loader2 className="animate-spin" /> : <Zap />}
              {loading ? 'Đang phân tích…' : hasResult ? 'Phân tích lại' : 'Phân tích'}
            </Button>
            <p className="text-center text-xs text-ink-tertiary">
              {hasInput ? 'Ctrl + Enter để phân tích' : 'Nhập SMILES hoặc vẽ/tải cấu trúc để bắt đầu'}
            </p>
          </div>
        </section>

        <div className="min-w-0 flex-[1_1_420px] space-y-4">
          <ResultsPane
            capabilities={endpointCapabilities}
            loading={loading}
            error={error && !(error.field && validation) ? error : null}
            result={result}
            batchResult={batchResult}
            compareResult={compareResult}
          />
          {canSave ? <SaveBar saving={saving} onSave={(ask) => void saveToSession(ask)} /> : null}
        </div>
      </div>
    </>
  );
}
