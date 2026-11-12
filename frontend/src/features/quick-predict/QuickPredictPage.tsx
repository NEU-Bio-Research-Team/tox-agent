import { Suspense, lazy, useEffect, useState } from 'react';
import { Link } from 'react-router';
import { CheckCircle, ChevronDown, ChevronRight, History, Info, Loader2, RefreshCw, XCircle, Zap } from 'lucide-react';
import { Navbar } from '../../app/shell/Navbar';
import { Footer } from '../../app/shell/Footer';
import { AnalysisPanel } from '../analysis/AnalysisPanel';
import { MoleculeDepiction } from '../analysis/MoleculeDepiction';
import { looksLikeSmiles } from '../composer/MessageComposer';
import { StructureImagePanel } from './StructureImagePanel';
import { Button } from '../../shared/ui/button';
import { Input } from '../../shared/ui/input';
import { Label } from '../../shared/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../shared/ui/tabs';
import { Textarea } from '../../shared/ui/textarea';
import {
  quickPredict,
  quickPredictBatch,
  quickPredictCompare,
  quickPredictCapabilities,
} from '../../shared/api/endpoints';
import { ApiError } from '../../shared/api/types';
import type {
  Endpoint,
  PredictCapabilities,
  QuickPredictBatchResult,
  QuickPredictCompareResult,
  QuickPredictResult,
} from '../../shared/api/types';
import { errorMessageVi } from '../../shared/lib/labels';
import { getEndpointSelection, getExpertModeEnabled, setEndpointSelection } from '../../shared/lib/preferences';

const StructureDrawPanel = lazy(() =>
  import('./StructureDrawPanel').then((m) => ({ default: m.StructureDrawPanel })),
);

type InputMode = 'text' | 'draw' | 'image';

const EXAMPLE_MOLECULES = [
  { name: 'Caffeine', icon: 'C', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { name: 'Aspirin', icon: 'A', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'Ethanol', icon: 'E', smiles: 'CCO' },
  { name: 'Haloperidol', icon: 'H', smiles: 'O=C(CCCN1CCC(c2ccc(Cl)cc2)(O)CC1)c1ccc(F)cc1' },
];

const BOX_STYLE = { borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' } as const;
const CAPTION_STYLE = { color: 'var(--text-faint)', letterSpacing: '0.05em' } as const;

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
 * The analysis page: the production site's layout over this control plane's
 * stateless prediction API.
 *
 * What is the production design: the title, the input card with its three
 * tabs, the examples, the current-SMILES and preview boxes, the Analyze
 * button and the advanced options. What is this codebase's and unchanged:
 * every request and what it means. Endpoints and models come from the
 * capability inventory rather than a list in the page; results are one panel
 * per endpoint with no combined label (ADR-0002); nothing is saved, so where
 * the production page listed past analyses this one points at Sessions.
 */
export function QuickPredictPage() {
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

  const ocrAvailable = caps?.ocr_available ?? false;
  const endpointCapabilities = caps?.endpoints ?? [];
  // A list is typed, so batch applies to the text tab only; a drawn or
  // recognised structure is always one molecule.
  const batchActive = batchMode && inputMode === 'text';
  const trimmedSmiles = smiles.trim();
  const batchLines = batchText.split('\n').map((line) => line.trim()).filter(Boolean);
  const previewSmiles = useDebounced(trimmedSmiles, 400);

  const overrides =
    expertMode && thresholdHerg.trim() ? { herg: Number(thresholdHerg) } : null;

  const changeSmiles = (value: string) => {
    setSmiles(value);
    if (error?.field) setError(null);
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
          setError({ message: 'Chọn ít nhất hai cặp endpoint/mô hình để so sánh.' });
          setAdvancedOpen(true);
          return;
        }
        setResult(null);
        setCompareResult(await quickPredictCompare({ smiles: trimmedSmiles, model_selection: selected, threshold_overrides: overrides }));
      } else {
        setResult(await quickPredict({ smiles: trimmedSmiles, endpoints, model_selection: modelSelection, threshold_overrides: overrides }));
      }
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
      if (err instanceof ApiError) {
        setError({ message: errorMessageVi(err.code, err.message) });
      } else {
        setError({ message: 'Không phân tích được. Thử lại.' });
      }
    } finally {
      setLoading(false);
    }
  };

  const hasInput = batchActive ? batchLines.length > 0 : Boolean(trimmedSmiles);
  const analyzeDisabled = loading || !hasInput || endpoints.length === 0;
  const hasResult = Boolean(result || batchResult || compareResult);

  // The page cannot validate chemistry — the predictor does. So this states
  // only what is known: the input is ready to send, or it does not read like
  // SMILES, or the predictor has already rejected it.
  const validation: { tone: 'valid' | 'invalid' | 'note'; message: string } | null = batchActive || !trimmedSmiles
    ? null
    : error?.field
      ? { tone: 'invalid', message: error.message }
      : looksLikeSmiles(trimmedSmiles)
        ? { tone: 'valid', message: 'Ready to call API. Detailed validation will be handled by the backend.' }
        : { tone: 'note', message: 'This does not read like a SMILES string. It can still be sent; the backend validates it.' };

  const validationColor =
    validation?.tone === 'valid' ? 'var(--accent-green)' : validation?.tone === 'invalid' ? 'var(--accent-red)' : 'var(--accent-yellow)';
  const validationBackground =
    validation?.tone === 'valid' ? 'rgba(34,197,94,0.08)' : validation?.tone === 'invalid' ? 'rgba(239,68,68,0.08)' : 'rgba(245,158,11,0.08)';
  const inputBorder =
    validation?.tone === 'valid'
      ? '1.5px solid var(--accent-green)'
      : validation?.tone === 'invalid'
        ? '1.5px solid var(--accent-red)'
        : '1.5px solid var(--border)';

  const tabFallback = (
    <div className="rounded-xl border p-4 text-sm" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
      Loading input tools...
    </div>
  );

  return (
    <div className="legacy-ui" style={{ minHeight: '100vh', backgroundColor: 'var(--bg)' }}>
      <Navbar />

      <main className="mx-auto max-w-5xl px-4 sm:px-6">
        <section className="pb-12 pt-12 md:pt-20">
          <div className="mb-10 text-center">
            <h1
              className="mb-3 bg-gradient-to-r from-[var(--accent-blue)] to-[var(--accent-green)] bg-clip-text text-4xl font-bold md:text-5xl"
              style={{ WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}
            >
              Drug Toxicity Analysis
            </h1>
            <p className="mb-1 text-base" style={{ color: 'var(--text-muted)' }}>
              Per-endpoint toxicity screening: hERG, Tox21 and ClinTox
            </p>
            <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
              Type SMILES, draw a molecule, or upload a structure image to start analysis
            </p>
          </div>

          <div
            className="rounded-2xl p-5 shadow-lg md:p-8"
            style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
          >
            <Tabs value={inputMode} onValueChange={(next) => setInputMode(next as InputMode)} className="mb-4">
              <TabsList className="grid h-10 w-full grid-cols-3" style={{ backgroundColor: 'var(--surface-alt)' }}>
                <TabsTrigger value="text">Type SMILES</TabsTrigger>
                <TabsTrigger value="draw">Draw Molecule</TabsTrigger>
                <TabsTrigger value="image">Upload Image</TabsTrigger>
              </TabsList>

              <TabsContent value="text" className="space-y-4 pt-3">
                <div className="flex justify-end">
                  <div
                    className="inline-flex rounded-full border p-0.5 text-xs"
                    role="group"
                    aria-label="Prediction mode"
                    style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' }}
                  >
                    {([['Single', false], ['Batch', true]] as const).map(([label, value]) => (
                      <button
                        key={label}
                        type="button"
                        aria-pressed={batchMode === value}
                        disabled={loading}
                        onClick={() => { setBatchMode(value); setError(null); }}
                        className="rounded-full px-3 py-1 font-medium transition-colors"
                        style={{
                          backgroundColor: batchMode === value ? 'var(--surface)' : 'transparent',
                          color: batchMode === value ? 'var(--text)' : 'var(--text-muted)',
                          boxShadow: batchMode === value ? '0 1px 2px rgba(15,23,42,0.12)' : 'none',
                        }}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                {batchMode ? (
                  <div>
                    <Label htmlFor="qp-batch" className="mb-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                      SMILES list <span className="font-normal" style={{ color: 'var(--text-faint)' }}>· one per line · {batchLines.length} molecule{batchLines.length === 1 ? '' : 's'}</span>
                    </Label>
                    <Textarea
                      id="qp-batch"
                      aria-label="SMILES list"
                      rows={6}
                      className="min-h-[140px] px-4 py-3 font-mono text-sm"
                      style={{ backgroundColor: 'var(--surface-alt)', border: '1.5px solid var(--border)', color: 'var(--text)', letterSpacing: '0.02em' }}
                      placeholder={'CCO\nCC(=O)Oc1ccccc1C(=O)O'}
                      value={batchText}
                      disabled={loading}
                      spellCheck={false}
                      autoCorrect="off"
                      autoComplete="off"
                      onChange={(e) => {
                        setBatchText(e.target.value);
                        if (error?.field) setError(null);
                      }}
                    />
                  </div>
                ) : (
                  <>
                    <div className="relative">
                      <div className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-faint)' }}>
                        <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                          <circle cx="12" cy="12" r="3" />
                          <path d="M12 1v6m0 6v6m0-6h6m-6 0H6" />
                        </svg>
                      </div>
                      <Textarea
                        aria-label="SMILES"
                        value={smiles}
                        onChange={(event) => changeSmiles(event.target.value)}
                        placeholder="CC(=O)Oc1ccccc1C(=O)O"
                        className="min-h-[80px] resize-none py-4 pl-12 pr-10 font-mono text-sm"
                        style={{ backgroundColor: 'var(--surface-alt)', border: inputBorder, color: 'var(--text)', letterSpacing: '0.02em' }}
                        aria-invalid={error?.field ? true : undefined}
                        disabled={loading}
                        spellCheck={false}
                        autoCorrect="off"
                        autoComplete="off"
                      />
                      {smiles && !loading && (
                        <button
                          type="button"
                          onClick={() => changeSmiles('')}
                          aria-label="Clear SMILES"
                          className="absolute right-3 top-1/2 -translate-y-1/2 rounded p-1 transition-colors hover:bg-[var(--surface)]"
                          style={{ color: 'var(--text-faint)' }}
                        >
                          x
                        </button>
                      )}
                    </div>

                    <div>
                      <p className="mb-2 text-xs uppercase" style={CAPTION_STYLE}>
                        Quick Examples:
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {EXAMPLE_MOLECULES.map((mol) => (
                          <button
                            type="button"
                            key={mol.name}
                            onClick={() => changeSmiles(mol.smiles)}
                            disabled={loading}
                            className="rounded-full border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-2 text-xs font-medium text-[var(--text-muted)] transition-all hover:border-[var(--accent-blue)] hover:bg-[var(--surface)] hover:text-[var(--text)]"
                          >
                            {mol.icon} {mol.name}
                          </button>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </TabsContent>

              <TabsContent value="draw" className="pt-3">
                <Suspense fallback={tabFallback}>
                  <StructureDrawPanel onSmilesExtracted={changeSmiles} disabled={loading} />
                </Suspense>
              </TabsContent>

              <TabsContent value="image" className="pt-3">
                <StructureImagePanel available={ocrAvailable} smiles={smiles} onSmilesChange={changeSmiles} disabled={loading} />
              </TabsContent>
            </Tabs>

            {validation && (
              <div
                className="mb-4 rounded-lg px-4 py-2 text-sm"
                style={{ backgroundColor: validationBackground, borderLeft: `2px solid ${validationColor}`, color: validationColor }}
              >
                <div className="flex items-center gap-2">
                  {validation.tone === 'valid' && <CheckCircle className="h-4 w-4 shrink-0" />}
                  {validation.tone === 'invalid' && <XCircle className="h-4 w-4 shrink-0" />}
                  {validation.tone === 'note' && <Info className="h-4 w-4 shrink-0" />}
                  <span>{validation.message}</span>
                </div>
              </div>
            )}

            {!batchActive && (
              <>
                <div className="mb-4 rounded-lg border px-4 py-3" style={BOX_STYLE}>
                  <p className="mb-1 text-xs uppercase" style={CAPTION_STYLE}>
                    Current SMILES for Analysis
                  </p>
                  <p className="break-all font-mono text-sm" style={{ color: 'var(--text)' }}>
                    {trimmedSmiles || 'No SMILES selected yet.'}
                  </p>
                </div>

                <div className="mb-4 rounded-lg border px-4 py-3" style={BOX_STYLE}>
                  <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div className="min-w-0 flex-1">
                      <p className="mb-1 text-xs uppercase" style={CAPTION_STYLE}>
                        Molecule Preview
                      </p>
                      {!trimmedSmiles ? (
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                          Enter or extract a SMILES string to render a structure preview.
                        </p>
                      ) : previewSmiles !== trimmedSmiles ? (
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                          Rendering preview...
                        </p>
                      ) : (
                        <div
                          className="mt-3 flex justify-center overflow-hidden rounded-xl border"
                          style={{ borderColor: 'var(--border)', backgroundColor: '#ffffff' }}
                        >
                          <MoleculeDepiction smiles={previewSmiles} size={260} />
                        </div>
                      )}
                    </div>
                    <div className="flex justify-end">
                      <Button type="button" variant="outline" onClick={() => changeSmiles('')} disabled={loading || !trimmedSmiles}>
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Retry
                      </Button>
                    </div>
                  </div>
                </div>
              </>
            )}

            <div className="mb-4 flex items-center justify-end">
              <Button
                onClick={() => void analyse()}
                disabled={analyzeDisabled}
                className="h-[48px] rounded-lg px-7 text-base font-semibold"
                style={{
                  backgroundColor: analyzeDisabled ? 'var(--border)' : 'var(--accent-blue)',
                  color: '#ffffff',
                  opacity: analyzeDisabled ? 0.5 : 1,
                }}
              >
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                {loading ? 'Analyzing...' : 'Analyze'}
              </Button>
            </div>

            <div>
              <button
                type="button"
                onClick={() => setAdvancedOpen((open) => !open)}
                aria-expanded={advancedOpen}
                aria-controls="qp-advanced"
                className="flex items-center gap-1 text-sm"
                style={{ color: 'var(--text-muted)' }}
              >
                {advancedOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                Advanced options
              </button>

              {advancedOpen && (
                <div id="qp-advanced" className="mt-4 space-y-4">
                  <fieldset>
                    <legend className="mb-1 block text-sm" style={{ color: 'var(--text-muted)' }}>
                      Endpoints &amp; models
                    </legend>
                    <p className="mb-2 text-xs" style={{ color: 'var(--text-faint)' }}>
                      Each endpoint is a separate measurement with its own model. Only what this deployment serves can be selected.
                    </p>
                    {endpointCapabilities.length === 0 && (
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        The endpoint inventory has not loaded.
                      </p>
                    )}
                    <div className="flex flex-wrap gap-2">
                      {endpointCapabilities.map((endpoint) => {
                        const selected = endpoints.includes(endpoint.id);
                        return (
                          <button key={endpoint.id} type="button" disabled={!endpoint.enabled || loading} aria-pressed={selected} title={endpoint.blocked_reason ?? undefined}
                            onClick={() => setEndpoints((current) => selected ? (current.length > 1 ? current.filter((item) => item !== endpoint.id) : current) : [...current, endpoint.id])}
                            className="rounded-lg border px-3 py-2 text-left text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                            style={{ borderColor: selected ? 'var(--accent-blue)' : 'var(--border)', backgroundColor: selected ? 'var(--accent-blue-muted)' : 'var(--surface-alt)', color: 'var(--text)' }}>
                            <span className="font-medium">{endpoint.display_name}</span>{!endpoint.enabled && <span className="ml-1 text-[10px]" style={{ color: 'var(--text-faint)' }}>unavailable</span>}
                          </button>
                        );
                      })}
                    </div>
                    <div className="mt-3 space-y-2">
                      {endpointCapabilities.filter((endpoint) => endpoints.includes(endpoint.id)).map((endpoint) => {
                        const models = endpoint.models ?? [];
                        if (!endpoint.enabled) return null;
                        return (
                          <div key={`${endpoint.id}-model`} className="rounded-lg border px-3 py-2 text-xs" style={BOX_STYLE}>
                            <div className="flex items-center justify-between gap-3">
                              <span className="font-medium" style={{ color: 'var(--text)' }}>{endpoint.display_name}</span>
                              {!compareMode && <select aria-label={`Model for ${endpoint.display_name}`} value={modelSelection[endpoint.id] ?? ''} disabled={loading} onChange={(event) => setModelSelection((current) => ({ ...current, [endpoint.id]: event.target.value }))} className="max-w-[240px] rounded-md px-2 py-1 text-xs" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}>
                                {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}
                              </select>}
                            </div>
                            {compareMode && <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1" style={{ color: 'var(--text)' }}>
                              {models.map((model) => {
                                const checked = (compareSelection[endpoint.id] ?? []).includes(model.model_id);
                                return <label key={model.model_id} className="flex items-center gap-1.5"><input type="checkbox" checked={checked} disabled={loading} onChange={() => setCompareSelection((current) => {
                                  const existing = current[endpoint.id] ?? [];
                                  const next = checked ? existing.filter((id) => id !== model.model_id) : [...existing, model.model_id];
                                  return { ...current, [endpoint.id]: next };
                                })} />{model.model_id}</label>;
                              })}
                            </div>}
                          </div>
                        );
                      })}
                    </div>
                    {endpoints.length > 1 && new Set(endpoints.map((endpoint) => modelSelection[endpoint])).size === 1 && (
                      <p className="mt-2 text-xs" style={{ color: 'var(--accent-blue)' }}>The selected endpoints share one model; the predictor runs inference once.</p>
                    )}
                  </fieldset>

                  {!batchActive && (
                    <label className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <input type="checkbox" checked={compareMode} disabled={loading} onChange={(event) => setCompareMode(event.target.checked)} />
                      Compare models (each model keeps its own result)
                    </label>
                  )}

                  {expertMode && (
                    <div>
                      <Label htmlFor="qp-threshold" className="mb-1 block text-sm font-normal" style={{ color: 'var(--text-muted)' }}>
                        hERG threshold override (expert)
                      </Label>
                      <p className="mb-2 text-xs" style={{ color: 'var(--text-faint)' }}>
                        The backend refuses the override unless the signed-in identity has the expert role.
                      </p>
                      <Input
                        id="qp-threshold"
                        className="h-9 max-w-[160px] font-mono text-sm"
                        style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)', color: 'var(--text)' }}
                        placeholder="e.g. 0.3"
                        inputMode="decimal"
                        value={thresholdHerg}
                        disabled={loading}
                        onChange={(e) => setThresholdHerg(e.target.value)}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {error && !(error.field && validation) && (
          <div
            role="alert"
            className="mb-6 rounded-lg px-4 py-3"
            style={{ backgroundColor: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.35)', color: 'var(--accent-red)' }}
          >
            {error.message}
          </div>
        )}

        <section aria-live="polite" className="space-y-4 pb-12">
          {compareResult ? (
            <div>
              <p className="mb-3 text-sm font-medium" style={{ color: 'var(--text)' }}>Comparison by model</p>
              <div className="grid gap-4 lg:grid-cols-2">
                {compareResult.comparisons.map((comparison) => <div key={`${comparison.endpoint}:${comparison.model_id}`} className="min-w-0 rounded-xl border p-3" style={{ borderColor: 'var(--border)' }}>
                  <p className="mb-2 font-mono text-xs" style={{ color: 'var(--accent-blue)' }}>{comparison.endpoint} · {comparison.model_id}</p>
                  <AnalysisPanel analysis={comparison.result} />
                </div>)}
              </div>
            </div>
          ) : batchResult ? (
            <>
              {batchResult.errors.length > 0 && (
                <div
                  className="rounded-xl border p-3 text-xs"
                  style={{ borderColor: 'rgba(239,68,68,0.35)', color: 'var(--accent-red)', backgroundColor: 'rgba(239,68,68,0.08)' }}
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
          ) : result ? (
            <AnalysisPanel analysis={result} />
          ) : null}

          {!loading && (
            <div className="rounded-xl" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-start gap-3">
                  <History className="mt-0.5 h-5 w-5 shrink-0" style={{ color: 'var(--accent-blue)' }} />
                  <div>
                    <h2 className="font-semibold" style={{ color: 'var(--text)' }}>
                      {hasResult ? 'This result is not saved' : 'Quick analysis keeps no history'}
                    </h2>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      Nothing on this page is stored. Open a session to keep analyses, ask the agent about them and get an audit trail.
                    </p>
                  </div>
                </div>
                <Button asChild variant="outline" className="shrink-0">
                  <Link to="/sessions">Open Sessions</Link>
                </Button>
              </div>
            </div>
          )}
        </section>
      </main>

      <Footer />
    </div>
  );
}
