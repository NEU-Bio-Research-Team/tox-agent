import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Hash, ImageUp, PenTool, Send, Settings2, X } from 'lucide-react';
import { Textarea } from '../ui/textarea';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover';
import { Checkbox } from '../ui/checkbox';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { ImageUploadDialog, type StagedImage } from './ImageUploadDialog';
import type { Endpoint, IntentHint } from '../../lib/api/types';
import { quickPredictCapabilities, type SendMessageInput } from '../../lib/api/endpoints';
import { getDraft, getEndpointSelection, getExpertModeEnabled, setDraft, setEndpointSelection } from '../../lib/preferences';

// react-ocl pulls the large openchemlib editor bundle. The ordinary text/
// SMILES composer must not download it until the user explicitly opens the
// 2D drawing dialog (W5-14).
const StructureEditorDialog = lazy(async () => {
  const module = await import('./StructureEditorDialog');
  return { default: module.StructureEditorDialog };
});

function StructureEditorLoadingDialog() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="status" aria-live="polite">
      <div className="rounded-xl border p-4 text-sm shadow-lg" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
        Đang tải trình vẽ cấu trúc…
      </div>
    </div>
  );
}

const INTENT_OPTIONS: Array<{ value: IntentHint; label: string }> = [
  { value: 'auto', label: 'Tự động (router quyết định)' },
  { value: 'analyze', label: 'Phân tích phân tử' },
  { value: 'ask_report', label: 'Hỏi về báo cáo hiện tại' },
  { value: 'research_evidence', label: 'Tìm bằng chứng khoa học' },
  { value: 'request_attribution', label: 'Attribution' },
];

//: A bare, whitespace-free token made only of characters SMILES notation
// actually uses. Best-effort UX only — the predictor is the real validator.
const SMILES_LIKE = /^[A-Za-z0-9@+\-=#$:()[\]\\/%.]+$/;

export function looksLikeSmiles(text: string): boolean {
  return text.length > 0 && !/\s/.test(text) && /[A-Za-z]/.test(text) && SMILES_LIKE.test(text);
}

export interface AnalysisContext {
  analysisId: string;
  label: string;
}

export interface SmilesPrefill {
  smiles: string;
  /** A monotonic signal permits the user to apply the same SMILES twice. */
  signal: number;
}

export function MessageComposer({
  sessionId,
  hasActiveAnalysis,
  disabled,
  focusSmilesSignal,
  openDrawSignal,
  openImageSignal,
  smilesPrefill,
  structureRecognitionAvailable,
  analysisContext,
  onClearAnalysisContext,
  onSend,
}: {
  /** Section 7.6: draft persists per session, keyed by this id — switching
   * sessions must never leak one session's unsent text into another's box. */
  sessionId: string;
  hasActiveAnalysis: boolean;
  disabled: boolean;
  /** Bumped by the parent to imperatively focus the SMILES field — e.g. when
   * a "Nhập SMILES" clarification button is pressed. */
  focusSmilesSignal?: number;
  /** Bumped by the parent's empty-state hero cards to open the draw/image
   * dialogs from outside the composer, mirroring focusSmilesSignal. */
  openDrawSignal?: number;
  openImageSignal?: number;
  /** Recognition/edit actions can fill the field, but never submit it. */
  smilesPrefill?: SmilesPrefill;
  /** `GET /health/ready`'s `capabilities.structure_recognition` — a
   * deployment fact (is `TOXAGENT_OCR_URL` configured?), not a permanent
   * limitation, so the upload dialog's copy must not hardcode "unsupported". */
  structureRecognitionAvailable?: boolean;
  /** Section 8.2.1's "Hỏi về phân tích này": a non-active analysis the user
   * is explicitly targeting the next message at, shown as a removable chip
   * so it's clear this differs from whatever is `active_analysis` today. */
  analysisContext?: AnalysisContext | null;
  onClearAnalysisContext?: () => void;
  onSend: (input: SendMessageInput) => Promise<boolean>;
}) {
  const [text, setTextState] = useState(() => getDraft(sessionId));
  const [smiles, setSmiles] = useState('');
  const [intentHint, setIntentHint] = useState<IntentHint>('auto');
  const [endpoints, setEndpoints] = useState<Endpoint[]>(() => getEndpointSelection() ?? ['herg', 'tox21']);
  const [tox21Tasks, setTox21Tasks] = useState<string[]>([]);
  const [thresholdHerg, setThresholdHerg] = useState('');
  const [clientMessageId, setClientMessageId] = useState(() => crypto.randomUUID());
  const [drawDialogOpen, setDrawDialogOpen] = useState(false);
  const [imageDialogOpen, setImageDialogOpen] = useState(false);
  const [stagedImage, setStagedImage] = useState<StagedImage | null>(null);
  const expertMode = getExpertModeEnabled();
  const smilesInputRef = useRef<HTMLInputElement>(null);
  const capabilities = useQuery({ queryKey: ['predict-capabilities'], queryFn: quickPredictCapabilities, staleTime: 5 * 60_000 });
  const endpointCapabilities = capabilities.data?.endpoints ?? [];

  useEffect(() => {
    if (!capabilities.data) return;
    const enabled = new Set(capabilities.data.endpoints?.filter((endpoint) => endpoint.enabled).map((endpoint) => endpoint.id) ?? capabilities.data.served_endpoints);
    setEndpoints((current) => {
      const compatible = current.filter((endpoint) => enabled.has(endpoint));
      const fallback = capabilities.data!.default_endpoints?.filter((endpoint) => enabled.has(endpoint)) ?? [];
      const next = compatible.length ? compatible : fallback;
      return next.length ? next : compatible;
    });
  }, [capabilities.data]);

  useEffect(() => { setEndpointSelection(endpoints); }, [endpoints]);

  const setText = (next: string) => {
    setTextState(next);
    setDraft(sessionId, next);
  };

  useEffect(() => {
    if (focusSmilesSignal !== undefined) smilesInputRef.current?.focus();
    // Only the signal changing should trigger a focus, not every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusSmilesSignal]);

  useEffect(() => {
    if (openDrawSignal !== undefined) setDrawDialogOpen(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openDrawSignal]);

  useEffect(() => {
    if (openImageSignal !== undefined) setImageDialogOpen(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openImageSignal]);

  useEffect(() => {
    if (!smilesPrefill) return;
    setSmiles(smilesPrefill.smiles);
    smilesInputRef.current?.focus();
  }, [smilesPrefill]);

  // Releases the object URL backing whichever image is staged when the
  // composer unmounts (e.g. the user switches sessions) without sending or
  // explicitly removing it — the object URL manager doesn't do this itself.
  useEffect(() => {
    return () => {
      if (stagedImage) URL.revokeObjectURL(stagedImage.previewUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stagedImage]);

  const wantsFreshSmiles = smiles.trim().length > 0 || looksLikeSmiles(text.trim());
  const needsTox21Target = wantsFreshSmiles && endpoints.includes('tox21') && tox21Tasks.length === 0;
  const canSend = !disabled && !needsTox21Target && (text.trim().length > 0 || smiles.trim().length > 0 || stagedImage !== null);

  const clearStagedImage = () => {
    if (stagedImage) URL.revokeObjectURL(stagedImage.previewUrl);
    setStagedImage(null);
  };

  const handleSend = async () => {
    if (!canSend) return;
    const trimmedText = text.trim();
    const trimmedSmiles = smiles.trim();
    // The main box's placeholder has always invited a bare SMILES; this is
    // what actually makes that true instead of it being routed as a chat
    // question the backend then can't find a molecule in.
    const autoDetectedSmiles = !trimmedSmiles && looksLikeSmiles(trimmedText);
    const effectiveSmiles = trimmedSmiles || (autoDetectedSmiles ? trimmedText : '');
    const effectiveText = autoDetectedSmiles ? '' : trimmedText;

    const input: SendMessageInput = {
      client_message_id: clientMessageId,
      intent_hint: intentHint,
      content: effectiveText ? [{ type: 'text', text: effectiveText }] : undefined,
      molecule: effectiveSmiles ? { smiles: effectiveSmiles } : undefined,
      analysis_options: effectiveSmiles
        ? {
            endpoints,
            threshold_overrides:
              expertMode && thresholdHerg.trim() ? { herg: Number(thresholdHerg) } : null,
            explanation_mode: effectiveSmiles ? 'required' : 'on_demand',
            explanation_targets: effectiveSmiles
              ? [
                  ...(endpoints.includes('herg') ? [{ endpoint: 'herg' as const }] : []),
                  ...tox21Tasks.map((task) => ({ endpoint: 'tox21' as const, task })),
                ]
              : [],
          }
        : undefined,
      // A new molecule in the same send always wins — the chip targets a
      // *different* analysis than whatever is active, and asking a fresh
      // question about a brand-new SMILES should never accidentally get
      // scoped to a stale one.
      analysis_id: !effectiveSmiles && analysisContext ? analysisContext.analysisId : undefined,
      image: stagedImage ? { mime_type: stagedImage.mimeType, data_base64: stagedImage.dataBase64 } : undefined,
    };
    const accepted = await onSend(input);
    if (accepted) {
      setText('');
      setSmiles('');
      clearStagedImage();
      // A fresh id for the *next* message; a failed send above keeps this
      // one so a retry of unedited content reuses the same idempotency key
      // instead of risking a duplicate if the original request actually
      // landed and only the response was lost.
      setClientMessageId(crypto.randomUUID());
    }
  };

  return (
    <div className="ta-glass rounded-[var(--radius-floating)] border p-3 shadow-[var(--shadow-float)] transition-shadow focus-within:shadow-[0_0_0_3px_var(--purple-glow),var(--shadow-float)]" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--line-strong)' }}>
      {analysisContext && (
        <div
          className="mb-2 flex items-center gap-1.5 self-start rounded-full px-2.5 py-1 text-xs font-medium"
          style={{ backgroundColor: 'var(--accent-blue-muted)', color: 'var(--accent-blue)', width: 'fit-content' }}
        >
          <span>Đang hỏi về {analysisContext.label}</span>
          {onClearAnalysisContext && (
            <button
              type="button"
              onClick={onClearAnalysisContext}
              aria-label="Bỏ ngữ cảnh phân tích, quay về analysis đang active"
              className="rounded-full hover:opacity-70"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      )}
      {stagedImage && (
        <div
          className="mb-2 flex items-center gap-2 self-start rounded-lg border p-1.5"
          style={{ borderColor: 'var(--border)', width: 'fit-content' }}
        >
          <img src={stagedImage.previewUrl} alt="" className="h-10 w-10 rounded object-cover" />
          <span className="max-w-[160px] truncate text-xs" style={{ color: 'var(--text-muted)' }}>
            {stagedImage.fileName}
          </span>
          <button
            type="button"
            onClick={clearStagedImage}
            aria-label="Bỏ ảnh đã chọn"
            className="rounded-full p-0.5 hover:opacity-70"
          >
            <X className="h-3.5 w-3.5" style={{ color: 'var(--text-faint)' }} />
          </button>
        </div>
      )}
      <Textarea
        placeholder={hasActiveAnalysis ? 'Hỏi về kết quả này…' : 'Nhập SMILES hoặc mô tả yêu cầu…'}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            void handleSend();
          }
        }}
        rows={2}
        className="min-h-12 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0"
      />
      <div className="flex flex-wrap items-center gap-1.5 border-t pt-2" style={{ borderColor: 'var(--line)' }}>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 px-2 text-xs"
          onClick={() => smilesInputRef.current?.focus()}
        >
          <Hash className="h-3.5 w-3.5" />
          SMILES
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 px-2 text-xs"
          onClick={() => setImageDialogOpen(true)}
        >
          <ImageUp className="h-3.5 w-3.5" />
          Ảnh
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 px-2 text-xs"
          onClick={() => setDrawDialogOpen(true)}
        >
          <PenTool className="h-3.5 w-3.5" />
          Vẽ cấu trúc
        </Button>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Input
          ref={smilesInputRef}
          placeholder="SMILES (tuỳ chọn)"
          value={smiles}
          onChange={(event) => setSmiles(event.target.value)}
          className="h-8 max-w-[220px] font-mono text-xs"
        />

        <Select value={intentHint} onValueChange={(value) => setIntentHint(value as IntentHint)}>
          <SelectTrigger className="h-8 w-[200px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {INTENT_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Tuỳ chọn nâng cao">
              <Settings2 className="h-4 w-4" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-72 space-y-3">
            <div>
              <p className="mb-1.5 text-xs font-medium" style={{ color: 'var(--text)' }}>
                Endpoint (khi có SMILES mới)
              </p>
              {endpointCapabilities.map((endpoint) => (
                <label key={endpoint.id} className="flex items-center gap-2 py-1 text-xs" title={endpoint.blocked_reason ?? undefined}>
                  <Checkbox
                    checked={endpoints.includes(endpoint.id)}
                    disabled={!endpoint.enabled}
                    onCheckedChange={(checked) =>
                      setEndpoints((prev) => {
                        const next = checked ? [...prev, endpoint.id] : prev.filter((item) => item !== endpoint.id);
                        return next.length ? next : prev;
                      })
                    }
                  />
                  <span className={endpoint.enabled ? '' : 'text-muted-foreground'}>{endpoint.display_name}</span>
                  {!endpoint.enabled && <span className="text-[10px] text-muted-foreground">không khả dụng</span>}
                </label>
              ))}
            </div>
            {endpoints.includes('tox21') && endpointCapabilities.find((endpoint) => endpoint.id === 'tox21')?.tasks.length ? (
              <div>
                <p className="mb-1 text-xs font-medium">Assay Tox21 để giải thích <span className="font-normal text-muted-foreground">(chọn ít nhất một)</span></p>
                <div className="grid grid-cols-2 gap-x-2">
                  {endpointCapabilities.find((endpoint) => endpoint.id === 'tox21')!.tasks.map((task) => (
                    <label key={task} className="flex items-center gap-1.5 py-0.5 text-xs">
                      <Checkbox checked={tox21Tasks.includes(task)} onCheckedChange={(checked) => setTox21Tasks((current) => checked ? [...current, task] : current.filter((item) => item !== task))} />
                      {task}
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
            {expertMode && (
              <div>
                <Label htmlFor="herg-threshold" className="text-xs">
                  hERG threshold override (expert — backend từ chối nếu token không có role expert)
                </Label>
                <Input
                  id="herg-threshold"
                  className="mt-1 h-8 text-xs"
                  placeholder="vd. 0.3"
                  value={thresholdHerg}
                  onChange={(event) => setThresholdHerg(event.target.value)}
                />
              </div>
            )}
          </PopoverContent>
        </Popover>

        <Button onClick={() => void handleSend()} disabled={!canSend} variant="primary-gloss" size="icon-circle" className="ml-auto" aria-label="Gửi">
          <Send className="h-3.5 w-3.5" />
        </Button>
      </div>

      {drawDialogOpen && (
        <Suspense fallback={<StructureEditorLoadingDialog />}>
          <StructureEditorDialog
            open={drawDialogOpen}
            onOpenChange={setDrawDialogOpen}
            onConfirm={(drawnSmiles) => {
              setSmiles(drawnSmiles);
              smilesInputRef.current?.focus();
            }}
          />
        </Suspense>
      )}
      <ImageUploadDialog
        open={imageDialogOpen}
        onOpenChange={setImageDialogOpen}
        available={structureRecognitionAvailable ?? false}
        onConfirm={(image) => {
          clearStagedImage();
          setStagedImage(image);
        }}
      />
    </div>
  );
}
