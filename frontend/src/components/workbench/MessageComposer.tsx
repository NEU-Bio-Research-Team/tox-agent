import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { Gauge, Hash, ImageUp, PenTool, Plus, Send, X } from 'lucide-react';
import { Textarea } from '../ui/textarea';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover';
import { Label } from '../ui/label';
import { ImageUploadDialog, type StagedImage } from './ImageUploadDialog';
import type { SendMessageInput } from '../../lib/api/endpoints';
import { getDraft, getExpertModeEnabled, setDraft } from '../../lib/preferences';
import { looksLikeSmiles, suggestMolecule } from '../../lib/smiles';

export { looksLikeSmiles };

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

export interface AnalysisContext {
  analysisId: string;
  label: string;
}

export interface SmilesPrefill {
  smiles: string;
  /** A monotonic signal permits the user to apply the same SMILES twice. */
  signal: number;
}

export interface TextPrefill {
  text: string;
  /** Same role as SmilesPrefill.signal. */
  signal: number;
}

/**
 * Text, one attach menu, send.
 *
 * The composer used to expose the router's own choices — an intent select, an
 * explanation-mode radio group, and an advanced popover of endpoints and Tox21
 * assays. Each had a default that was right for nearly every message (`auto`,
 * `on_demand`, the deployment's default endpoints), and the one combination
 * that was not — a required Tox21 explanation with no assay — could only be
 * reached through them. The router decides the intent; an explanation is
 * something the user asks for in words (I03).
 */
export function MessageComposer({
  sessionId,
  hasActiveAnalysis,
  disabled,
  focusSmilesSignal,
  smilesPrefill,
  textPrefill,
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
  /** Bumped by the parent to open and focus the SMILES field — e.g. when a
   * "Nhập SMILES" clarification button is pressed. */
  focusSmilesSignal?: number;
  /** Recognition/edit actions can fill the field, but never submit it. */
  smilesPrefill?: SmilesPrefill;
  /** An example prompt picked from the empty state. Fills, never submits. */
  textPrefill?: TextPrefill;
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
  const [smilesFieldOpen, setSmilesFieldOpen] = useState(false);
  const [smilesFocusRequest, setSmilesFocusRequest] = useState(0);
  // A molecule detected in the text that the user removed from the chip: the
  // message then goes as text only.
  const [dismissedDetection, setDismissedDetection] = useState<string | null>(null);
  // Some input methods use Enter to accept a candidate. Without this the
  // composer submitted a half-typed draft mid-composition (I21).
  const [composing, setComposing] = useState(false);
  const [thresholdHerg, setThresholdHerg] = useState('');
  const [clientMessageId, setClientMessageId] = useState(() => crypto.randomUUID());
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const [drawDialogOpen, setDrawDialogOpen] = useState(false);
  const [imageDialogOpen, setImageDialogOpen] = useState(false);
  const [stagedImage, setStagedImage] = useState<StagedImage | null>(null);
  const expertMode = getExpertModeEnabled();
  const smilesInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const setText = (next: string) => {
    setTextState(next);
    setDraft(sessionId, next);
  };

  const openSmilesField = () => {
    setSmilesFieldOpen(true);
    setSmilesFocusRequest((n) => n + 1);
  };

  // Runs after the render that mounted the field, so the ref is set.
  useEffect(() => {
    if (smilesFocusRequest > 0) smilesInputRef.current?.focus();
  }, [smilesFocusRequest]);

  useEffect(() => {
    if (focusSmilesSignal !== undefined) openSmilesField();
    // Only the signal changing should trigger a focus, not every render.
     
  }, [focusSmilesSignal]);

  useEffect(() => {
    if (!smilesPrefill) return;
    setSmiles(smilesPrefill.smiles);
    openSmilesField();
  }, [smilesPrefill]);

  useEffect(() => {
    if (!textPrefill) return;
    setText(textPrefill.text);
    textareaRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [textPrefill]);

  // Releases the object URL backing whichever image is staged when the
  // composer unmounts (e.g. the user switches sessions) without sending or
  // explicitly removing it — the object URL manager doesn't do this itself.
  useEffect(() => {
    return () => {
      if (stagedImage) URL.revokeObjectURL(stagedImage.previewUrl);
    };
     
  }, [stagedImage]);

  // The molecule the composer will send, and the text it will keep alongside
  // it. Both, always (I04): replacing a question with the molecule it mentions
  // is what produced `research_subject_missing` and silent text loss.
  const suggestion = suggestMolecule(text);
  const trimmedSmilesField = smiles.trim();
  const detectedSmiles = suggestion.smiles && suggestion.smiles !== dismissedDetection ? suggestion.smiles : '';
  const effectiveSmiles = trimmedSmilesField || detectedSmiles;
  // Ambiguity is a question for the user, not a coin flip. Picking one of the
  // candidates fills the SMILES field, which clears the block.
  const ambiguousMolecule = !trimmedSmilesField && suggestion.candidates.length > 1;
  const smilesFieldVisible = smilesFieldOpen || trimmedSmilesField.length > 0;

  const hasSomethingToSend =
    text.trim().length > 0 || trimmedSmilesField.length > 0 || stagedImage !== null;
  const canSend = !disabled && !ambiguousMolecule && hasSomethingToSend;

  const clearStagedImage = () => {
    if (stagedImage) URL.revokeObjectURL(stagedImage.previewUrl);
    setStagedImage(null);
  };

  const closeSmilesField = () => {
    setSmiles('');
    setSmilesFieldOpen(false);
  };

  const handleSend = async () => {
    if (!canSend) return;
    const trimmedText = text.trim();
    // The question survives the molecule. Only a message that is *nothing but*
    // a bare SMILES has no question left to keep — a sentence that mentions
    // one keeps both, which is what `research_subject_missing` needed and what
    // stops `hello` from silently becoming a molecule (I04).
    const effectiveText =
      !trimmedSmilesField && detectedSmiles && suggestion.isBareMolecule ? '' : trimmedText;

    const input: SendMessageInput = {
      client_message_id: clientMessageId,
      intent_hint: 'auto',
      content: effectiveText ? [{ type: 'text', text: effectiveText }] : undefined,
      molecule: effectiveSmiles ? { smiles: effectiveSmiles } : undefined,
      // Sent for an image too (I09). An image is an input, not a different
      // product. Endpoints are left to the deployment's defaults, which is
      // what the server applies when none are requested.
      analysis_options: effectiveSmiles || stagedImage
        ? {
            threshold_overrides:
              expertMode && thresholdHerg.trim() ? { herg: Number(thresholdHerg) } : null,
            explanation_mode: 'on_demand',
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
      closeSmilesField();
      setDismissedDetection(null);
      clearStagedImage();
      // A fresh id for the *next* message; a failed send above keeps this
      // one so a retry of unedited content reuses the same idempotency key
      // instead of risking a duplicate if the original request actually
      // landed and only the response was lost.
      setClientMessageId(crypto.randomUUID());
    }
  };

  const attachItem = (label: string, Icon: typeof Hash, onSelect: () => void) => (
    <button
      type="button"
      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-[var(--purple-50)]"
      onClick={() => {
        setAttachMenuOpen(false);
        onSelect();
      }}
    >
      <Icon className="h-4 w-4" style={{ color: 'var(--text-muted)' }} />
      {label}
    </button>
  );

  const chipStyle = { backgroundColor: 'var(--accent-blue-muted)', color: 'var(--accent-blue)', width: 'fit-content' } as const;

  return (
    <div className="ta-glass rounded-[var(--radius-floating)] border p-3 shadow-[var(--shadow-float)] transition-shadow focus-within:shadow-[0_0_0_3px_var(--purple-glow),var(--shadow-float)]" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--line-strong)' }}>
      {analysisContext && (
        <div className="mb-2 flex items-center gap-1.5 self-start rounded-full px-2.5 py-1 text-xs font-medium" style={chipStyle}>
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
        ref={textareaRef}
        placeholder={hasActiveAnalysis ? 'Hỏi về kết quả này…' : 'Nhập SMILES hoặc mô tả yêu cầu…'}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onCompositionStart={() => setComposing(true)}
        onCompositionEnd={() => setComposing(false)}
        onKeyDown={(event) => {
          // `nativeEvent.isComposing` covers browsers that fire keydown during
          // composition; the state flag covers those that do not set it. Enter
          // while composing belongs to the input method, not to us (I21).
          if (composing || event.nativeEvent.isComposing) return;
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            void handleSend();
          }
        }}
        rows={2}
        className="min-h-12 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0"
      />

      {(smilesFieldVisible || (detectedSmiles && !trimmedSmilesField) || ambiguousMolecule) && (
        <div className="mb-2 flex flex-wrap items-center gap-2">
          {smilesFieldVisible && (
            <div className="flex items-center gap-1 rounded-full py-0.5 pl-2.5 pr-1" style={chipStyle}>
              <Hash className="h-3 w-3 shrink-0" />
              <Input
                ref={smilesInputRef}
                aria-label="SMILES"
                placeholder="SMILES"
                value={smiles}
                onChange={(event) => setSmiles(event.target.value)}
                className="h-6 w-[220px] border-0 bg-transparent px-1 font-mono text-xs shadow-none focus-visible:ring-0"
              />
              <button type="button" onClick={closeSmilesField} aria-label="Bỏ SMILES" className="rounded-full p-0.5 hover:opacity-70">
                <X className="h-3 w-3" />
              </button>
            </div>
          )}
          {detectedSmiles && !trimmedSmilesField && (
            <div className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs" style={chipStyle}>
              <Hash className="h-3 w-3 shrink-0" />
              <span className="max-w-[260px] truncate font-mono">{detectedSmiles}</span>
              <button
                type="button"
                onClick={() => setDismissedDetection(detectedSmiles)}
                aria-label="Không phân tích chuỗi này như phân tử"
                className="rounded-full hover:opacity-70"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          )}
          {ambiguousMolecule && (
            <>
              <p role="status" className="text-[11px] leading-snug" style={{ color: 'var(--text-muted)' }}>
                Câu này có {suggestion.candidates.length} chuỗi giống SMILES. Chọn chuỗi cần phân tích:
              </p>
              {suggestion.candidates.map((candidate) => (
                <button
                  key={candidate}
                  type="button"
                  onClick={() => {
                    setSmiles(candidate);
                    setSmilesFieldOpen(true);
                  }}
                  className="rounded-full border px-2.5 py-0.5 font-mono text-xs hover:bg-[var(--purple-50)]"
                  style={{ borderColor: 'var(--line)' }}
                >
                  {candidate}
                </button>
              ))}
            </>
          )}
        </div>
      )}

      <div className="flex items-center gap-1.5 border-t pt-2" style={{ borderColor: 'var(--line)' }}>
        <Popover open={attachMenuOpen} onOpenChange={setAttachMenuOpen}>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Thêm SMILES, ảnh hoặc bản vẽ">
              <Plus className="h-4 w-4" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-48 p-1">
            {attachItem('SMILES', Hash, openSmilesField)}
            {attachItem('Ảnh', ImageUp, () => setImageDialogOpen(true))}
            {attachItem('Vẽ cấu trúc', PenTool, () => setDrawDialogOpen(true))}
          </PopoverContent>
        </Popover>

        {expertMode && (
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 gap-1 px-2 text-xs"
                aria-label="Ngưỡng hERG (chuyên gia)"
              >
                <Gauge className="h-3.5 w-3.5" />
                {thresholdHerg.trim() ? `hERG ${thresholdHerg.trim()}` : null}
              </Button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-64">
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
            </PopoverContent>
          </Popover>
        )}

        <Button
          onClick={() => void handleSend()}
          disabled={!canSend}
          variant="primary-gloss"
          size="icon-circle"
          className="ml-auto"
          aria-label="Gửi"
        >
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
              openSmilesField();
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
