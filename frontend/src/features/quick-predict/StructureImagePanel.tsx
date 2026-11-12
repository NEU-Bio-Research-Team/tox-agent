import { useEffect, useRef, useState, type DragEvent } from 'react';
import { AlertCircle, CheckCircle2, ImagePlus, Loader2, UploadCloud } from 'lucide-react';
import { Button } from '../../shared/ui/button';
import { Input } from '../../shared/ui/input';
import { recognizeStructure } from '../../shared/api/endpoints';
import { ApiError, type RecognizedStructure } from '../../shared/api/types';
import { errorMessageVi } from '../../shared/lib/labels';
import { MAX_IMAGE_BYTES, isAcceptedMimeType, type AcceptedMimeType } from '../composer/structureImage';

function readBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== 'string') return reject(new Error('unreadable'));
      resolve(result.slice(result.indexOf(',') + 1));
    };
    reader.onerror = () => reject(new Error('unreadable'));
    reader.readAsDataURL(file);
  });
}

/**
 * The "Upload Image" tab. The production layout — drop zone, preview, an
 * explicit extract step — over this control plane's recognition endpoint.
 *
 * Recognition is a guess, so its result lands in a field the user can correct
 * before analysing, with the confidence the recogniser reported beside it.
 * Whether recognition exists at all is a fact about the deployment, read from
 * the capability inventory by the caller; when it does not, the tab says so
 * instead of offering an upload that can only fail.
 */
export function StructureImagePanel({
  available,
  smiles,
  onSmilesChange,
  disabled = false,
}: {
  available: boolean;
  smiles: string;
  onSmilesChange: (smiles: string) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<(File & { type: AcceptedMimeType }) | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recognized, setRecognized] = useState<RecognizedStructure | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const locked = disabled || extracting || !available;

  const choose = (candidate: File | undefined) => {
    setError(null);
    setRecognized(null);
    if (!candidate) return;
    if (!isAcceptedMimeType(candidate.type)) {
      setError('Only PNG, JPEG or WebP images are supported.');
      return;
    }
    if (candidate.size > MAX_IMAGE_BYTES) {
      setError(`The image exceeds the ${(MAX_IMAGE_BYTES / 1_000_000).toFixed(0)}MB limit.`);
      return;
    }
    setFile(candidate as File & { type: AcceptedMimeType });
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    if (!locked) choose(event.dataTransfer.files?.[0]);
  };

  const extract = async () => {
    if (!file || locked) return;
    setExtracting(true);
    setError(null);
    setRecognized(null);
    try {
      const result = await recognizeStructure({ mime_type: file.type, data_base64: await readBase64(file) });
      setRecognized(result);
      onSmilesChange(result.canonical_smiles);
    } catch (err) {
      setError(err instanceof ApiError ? errorMessageVi(err.code, err.message) : 'The image could not be recognised.');
    } finally {
      setExtracting(false);
    }
  };

  return (
    <div className="space-y-3">
      {!available && (
        <div
          className="rounded-xl border p-4 text-sm"
          style={{ borderColor: 'var(--border)', color: 'var(--text-muted)', backgroundColor: 'var(--surface-alt)' }}
        >
          Structure recognition from images is not enabled on this deployment. Type a SMILES or draw the molecule
          instead.
        </div>
      )}

      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!locked) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className="rounded-xl border-2 border-dashed p-6 transition-colors"
        style={{
          borderColor: dragActive ? 'var(--accent-blue)' : 'var(--border)',
          backgroundColor: dragActive ? 'rgba(59,130,246,0.08)' : 'var(--surface-alt)',
          opacity: available ? 1 : 0.6,
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="hidden"
          aria-label="Structure image"
          disabled={locked}
          onChange={(event) => {
            choose(event.target.files?.[0]);
            event.target.value = '';
          }}
        />
        <div className="flex flex-col items-center gap-2 text-center">
          <UploadCloud className="h-8 w-8" style={{ color: 'var(--accent-blue)' }} />
          <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
            Drag and drop a molecule image here
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Allowed: PNG/JPG/JPEG/WebP, max 5MB
          </p>
          <Button type="button" variant="outline" onClick={() => inputRef.current?.click()} disabled={locked}>
            <ImagePlus className="mr-2 h-4 w-4" />
            Choose Image
          </Button>
        </div>
      </div>

      {previewUrl && (
        <div className="overflow-hidden rounded-xl border" style={{ borderColor: 'var(--border)' }}>
          <img src={previewUrl} alt="Molecule upload preview" className="max-h-[280px] w-full bg-white object-contain" />
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button type="button" onClick={() => void extract()} disabled={!file || locked}>
          {extracting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Extracting...
            </>
          ) : (
            <>
              <CheckCircle2 className="mr-2 h-4 w-4" />
              Extract SMILES from Image
            </>
          )}
        </Button>
        {file && (
          <p className="min-w-0 truncate text-xs" style={{ color: 'var(--text-muted)' }}>
            {file.name}
          </p>
        )}
      </div>

      {error && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-lg border px-4 py-3 text-sm"
          style={{ borderColor: 'rgba(239,68,68,0.35)', color: 'var(--accent-red)', backgroundColor: 'rgba(239,68,68,0.08)' }}
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {recognized && (
        <div className="space-y-2 rounded-lg border px-4 py-3 text-sm" style={{ borderColor: 'var(--border)', color: 'var(--text)' }}>
          <div className="flex items-center gap-2 font-medium">
            <CheckCircle2 className="h-4 w-4" style={{ color: 'var(--accent-green)' }} />
            Extraction Success
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {recognized.confidence != null
              ? `Confidence: ${(recognized.confidence * 100).toFixed(1)}%.`
              : 'The recogniser reported no confidence.'}{' '}
            Check the structure and correct the SMILES if needed.
          </p>
          <Input
            className="font-mono text-xs"
            value={smiles}
            onChange={(event) => onSmilesChange(event.target.value)}
            aria-label="Recognized SMILES (editable)"
            disabled={disabled}
            spellCheck={false}
          />
        </div>
      )}
    </div>
  );
}
