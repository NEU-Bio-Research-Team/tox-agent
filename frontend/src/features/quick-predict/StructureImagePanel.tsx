import { useEffect, useRef, useState, type DragEvent } from 'react';
import { AlertCircle, CheckCircle2, ImagePlus, Loader2, UploadCloud } from 'lucide-react';
import { Button } from '../../shared/ui/button';
import { Callout } from '../../shared/ui/callout';
import { Input } from '../../shared/ui/input';
import { cn } from '../../shared/ui/utils';
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
      setError('Chỉ hỗ trợ ảnh PNG, JPEG hoặc WebP.');
      return;
    }
    if (candidate.size > MAX_IMAGE_BYTES) {
      setError(`Ảnh vượt giới hạn ${(MAX_IMAGE_BYTES / 1_000_000).toFixed(0)}MB.`);
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
      setError(err instanceof ApiError ? errorMessageVi(err.code, err.message) : 'Không nhận diện được ảnh.');
    } finally {
      setExtracting(false);
    }
  };

  return (
    <div className="space-y-3">
      {!available && (
        <Callout tone="neutral">Triển khai này chưa bật nhận diện cấu trúc từ ảnh. Hãy nhập SMILES hoặc vẽ phân tử.</Callout>
      )}

      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!locked) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={cn(
          'rounded-xl border-2 border-dashed p-6 transition-colors',
          dragActive ? 'border-brand bg-brand-soft' : 'border-line-strong bg-canvas-subtle',
          !available && 'opacity-60',
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="hidden"
          aria-label="Ảnh cấu trúc"
          disabled={locked}
          onChange={(event) => {
            choose(event.target.files?.[0]);
            event.target.value = '';
          }}
        />
        <div className="flex flex-col items-center gap-2 text-center">
          <UploadCloud className="size-8 text-link" />
          <p className="text-sm font-medium text-ink">Kéo thả ảnh phân tử vào đây</p>
          <p className="text-xs text-ink-secondary">PNG, JPG hoặc WebP, tối đa 5MB</p>
          <Button type="button" variant="outline" onClick={() => inputRef.current?.click()} disabled={locked}>
            <ImagePlus />
            Chọn ảnh
          </Button>
        </div>
      </div>

      {previewUrl && (
        <div className="overflow-hidden rounded-xl border border-line">
          <img src={previewUrl} alt="Ảnh phân tử đã tải lên" className="max-h-[280px] w-full bg-depiction object-contain" />
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button type="button" onClick={() => void extract()} disabled={!file || locked}>
          {extracting ? (
            <>
              <Loader2 className="animate-spin" />
              Đang nhận diện…
            </>
          ) : (
            <>
              <CheckCircle2 />
              Trích SMILES từ ảnh
            </>
          )}
        </Button>
        {file && (
          <p className="min-w-0 truncate text-xs text-ink-secondary">{file.name}</p>
        )}
      </div>

      {error && (
        <Callout tone="danger" role="alert" icon={<AlertCircle />}>{error}</Callout>
      )}

      {recognized && (
        <div className="space-y-2 rounded-lg border border-line px-4 py-3 text-sm text-ink">
          <div className="flex items-center gap-2 font-medium">
            <CheckCircle2 className="size-4 text-success" />
            Đã trích SMILES
          </div>
          <p className="text-xs text-ink-secondary">
            {recognized.confidence != null
              ? `Độ tin cậy: ${(recognized.confidence * 100).toFixed(1)}%.`
              : 'Bộ nhận diện không báo độ tin cậy.'}{' '}
            Kiểm tra cấu trúc và sửa SMILES nếu cần. Nhận diện chỉ là một phỏng đoán.
          </p>
          <Input
            className="font-mono text-xs"
            value={smiles}
            onChange={(event) => onSmilesChange(event.target.value)}
            aria-label="SMILES nhận diện được (có thể sửa)"
            disabled={disabled}
            spellCheck={false}
          />
        </div>
      )}
    </div>
  );
}
