import { useRef, useState } from 'react';
import { ImageUp } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '../../shared/ui/dialog';
import { Button } from '../../shared/ui/button';
import { MAX_IMAGE_BYTES as MAX_BYTES, isAcceptedMimeType, type StagedImage } from './structureImage';

export type { StagedImage };

/**
 * Whether the upload actually gets recognised (toxocr/, ADR 0006) or answers
 * `capability_unavailable` is a deployment fact — `TOXAGENT_OCR_URL` set or
 * not — never something this dialog hardcodes. `available` comes from
 * `GET /health/ready`'s `capabilities.structure_recognition`, so its copy
 * stays honest either way instead of asserting a permanent limitation that
 * may not hold for this deployment.
 */
export function ImageUploadDialog({
  open,
  onOpenChange,
  available,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** See `capabilities.structure_recognition` above. */
  available: boolean;
  onConfirm: (image: StagedImage) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (file: File | undefined) => {
    setError(null);
    if (!file) return;
    const { type: mimeType } = file;
    if (!isAcceptedMimeType(mimeType)) {
      setError('Chỉ hỗ trợ ảnh PNG, JPEG hoặc WEBP.');
      return;
    }
    if (file.size > MAX_BYTES) {
      setError(`Ảnh vượt quá giới hạn ${(MAX_BYTES / 1_000_000).toFixed(0)}MB.`);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== 'string') {
        setError('Không đọc được tệp ảnh.');
        return;
      }
      onConfirm({
        mimeType,
        dataBase64: result.slice(result.indexOf(',') + 1),
        previewUrl: URL.createObjectURL(file),
        fileName: file.name,
        sizeBytes: file.size,
      });
      onOpenChange(false);
    };
    reader.onerror = () => setError('Không đọc được tệp ảnh.');
    reader.readAsDataURL(file);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Tải ảnh cấu trúc</DialogTitle>
          <DialogDescription>
            {available
              ? 'Ảnh sẽ được nhận diện thành SMILES rồi phân tích như một cấu trúc nhập tay.'
              : 'Bản này chưa bật nhận diện cấu trúc từ ảnh — ảnh vẫn được gửi kèm tin nhắn, nhưng ToxAgent sẽ trả ' +
                'lời là chưa hỗ trợ. Dùng SMILES hoặc vẽ cấu trúc để phân tích ngay.'}
          </DialogDescription>
        </DialogHeader>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed p-8 text-center transition-colors"
          style={{ borderColor: 'var(--line)' }}
        >
          <ImageUp className="h-8 w-8 text-ink-tertiary" />
          <span className="text-sm font-medium text-ink">
            Chọn ảnh cấu trúc
          </span>
          <span className="text-xs text-ink-tertiary">
            PNG, JPEG hoặc WEBP — tối đa 5MB
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="hidden"
          onChange={(event) => handleFile(event.target.files?.[0])}
        />

        {error && (
          <p className="text-xs text-danger">
            {error}
          </p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Huỷ
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
