import { ScanLine } from 'lucide-react';
import type { StructureRecognitionTextContent } from '../../lib/api/types';
import { Button } from '../ui/button';

export function StructureRecognitionCard({
  content,
  onUseSmiles,
}: {
  content: StructureRecognitionTextContent;
  /** Prefill only: the user retains the final edit and submit action. */
  onUseSmiles: (smiles: string) => void;
}) {
  const confidence = content.confidence === undefined ? null : Math.round(content.confidence * 100);

  return (
    <div
      className="my-2 max-w-lg rounded-xl border p-3"
      style={{ backgroundColor: 'var(--surface-alt)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center gap-2">
        <ScanLine className="h-4 w-4 shrink-0" style={{ color: 'var(--accent-blue)' }} />
        <h3 className="text-sm font-medium" style={{ color: 'var(--text)' }}>
          Đã nhận diện cấu trúc từ ảnh
        </h3>
      </div>
      <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
        SMILES nhận diện
      </p>
      <code className="mt-1 block break-all rounded bg-black/10 px-2 py-1 font-mono text-xs" style={{ color: 'var(--text)' }}>
        {content.smiles}
      </code>
      {content.canonical_smiles !== content.smiles && (
        <p className="mt-1 break-all font-mono text-xs" style={{ color: 'var(--text-faint)' }}>
          Canonical: {content.canonical_smiles}
        </p>
      )}
      <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
        {confidence === null
          ? 'Dịch vụ OCR không cung cấp độ tin cậy cho lần nhận diện này.'
          : `Độ tin cậy nhận diện: ${confidence}%`}
      </p>
      <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
        Độ tin cậy này chỉ phản ánh nhận diện ảnh, không phải độ tin cậy dự đoán độc tính hay đánh giá an toàn.
      </p>
      <Button size="sm" variant="outline" className="mt-3" onClick={() => onUseSmiles(content.smiles)}>
        Chỉnh sửa SMILES để phân tích mới
      </Button>
    </div>
  );
}
