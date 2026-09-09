import { Hash, ImageUp, PenTool } from 'lucide-react';

interface InputOption {
  key: 'smiles' | 'image' | 'draw';
  icon: typeof Hash;
  title: string;
  description: string;
  badge?: string;
}

const OPTIONS: InputOption[] = [
  {
    key: 'smiles',
    icon: Hash,
    title: 'Nhập SMILES',
    description: 'Dán chuỗi SMILES để phân tích hERG, Tox21 và ClinTox.',
  },
  {
    key: 'image',
    icon: ImageUp,
    title: 'Tải ảnh cấu trúc',
    description: 'Tải ảnh cấu trúc hoá học lên để nhận diện.',
  },
  {
    key: 'draw',
    icon: PenTool,
    title: 'Vẽ cấu trúc',
    description: 'Vẽ cấu trúc 2D bằng công cụ có sẵn và chuyển thành SMILES.',
  },
];

export function EmptyStateHero({
  onPickSmiles,
  onPickImage,
  onPickDraw,
}: {
  onPickSmiles: () => void;
  onPickImage: () => void;
  onPickDraw: () => void;
}) {
  const handlers: Record<InputOption['key'], () => void> = {
    smiles: onPickSmiles,
    image: onPickImage,
    draw: onPickDraw,
  };

  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-7 px-4 py-10 text-center">
      <div className="flex max-w-xl flex-col items-center gap-3">
        <h1 className="text-[32px] font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
          Bạn muốn phân tích gì?
        </h1>
        <p className="text-sm md:text-base" style={{ color: 'var(--ink-secondary)' }}>
          Dán SMILES, tải ảnh cấu trúc hoặc vẽ phân tử để bắt đầu một phân tích có thể kiểm tra.
        </p>
      </div>

      <div className="flex w-full max-w-xl flex-wrap justify-center gap-2">
        {OPTIONS.map((option) => (
          <button
            key={option.key}
            type="button"
            onClick={handlers[option.key]}
            className="group flex items-center gap-2 rounded-full border px-4 py-2.5 text-left transition-colors hover:bg-[var(--purple-50)]"
            style={{ backgroundColor: 'var(--surface-solid)', borderColor: 'var(--line)' }}
          >
            <span
              className="flex h-7 w-7 items-center justify-center rounded-full"
              style={{ backgroundColor: 'var(--purple-100)' }}
            >
              <option.icon className="h-4 w-4" style={{ color: 'var(--purple-600)' }} />
            </span>
            <span className="flex items-center gap-1.5 text-sm font-medium" style={{ color: 'var(--ink)' }}>
              {option.title}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
