import type { ComponentType } from 'react';
import { Hash, ImageUp, PenTool } from 'lucide-react';
import logoImage from '../../assets/logo-tox.png';
import { useRotatingText } from '../../hooks/useRotatingText';

const TAGLINES = [
  'Đưa một SMILES, nhận về bằng chứng — không phải một con số duy nhất.',
  'hERG, Tox21, ClinTox — luôn tách biệt, luôn kèm nguồn và ngưỡng.',
  'Mọi claim trong câu trả lời đều trỏ về đúng một observation kiểm chứng được.',
  'Vẽ cấu trúc hoặc dán SMILES — bắt đầu phân tích trong vài giây.',
] as const;

interface InputOption {
  key: 'smiles' | 'image' | 'draw';
  icon: ComponentType<{ className?: string; style?: React.CSSProperties }>;
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
    badge: 'Sắp ra mắt',
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
  const { text, index } = useRotatingText(TAGLINES, 4500);

  const handlers: Record<InputOption['key'], () => void> = {
    smiles: onPickSmiles,
    image: onPickImage,
    draw: onPickDraw,
  };

  return (
    <div className="flex h-full min-h-[520px] flex-col items-center justify-center gap-10 px-4 py-10 text-center">
      <div className="flex flex-col items-center gap-4">
        <img src={logoImage} alt="" className="h-16 w-16" />
        <h1 className="text-4xl font-bold tracking-tight md:text-5xl" style={{ color: 'var(--text)' }}>
          ToxAgent
        </h1>
        <p
          key={index}
          className="animate-in fade-in slide-in-from-bottom-1 max-w-lg text-sm duration-500 md:text-base"
          style={{ color: 'var(--text-muted)' }}
        >
          {text}
        </p>
      </div>

      <div className="grid w-full max-w-3xl gap-4 sm:grid-cols-3">
        {OPTIONS.map((option) => (
          <button
            key={option.key}
            type="button"
            onClick={handlers[option.key]}
            className="group flex flex-col items-center gap-3 rounded-2xl border p-6 text-center transition-colors hover:shadow-sm"
            style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
            onMouseEnter={(event) => (event.currentTarget.style.borderColor = 'var(--accent-blue)')}
            onMouseLeave={(event) => (event.currentTarget.style.borderColor = 'var(--border)')}
          >
            <span
              className="flex h-12 w-12 items-center justify-center rounded-full"
              style={{ backgroundColor: 'var(--accent-blue-muted)' }}
            >
              <option.icon className="h-6 w-6" style={{ color: 'var(--accent-blue)' }} />
            </span>
            <span className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text)' }}>
              {option.title}
              {option.badge && (
                <span
                  className="rounded-full px-1.5 py-0.5 text-[10px] font-medium"
                  style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-faint)' }}
                >
                  {option.badge}
                </span>
              )}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
              {option.description}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
