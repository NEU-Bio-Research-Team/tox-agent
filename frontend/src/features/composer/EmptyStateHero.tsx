import { Activity, FlaskConical, MessageCircleQuestion, ScanSearch, ShieldCheck, ChevronRight } from 'lucide-react';

/** Example requests. Each is a sentence with exactly one molecule in it, so
 * the composer keeps the question and extracts the SMILES (I04). */
export const EMPTY_STATE_EXAMPLES = [
  'Phân tích aspirin CC(=O)Oc1ccccc1C(=O)O',
  'Caffeine CN1C=NC2=C1C(=O)N(C(=O)N2C)C có nguy cơ ức chế hERG không?',
  'CC(=O)Nc1ccc(O)cc1 có tín hiệu Tox21 nào đáng chú ý?',
];

const EXAMPLE_PRESENTATION = [
  { icon: FlaskConical, caption: 'Phân tích một phân tử' },
  { icon: MessageCircleQuestion, caption: 'Hỏi về một nguy cơ cụ thể' },
  { icon: Activity, caption: 'Tìm tín hiệu cần chú ý' },
] as const;

const HOW_IT_WORKS = [
  { icon: MessageCircleQuestion, label: 'Bạn hỏi' },
  { icon: ScanSearch, label: 'Mô hình đo' },
  { icon: ShieldCheck, label: 'Đối chiếu nguồn' },
] as const;

/**
 * A greeting, three examples to start from, and a one-line picture of what
 * happens to a question. The input methods (SMILES, image, drawing) live in the
 * composer's attach menu; repeating them here as cards gave the same three
 * actions two entry points.
 */
export function EmptyStateHero({ onPickExample }: { onPickExample: (text: string) => void }) {
  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-7 px-4 py-10 text-center">
      <div className="flex max-w-xl flex-col items-center gap-3">
        <h2 className="text-3xl font-semibold tracking-tight text-ink">Bạn muốn phân tích gì?</h2>
        <p className="text-sm text-ink-secondary md:text-base">
          Hỏi bằng lời thường, kèm công thức SMILES nếu có. Dùng nút + để thêm ảnh hay vẽ cấu trúc.
        </p>
      </div>

      <ul className="flex w-full max-w-xl flex-col gap-2.5" aria-label="Ví dụ">
        {EMPTY_STATE_EXAMPLES.map((example, index) => {
          const { icon: Icon, caption } = EXAMPLE_PRESENTATION[index] ?? EXAMPLE_PRESENTATION[0];
          return (
            <li key={example}>
              <button
                type="button"
                onClick={() => onPickExample(example)}
                className="flex w-full items-center gap-3.5 rounded-xl border border-line bg-surface-solid px-4 py-3 text-left transition-colors hover:border-brand-border hover:bg-brand-soft"
              >
                <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-brand-soft-strong text-brand-ink" aria-hidden="true">
                  <Icon className="size-4" />
                </span>
                <span className="min-w-0">
                  <span className="block text-sm text-ink">{example}</span>
                  <span className="mt-0.5 block text-xs text-ink-tertiary">{caption}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>

      <ol className="flex flex-wrap items-center justify-center gap-x-2 gap-y-2 text-xs text-ink-secondary" aria-label="Cách ToxAgent trả lời">
        {HOW_IT_WORKS.map((step, index) => (
          <li key={step.label} className="flex items-center gap-2">
            {index > 0 ? <ChevronRight className="size-3.5 text-ink-tertiary" aria-hidden="true" /> : null}
            <span className="flex items-center gap-1.5 rounded-full border border-line bg-canvas-subtle px-2.5 py-1">
              <step.icon className="size-3.5 text-brand-ink" aria-hidden="true" />
              {step.label}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
