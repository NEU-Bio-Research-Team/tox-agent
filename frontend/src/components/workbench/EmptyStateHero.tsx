/** Example requests. Each is a sentence with exactly one molecule in it, so
 * the composer keeps the question and extracts the SMILES (I04). */
export const EMPTY_STATE_EXAMPLES = [
  'Phân tích aspirin CC(=O)Oc1ccccc1C(=O)O',
  'Caffeine CN1C=NC2=C1C(=O)N(C(=O)N2C)C có nguy cơ ức chế hERG không?',
  'CC(=O)Nc1ccc(O)cc1 có tín hiệu Tox21 nào đáng chú ý?',
];

/**
 * One line of guidance and a few example prompts. The input methods (SMILES,
 * image, drawing) live in the composer's attach menu; repeating them here as
 * cards gave the same three actions two entry points.
 */
export function EmptyStateHero({ onPickExample }: { onPickExample: (text: string) => void }) {
  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-6 px-4 py-10 text-center">
      <div className="flex max-w-xl flex-col items-center gap-3">
        <h1 className="text-[32px] font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
          Bạn muốn phân tích gì?
        </h1>
        <p className="text-sm md:text-base" style={{ color: 'var(--ink-secondary)' }}>
          Nhập câu hỏi kèm SMILES, hoặc dùng nút + để thêm ảnh hay vẽ cấu trúc.
        </p>
      </div>

      <ul className="flex w-full max-w-xl flex-col gap-2" aria-label="Ví dụ">
        {EMPTY_STATE_EXAMPLES.map((example) => (
          <li key={example}>
            <button
              type="button"
              onClick={() => onPickExample(example)}
              className="w-full rounded-xl border px-4 py-2.5 text-left text-sm transition-colors hover:bg-[var(--purple-50)]"
              style={{ backgroundColor: 'var(--surface-solid)', borderColor: 'var(--line)', color: 'var(--ink)' }}
            >
              {example}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
