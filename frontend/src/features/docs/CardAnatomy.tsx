import { StatusPill } from '../../shared/ui/status-pill';

function Marker({ n }: { n: number }) {
  return (
    <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-brand-ink text-2xs font-bold text-canvas" aria-hidden="true">
      {n}
    </span>
  );
}

const PARTS = [
  { title: 'Phép đo.', body: 'Tên và điều nó đo. Mỗi thẻ chỉ nói về đúng một phép đo.' },
  { title: 'Nhãn.', body: 'Lớp mô hình gán, theo ngưỡng ở mục 4. Đây không phải kết luận về độc tính.' },
  { title: 'Xác suất.', body: 'Đầu ra của mô hình, chưa được hiệu chuẩn. Dùng để so sánh với ngưỡng.' },
  { title: 'Ngưỡng và mô hình.', body: 'Ngưỡng đã áp dụng và mô hình đã tạo ra con số.' },
] as const;

/** The annotated figure of a result card in "Đọc kết quả". It is a drawing with
 * example values, not a live result, so it carries no data from the predictor. */
export function CardAnatomy() {
  return (
    <figure className="not-prose my-2 flex flex-wrap items-start gap-5 rounded-xl border border-line bg-canvas-subtle p-5">
      <div className="min-w-[280px] flex-[1_1_300px] space-y-3 rounded-xl border border-line bg-card p-4" aria-label="Ví dụ một thẻ kết quả">
        <div className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2">
            <Marker n={1} />
            <span className="text-15 font-semibold text-ink">hERG</span>
          </span>
          <span className="flex items-center gap-1.5">
            <StatusPill tone="warning">blocker</StatusPill>
            <Marker n={2} />
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-3xl font-bold leading-none tracking-tight text-ink tabular-nums">73.00%</span>
          <Marker n={3} />
        </div>
        <div className="relative h-2 rounded-full bg-line">
          <div className="h-full w-[73%] rounded-full bg-brand" />
          <div className="absolute -top-1 left-1/2 h-4 w-0.5 rounded-sm bg-ink" />
        </div>
        <p className="flex items-center gap-2 font-mono text-xs text-ink-secondary">
          ngưỡng 0.5 · model m <Marker n={4} />
        </p>
      </div>
      <ol className="m-0 min-w-[240px] flex-[1_1_260px] list-none space-y-3 p-0 text-sm leading-relaxed text-ink">
        {PARTS.map((part, index) => (
          <li key={part.title} className="flex gap-2.5">
            <span className="mt-0.5"><Marker n={index + 1} /></span>
            <span>
              <strong className="font-semibold">{part.title}</strong> {part.body}
            </span>
          </li>
        ))}
      </ol>
    </figure>
  );
}
