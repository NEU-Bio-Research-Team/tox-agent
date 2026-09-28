import type { ReportExplanation } from '../../../../shared/api/types';

/** The palette from `toxpred/application/depiction.py`. Duplicated rather than
 * fetched: the legend has to describe the colours the *stored* figure was drawn
 * with, and the figure carries its palette version in its provenance so a
 * mismatch is auditable. */
const POSITIVE = '#D73027';
const NEGATIVE = '#1A9850';
const NEUTRAL = '#BDBDBD';

/** Signed contribution as text, always with its sign. Colour is one channel and
 * a substantial minority of readers cannot use this particular pair. */
function signed(value: unknown): string {
  const number = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(number)) return '—';
  return `${number > 0 ? '+' : ''}${number.toFixed(4)}`;
}

function targetPhrase(explanation: ReportExplanation): string {
  return explanation.task ? `${explanation.task} active` : explanation.endpoint;
}

/**
 * XAI-02's accessibility half: the legend and the contributor table.
 *
 * The figure now encodes direction as hue, which is the right primary channel
 * and the wrong only channel. So the same two facts appear here three more
 * times: named in the legend ("increases"/"decreases", with the target spelled
 * out rather than left as "positive"), signed in the table, and ordered by
 * magnitude so the strongest contributor is first in both directions.
 *
 * `unmapped_importance` is shown even at zero. "None of the attribution mass
 * fell outside the structure" and "the explainer did not tell us" are different
 * facts, and an explanation narrative is tempted to omit both.
 */
export function ExplanationLegend({ explanation }: { explanation: ReportExplanation }) {
  const target = targetPhrase(explanation);
  const positive = explanation.extracted_highlights?.positive_contributors ?? [];
  const negative = explanation.extracted_highlights?.negative_contributors ?? [];
  const unmapped = explanation.extracted_highlights?.unmapped_importance;

  return (
    <div className="space-y-3">
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs" style={{ color: 'var(--text-muted)' }}>
        {[
          { color: POSITIVE, label: `+ tăng ${target}` },
          { color: NEGATIVE, label: `− giảm ${target}` },
          { color: NEUTRAL, label: 'gần bằng 0' },
        ].map((entry) => (
          <li key={entry.label} className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block h-3 w-3 rounded-sm border"
              style={{ backgroundColor: entry.color, borderColor: 'var(--border-subtle)' }}
            />
            {entry.label}
          </li>
        ))}
      </ul>

      {(positive.length > 0 || negative.length > 0) && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[380px] border-collapse text-xs">
            <caption className="pb-1 text-left" style={{ color: 'var(--text-muted)' }}>
              Đóng góp mạnh nhất theo từng chiều. Đây là hành vi của model, không
              phải cơ chế hoá học.
            </caption>
            <thead>
              <tr className="text-left">
                <th className="border-b py-1 pr-3 font-medium">Chiều</th>
                <th className="border-b py-1 pr-3 font-medium">Atom</th>
                <th className="border-b py-1 pr-3 font-medium">Chỉ số</th>
                <th className="border-b py-1 font-medium">Đóng góp có dấu</th>
              </tr>
            </thead>
            <tbody>
              {[
                ...positive.map((c) => ({ contributor: c, direction: '+' as const })),
                ...negative.map((c) => ({ contributor: c, direction: '−' as const })),
              ].map(({ contributor, direction }, index) => (
                <tr key={`${direction}-${String(contributor.atom_index)}-${index}`}>
                  <td className="border-b py-1 pr-3">
                    <span
                      aria-hidden
                      className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm"
                      style={{ backgroundColor: direction === '+' ? POSITIVE : NEGATIVE }}
                    />
                    {direction === '+' ? `tăng ${target}` : `giảm ${target}`}
                  </td>
                  <td className="border-b py-1 pr-3">{String(contributor.symbol ?? '?')}</td>
                  <td className="border-b py-1 pr-3 tabular-nums">
                    {String(contributor.atom_index ?? '—')}
                  </td>
                  <td className="border-b py-1 tabular-nums">
                    {signed(contributor.signed_contribution)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {unmapped === null || unmapped === undefined
          ? 'Explainer không báo phần đóng góp không gán được vào cấu trúc.'
          : `Phần đóng góp không gán được vào cấu trúc: ${(unmapped * 100).toFixed(1)}%.`}
        {explanation.method ? ` Phương pháp: ${explanation.method}.` : ''}
      </p>
    </div>
  );
}
