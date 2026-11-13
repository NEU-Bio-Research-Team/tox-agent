import { CircleSlash } from 'lucide-react';

const REASON_VI: Record<string, string> = {
  clintox: 'ClinTox không được phục vụ — thiếu tokenizer cho checkpoint hiện tại.',
};

/** A measurement the deployment does not serve is shown as unavailable, not
 * hidden, and not drawn as a failure or as a negative result. */
export function EndpointUnavailableCard({ endpoint }: { endpoint: string }) {
  return (
    <article className="rounded-xl border border-dashed border-line-strong bg-surface-muted p-5">
      <div className="flex items-center gap-2">
        <CircleSlash className="size-4 text-ink-tertiary" aria-hidden="true" />
        <h3 className="text-sm font-semibold uppercase text-ink-secondary">{endpoint} — không khả dụng</h3>
      </div>
      <p className="mt-1 text-13 text-ink-secondary">
        {REASON_VI[endpoint] ?? 'Endpoint này hiện không được phục vụ cho phân tử này.'}
      </p>
    </article>
  );
}
