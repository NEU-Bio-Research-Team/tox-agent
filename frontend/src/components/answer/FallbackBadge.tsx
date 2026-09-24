import { AlertOctagon } from 'lucide-react';

/** N-6: is_fallback means the model failed twice and ToxAgent built a
 * deterministic answer instead. That's product information, not an internal
 * detail — shown at both the top and the bottom of the answer. */
export function FallbackBadge() {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-semibold"
      style={{ borderColor: 'var(--accent-yellow)', color: 'var(--accent-yellow)' }}
    >
      <AlertOctagon className="h-3 w-3" />
      ĐÁP ÁN DỰ PHÒNG
    </span>
  );
}
