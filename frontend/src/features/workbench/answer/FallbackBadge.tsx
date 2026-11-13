import { Bot } from 'lucide-react';
import { StatusPill } from '../../../shared/ui/status-pill';

/** N-6: is_fallback means the model failed twice and ToxAgent built a
 * deterministic answer instead. That's product information, not an internal
 * detail — shown at both the top and the bottom of the answer, in plain words. */
export function FallbackBadge() {
  return (
    <StatusPill tone="warning">
      <Bot className="size-3" aria-hidden="true" />
      Bản tóm tắt tự động
    </StatusPill>
  );
}
